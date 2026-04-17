#include "block_ops_batched.cuh"
#include <mma.h>
using namespace nvcuda;

// ============================================================
// Batched shifted-copy 9-GEMM 3x3 convolution.
//
// This is the MINIMAL correct extension of block_conv_3x3_shifted
// to batch B: an outer loop that processes one batch element per
// pass, using the same M-split (8 warps * 16 output channels)
// and 4-accumulator-per-warp structure as the existing kernel.
//
// Per pass (= per batch element):
//   - Build shifted[C_in, 64] FP16 in global scratch for each
//     of 9 shifts.
//   - Load weight slice W_s[C_out, C_in] FP16 into shared memory.
//   - GEMM: 8 warps, each 16-row M-slab, 4 accumulators across
//     4 N-tiles (covering all 64 spatial positions for this
//     batch element).
//   - Store 4 accumulators to act_out[b, :, :] FP32.
//
// WHY THIS DESIGN:
//   This convention (M = C_out, N = spatial) matches the existing
//   kernel exactly. The primary benefit measured by this
//   benchmark is kernel-launch overhead amortization (1 launch
//   for B positions vs. B launches). This is a CONSERVATIVE
//   baseline; Phase 4 may benefit further from (1) keeping more
//   accumulators alive per warp to amortize weight loads, or
//   (2) re-splitting warps across spatial*batch dimension to
//   improve utilization at small B. If this baseline clears the
//   go/no-go threshold, Phase 4 will too.
//
//   act_in and act_out are FP32 in global memory (matching the
//   existing kernel's activation precision). w_smem is one
//   [C_out, C_in] FP16 tile. shifted is [C_in, 64] FP16 in
//   global scratch (reused across passes).
// ============================================================

__device__ static const int KY_TABLE_B[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
__device__ static const int KX_TABLE_B[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

__device__ void block_conv_3x3_shifted_batched(
    const float* __restrict__ act_in,
    half* const*             W_s,
    float* __restrict__      act_out,
    half* __restrict__       w_smem,
    float* __restrict__      acc_smem,
    half* __restrict__       shifted,
    int C_in, int C_out, int B
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    const int K_tiles  = (C_in + 15) / 16;
    const bool a_direct = (C_in % 16 == 0);
    const int M_start  = warp_id * 16;
    const bool warp_has_M = (M_start < C_out);

    // Per-warp staging for A fragment boundary loads (C_in % 16 != 0).
    // Placed after the weight tile in the same smem region.
    half* a_staging = w_smem + (size_t)C_in * C_out + warp_id * 256;

    for (int pass = 0; pass < B; pass++) {
        // 4 accumulators live across 9 shifts within this pass.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
        #pragma unroll
        for (int t = 0; t < 4; t++) wmma::fill_fragment(acc[t], 0.0f);

        for (int s = 0; s < 9; s++) {
            const int ky = KY_TABLE_B[s];
            const int kx = KX_TABLE_B[s];
            const half* W_slice = W_s[s];

            // ----- Build shifted[C_in, 64] FP16 for this (pass, shift) -----
            __syncthreads();
            const int total_sh = C_in * 64;
            for (int i = tid; i < total_sh; i += 256) {
                int c = i >> 6;        // i / 64
                int n = i & 63;        // i % 64
                int oy = n >> 3;
                int ox = n & 7;
                int iy = oy + ky;
                int ix = ox + kx;
                half v = __float2half(0.0f);
                if ((unsigned)iy < 8u && (unsigned)ix < 8u) {
                    v = __float2half(act_in[pass * C_in * 64 + c * 64 + iy * 8 + ix]);
                }
                shifted[i] = v;
            }

            // ----- Load weight slice W_s[C_out, C_in] into shared memory -----
            const int total_w = C_in * C_out;
            for (int i = tid; i < total_w; i += 256) {
                w_smem[i] = W_slice[i];
            }
            __syncthreads();

            if (!warp_has_M) continue;

            // ----- GEMM: 4 N-tiles × K_tiles = per-warp tile reduction -----
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                int N_start = t * 16;

                for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    int K_start = k_tile * 16;

                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    if (a_direct && K_start + 16 <= C_in) {
                        wmma::load_matrix_sync(a_frag, w_smem + M_start * C_in + K_start, C_in);
                    } else {
                        for (int i = lane; i < 256; i += 32) {
                            int r = i / 16, c = i % 16;
                            int m_idx = M_start + r, k_idx = K_start + c;
                            half vw = __float2half(0.0f);
                            if (m_idx < C_out && k_idx < C_in) {
                                vw = w_smem[m_idx * C_in + k_idx];
                            }
                            a_staging[i] = vw;
                        }
                        __syncwarp();
                        wmma::load_matrix_sync(a_frag, a_staging, 16);
                    }

                    // shifted layout: [C_in, 64] row-major, leading dim 64.
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, shifted + K_start * 64 + N_start, 64);

                    wmma::mma_sync(acc[t], a_frag, b_frag, acc[t]);
                }
            }
        }

        // ----- Store 4 accumulators for this pass to act_out[pass, :, :] -----
        if (warp_has_M) {
            float* warp_staging = acc_smem + warp_id * 256;
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                int N_start = t * 16;

                wmma::store_matrix_sync(warp_staging, acc[t], 16, wmma::mem_row_major);
                __syncwarp();

                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16;      // output channel offset within tile (0..15)
                    int c = i % 16;      // N offset within tile (0..15)
                    int oc = M_start + r;
                    if (oc < C_out) {
                        act_out[pass * C_out * 64 + oc * 64 + N_start + c] = warp_staging[i];
                    }
                }
                __syncwarp();
            }
        }
    }

    __syncthreads();
}
