#pragma once

#include "common.cuh"
#include <cuda_fp16.h>

#ifdef __CUDACC__

#include <mma.h>

// ============================================================
// Batched shifted-copy 9-GEMM 3x3 convolution (Tensor Core)
//
// Two kernels are exposed here, both extending
// block_conv_3x3_shifted (block_ops.cu:334) to batch B > 1.
// They share memory layout — activations in global memory FP32,
// shifted buffer in global FP16, weight slice staged through
// shared memory — but differ in GEMM-fusion strategy:
//
//   Kernel 2 (baseline):
//     block_conv_3x3_shifted_batched
//     Outer loop = B. Each pass processes ONE batch element
//     with the existing 4-accumulator-per-warp layout. No
//     weight reuse across positions; per-position cost is
//     identical to the existing single-batch kernel. Measures
//     only kernel-launch overhead amortization.
//
//   Kernel 3 (chunked):
//     block_conv_3x3_shifted_batched_chunked<CHUNK>
//     Outer loop = B/CHUNK. Each pass processes CHUNK batch
//     elements simultaneously, holding 4*CHUNK accumulators
//     alive per warp across 9 shifts. Each weight slice is
//     loaded once per (pass, shift) and reused across all
//     CHUNK positions in the GEMM. Inner-loop order also
//     swapped (k_tile outer, n_tile inner) so a_frag is loaded
//     once per k_tile and reused across 4*CHUNK n_tiles. This
//     is the kernel whose measurements actually test the v2
//     Phase 4 thesis.
//
// Caller must launch with blockDim.x == 256 (8 warps).
// Assumes C_in and C_out multiples of 16. B >= 1.
// For the chunked kernel, B must be divisible by CHUNK.
// ============================================================

// Shift offset helpers: s in [0,9) → (ky, kx) in [-1,1]^2.
// Used by both kernels. Using arithmetic rather than a __device__
// table to avoid ODR issues across translation units.
__device__ __forceinline__ int conv3x3_ky(int s) { return (s / 3) - 1; }
__device__ __forceinline__ int conv3x3_kx(int s) { return (s % 3) - 1; }

// -------- Kernel 2: baseline batched (B passes, CHUNK=1 semantics) --------

__device__ void block_conv_3x3_shifted_batched(
    const float* __restrict__ act_in,   // [B, C_in, 8, 8] FP32 in global
    half* const*             W_s,       // 9 x [C_out, C_in] FP16 device ptrs
    float* __restrict__      act_out,   // [B, C_out, 8, 8] FP32 in global
    half* __restrict__       w_smem,    // shared: [C_out * C_in + 8*256] FP16 staging
    float* __restrict__      acc_smem,  // shared: 8 warps * 256 FP32
    half* __restrict__       shifted,   // [C_in, 64] FP16 in global scratch
    int C_in, int C_out, int B);

// -------- Kernel 3: chunked batched (B/CHUNK passes) --------
//
// Template-parameterized for compile-time CHUNK so the compiler can
// unroll the 4*CHUNK accumulator array and keep it in registers.
// Instantiate with small CHUNK values only — register pressure grows
// linearly with CHUNK. Budget: ~32*CHUNK regs/thread for accumulators
// alone, out of 255/thread limit. CHUNK=1,2,4 are safe; CHUNK=8 spills.

template<int CHUNK>
__device__ void block_conv_3x3_shifted_batched_chunked(
    const float* __restrict__ act_in,
    half* const*             W_s,
    float* __restrict__      act_out,
    half* __restrict__       w_smem,
    float* __restrict__      acc_smem,
    half* __restrict__       shifted,   // [C_in, 64*CHUNK] FP16 in global scratch
    int C_in, int C_out, int B
) {
    namespace wmma = nvcuda::wmma;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    const int K_tiles  = (C_in + 15) / 16;
    const bool a_direct = (C_in % 16 == 0);
    const int M_start  = warp_id * 16;
    const bool warp_has_M = (M_start < C_out);

    half* a_staging = w_smem + (size_t)C_in * C_out + warp_id * 256;

    // B must be divisible by CHUNK — caller's contract.
    const int n_passes = B / CHUNK;
    const int N_cols   = 64 * CHUNK;

    for (int pass = 0; pass < n_passes; pass++) {
        // 4*CHUNK accumulators live across 9 shifts within this pass.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4 * CHUNK];
        #pragma unroll
        for (int t = 0; t < 4 * CHUNK; t++) wmma::fill_fragment(acc[t], 0.0f);

        for (int s = 0; s < 9; s++) {
            const int ky = conv3x3_ky(s);
            const int kx = conv3x3_kx(s);
            const half* W_slice = W_s[s];

            // ----- Build shifted[C_in, 64*CHUNK] FP16 for (pass, shift) -----
            __syncthreads();
            const int total_sh = C_in * N_cols;
            for (int i = tid; i < total_sh; i += 256) {
                int c = i / N_cols;
                int r = i - c * N_cols;               // in [0, 64*CHUNK)
                int local_batch = r / 64;
                int n = r - local_batch * 64;         // spatial 0..63
                int oy = n >> 3;
                int ox = n & 7;
                int iy = oy + ky;
                int ix = ox + kx;
                int actual_batch = pass * CHUNK + local_batch;
                half v = __float2half(0.0f);
                if ((unsigned)iy < 8u && (unsigned)ix < 8u) {
                    v = __float2half(act_in[actual_batch * C_in * 64 + c * 64 + iy * 8 + ix]);
                }
                shifted[i] = v;
            }

            // ----- Load weight slice W_s[C_out, C_in] -> w_smem -----
            const int total_w = C_in * C_out;
            for (int i = tid; i < total_w; i += 256) {
                w_smem[i] = W_slice[i];
            }
            __syncthreads();

            if (!warp_has_M) continue;

            // ----- GEMM with k_tile OUTER, n_tile INNER -----
            // a_frag loaded once per k_tile, reused across 4*CHUNK n_tiles.
            // This is the arithmetic-intensity win over the baseline kernel.
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

                #pragma unroll
                for (int n = 0; n < 4 * CHUNK; n++) {
                    int N_start = n * 16;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, shifted + K_start * N_cols + N_start, N_cols);
                    wmma::mma_sync(acc[n], a_frag, b_frag, acc[n]);
                }
            }
        }

        // ----- Store 4*CHUNK accumulators to act_out -----
        if (warp_has_M) {
            float* warp_staging = acc_smem + warp_id * 256;
            #pragma unroll
            for (int n = 0; n < 4 * CHUNK; n++) {
                int N_start = n * 16;
                int local_batch = N_start / 64;
                int spatial_offset = N_start - local_batch * 64;  // multiple of 16
                int actual_batch = pass * CHUNK + local_batch;

                wmma::store_matrix_sync(warp_staging, acc[n], 16, wmma::mem_row_major);
                __syncwarp();

                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16;
                    int c = i % 16;
                    int oc = M_start + r;
                    if (oc < C_out) {
                        act_out[actual_batch * C_out * 64 + oc * 64 + spatial_offset + c]
                            = warp_staging[i];
                    }
                }
                __syncwarp();
            }
        }
    }

    __syncthreads();
}

// -------- Smem / scratch size helpers --------

__host__ __device__ inline size_t batched_conv_w_smem_bytes(int C_in, int C_out) {
    // Weight tile + 8 warps * 256-half a_staging.
    return ((size_t)C_in * C_out + 8 * 256) * sizeof(half);
}

__host__ __device__ inline size_t batched_conv_acc_smem_bytes() {
    return (size_t)8 * 256 * sizeof(float);
}

__host__ __device__ inline size_t batched_conv_shifted_bytes(int C_in, int /*B*/) {
    // Kernel 2 (baseline): shifted is [C_in, 64] independent of B.
    return (size_t)C_in * 64 * sizeof(half);
}

__host__ __device__ inline size_t batched_conv_shifted_bytes_chunked(int C_in, int CHUNK) {
    // Kernel 3 (chunked): shifted is [C_in, 64*CHUNK] per pass.
    return (size_t)C_in * 64 * CHUNK * sizeof(half);
}

#endif // __CUDACC__
