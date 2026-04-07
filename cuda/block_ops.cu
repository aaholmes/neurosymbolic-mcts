#include "block_ops.cuh"
#include <cfloat>
#include <mma.h>
using namespace nvcuda;

// All functions assume blockDim.x == 256 and all 256 threads participate.
// Work is striped: thread t owns elements t, t+256, t+512, ...

__device__ void block_bn_relu(
    float* data,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int channels,
    bool relu
) {
    int tid = threadIdx.x;
    int total = channels * 64;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int ch = idx / 64;
        float x = data[idx];
        float norm = (x - running_mean[ch]) * rsqrtf(running_var[ch] + 1e-5f);
        float result = norm * gamma[ch] + beta[ch];
        if (relu && result < 0.0f) result = 0.0f;
        data[idx] = result;
    }
    __syncthreads();
}

__device__ void block_bn_relu_1ch(
    float* data,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    bool relu
) {
    int tid = threadIdx.x;
    float g = gamma[0], b = beta[0], m = running_mean[0];
    float inv_std = rsqrtf(running_var[0] + 1e-5f);
    for (int idx = tid; idx < 64; idx += blockDim.x) {
        float x = data[idx];
        float result = (x - m) * inv_std * g + b;
        if (relu && result < 0.0f) result = 0.0f;
        data[idx] = result;
    }
    __syncthreads();
}

// Shared memory weight buffer for block_conv_3x3.
// Caller must provide smem_weights pointing to at least C_OUT * 9 floats
// in shared memory. For C_OUT=128: 1152 floats = 4608 bytes.
// This is provided by block_forward.cu's shared memory layout.
__device__ void block_conv_3x3_smem_w(
    const float* __restrict__ input_smem,
    const float* __restrict__ weights,
    float* __restrict__ output_smem,
    float* __restrict__ smem_weights,
    int C_in, int C_out
) {
    int tid = threadIdx.x;

    // Strategy: iterate over input channels. For each input channel:
    //   1. All 256 threads cooperatively load C_out*9 weights into shared memory
    //   2. Each thread computes its output elements using cached weights
    //   3. Accumulate partial results across input channels
    //
    // Weight load: C_out*9 = 1152 floats. 256 threads → 5 passes of 256 = 1280 slots.
    // We stripe: thread t loads weights[t], weights[t+256], ...
    //
    // Each thread owns C_out*64/256 = 32 output elements (for C_out=128).
    // These are unrolled as 32 scalar accumulators.

    // Precompute output element assignments
    int spat[32];
    int oy[32], ox[32];
    int out_ch[32];
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int idx = tid + k * 256;
        out_ch[k] = idx / 64;
        spat[k] = idx % 64;
        oy[k] = spat[k] / 8;
        ox[k] = spat[k] % 8;
    }

    // 32 accumulators
    float a[32];
    #pragma unroll
    for (int k = 0; k < 32; k++) a[k] = 0.0f;

    // Weight buffer in shared memory: smem_weights[out_ch * 9 + ki]
    // Total: C_out * 9 floats

    for (int c = 0; c < C_in; c++) {
        // Step 1: Load this input channel's weights for ALL output channels
        // weights layout in global: [out_ch][c][ki] → linear offset out_ch*C_in*9 + c*9 + ki
        // We want: smem_weights[out_ch * 9 + ki]
        const int W_PER_THREAD = (C_out * 9 + 255) / 256;  // 5 for C_out=128
        #pragma unroll
        for (int i = 0; i < W_PER_THREAD; i++) {
            int wi = tid + i * 256;
            if (wi < C_out * 9) {
                int out_ch_w = wi / 9;
                int ki = wi % 9;
                smem_weights[wi] = weights[out_ch_w * C_in * 9 + c * 9 + ki];
            }
        }
        __syncthreads();

        // Step 2: Compute convolutions using shared memory weights
        const float* ic = input_smem + c * 64;

        #pragma unroll
        for (int k = 0; k < 32; k++) {
            int oc = out_ch[k];
            int oy_k = oy[k], ox_k = ox[k];
            const float* w = smem_weights + oc * 9;
            // Center always valid
            float acc = w[4] * ic[oy_k * 8 + ox_k];
            // Row above
            if (oy_k > 0) {
                if (ox_k > 0) acc += w[0] * ic[(oy_k-1)*8 + (ox_k-1)];
                acc += w[1] * ic[(oy_k-1)*8 +  ox_k     ];
                if (ox_k < 7) acc += w[2] * ic[(oy_k-1)*8 + (ox_k+1)];
            }
            // Same row
            if (ox_k > 0) acc += w[3] * ic[ oy_k   *8 + (ox_k-1)];
            if (ox_k < 7) acc += w[5] * ic[ oy_k   *8 + (ox_k+1)];
            // Row below
            if (oy_k < 7) {
                if (ox_k > 0) acc += w[6] * ic[(oy_k+1)*8 + (ox_k-1)];
                acc += w[7] * ic[(oy_k+1)*8 +  ox_k     ];
                if (ox_k < 7) acc += w[8] * ic[(oy_k+1)*8 + (ox_k+1)];
            }
            a[k] += acc;
        }
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int idx = tid + k * 256;
        if (idx < C_out * 64)
            output_smem[idx] = a[k];
    }
    __syncthreads();
}

// Convenience wrapper: allocates smem_weights from caller's shared memory pool.
// For use when caller has extra shared memory available.
// The smem_weights pointer must point to at least C_out * 9 floats.
__device__ void block_conv_3x3(
    const float* __restrict__ input_smem,
    const float* __restrict__ weights,
    float* __restrict__ output_smem,
    int C_in, int C_out
) {
    // This version reads weights directly from global memory (no smem caching).
    // Use block_conv_3x3_smem_w() for the optimized version.
    int tid = threadIdx.x;
    int total = C_out * 64;

    for (int out_idx = tid; out_idx < total; out_idx += blockDim.x) {
        int out_ch = out_idx / 64;
        int spatial = out_idx % 64;
        int oy = spatial / 8;
        int ox = spatial % 8;
        float acc = 0.0f;
        const float* w_base = weights + out_ch * C_in * 9;

        for (int c = 0; c < C_in; c++) {
            float w0 = w_base[c * 9 + 0], w1 = w_base[c * 9 + 1], w2 = w_base[c * 9 + 2];
            float w3 = w_base[c * 9 + 3], w4 = w_base[c * 9 + 4], w5 = w_base[c * 9 + 5];
            float w6 = w_base[c * 9 + 6], w7 = w_base[c * 9 + 7], w8 = w_base[c * 9 + 8];

            const float* ic = input_smem + c * 64;

            if (oy > 0) {
                if (ox > 0) acc += w0 * ic[(oy-1)*8+(ox-1)];
                              acc += w1 * ic[(oy-1)*8+ox    ];
                if (ox < 7) acc += w2 * ic[(oy-1)*8+(ox+1)];
            }
            {
                if (ox > 0) acc += w3 * ic[ oy   *8+(ox-1)];
                              acc += w4 * ic[ oy   *8+ox    ];
                if (ox < 7) acc += w5 * ic[ oy   *8+(ox+1)];
            }
            if (oy < 7) {
                if (ox > 0) acc += w6 * ic[(oy+1)*8+(ox-1)];
                              acc += w7 * ic[(oy+1)*8+ox    ];
                if (ox < 7) acc += w8 * ic[(oy+1)*8+(ox+1)];
            }
        }
        output_smem[out_idx] = acc;
    }
    __syncthreads();
}

// ============================================================
// Tensor Core 3x3 convolution using wmma 16×16×16 FP16
//
// Reformulates conv3x3 as GEMM:
//   C[C_out, 64] = W[C_out, C_in*9] × im2col[C_in*9, 64]
//
// - Weights pre-converted to FP16 (global memory, read-only)
// - im2col tile built on-the-fly from FP32 shared-memory activations
// - FP32 accumulator, results written as FP32 to output_smem
//
// Warp mapping: 8 warps (256 threads), warp w owns M rows [w*16, w*16+16).
// Each warp iterates over 4 N_tiles (covering N=64) and 72 K_tiles (K=1152).
// smem_staging: 8 * 256 halves, each warp gets its own 256-half region.
// ============================================================
__device__ void block_conv_3x3_tc(
    const float* __restrict__ input_smem,
    const half* __restrict__ weights_h,
    float* __restrict__ output_smem,
    half* __restrict__ smem_staging,
    int C_in, int C_out
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    int K = C_in * 9;
    int K_tiles = (K + 15) / 16;

    // Each warp gets its own staging area: 256 halves
    half* my_staging = smem_staging + warp_id * 256;

    // Warp w owns output rows [M_start, M_start+16)
    int M_start = warp_id * 16;

    // Handle C_out not being exactly 128 (e.g., could be less)
    if (M_start >= C_out) {
        __syncthreads();
        return;
    }

    // wmma::load_matrix_sync requires leading dimension to be a multiple of 8
    // for half precision. When K % 16 != 0, we must load A through staging.
    bool a_direct = (K % 16 == 0);

    for (int n_tile = 0; n_tile < 4; n_tile++) {
        int N_start = n_tile * 16;

        // Initialize FP32 accumulator fragment
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
            int K_start = k_tile * 16;

            // --- Load A fragment (weights) ---
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            if (a_direct && K_start + 16 <= K) {
                // Fast path: K is aligned and tile is fully within bounds
                wmma::load_matrix_sync(a_frag, weights_h + M_start * K + K_start, K);
            } else {
                // Slow path: K not aligned or boundary tile. Load via staging with ld=16.
                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16;  // row [0,16)
                    int c = i % 16;  // col [0,16)
                    int m_idx = M_start + r;
                    int k_idx = K_start + c;
                    half val = __float2half(0.0f);
                    if (m_idx < C_out && k_idx < K)
                        val = weights_h[m_idx * K + k_idx];
                    my_staging[i] = val;
                }
                __syncwarp();
                wmma::load_matrix_sync(a_frag, my_staging, 16);
            }

            // --- Build B tile (im2col gather) into my_staging ---
            // B[k, n]: k indexes (c_in, kernel_pos), n indexes spatial
            for (int i = lane; i < 256; i += 32) {
                int k_local = i / 16;  // row within tile [0,16)
                int n_local = i % 16;  // col within tile [0,16)
                int k_idx = K_start + k_local;
                int n_idx = N_start + n_local;
                half val = __float2half(0.0f);
                if (k_idx < K && n_idx < 64) {
                    int c = k_idx / 9;
                    int ki = k_idx % 9;
                    int ky = ki / 3 - 1;
                    int kx = ki % 3 - 1;
                    int oy = n_idx / 8;
                    int ox = n_idx % 8;
                    int iy = oy + ky;
                    int ix = ox + kx;
                    if (iy >= 0 && iy < 8 && ix >= 0 && ix < 8)
                        val = __float2half(input_smem[c * 64 + iy * 8 + ix]);
                }
                my_staging[i] = val;
            }
            __syncwarp();

            // Load B fragment from shared memory staging
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, my_staging, 16);

            // Matrix multiply-accumulate
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        // Store accumulator to output shared memory
        // output_smem is [C_out, 64] row-major, leading dim = 64
        wmma::store_matrix_sync(output_smem + M_start * 64 + N_start, acc, 64,
                                wmma::mem_row_major);
    }
    __syncthreads();
}

// ============================================================
// Shifted-copy conv3x3: 9 dense GEMMs, no im2col gather
//
// For each kernel position s (ky, kx):
//   1. Build shifted[C_in, 64] FP16 from input FP32 (all 256 threads)
//   2. Dense GEMM: output += W_s[C_out, C_in] × shifted[C_in, 64]
//
// Both A (weights) and B (shifted) are contiguous with aligned ld.
// ============================================================

// Kernel offset table: s → (ky, kx) for s in 0..8
// s=0: (-1,-1), s=1: (-1,0), s=2: (-1,1)
// s=3: (0,-1),  s=4: (0,0),  s=5: (0,1)
// s=6: (1,-1),  s=7: (1,0),  s=8: (1,1)
__device__ static const int KY_TABLE[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
__device__ static const int KX_TABLE[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

__device__ void block_conv_3x3_shifted(
    const float* __restrict__ input_smem,
    half* const* W_s,
    float* __restrict__ output_smem,
    half* __restrict__ shifted,
    int C_in, int C_out
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int total_in = C_in * 64;
    int K_tiles = (C_in + 15) / 16;
    bool a_direct = (C_in % 16 == 0);

    int M_start = warp_id * 16;
    if (M_start >= C_out) {
        // This warp has no work — must still participate in __syncthreads
        for (int n_tile = 0; n_tile < 4; n_tile++) {
            for (int s = 0; s < 9; s++) {
                __syncthreads();
                __syncthreads();
            }
        }
        __syncthreads();
        return;
    }

    // Per-warp staging for A fragment boundary loads (when C_in % 16 != 0)
    // Reuse a portion of the shifted buffer that the current warp isn't reading
    // (shifted is only read during B loads, which happen after A loads)
    half* a_staging = shifted + total_in + warp_id * 256;  // 256 halves past the shifted data

    for (int n_tile = 0; n_tile < 4; n_tile++) {
        int N_start = n_tile * 16;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (int s = 0; s < 9; s++) {
            int ky = KY_TABLE[s];
            int kx = KX_TABLE[s];
            const half* W_slice = W_s[s];

            // --- Build shifted FP16 copy (all 256 threads) ---
            __syncthreads();
            for (int i = tid; i < total_in; i += 256) {
                int c = i >> 6;       // i / 64
                int n = i & 63;       // i % 64
                int oy = n >> 3;      // n / 8
                int ox = n & 7;       // n % 8
                int iy = oy + ky;
                int ix = ox + kx;
                half val = __float2half(0.0f);
                if ((unsigned)iy < 8u && (unsigned)ix < 8u)
                    val = __float2half(input_smem[c * 64 + iy * 8 + ix]);
                shifted[i] = val;
            }
            __syncthreads();

            // --- Dense GEMM: acc += W_slice[C_out, C_in] × shifted[C_in, 64] ---
            for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
                int K_start = k_tile * 16;

                // Load A fragment (weights) from contiguous global FP16
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                if (a_direct && K_start + 16 <= C_in) {
                    wmma::load_matrix_sync(a_frag, W_slice + M_start * C_in + K_start, C_in);
                } else {
                    // Boundary: load via staging with ld=16
                    for (int i = lane; i < 256; i += 32) {
                        int r = i / 16, c = i % 16;
                        int m_idx = M_start + r, k_idx = K_start + c;
                        half v = __float2half(0.0f);
                        if (m_idx < C_out && k_idx < C_in)
                            v = W_slice[m_idx * C_in + k_idx];
                        a_staging[i] = v;
                    }
                    __syncwarp();
                    wmma::load_matrix_sync(a_frag, a_staging, 16);
                }

                // Load B fragment (shifted activations) from contiguous shared FP16
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, shifted + K_start * 64 + N_start, 64);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
        }

        // Store accumulated result
        wmma::store_matrix_sync(output_smem + M_start * 64 + N_start, acc, 64,
                                wmma::mem_row_major);
    }
    __syncthreads();
}

__device__ void block_1x1_conv(
    const float* __restrict__ input_smem,
    const float* __restrict__ weights,
    float* __restrict__ output_smem,
    int C_in, int C_out
) {
    int tid = threadIdx.x;
    int total = C_out * 64;
    for (int out_idx = tid; out_idx < total; out_idx += blockDim.x) {
        int out_ch = out_idx / 64;
        int spatial = out_idx % 64;
        float acc = 0.0f;
        const float* w_row = weights + out_ch * C_in;
        for (int c = 0; c < C_in; c++) {
            acc += w_row[c] * input_smem[c * 64 + spatial];
        }
        output_smem[out_idx] = acc;
    }
    __syncthreads();
}

__device__ void block_se_block(
    float* data,
    const float* fc1_w,
    const float* fc2_w,
    int channels,
    int inner,
    float* smem_avg,
    float* smem_fc1
) {
    int tid = threadIdx.x;

    // 1. Global avg pool: each thread accumulates sums for its channels
    //    in registers, then writes once to shared memory. No atomics needed.
    for (int c = tid; c < channels; c += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < 64; i++) sum += data[c * 64 + i];
        smem_avg[c] = sum / 64.0f;
    }
    __syncthreads();

    // 2. FC1: smem_fc1[j] = ReLU(sum_c(fc1_w[j,c] * smem_avg[c]))
    for (int j = tid; j < inner; j += blockDim.x) {
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) sum += fc1_w[j * channels + c] * smem_avg[c];
        smem_fc1[j] = sum > 0.0f ? sum : 0.0f;
    }
    __syncthreads();

    // 3. FC2 + sigmoid + channel-wise scale (in-place on data)
    for (int c = tid; c < channels; c += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < inner; j++) sum += fc2_w[c * inner + j] * smem_fc1[j];
        float scale = 1.0f / (1.0f + expf(-sum));  // sigmoid
        for (int i = 0; i < 64; i++) data[c * 64 + i] *= scale;
    }
    __syncthreads();
}

__device__ void block_log_softmax(
    float* data,
    int size,
    float* smem_reduce
) {
    int tid = threadIdx.x;

    // 1. Find global max (numerical stability)
    float local_max = -FLT_MAX;
    for (int i = tid; i < size; i += blockDim.x) {
        if (data[i] > local_max) local_max = data[i];
    }
    smem_reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && smem_reduce[tid + stride] > smem_reduce[tid])
            smem_reduce[tid] = smem_reduce[tid + stride];
        __syncthreads();
    }
    float global_max = smem_reduce[0];
    __syncthreads();

    // 2. Sum exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) local_sum += expf(data[i] - global_max);
    smem_reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem_reduce[tid] += smem_reduce[tid + stride];
        __syncthreads();
    }
    float log_total = logf(smem_reduce[0]);
    __syncthreads();

    // 3. Write log-softmax values
    for (int i = tid; i < size; i += blockDim.x) {
        data[i] = (data[i] - global_max) - log_total;
    }
    __syncthreads();
}

__device__ void block_board_to_planes(
    const BoardState* bs,
    float* planes
) {
    int tid = threadIdx.x;
    int stm = bs->w_to_move ? 0 : 1;
    int opp = 1 - stm;
    bool flip = !bs->w_to_move;

    // Zero all 17 planes
    for (int i = tid; i < 17 * 64; i += blockDim.x) planes[i] = 0.0f;
    __syncthreads();

    // Planes 0-5: STM pieces; planes 6-11: opponent pieces
    for (int piece = 0; piece < 6; piece++) {
        uint64_t stm_bb = bs->pieces[stm * 6 + piece];
        uint64_t opp_bb = bs->pieces[opp * 6 + piece];
        for (int sq = tid; sq < 64; sq += blockDim.x) {
            int mapped = flip ? (sq ^ 56) : sq;
            if ((stm_bb >> sq) & 1) planes[piece * 64 + mapped] = 1.0f;
            if ((opp_bb >> sq) & 1) planes[(6 + piece) * 64 + mapped] = 1.0f;
        }
    }
    __syncthreads();

    // Planes 12-16: en passant + castling rights (thread 0 only)
    if (tid == 0) {
        if (bs->en_passant != EN_PASSANT_NONE) {
            int mapped = flip ? (bs->en_passant ^ 56) : bs->en_passant;
            planes[12 * 64 + mapped] = 1.0f;
        }
        uint8_t stm_ks, stm_qs, opp_ks, opp_qs;
        if (bs->w_to_move) {
            stm_ks = bs->castling & CASTLE_WK;
            stm_qs = bs->castling & CASTLE_WQ;
            opp_ks = bs->castling & CASTLE_BK;
            opp_qs = bs->castling & CASTLE_BQ;
        } else {
            stm_ks = bs->castling & CASTLE_BK;
            stm_qs = bs->castling & CASTLE_BQ;
            opp_ks = bs->castling & CASTLE_WK;
            opp_qs = bs->castling & CASTLE_WQ;
        }
        if (stm_ks) for (int i = 0; i < 64; i++) planes[13 * 64 + i] = 1.0f;
        if (stm_qs) for (int i = 0; i < 64; i++) planes[14 * 64 + i] = 1.0f;
        if (opp_ks) for (int i = 0; i < 64; i++) planes[15 * 64 + i] = 1.0f;
        if (opp_qs) for (int i = 0; i < 64; i++) planes[16 * 64 + i] = 1.0f;
    }
    __syncthreads();
}
