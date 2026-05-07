#include "block_ops.cuh"
#include <cfloat>
#include <mma.h>
using namespace nvcuda;

// All functions assume blockDim.x == 256 and all 256 threads participate.
// Work is striped: thread t owns elements t, t+256, t+512, ...
//
// Activation buffers are FP16 in smem. Internal arithmetic is FP32.

__device__ void block_bn_relu(
    __half* data,
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
        float x = __half2float(data[idx]);
        float norm = (x - running_mean[ch]) * rsqrtf(running_var[ch] + 1e-5f);
        float result = norm * gamma[ch] + beta[ch];
        if (relu && result < 0.0f) result = 0.0f;
        data[idx] = __float2half(result);
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
__device__ void block_conv_3x3_smem_w(
    const __half* __restrict__ input_smem,
    const float* __restrict__ weights,
    __half* __restrict__ output_smem,
    float* __restrict__ smem_weights,
    int C_in, int C_out
) {
    int tid = threadIdx.x;

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

    // 32 FP32 accumulators
    float a[32];
    #pragma unroll
    for (int k = 0; k < 32; k++) a[k] = 0.0f;

    for (int c = 0; c < C_in; c++) {
        // Step 1: Load this input channel's weights for ALL output channels
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
        const __half* ic = input_smem + c * 64;

        #pragma unroll
        for (int k = 0; k < 32; k++) {
            int oc = out_ch[k];
            int oy_k = oy[k], ox_k = ox[k];
            const float* w = smem_weights + oc * 9;
            // Center always valid
            float acc = w[4] * __half2float(ic[oy_k * 8 + ox_k]);
            // Row above
            if (oy_k > 0) {
                if (ox_k > 0) acc += w[0] * __half2float(ic[(oy_k-1)*8 + (ox_k-1)]);
                acc += w[1] * __half2float(ic[(oy_k-1)*8 +  ox_k     ]);
                if (ox_k < 7) acc += w[2] * __half2float(ic[(oy_k-1)*8 + (ox_k+1)]);
            }
            // Same row
            if (ox_k > 0) acc += w[3] * __half2float(ic[ oy_k   *8 + (ox_k-1)]);
            if (ox_k < 7) acc += w[5] * __half2float(ic[ oy_k   *8 + (ox_k+1)]);
            // Row below
            if (oy_k < 7) {
                if (ox_k > 0) acc += w[6] * __half2float(ic[(oy_k+1)*8 + (ox_k-1)]);
                acc += w[7] * __half2float(ic[(oy_k+1)*8 +  ox_k     ]);
                if (ox_k < 7) acc += w[8] * __half2float(ic[(oy_k+1)*8 + (ox_k+1)]);
            }
            a[k] += acc;
        }
        __syncthreads();
    }

    // Write FP16 results
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int idx = tid + k * 256;
        if (idx < C_out * 64)
            output_smem[idx] = __float2half(a[k]);
    }
    __syncthreads();
}

// Direct conv3x3 reading weights from global memory each call.
// Used by per-op tests; not on the MCTS hot path.
__device__ void block_conv_3x3(
    const __half* __restrict__ input_smem,
    const float* __restrict__ weights,
    __half* __restrict__ output_smem,
    int C_in, int C_out
) {
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

            const __half* ic = input_smem + c * 64;

            if (oy > 0) {
                if (ox > 0) acc += w0 * __half2float(ic[(oy-1)*8+(ox-1)]);
                              acc += w1 * __half2float(ic[(oy-1)*8+ox    ]);
                if (ox < 7) acc += w2 * __half2float(ic[(oy-1)*8+(ox+1)]);
            }
            {
                if (ox > 0) acc += w3 * __half2float(ic[ oy   *8+(ox-1)]);
                              acc += w4 * __half2float(ic[ oy   *8+ox    ]);
                if (ox < 7) acc += w5 * __half2float(ic[ oy   *8+(ox+1)]);
            }
            if (oy < 7) {
                if (ox > 0) acc += w6 * __half2float(ic[(oy+1)*8+(ox-1)]);
                              acc += w7 * __half2float(ic[(oy+1)*8+ox    ]);
                if (ox < 7) acc += w8 * __half2float(ic[(oy+1)*8+(ox+1)]);
            }
        }
        output_smem[out_idx] = __float2half(acc);
    }
    __syncthreads();
}

// ============================================================
// Tensor Core 3x3 conv via wmma 16×16×16 FP16 (FP32 accumulator).
//
// Input/output are FP16. Per-warp FP32 staging is carved from the
// smem_staging region (the same buffer that holds the FP16 im2col
// gather) once the K-loop is finished — they don't conflict in time.
// ============================================================
__device__ void block_conv_3x3_tc(
    const __half* __restrict__ input_smem,
    const half* __restrict__ weights_h,
    __half* __restrict__ output_smem,
    half* __restrict__ smem_staging,
    int C_in, int C_out
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    int K = C_in * 9;
    int K_tiles = (K + 15) / 16;

    // Each warp gets its own 256-half region in smem_staging
    half* my_staging = smem_staging + warp_id * 256;
    int M_start = warp_id * 16;

    if (M_start >= C_out) {
        // Active path does 2 __syncthreads after the main K-loops (one
        // before per-warp FP32 store-staging, one final). Match that count
        // here so block-wide barriers stay aligned at C_out < 128.
        __syncthreads();
        __syncthreads();
        return;
    }

    bool a_direct = (K % 16 == 0);

    // 4 accumulators (one per N_tile), built across all K_tiles below.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    for (int t = 0; t < 4; t++) wmma::fill_fragment(acc[t], 0.0f);

    for (int n_tile = 0; n_tile < 4; n_tile++) {
        int N_start = n_tile * 16;

        for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
            int K_start = k_tile * 16;

            // --- Load A fragment (weights) ---
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            if (a_direct && K_start + 16 <= K) {
                wmma::load_matrix_sync(a_frag, weights_h + M_start * K + K_start, K);
            } else {
                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16;
                    int c = i % 16;
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
            for (int i = lane; i < 256; i += 32) {
                int k_local = i / 16;
                int n_local = i % 16;
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
                        val = input_smem[c * 64 + iy * 8 + ix];   // already FP16
                }
                my_staging[i] = val;
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, my_staging, 16);

            wmma::mma_sync(acc[n_tile], a_frag, b_frag, acc[n_tile]);
        }
    }

    // --- Store all 4 accumulators as FP16 ---
    // Repurpose smem_staging as per-warp FP32 staging (8 warps × 256 floats =
    // 2048 floats = 4096 halves, fits inside the 8192-half region).
    __syncthreads();
    float* warp_fp32_stage = ((float*)smem_staging) + warp_id * 256;

    for (int n_tile = 0; n_tile < 4; n_tile++) {
        wmma::store_matrix_sync(warp_fp32_stage, acc[n_tile], 16, wmma::mem_row_major);
        __syncwarp();
        // Convert 16×16 FP32 → FP16, write to output_smem at [M_start, N_start]
        int N_start = n_tile * 16;
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16;
            int c = i % 16;
            int oc = M_start + r;
            int n  = N_start + c;
            if (oc < C_out && n < 64)
                output_smem[oc * 64 + n] = __float2half(warp_fp32_stage[i]);
        }
        __syncwarp();
    }
    __syncthreads();
}

// ============================================================
// Shifted-copy conv3x3: 9 dense GEMMs, FP16 in/out.
// ============================================================

__device__ static const int KY_TABLE[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
__device__ static const int KX_TABLE[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

__device__ void block_conv_3x3_shifted(
    const __half* __restrict__ input_smem,
    half* const* W_s,
    __half* __restrict__ output_smem,
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
    bool active = (M_start < C_out);
    // ALL 256 threads must participate in the cooperative shifted-buffer
    // build below (each thread strides by 256 over the [C_in, 64] input).
    // A hard early-return for inactive warps would leave half the buffer
    // unwritten and produce wrong mma results when C_out < 128. So we keep
    // every warp live through the shift loop and only gate the per-warp
    // mma + store on `active`.

    // Per-warp staging for A-fragment boundary loads (when C_in % 16 != 0).
    half* a_staging = shifted + total_in + warp_id * 256;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    if (active) {
        for (int t = 0; t < 4; t++) wmma::fill_fragment(acc[t], 0.0f);
    }

    for (int s = 0; s < 9; s++) {
        int ky = KY_TABLE[s];
        int kx = KX_TABLE[s];
        const half* W_slice = active ? W_s[s] : nullptr;

        // --- Build shifted FP16 copy ONCE per kernel position (all 256 threads) ---
        __syncthreads();
        for (int i = tid; i < total_in; i += 256) {
            int c = i >> 6;
            int n = i & 63;
            int oy = n >> 3;
            int ox = n & 7;
            int iy = oy + ky;
            int ix = ox + kx;
            half val = __float2half(0.0f);
            if ((unsigned)iy < 8u && (unsigned)ix < 8u)
                val = input_smem[c * 64 + iy * 8 + ix];
            shifted[i] = val;
        }
        __syncthreads();

        if (active) {
            for (int n_tile = 0; n_tile < 4; n_tile++) {
                int N_start = n_tile * 16;

                for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    int K_start = k_tile * 16;

                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    if (a_direct && K_start + 16 <= C_in) {
                        wmma::load_matrix_sync(a_frag, W_slice + M_start * C_in + K_start, C_in);
                    } else {
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

                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, shifted + K_start * 64 + N_start, 64);

                    wmma::mma_sync(acc[n_tile], a_frag, b_frag, acc[n_tile]);
                }
            }
        }
    }

    // --- Store all 4 accumulators as FP16 ---
    // After the 9-shift loop, the shifted buffer is dead. Repurpose it as
    // per-warp FP32 staging (active warps only).
    __syncthreads();
    if (active) {
        float* warp_fp32_stage = ((float*)shifted) + warp_id * 256;
        for (int n_tile = 0; n_tile < 4; n_tile++) {
            wmma::store_matrix_sync(warp_fp32_stage, acc[n_tile], 16, wmma::mem_row_major);
            __syncwarp();
            int N_start = n_tile * 16;
            for (int i = lane; i < 256; i += 32) {
                int r = i / 16;
                int c = i % 16;
                int oc = M_start + r;
                int n  = N_start + c;
                if (oc < C_out && n < 64)
                    output_smem[oc * 64 + n] = __float2half(warp_fp32_stage[i]);
            }
            __syncwarp();
        }
    }
    __syncthreads();
}

// ============================================================
// Batched shifted-copy conv at B=2. Same algorithm as the single-batch
// version, but the inner WMMA loop loads A once and applies it to both
// batch elements before advancing — halving weight-memory traffic.
// ============================================================
__device__ void block_conv_3x3_shifted_b2(
    const __half* __restrict__ input_smem_b2,
    half* const* W_s,
    __half* __restrict__ output_smem_b2,
    half* __restrict__ shifted_b2,
    int C_in, int C_out,
    int input_batch_stride,
    int output_batch_stride
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int total_in_per_b = C_in * 64;        // tight packing in shifted_b2
    int total_out_per_b = C_out * 64;      // tight packing in output (we write out_b at stride)
    int K_tiles = (C_in + 15) / 16;
    bool a_direct = (C_in % 16 == 0);

    int M_start = warp_id * 16;
    bool active = (M_start < C_out);

    // Per-warp staging for A-fragment boundary loads (when C_in % 16 != 0).
    // Lives just past the doubled shifted buffer.
    half* a_staging = shifted_b2 + 2 * total_in_per_b + warp_id * 256;

    // 8 accumulators per active warp: 4 N-tiles × 2 batches.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    if (active) {
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            #pragma unroll
            for (int t = 0; t < 4; t++) wmma::fill_fragment(acc[b][t], 0.0f);
        }
    }

    for (int s = 0; s < 9; s++) {
        int ky = KY_TABLE[s];
        int kx = KX_TABLE[s];
        const half* W_slice = active ? W_s[s] : nullptr;

        // ALL 256 threads cooperate: build shifted FP16 copy for both batches.
        __syncthreads();
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            // Read using caller's stride (e.g. 8192 if buf sized for 6×128)
            const __half* in_b   = input_smem_b2 + b * input_batch_stride;
            // Write tight-packed in shifted (we control its layout)
            half*         out_b  = shifted_b2    + b * total_in_per_b;
            for (int i = tid; i < total_in_per_b; i += 256) {
                int c = i >> 6;
                int n = i & 63;
                int oy = n >> 3;
                int ox = n & 7;
                int iy = oy + ky;
                int ix = ox + kx;
                half val = __float2half(0.0f);
                if ((unsigned)iy < 8u && (unsigned)ix < 8u)
                    val = in_b[c * 64 + iy * 8 + ix];
                out_b[i] = val;
            }
        }
        __syncthreads();

        if (active) {
            for (int n_tile = 0; n_tile < 4; n_tile++) {
                int N_start = n_tile * 16;

                for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    int K_start = k_tile * 16;

                    // Load A ONCE per (n_tile, k_tile).
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    if (a_direct && K_start + 16 <= C_in) {
                        wmma::load_matrix_sync(a_frag, W_slice + M_start * C_in + K_start, C_in);
                    } else {
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

                    // Apply A to BOTH batches.
                    #pragma unroll
                    for (int b = 0; b < 2; b++) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                        wmma::load_matrix_sync(
                            b_frag,
                            shifted_b2 + b * total_in_per_b + K_start * 64 + N_start,
                            64);
                        wmma::mma_sync(acc[b][n_tile], a_frag, b_frag, acc[b][n_tile]);
                    }
                }
            }
        }
    }

    // --- Store all accumulators as FP16 ---
    // After the 9-shift loop, the shifted buffer is dead. Repurpose it as
    // per-warp FP32 staging (active warps only). We have 2 × C_in × 64 halves
    // available; need 8 warps × 512 halves = 4096 halves. Fits when C_in ≥ 32.
    __syncthreads();
    if (active) {
        float* warp_fp32_stage = ((float*)shifted_b2) + warp_id * 256;
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            __half* out_b = output_smem_b2 + b * output_batch_stride;
            for (int n_tile = 0; n_tile < 4; n_tile++) {
                wmma::store_matrix_sync(warp_fp32_stage, acc[b][n_tile], 16,
                                        wmma::mem_row_major);
                __syncwarp();
                int N_start = n_tile * 16;
                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16;
                    int c = i % 16;
                    int oc = M_start + r;
                    int n  = N_start + c;
                    if (oc < C_out && n < 64)
                        out_b[oc * 64 + n] = __float2half(warp_fp32_stage[i]);
                }
                __syncwarp();
            }
        }
    }
    __syncthreads();
}

__device__ void block_1x1_conv(
    const __half* __restrict__ input_smem,
    const float* __restrict__ weights,
    __half* __restrict__ output_smem,
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
            acc += w_row[c] * __half2float(input_smem[c * 64 + spatial]);
        }
        output_smem[out_idx] = __float2half(acc);
    }
    __syncthreads();
}

__device__ void block_se_block(
    __half* data,
    const float* fc1_w,
    const float* fc2_w,
    int channels,
    int inner,
    float* smem_avg,
    float* smem_fc1
) {
    int tid = threadIdx.x;

    // 1. Global avg pool — FP32 accumulation, FP32 output to smem_avg
    for (int c = tid; c < channels; c += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < 64; i++) sum += __half2float(data[c * 64 + i]);
        smem_avg[c] = sum / 64.0f;
    }
    __syncthreads();

    // 2. FC1 — FP32 in, FP32 out
    for (int j = tid; j < inner; j += blockDim.x) {
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) sum += fc1_w[j * channels + c] * smem_avg[c];
        smem_fc1[j] = sum > 0.0f ? sum : 0.0f;
    }
    __syncthreads();

    // 3. FC2 + sigmoid + channel-wise scale (in-place on FP16 data)
    for (int c = tid; c < channels; c += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < inner; j++) sum += fc2_w[c * inner + j] * smem_fc1[j];
        float scale = 1.0f / (1.0f + expf(-sum));  // sigmoid
        for (int i = 0; i < 64; i++) {
            float x = __half2float(data[c * 64 + i]);
            data[c * 64 + i] = __float2half(x * scale);
        }
    }
    __syncthreads();
}

__device__ void block_log_softmax(
    float* data,
    int size,
    float* smem_reduce
) {
    int tid = threadIdx.x;

    // 1. Find global max
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

    for (int i = tid; i < size; i += blockDim.x) {
        data[i] = (data[i] - global_max) - log_total;
    }
    __syncthreads();
}

__device__ void block_board_to_planes(
    const BoardState* bs,
    __half* planes
) {
    int tid = threadIdx.x;
    int stm = bs->w_to_move ? 0 : 1;
    int opp = 1 - stm;
    bool flip = !bs->w_to_move;

    const __half H_ZERO = __float2half(0.0f);
    const __half H_ONE  = __float2half(1.0f);

    for (int i = tid; i < 17 * 64; i += blockDim.x) planes[i] = H_ZERO;
    __syncthreads();

    for (int piece = 0; piece < 6; piece++) {
        uint64_t stm_bb = bs->pieces[stm * 6 + piece];
        uint64_t opp_bb = bs->pieces[opp * 6 + piece];
        for (int sq = tid; sq < 64; sq += blockDim.x) {
            int mapped = flip ? (sq ^ 56) : sq;
            if ((stm_bb >> sq) & 1) planes[piece * 64 + mapped] = H_ONE;
            if ((opp_bb >> sq) & 1) planes[(6 + piece) * 64 + mapped] = H_ONE;
        }
    }
    __syncthreads();

    if (tid == 0) {
        if (bs->en_passant != EN_PASSANT_NONE) {
            int mapped = flip ? (bs->en_passant ^ 56) : bs->en_passant;
            planes[12 * 64 + mapped] = H_ONE;
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
        if (stm_ks) for (int i = 0; i < 64; i++) planes[13 * 64 + i] = H_ONE;
        if (stm_qs) for (int i = 0; i < 64; i++) planes[14 * 64 + i] = H_ONE;
        if (opp_ks) for (int i = 0; i < 64; i++) planes[15 * 64 + i] = H_ONE;
        if (opp_qs) for (int i = 0; i < 64; i++) planes[16 * 64 + i] = H_ONE;
    }
    __syncthreads();
}
