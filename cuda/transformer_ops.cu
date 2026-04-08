#include "transformer_ops.cuh"
#include <mma.h>
#include <cfloat>
using namespace nvcuda;

// ============================================================
// Dense GEMM using wmma 16×16×16 FP16
//
// C[M, N] += A[M, K] × B[K, N]
// A: FP16 global memory (weights), row-major with leading dim ld_a
// B: FP32 shared memory (activations), row-major with leading dim N
// C: FP32 shared memory (output), row-major with leading dim N
//
// B is converted to FP16 on-the-fly into workspace for wmma loads.
// Workspace needs K * 16 halves (one K-column tile of 16 rows at a time,
// but actually we tile both M and N and load B per (K_tile, N_tile)).
//
// Warp mapping: 8 warps assigned to (M_tile, N_tile) pairs.
// M_tiles = ceil(M/16), N_tiles = ceil(N/16).
// Total work = M_tiles × N_tiles, distributed round-robin across 8 warps.
// ============================================================

__device__ void tf_gemm_acc(
    const half* __restrict__ A_global,
    const float* __restrict__ B_smem,
    float* __restrict__ C_smem,
    half* __restrict__ workspace,
    int M, int N, int K, int ld_a
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int M_tiles = (M + 15) / 16;
    int N_tiles = (N + 15) / 16;
    int K_tiles = (K + 15) / 16;
    int total_tiles = M_tiles * N_tiles;
    bool a_aligned = (ld_a % 16 == 0);

    // Per-warp staging area for A boundary loads
    half* a_staging = workspace + warp_id * 256;
    // Shared B staging: after a_staging area (8 warps × 256 = 2048 halves)
    // Each warp loads its own B tile into its staging area
    half* b_staging = workspace + 8 * 256 + warp_id * 256;

    // Each warp handles one or more (M_tile, N_tile) output tiles
    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += 8) {
        int m_tile = tile_idx / N_tiles;
        int n_tile = tile_idx % N_tiles;
        int M_start = m_tile * 16;
        int N_start = n_tile * 16;

        // Load existing accumulator from C
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::load_matrix_sync(acc, C_smem + M_start * N + N_start, N, wmma::mem_row_major);

        for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
            int K_start = k_tile * 16;

            // Load A fragment (weights from global FP16)
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            if (a_aligned && K_start + 16 <= K && M_start + 16 <= M) {
                wmma::load_matrix_sync(a_frag, A_global + M_start * ld_a + K_start, ld_a);
            } else {
                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16, c = i % 16;
                    int m = M_start + r, k = K_start + c;
                    a_staging[i] = (m < M && k < K) ? A_global[m * ld_a + k] : __float2half(0.0f);
                }
                __syncwarp();
                wmma::load_matrix_sync(a_frag, a_staging, 16);
            }

            // Load B fragment (activations from shared FP32 → convert to FP16)
            for (int i = lane; i < 256; i += 32) {
                int r = i / 16, c = i % 16;
                int k = K_start + r, n = N_start + c;
                b_staging[i] = (k < K && n < N) ? __float2half(B_smem[k * N + n]) : __float2half(0.0f);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, b_staging, 16);

            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        // Store accumulator back to C
        wmma::store_matrix_sync(C_smem + M_start * N + N_start, acc, N, wmma::mem_row_major);
    }
    __syncthreads();
}

__device__ void tf_gemm(
    const half* __restrict__ A_global,
    const float* __restrict__ B_smem,
    float* __restrict__ C_smem,
    half* __restrict__ workspace,
    int M, int N, int K, int ld_a
) {
    // Zero C first
    int tid = threadIdx.x;
    int total = M * N;
    for (int i = tid; i < total; i += blockDim.x) C_smem[i] = 0.0f;
    __syncthreads();

    tf_gemm_acc(A_global, B_smem, C_smem, workspace, M, N, K, ld_a);
}

// ============================================================
// LayerNorm: per-token normalization
// ============================================================
__device__ void tf_layer_norm(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int num_tokens, int d_model,
    float* __restrict__ smem_reduce
) {
    int tid = threadIdx.x;
    int total = num_tokens * d_model;

    // Process each token
    for (int tok = 0; tok < num_tokens; tok++) {
        const float* x = input + tok * d_model;
        float* y = output + tok * d_model;

        // Mean
        float local_sum = 0.0f;
        for (int d = tid; d < d_model; d += blockDim.x) local_sum += x[d];
        smem_reduce[tid] = local_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) smem_reduce[tid] += smem_reduce[tid + s];
            __syncthreads();
        }
        float mean = smem_reduce[0] / (float)d_model;
        __syncthreads();

        // Variance
        float local_var = 0.0f;
        for (int d = tid; d < d_model; d += blockDim.x) {
            float diff = x[d] - mean;
            local_var += diff * diff;
        }
        smem_reduce[tid] = local_var;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) smem_reduce[tid] += smem_reduce[tid + s];
            __syncthreads();
        }
        float inv_std = rsqrtf(smem_reduce[0] / (float)d_model + 1e-5f);
        __syncthreads();

        // Normalize
        for (int d = tid; d < d_model; d += blockDim.x) {
            y[d] = (x[d] - mean) * inv_std * gamma[d] + beta[d];
        }
        __syncthreads();
    }
}

// ============================================================
// Row-wise softmax
// ============================================================
__device__ void tf_softmax_rows(
    float* __restrict__ data,
    int rows, int cols,
    float* __restrict__ smem_reduce
) {
    int tid = threadIdx.x;

    for (int row = 0; row < rows; row++) {
        float* r = data + row * cols;

        // Max
        float local_max = -FLT_MAX;
        for (int c = tid; c < cols; c += blockDim.x)
            if (r[c] > local_max) local_max = r[c];
        smem_reduce[tid] = local_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && smem_reduce[tid + s] > smem_reduce[tid])
                smem_reduce[tid] = smem_reduce[tid + s];
            __syncthreads();
        }
        float row_max = smem_reduce[0];
        __syncthreads();

        // Sum exp
        float local_sum = 0.0f;
        for (int c = tid; c < cols; c += blockDim.x) {
            r[c] = expf(r[c] - row_max);
            local_sum += r[c];
        }
        smem_reduce[tid] = local_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) smem_reduce[tid] += smem_reduce[tid + s];
            __syncthreads();
        }
        float inv_sum = 1.0f / smem_reduce[0];
        __syncthreads();

        // Normalize
        for (int c = tid; c < cols; c += blockDim.x) r[c] *= inv_sum;
        __syncthreads();
    }
}

// ============================================================
// Simple element-wise ops
// ============================================================
__device__ void tf_add_bias(float* data, const float* bias, int M, int N) {
    int tid = threadIdx.x;
    for (int i = tid; i < M * N; i += blockDim.x)
        data[i] += bias[i % N];
    __syncthreads();
}

__device__ void tf_residual_add(float* dst, const float* src, int count) {
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += blockDim.x) dst[i] += src[i];
    __syncthreads();
}

__device__ void tf_relu(float* data, int count) {
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += blockDim.x)
        if (data[i] < 0.0f) data[i] = 0.0f;
    __syncthreads();
}

// ============================================================
// Board encoding: BoardState → [64, 17] token features
// Token t (square t) has 17 features: 12 piece planes + 1 EP + 4 castling
// STM-relative: pieces[0-5]=STM, pieces[6-11]=OPP, flipped for black
// ============================================================
__device__ void tf_board_to_tokens(
    const BoardState* bs,
    float* tokens
) {
    int tid = threadIdx.x;
    int stm = bs->w_to_move ? 0 : 1;
    int opp = 1 - stm;
    bool flip = !bs->w_to_move;

    // Zero all 64 × 17 = 1088 floats
    for (int i = tid; i < TF_NUM_TOKENS * NN_INPUT_CHANNELS; i += blockDim.x)
        tokens[i] = 0.0f;
    __syncthreads();

    // Piece planes (channels 0-11)
    for (int piece = 0; piece < 6; piece++) {
        uint64_t stm_bb = bs->pieces[stm * 6 + piece];
        uint64_t opp_bb = bs->pieces[opp * 6 + piece];
        for (int sq = tid; sq < 64; sq += blockDim.x) {
            int mapped = flip ? (sq ^ 56) : sq;
            if ((stm_bb >> sq) & 1) tokens[mapped * NN_INPUT_CHANNELS + piece] = 1.0f;
            if ((opp_bb >> sq) & 1) tokens[mapped * NN_INPUT_CHANNELS + 6 + piece] = 1.0f;
        }
    }
    __syncthreads();

    // EP + castling (channels 12-16), thread 0 only
    if (tid == 0) {
        if (bs->en_passant != EN_PASSANT_NONE) {
            int mapped = flip ? (bs->en_passant ^ 56) : bs->en_passant;
            tokens[mapped * NN_INPUT_CHANNELS + 12] = 1.0f;
        }
        uint8_t stm_ks, stm_qs, opp_ks, opp_qs;
        if (bs->w_to_move) {
            stm_ks = bs->castling & CASTLE_WK; stm_qs = bs->castling & CASTLE_WQ;
            opp_ks = bs->castling & CASTLE_BK; opp_qs = bs->castling & CASTLE_BQ;
        } else {
            stm_ks = bs->castling & CASTLE_BK; stm_qs = bs->castling & CASTLE_BQ;
            opp_ks = bs->castling & CASTLE_WK; opp_qs = bs->castling & CASTLE_WQ;
        }
        // Castling is a per-square feature: set on ALL squares
        for (int sq = 0; sq < 64; sq++) {
            if (stm_ks) tokens[sq * NN_INPUT_CHANNELS + 13] = 1.0f;
            if (stm_qs) tokens[sq * NN_INPUT_CHANNELS + 14] = 1.0f;
            if (opp_ks) tokens[sq * NN_INPUT_CHANNELS + 15] = 1.0f;
            if (opp_qs) tokens[sq * NN_INPUT_CHANNELS + 16] = 1.0f;
        }
    }
    __syncthreads();
}
