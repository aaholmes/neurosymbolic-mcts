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
// GEMM with both operands in shared memory FP32
// C[M,N] = A[M,K] × B[K,N]
// ============================================================
__device__ void tf_gemm_smem(
    const float* __restrict__ A_smem,
    const float* __restrict__ B_smem,
    float* __restrict__ C_smem,
    half* __restrict__ workspace,
    int M, int N, int K,
    int ld_a, int ld_b
) {
    if (ld_a == 0) ld_a = K;
    if (ld_b == 0) ld_b = N;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int M_tiles = (M + 15) / 16;
    int N_tiles = (N + 15) / 16;
    int K_tiles = (K + 15) / 16;
    int total_tiles = M_tiles * N_tiles;

    half* a_staging = workspace + warp_id * 256;
    half* b_staging = workspace + 8 * 256 + warp_id * 256;

    for (int i = tid; i < M * N; i += blockDim.x) C_smem[i] = 0.0f;
    __syncthreads();

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += 8) {
        int m_tile = tile_idx / N_tiles;
        int n_tile = tile_idx % N_tiles;
        int M_start = m_tile * 16, N_start = n_tile * 16;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
            int K_start = k_tile * 16;

            for (int i = lane; i < 256; i += 32) {
                int r = i / 16, c = i % 16;
                int m = M_start + r, k = K_start + c;
                a_staging[i] = (m < M && k < K) ? __float2half(A_smem[m * ld_a + k]) : __float2half(0.0f);
            }
            for (int i = lane; i < 256; i += 32) {
                int r = i / 16, c = i % 16;
                int k = K_start + r, n = N_start + c;
                b_staging[i] = (k < K && n < N) ? __float2half(B_smem[k * ld_b + n]) : __float2half(0.0f);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_staging, 16);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, b_staging, 16);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        wmma::store_matrix_sync(C_smem + M_start * N + N_start, acc, N, wmma::mem_row_major);
    }
    __syncthreads();
}

// ============================================================
// GEMM: C[M,N] = A[M,K] × B[N,K]^T with scaling
// B is stored as [N, K] and accessed transposed
// ============================================================
__device__ void tf_gemm_smem_abt(
    const float* __restrict__ A_smem,
    const float* __restrict__ B_smem,
    float* __restrict__ C_smem,
    half* __restrict__ workspace,
    int M, int N, int K,
    float scale,
    int ld_a, int ld_b
) {
    if (ld_a == 0) ld_a = K;
    if (ld_b == 0) ld_b = K;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int M_tiles = (M + 15) / 16;
    int N_tiles = (N + 15) / 16;
    int K_tiles = (K + 15) / 16;
    int total_tiles = M_tiles * N_tiles;

    half* a_staging = workspace + warp_id * 256;
    half* b_staging = workspace + 8 * 256 + warp_id * 256;

    for (int i = tid; i < M * N; i += blockDim.x) C_smem[i] = 0.0f;
    __syncthreads();

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += 8) {
        int m_tile = tile_idx / N_tiles;
        int n_tile = tile_idx % N_tiles;
        int M_start = m_tile * 16, N_start = n_tile * 16;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
            int K_start = k_tile * 16;

            for (int i = lane; i < 256; i += 32) {
                int r = i / 16, c = i % 16;
                int m = M_start + r, k = K_start + c;
                a_staging[i] = (m < M && k < K) ? __float2half(A_smem[m * ld_a + k]) : __float2half(0.0f);
            }
            // B[N, K] transposed: staging[k_local, n_local] = B[n, k]
            for (int i = lane; i < 256; i += 32) {
                int r = i / 16, c = i % 16;
                int k = K_start + r, n = N_start + c;
                b_staging[i] = (k < K && n < N) ? __float2half(B_smem[n * ld_b + k]) : __float2half(0.0f);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_staging, 16);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, b_staging, 16);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        // Apply scale and store
        if (scale != 1.0f) {
            for (int i = 0; i < acc.num_elements; i++) acc.x[i] *= scale;
        }
        wmma::store_matrix_sync(C_smem + M_start * N + N_start, acc, N, wmma::mem_row_major);
    }
    __syncthreads();
}

// ============================================================
// Linear layer: output[M,N] = input[M,K] × weight^T[K,N] + bias[N]
//
// Reformulated as: C_T[N,M] = W[N,K] × input_T[K,M]
//   where input_T is input transposed (handled during B staging)
//   and C_T is stored transposed to get output[M,N]
//
// A = W[N,K] in global FP16 with ld=K
// B = input^T[K,M] created by transposed read from input[M,K] in smem FP32
// C = output transposed [N,M] → stored as output[M,N]
// ============================================================
__device__ void tf_linear(
    const float* __restrict__ input_smem,
    const half* __restrict__ weight_fp16,
    float* __restrict__ output_smem,
    const float* __restrict__ bias,
    half* __restrict__ workspace,
    int M, int N, int K,
    bool accumulate
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // GEMM dimensions: C_T[N, M] = A[N, K] × B_T[K, M]
    int M_tiles = (N + 15) / 16;   // N is the "M" dimension of the GEMM
    int N_tiles = (M + 15) / 16;   // M is the "N" dimension of the GEMM
    int K_tiles = (K + 15) / 16;
    int total_tiles = M_tiles * N_tiles;
    bool a_aligned = (K % 16 == 0);

    half* a_staging = workspace + warp_id * 256;
    half* b_staging = workspace + 8 * 256 + warp_id * 256;

    // Zero output if not accumulating
    if (!accumulate) {
        for (int i = tid; i < M * N; i += blockDim.x) output_smem[i] = 0.0f;
        __syncthreads();
    }

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += 8) {
        int n_tile = tile_idx / N_tiles;  // output row tile (in N dimension)
        int m_tile = tile_idx % N_tiles;  // output col tile (in M dimension)
        int N_start = n_tile * 16;  // row start in W (= output feature)
        int M_start = m_tile * 16;  // col start (= token index)

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
            int K_start = k_tile * 16;

            // A fragment: W[N, K] with ld=K
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            if (a_aligned && K_start + 16 <= K && N_start + 16 <= N) {
                wmma::load_matrix_sync(a_frag, weight_fp16 + N_start * K + K_start, K);
            } else {
                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16, c = i % 16;
                    int n = N_start + r, k = K_start + c;
                    a_staging[i] = (n < N && k < K) ? weight_fp16[n * K + k] : __float2half(0.0f);
                }
                __syncwarp();
                wmma::load_matrix_sync(a_frag, a_staging, 16);
            }

            // B fragment: input^T[K, M] — transposed read from input[M, K]
            // B[k, m] = input[m, k] → staging[k_local * 16 + m_local] = input[(M_start+m_local)*K + (K_start+k_local)]
            for (int i = lane; i < 256; i += 32) {
                int r = i / 16, c = i % 16;  // r = k_local, c = m_local
                int k = K_start + r, m = M_start + c;
                b_staging[i] = (k < K && m < M) ? __float2half(input_smem[m * K + k]) : __float2half(0.0f);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, b_staging, 16);

            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        // acc holds C_T[N_start:N_start+16, M_start:M_start+16]
        // We need output[M_start:M_start+16, N_start:N_start+16] = C_T^T
        // Store to output_smem directly with transposed indexing.
        // Use shared memory as temp: store row-major [16,16], then scatter-transpose.
        // Reuse b_staging region as temp (512 halves = 256 floats, just barely fits 16×16)
        // Actually: need contiguous 256 floats. Use output_smem itself as temp
        // by storing to a region we'll overwrite anyway.
        //
        // Simplest correct approach: store to output directly using row-major layout,
        // but with transposed coordinates. wmma::store_matrix_sync writes [16,16] to
        // memory with given leading dimension. If we store to output[N_start, M_start]
        // with ld=M, we get C_T stored in the output's row-major layout at the
        // wrong position. Instead, just store to shared temp and transpose.
        //
        // Use the first 256 floats of the shared workspace (safe because all warps
        // are processing different tiles and we sync after the full loop).
        // Each warp needs its own temp. Use b_staging reinterpreted:
        // b_staging has 256 halves = 512 bytes = 128 floats. NOT enough.
        //
        // Real fix: store row-major to output at transposed position.
        // C_T[n, m] = acc[n_local, m_local] at output[m, n].
        // wmma::store_matrix_sync stores acc[r][c] to addr[r * ld + c].
        // If we store to output + N_start * M + M_start with ld = M, we get:
        //   output[N_start + r, M_start + c] (if output were [N, M])
        // But output is [M, N], so this writes to the wrong place.
        //
        // Must use element-by-element extraction. wmma fragments expose .x[] array.
        // acc.num_elements = 8 per thread. The mapping from acc.x[i] to (row, col)
        // is architecture-dependent but we can store to temp and read back.
        //
        // Allocate temp in output_smem itself at a safe location:
        // output_smem[M_start * N + N_start] is where this tile's output goes.
        // Store C_T there temporarily, then transpose in-place.
        float* tile_out = output_smem + M_start * N + N_start;

        // Store C_T row-major into tile_out with ld=N (the output's full width)
        // This puts C_T[r,c] at tile_out[r * N + c] = output[(M_start) * N + N_start + r * N + c]
        //                                            = output[M_start + r, N_start + c] if we read as [rows, cols]
        // But we want output[m, n] = C_T[n, m]. With M_start as token offset and N_start as feature offset:
        // We're computing C_T[N, M] = W[N, K] × input^T[K, M]
        // C_T[n_local, m_local] should go to output[(M_start + m_local) * N + (N_start + n_local)]
        // wmma::store with base = output + M_start * N + N_start, ld = N stores:
        //   acc[r, c] → base[r * N + c] = output[M_start * N + N_start + r * N + c]
        //                                = output[(M_start + r) * N + (N_start + c)]
        // This puts r in the M dimension and c in the N dimension.
        // acc[r, c] = C_T[N_start + r, M_start + c]
        // So output[(M_start + r), (N_start + c)] = C_T[N_start + r, M_start + c]
        // We want output[m, n] = C_T[n, m], i.e., output[M_start + c, N_start + r] = C_T[N_start + r, M_start + c]
        // But we're getting output[M_start + r, N_start + c] = C_T[N_start + r, M_start + c]
        // This is output[M_start + r, N_start + c] = C_T[N_start + r, M_start + c]
        // i.e., m = M_start + r, n = N_start + c, and the stored value is C_T[n, m] — wait, no:
        //   C_T[N_start + r, M_start + c] — n_idx = N_start + r, m_idx = M_start + c
        //   Stored at output[M_start + r, N_start + c] — m = M_start + r, n = N_start + c
        //   So output[m, n] gets C_T[N_start + r, M_start + c] where m = M_start + r, n = N_start + c
        //   = C_T[n - N_start + N_start + (m - M_start - r + r)...
        // This is confusing. Let me just check: does r map to the m dimension or n dimension?
        //
        // The GEMM is C_T[N, M] = A[N, K] × B[K, M]. A is weight [N, K], B is input^T [K, M].
        // wmma computes C_T with M-tiles mapping to the "M" dimension of the GEMM (= N of weight)
        // and N-tiles mapping to the "N" dimension (= M of input).
        //
        // Wait — I got the tile mapping confused. Let me re-read the tile assignment:
        //   n_tile = tile_idx / N_tiles (maps to GEMM's M dimension = weight's N)
        //   m_tile = tile_idx % N_tiles (maps to GEMM's N dimension = input's M)
        // So N_start is the start in the weight/output-feature dimension
        // And M_start is the start in the token dimension
        //
        // The accumulator acc[r, c] contains C_T[N_start + r, M_start + c]
        // We want output[M_start + c, N_start + r] (transpose of C_T)
        //
        // If we store with ld=N at output + M_start * N + N_start:
        //   output[(M_start)*N + N_start + r*N + c] = acc[r,c]
        //   = output[M_start + r][N_start + c]   (interpreting as [M, N] matrix)
        //   = C_T[N_start + r, M_start + c]
        //
        // But we want output[m][n] = C_T[n, m], so:
        //   output[M_start + c][N_start + r] = C_T[N_start + r, M_start + c]
        //
        // The store puts it at [M_start + r][N_start + c] but we need [M_start + c][N_start + r].
        // r and c are swapped. The store is WRONG for transposition.
        //
        // For correct transposition, store at output + N_start * M + M_start with ld=M:
        // But wait, output is [M, N], ld should be N. We can't change ld per tile.
        //
        // The only correct approach without a temp buffer:
        // Store row-major with ld=16 to a per-warp temp, then scatter-transpose.
        // We need 256 floats of temp per warp.
        // Solution: increase workspace to have 256 floats per warp for tile_temp.
        // For now, just use shared memory outside the workspace:
        // The reduce buffer at TF_REDUCE_OFFSET has 256 floats, not used during GEMMs.
        // But that's shared across all warps. Not safe.
        //
        // Practical fix: use the output buffer itself. Store the raw 16×16 tile
        // at the output location (wrong order), then do an in-place transpose.
        // But 16×16 in-place transpose within shared memory is non-trivial.
        //
        // SIMPLEST FIX: just do element-by-element accumulation via scalar loop.
        // 256 elements / 32 lanes = 8 per thread. Very fast.

        // Store acc to per-warp temp using b_staging as half-precision temp
        // (we only need the values, precision loss is acceptable for store/reload)
        // Actually NO — we need FP32 precision for the output.
        //
        // Just compute the output elements directly from the accumulator.
        // wmma fragment .x[] elements have architecture-dependent mapping,
        // so we must store to temp memory first.
        //
        // Use output_smem as temp: store at the CORRECT transposed position row by row.
        // wmma doesn't support this directly. Use cooperative store:

        // FINAL APPROACH: store to a temporary 16x16 tile in shared memory,
        // then transpose-scatter into output. Allocate 256 floats per warp
        // in the workspace area beyond the staging regions.
        // Current workspace: 8*256 (a_staging) + 8*256 (b_staging) = 4096 halves = 2048 floats
        // Add 8*256 = 2048 more floats for tile_temp. Total workspace: 4096 floats = 16 KB.
        // This fits in TF_WORKSPACE_SIZE (6144 floats = 24 KB).
        float* tile_temp = (float*)(workspace + 2 * 8 * 256) + warp_id * 256;
        wmma::store_matrix_sync(tile_temp, acc, 16, wmma::mem_row_major);

        // Transpose: tile_temp[r * 16 + c] = C_T[N_start+r, M_start+c]
        //          → output[(M_start+c) * N + (N_start+r)]
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, c = i % 16;
            int out_m = M_start + c, out_n = N_start + r;
            if (out_m < M && out_n < N) {
                if (accumulate)
                    output_smem[out_m * N + out_n] += tile_temp[i];
                else
                    output_smem[out_m * N + out_n] = tile_temp[i];
            }
        }
        __syncwarp();
    }

    // Add bias if provided
    if (bias) {
        __syncthreads();
        for (int i = tid; i < M * N; i += blockDim.x)
            output_smem[i] += bias[i % N];
    }
    __syncthreads();
}

// ============================================================
// LayerNorm: per-token normalization — all tokens in parallel
//
// 256 threads, 64 tokens → 4 threads per token.
// Each thread handles 32 elements of d_model=128.
// Reductions via warp shuffle across 4-thread groups.
// No __syncthreads needed for reduction (all within-warp).
// ============================================================

// Helper: reduce sum across G consecutive threads within a warp
__device__ __forceinline__ float group_reduce_sum(float val, int G) {
    for (int s = G / 2; s > 0; s >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, s);
    return val;
}

// Helper: reduce max across G consecutive threads within a warp
__device__ __forceinline__ float group_reduce_max(float val, int G) {
    for (int s = G / 2; s > 0; s >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, s);
        val = fmaxf(val, other);
    }
    return val;
}

// Helper: broadcast from lane 0 of each G-thread group
__device__ __forceinline__ float group_broadcast(float val, int group_lane, int G) {
    return __shfl_sync(0xFFFFFFFF, val, (threadIdx.x / G) * G);
}

__device__ void tf_layer_norm(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int num_tokens, int d_model,
    float* __restrict__ smem_reduce  // unused in parallel version but kept for API compat
) {
    int tid = threadIdx.x;

    // For d_model=128, num_tokens=64: 4 threads per token, 32 elements per thread
    // For other sizes: fall back to serial if ratio doesn't work cleanly
    int threads_per_tok = blockDim.x / num_tokens;  // 256/64 = 4
    int tok = tid / threads_per_tok;
    int group_lane = tid % threads_per_tok;
    int elems_per_thread = d_model / threads_per_tok;  // 128/4 = 32
    int d_start = group_lane * elems_per_thread;

    if (tok >= num_tokens) { __syncthreads(); return; }

    const float* x = input + tok * d_model;
    float* y = output + tok * d_model;

    // Mean: each thread sums its 32 elements
    float local_sum = 0.0f;
    for (int d = 0; d < elems_per_thread; d++)
        local_sum += x[d_start + d];

    float total_sum = group_reduce_sum(local_sum, threads_per_tok);
    float mean = group_broadcast(total_sum, group_lane, threads_per_tok) / (float)d_model;

    // Variance
    float local_var = 0.0f;
    for (int d = 0; d < elems_per_thread; d++) {
        float diff = x[d_start + d] - mean;
        local_var += diff * diff;
    }

    float total_var = group_reduce_sum(local_var, threads_per_tok);
    float inv_std = rsqrtf(group_broadcast(total_var, group_lane, threads_per_tok) / (float)d_model + 1e-5f);

    // Normalize
    for (int d = 0; d < elems_per_thread; d++) {
        int idx = d_start + d;
        y[idx] = (x[idx] - mean) * inv_std * gamma[idx] + beta[idx];
    }
    __syncthreads();
}

// ============================================================
// Row-wise softmax — all rows in parallel
//
// 256 threads, 64 rows → 4 threads per row.
// Each thread handles 16 elements (cols=64, 64/4=16).
// Reductions via warp shuffle across 4-thread groups.
// ============================================================
__device__ void tf_softmax_rows(
    float* __restrict__ data,
    int rows, int cols,
    float* __restrict__ smem_reduce  // unused in parallel version
) {
    int tid = threadIdx.x;

    int threads_per_row = blockDim.x / rows;  // 256/64 = 4
    int row = tid / threads_per_row;
    int group_lane = tid % threads_per_row;
    int elems_per_thread = cols / threads_per_row;  // 64/4 = 16
    int c_start = group_lane * elems_per_thread;

    if (row >= rows) { __syncthreads(); return; }

    float* r = data + row * cols;

    // Max
    float local_max = -FLT_MAX;
    for (int c = 0; c < elems_per_thread; c++)
        if (r[c_start + c] > local_max) local_max = r[c_start + c];

    float row_max = group_reduce_max(local_max, threads_per_row);
    row_max = group_broadcast(row_max, group_lane, threads_per_row);

    // Exp and sum
    float local_sum = 0.0f;
    for (int c = 0; c < elems_per_thread; c++) {
        float val = expf(r[c_start + c] - row_max);
        r[c_start + c] = val;
        local_sum += val;
    }

    float total_sum = group_reduce_sum(local_sum, threads_per_row);
    float inv_sum = 1.0f / group_broadcast(total_sum, group_lane, threads_per_row);

    // Normalize
    for (int c = 0; c < elems_per_thread; c++)
        r[c_start + c] *= inv_sum;

    __syncthreads();
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
// ============================================================
// LayerNorm with FP16 output — compute in FP32, convert at final write
// ============================================================
__device__ void tf_layer_norm_f16out(
    const float* __restrict__ input,
    half* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int num_tokens, int d_model,
    float* __restrict__ smem_reduce
) {
    int tid = threadIdx.x;
    int threads_per_tok = blockDim.x / num_tokens;
    int tok = tid / threads_per_tok;
    int group_lane = tid % threads_per_tok;
    int elems_per_thread = d_model / threads_per_tok;
    int d_start = group_lane * elems_per_thread;

    if (tok >= num_tokens) { __syncthreads(); return; }

    const float* x = input + tok * d_model;
    half* y = output + tok * d_model;

    float local_sum = 0.0f;
    for (int d = 0; d < elems_per_thread; d++)
        local_sum += x[d_start + d];
    float total_sum = group_reduce_sum(local_sum, threads_per_tok);
    float mean = group_broadcast(total_sum, group_lane, threads_per_tok) / (float)d_model;

    float local_var = 0.0f;
    for (int d = 0; d < elems_per_thread; d++) {
        float diff = x[d_start + d] - mean;
        local_var += diff * diff;
    }
    float total_var = group_reduce_sum(local_var, threads_per_tok);
    float inv_std = rsqrtf(group_broadcast(total_var, group_lane, threads_per_tok) / (float)d_model + 1e-5f);

    for (int d = 0; d < elems_per_thread; d++) {
        int idx = d_start + d;
        y[idx] = __float2half((x[idx] - mean) * inv_std * gamma[idx] + beta[idx]);
    }
    __syncthreads();
}

// ============================================================
// Linear layer with FP16 input (already in shared memory)
// Uses dedicated staging region that doesn't overlap workspace.
// ============================================================
__device__ void tf_linear_f16in(
    const half* __restrict__ input_smem,
    const half* __restrict__ weight_fp16,
    float* __restrict__ output_smem,
    const float* __restrict__ bias,
    half* __restrict__ staging,
    int M, int N, int K,
    bool accumulate
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    int M_tiles = (N + 15) / 16;
    int N_tiles = (M + 15) / 16;
    int K_tiles = (K + 15) / 16;
    int total_tiles = M_tiles * N_tiles;
    bool a_aligned = (K % 16 == 0);

    half* a_staging = staging + warp_id * 256;
    half* b_staging = staging + 8 * 256 + warp_id * 256;

    if (!accumulate) {
        for (int i = tid; i < M * N; i += blockDim.x) output_smem[i] = 0.0f;
        __syncthreads();
    }

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += 8) {
        int n_tile = tile_idx / N_tiles;
        int m_tile = tile_idx % N_tiles;
        int N_start = n_tile * 16;
        int M_start = m_tile * 16;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
            int K_start = k_tile * 16;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            if (a_aligned && K_start + 16 <= K && N_start + 16 <= N) {
                wmma::load_matrix_sync(a_frag, weight_fp16 + N_start * K + K_start, K);
            } else {
                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16, c = i % 16;
                    int n = N_start + r, k = K_start + c;
                    a_staging[i] = (n < N && k < K) ? weight_fp16[n * K + k] : __float2half(0.0f);
                }
                __syncwarp();
                wmma::load_matrix_sync(a_frag, a_staging, 16);
            }

            // B fragment: input is already FP16 — skip FP32→FP16 conversion
            for (int i = lane; i < 256; i += 32) {
                int r = i / 16, c = i % 16;
                int k = K_start + r, m = M_start + c;
                b_staging[i] = (k < K && m < M) ? input_smem[m * K + k] : __float2half(0.0f);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, b_staging, 16);

            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        float* tile_temp = (float*)(staging + 2 * 8 * 256) + warp_id * 256;
        wmma::store_matrix_sync(tile_temp, acc, 16, wmma::mem_row_major);

        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, c = i % 16;
            int out_m = M_start + c, out_n = N_start + r;
            if (out_m < M && out_n < N) {
                if (accumulate)
                    output_smem[out_m * N + out_n] += tile_temp[i];
                else
                    output_smem[out_m * N + out_n] = tile_temp[i];
            }
        }
        __syncwarp();
    }

    if (bias) {
        __syncthreads();
        for (int i = tid; i < M * N; i += blockDim.x)
            output_smem[i] += bias[i % N];
    }
    __syncthreads();
}

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
