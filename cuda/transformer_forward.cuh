#pragma once

#include "common.cuh"
#include "transformer_weights.cuh"

#ifdef __CUDACC__

// ============================================================
// Shared memory layout for transformer forward pass
//
// buf_x[64×128 FP32]     @ +0       (32 KB) — residual stream
// buf_out[64×128 FP16]   @ +8192    (16 KB) — layer output (FP16 to save space)
// workspace[6144 FP32]   @ +12288   (24 KB) — Q/K/V/attn/FFN intermediate
// staging[4096 mixed]    @ +18432   (16 KB) — dedicated TC staging (a/b/tile_temp)
// reduce[256 FP32]       @ +22528   (1 KB)  — LayerNorm/softmax reduction
// Total: 22,784 floats = 91,136 bytes ≈ 89 KB
// ============================================================

constexpr int TF_BUF_SIZE           = TF_NUM_TOKENS * NN_HIDDEN_DIM;  // 8192 floats = 32 KB
constexpr int TF_BUF_OUT_SIZE_F     = TF_BUF_SIZE / 2;               // 4096 floats = 8192 halves = 16 KB
constexpr int TF_WORKSPACE_SIZE     = 6144;                           // 24 KB
constexpr int TF_STAGING_SIZE       = 4096;                           // 16 KB (TC staging)
constexpr int TF_REDUCE_SIZE        = 256;                            // 1 KB

constexpr int TF_BUF_X_OFFSET     = 0;                                           // 0
constexpr int TF_BUF_OUT_OFFSET   = TF_BUF_X_OFFSET + TF_BUF_SIZE;             // 8192
constexpr int TF_WORKSPACE_OFFSET = TF_BUF_OUT_OFFSET + TF_BUF_OUT_SIZE_F;     // 12288
constexpr int TF_STAGING_OFFSET   = TF_WORKSPACE_OFFSET + TF_WORKSPACE_SIZE;    // 18432
constexpr int TF_REDUCE_OFFSET    = TF_STAGING_OFFSET + TF_STAGING_SIZE;        // 22528

constexpr int TF_SMEM_FLOATS = TF_REDUCE_OFFSET + TF_REDUCE_SIZE;               // 22784
constexpr int TF_SMEM_BYTES  = TF_SMEM_FLOATS * 4;                              // 91136

// Full transformer forward pass.
// All 256 threads must participate.
__device__ void transformer_forward(
    const BoardState* bs,
    float q_result,
    const TransformerWeights* weights,
    const TransformerWeightsHalf* half_w,
    float* smem,
    float* policy_out,     // [NN_POLICY_SIZE] global memory
    float* value_out,
    float* k_out
);

#endif // __CUDACC__

int transformer_smem_bytes();
