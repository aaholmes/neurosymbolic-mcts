#pragma once

#include "common.cuh"
#include "nn_weights.cuh"

#ifdef __CUDACC__

// ============================================================
// Shared memory layout for oracle_net_forward_block
//
// buf1 (smem+0):          work buffer [128, 64] = 32 KB
//   — board planes at start, policy conv output, v_feat during value head
// buf2 (smem+8192):       backbone buffer [128, 64] = 32 KB
//   — start conv output, kept as backbone through policy head
// smem_reduce (+16384):   256 floats = 1 KB, multipurpose:
//   — SE avg pool  [0:128]
//   — SE FC1 out   [128:136]
//   — log-softmax reduction [0:256]
//   — value FC1 out [0:256]
// smem_weights (+16640):  1152 floats = 4.5 KB (scalar conv path)
//   — 3x3 conv weight tile cache: C_out * 9 = 128 * 9 = 1152
//   — loaded once per input channel, reused by all 256 threads
// smem_staging (+16640):  1024 floats = 4 KB (TC im2col path, overlaps smem_weights)
//   — 8 warps × 256 halves = 2048 halves = 4096 bytes = 1024 floats
//   — per-warp FP16 im2col tile for wmma load
// smem_shifted (+16640):  4096 floats = 16 KB (shifted-copy path, overlaps above)
//   — [128, 64] FP16 shifted input copy = 8192 halves = 16,384 bytes
// Total: 20736 floats = 82,944 bytes ≈ 81 KB (fits in 96 KB per SM)
// ============================================================

constexpr int BLOCK_BUF_SIZE    = NN_HIDDEN_DIM * 64;   // 8192 floats per buffer
constexpr int BLOCK_REDUCE_SIZE = 256;                    // reduction workspace
constexpr int BLOCK_WEIGHTS_SIZE = NN_HIDDEN_DIM * 9;    // 1152 floats for scalar conv weight tile
constexpr int BLOCK_STAGING_SIZE = 1024;                  // 8 warps × 256 halves = 1024 floats (TC im2col path)
constexpr int BLOCK_SHIFTED_SIZE = NN_HIDDEN_DIM * 64 / 2;  // 4096 floats = 8192 halves = 16 KB (shifted-copy path)

constexpr int BLOCK_BUF1_OFFSET   = 0;
constexpr int BLOCK_BUF2_OFFSET   = BLOCK_BUF1_OFFSET + BLOCK_BUF_SIZE;     // 8192
constexpr int BLOCK_REDUCE_OFFSET = BLOCK_BUF2_OFFSET + BLOCK_BUF_SIZE;     // 16384
constexpr int BLOCK_WEIGHTS_OFFSET = BLOCK_REDUCE_OFFSET + BLOCK_REDUCE_SIZE; // 16640
constexpr int BLOCK_STAGING_OFFSET = BLOCK_WEIGHTS_OFFSET;                    // 16640 (overlaps)
constexpr int BLOCK_SHIFTED_OFFSET = BLOCK_WEIGHTS_OFFSET;                    // 16640 (overlaps)

// Use max of all overlapping auxiliary regions for total size
constexpr int BLOCK_AUX_SIZE = BLOCK_SHIFTED_SIZE;  // 4096 (largest of weights/staging/shifted)
constexpr int BLOCK_SMEM_FLOATS = BLOCK_WEIGHTS_OFFSET + BLOCK_AUX_SIZE;  // 20736
constexpr int BLOCK_SMEM_BYTES  = BLOCK_SMEM_FLOATS * 4;                  // 82,944 bytes

// ============================================================
// Full SE-ResNet forward pass using 256-thread block cooperation.
//
// All 256 threads must participate. Activations live in shared memory.
// The residual shortcut for each residual block is saved in per-thread
// registers (32 floats/thread) to avoid needing a third smem buffer.
//
// smem:       caller-provided, at least BLOCK_SMEM_FLOATS floats
// policy_out: global memory [NN_POLICY_SIZE] — written as log-probs
// value_out:  written by thread 0, valid after function returns
// k_out:      written by thread 0, valid after function returns
// half_w:     optional FP16 conv weights for Tensor Core im2col path (nullptr = skip)
// shifted_w:  optional per-position FP16 weights for shifted-copy path (nullptr = skip)
// Priority: shifted_w > half_w > scalar fallback
// ============================================================
__device__ void oracle_net_forward_block(
    const BoardState* bs,
    float q_result,
    const OracleNetWeights* weights,
    float* smem,
    float* policy_out,
    float* value_out,
    float* k_out,
    const ConvWeightsHalf* half_w = nullptr,
    const ConvWeightsShifted* shifted_w = nullptr
);

#endif // __CUDACC__

// Returns BLOCK_SMEM_BYTES (for use in kernel launch dynamic smem size).
int block_smem_bytes();
