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
// smem_weights (+16640):  1152 floats = 4.5 KB
//   — 3x3 conv weight tile cache: C_out * 9 = 128 * 9 = 1152
//   — loaded once per input channel, reused by all 256 threads
// Total: 17792 floats = 71,168 bytes ≈ 69.5 KB (fits in 96 KB per SM)
// ============================================================

constexpr int BLOCK_BUF_SIZE    = NN_HIDDEN_DIM * 64;   // 8192 floats per buffer
constexpr int BLOCK_REDUCE_SIZE = 256;                    // reduction workspace
constexpr int BLOCK_WEIGHTS_SIZE = NN_HIDDEN_DIM * 9;    // 1152 floats for conv weight tile

constexpr int BLOCK_BUF1_OFFSET   = 0;
constexpr int BLOCK_BUF2_OFFSET   = BLOCK_BUF1_OFFSET + BLOCK_BUF_SIZE;     // 8192
constexpr int BLOCK_REDUCE_OFFSET = BLOCK_BUF2_OFFSET + BLOCK_BUF_SIZE;     // 16384
constexpr int BLOCK_WEIGHTS_OFFSET = BLOCK_REDUCE_OFFSET + BLOCK_REDUCE_SIZE; // 16640

constexpr int BLOCK_SMEM_FLOATS = BLOCK_WEIGHTS_OFFSET + BLOCK_WEIGHTS_SIZE;  // 17792
constexpr int BLOCK_SMEM_BYTES  = BLOCK_SMEM_FLOATS * 4;                      // 71,168 bytes

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
// ============================================================
__device__ void oracle_net_forward_block(
    const BoardState* bs,
    float q_result,
    const OracleNetWeights* weights,
    float* smem,
    float* policy_out,
    float* value_out,
    float* k_out
);

#endif // __CUDACC__

// Returns BLOCK_SMEM_BYTES (for use in kernel launch dynamic smem size).
int block_smem_bytes();
