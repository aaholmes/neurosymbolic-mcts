#pragma once

#include "common.cuh"
#include "nn_weights.cuh"

#ifdef __CUDACC__

// ============================================================
// Scratch buffer layout for one forward pass
// Each explorer warp needs its own scratch space.
// ============================================================

// Sizes in floats
constexpr int SCRATCH_BUF_SIZE = NN_HIDDEN_DIM * 64;         // 128*64 = 8192
constexpr int SCRATCH_COL_SIZE = NN_HIDDEN_DIM * 9 * 64;     // 128*9*64 = 73728 (largest im2col)
constexpr int SCRATCH_PLANES_SIZE = NN_INPUT_CHANNELS * 64;   // 17*64 = 1088
constexpr int SCRATCH_POLICY_SIZE = NN_POLICY_SIZE;            // 4672

// Total scratch per warp (in floats)
constexpr int SCRATCH_TOTAL_FLOATS =
    SCRATCH_BUF_SIZE * 2 +      // buf1 + buf2 (backbone activations)
    SCRATCH_COL_SIZE +           // im2col column buffer
    SCRATCH_PLANES_SIZE +        // input planes
    SCRATCH_POLICY_SIZE;         // policy output

// Total scratch per warp in bytes
constexpr int SCRATCH_TOTAL_BYTES = SCRATCH_TOTAL_FLOATS * sizeof(float);
// = (8192*2 + 73728 + 1088 + 4672) * 4 = 383,488 bytes ≈ 375 KB

// ============================================================
// Full OracleNet forward pass (device-side, warp-cooperative)
// ============================================================

// Run the full SE-ResNet forward pass on a single position.
// Called cooperatively by all 32 threads in a warp.
//
// Inputs:
//   bs        — the board position to evaluate
//   q_result  — PeSTO material balance in pawn units (from PE q-search)
//   weights   — shared NN weights in global memory (read-only)
//   scratch   — per-warp scratch buffer (SCRATCH_TOTAL_FLOATS floats)
//
// Outputs (written by thread 0, valid after __syncwarp):
//   policy_out — [4672] log-probabilities (AlphaZero move encoding)
//   value_out  — final value: tanh(v_logit + k * q_result)
//   k_out      — confidence scalar: 0.47 * softplus(k_logit)
__device__ void oracle_net_forward(
    const BoardState* bs,
    float q_result,
    const OracleNetWeights* weights,
    float* scratch,
    float* policy_out,
    float* value_out,
    float* k_out
);

#endif // __CUDACC__

// ============================================================
// Host-side API
// ============================================================

// Allocate scratch buffers for N warps.
// Returns pointer to contiguous GPU memory (N × SCRATCH_TOTAL_FLOATS floats).
float* alloc_nn_scratch(int num_warps);

// Free scratch buffers.
void free_nn_scratch(float* d_scratch);
