#pragma once

#include "common.cuh"

#ifdef __CUDACC__

// ============================================================
// Warp-Cooperative Neural Network Operations
//
// All functions are __device__ and expect to be called by a full
// warp (32 threads). Work is distributed across lanes via
// threadIdx.x % 32. All threads must call each function together
// (implicit warp synchronization via lockstep execution).
//
// These implement the building blocks of OracleNet's SE-ResNet:
// GEMM, BatchNorm, ReLU, SE attention, im2col for 3x3 conv.
// ============================================================

// --- Matrix multiply: C[M,N] = A[M,K] × B[K,N] ---
// Each of 32 threads computes ceil(M*N/32) output elements.
__device__ void warp_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

// --- im2col for 3x3 convolution with padding=1 ---
// Transforms [C_in, 8, 8] → [C_in*9, 64] column matrix.
__device__ void warp_im2col_3x3(
    const float* input,  // [C_in, 8, 8]
    float* col,          // [C_in*9, 64]
    int C_in
);

// --- BatchNorm + optional ReLU (fused) ---
// Operates in-place on [channels, 64] data.
// BN formula: (x - mean) / sqrt(var + eps) * gamma + beta
__device__ void warp_bn_relu(
    float* data,             // [channels, 64] — modified in-place
    const float* gamma,      // [channels]
    const float* beta,       // [channels]
    const float* running_mean, // [channels]
    const float* running_var,  // [channels]
    int channels,
    bool relu
);

// --- Single-channel BN + optional ReLU ---
__device__ void warp_bn_relu_1ch(
    float* data,             // [1, 64]
    const float* gamma,      // [1]
    const float* beta,       // [1]
    const float* running_mean, // [1]
    const float* running_var,  // [1]
    bool relu
);

// --- Elementwise add + ReLU: out[i] = max(0, a[i] + b[i]) ---
__device__ void warp_add_relu(
    const float* a, const float* b, float* out,
    int size
);

// --- Squeeze-and-Excitation block ---
// Global avg pool → FC(channels→inner) + ReLU → FC(inner→channels) + Sigmoid → Scale
__device__ void warp_se_block(
    float* data,             // [channels, 64] — modified in-place
    const float* fc1_w,      // [inner, channels] (no bias)
    const float* fc2_w,      // [channels, inner] (no bias)
    int channels,
    int inner                // reduction dim (8 for 128/16)
);

// --- Log-softmax over a 1D array ---
__device__ void warp_log_softmax(
    const float* input,
    float* output,
    int size
);

// --- Board encoding: BoardState → [17, 8, 8] float planes ---
// STM-relative (flips ranks for Black)
__device__ void warp_board_to_planes(
    const BoardState* bs,
    float* planes           // [17*64] output
);

// --- AlphaZero move encoding ---
// Maps a GPUMove to a policy tensor index (0..4671).
// Uses 73-plane encoding: 56 queen slides + 8 knight + 9 underpromotions.
// For Black (w_to_move=false), flips the move vertically before encoding.
__device__ int move_to_policy_index(GPUMove mv, bool w_to_move);

#endif // __CUDACC__
