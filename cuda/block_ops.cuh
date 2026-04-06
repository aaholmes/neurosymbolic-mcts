#pragma once

#include "common.cuh"

#ifdef __CUDACC__

// ============================================================
// Block-Cooperative Neural Network Operations
//
// All functions are __device__ and expect to be called by all
// threads of a 256-thread block (blockDim.x == 256).
// Work is striped across threads via threadIdx.x.
// All functions end with __syncthreads().
//
// Activation buffers (data, input_smem, output_smem) live in
// shared memory for cache efficiency.
// ============================================================

// --- BatchNorm + optional ReLU, in-place on [channels, 64] ---
__device__ void block_bn_relu(
    float* data,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int channels,
    bool relu
);

// --- Single-channel BN + optional ReLU on [1, 64] ---
__device__ void block_bn_relu_1ch(
    float* data,             // [64] in shared memory
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    bool relu
);

// --- 3x3 convolution (direct, no im2col) from shared memory ---
// input_smem:  [C_in, 64] in shared memory
// weights:     [C_out, C_in, 9] in global memory (OIHW, 3x3 kernel flattened)
// output_smem: [C_out, 64] in shared memory
// input_smem and output_smem must NOT alias.
__device__ void block_conv_3x3(
    const float* __restrict__ input_smem,
    const float* __restrict__ weights,
    float* __restrict__ output_smem,
    int C_in, int C_out
);

// --- 1x1 convolution (direct) from shared memory ---
// input_smem:  [C_in, 64] in shared memory
// weights:     [C_out, C_in] in global memory
// output_smem: [C_out, 64] in shared memory
// input_smem and output_smem must NOT alias.
__device__ void block_1x1_conv(
    const float* __restrict__ input_smem,
    const float* __restrict__ weights,
    float* __restrict__ output_smem,
    int C_in, int C_out
);

// --- Squeeze-and-Excitation block, in-place on [channels, 64] ---
// smem_avg:  [channels] shared workspace for global avg pool results
// smem_fc1:  [inner]    shared workspace for FC1 output
__device__ void block_se_block(
    float* data,
    const float* fc1_w,    // [inner, channels]
    const float* fc2_w,    // [channels, inner]
    int channels,
    int inner,
    float* smem_avg,       // shared float[channels]
    float* smem_fc1        // shared float[inner]
);

// --- Log-softmax in-place over 1D array ---
// data may live in global or shared memory.
// smem_reduce: shared float[256] workspace for parallel reduction.
__device__ void block_log_softmax(
    float* data,
    int size,
    float* smem_reduce     // shared float[blockDim.x]
);

// --- Board encoding: BoardState -> [17, 8, 8] float planes ---
// planes lives in shared memory; output is STM-relative (flips for Black).
__device__ void block_board_to_planes(
    const BoardState* bs,
    float* planes           // [17*64] in shared memory
);

#endif // __CUDACC__
