#pragma once

#include "common.cuh"
#include <cuda_fp16.h>

#ifdef __CUDACC__

// ============================================================
// Block-Cooperative Neural Network Operations (FP16 activations)
//
// All functions are __device__ and expect to be called by all
// threads of a 256-thread block (blockDim.x == 256).
// Work is striped across threads via threadIdx.x.
// All functions end with __syncthreads().
//
// Activation buffers (data, input_smem, output_smem) are FP16 in
// shared memory. Internal arithmetic is FP32 (load FP16 → FP32
// register → compute → __float2half → store) to preserve precision
// in BN, SE, and conv accumulation.
//
// The single-channel BN (block_bn_relu_1ch) and log-softmax stay
// FP32 because they operate on the value-head feature vector and
// the policy log-probs respectively — both are precision-critical
// and routed outside the FP16 activation buffers.
// ============================================================

// --- BatchNorm + optional ReLU, in-place on [channels, 64] FP16 ---
__device__ void block_bn_relu(
    __half* data,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int channels,
    bool relu
);

// --- Single-channel BN + optional ReLU on [1, 64] FP32 ---
// Used by the value head, which is kept FP32 end-to-end.
__device__ void block_bn_relu_1ch(
    float* data,             // [64] FP32 in shared memory
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    bool relu
);

// --- 3x3 convolution (direct, no im2col) FP16 in/out ---
// input_smem:  [C_in, 64] FP16 in shared memory
// weights:     [C_out, C_in, 9] FP32 in global memory (OIHW)
// output_smem: [C_out, 64] FP16 in shared memory
// Internal accumulator is FP32; FP16-load/FP32-compute/FP16-store.
__device__ void block_conv_3x3(
    const __half* __restrict__ input_smem,
    const float* __restrict__ weights,
    __half* __restrict__ output_smem,
    int C_in, int C_out
);

__device__ void block_conv_3x3_smem_w(
    const __half* __restrict__ input_smem,
    const float* __restrict__ weights,
    __half* __restrict__ output_smem,
    float* __restrict__ smem_weights,
    int C_in, int C_out
);

// --- 3x3 convolution via Tensor Core (wmma FP16) ---
// input_smem:    [C_in, 64] FP16 in shared memory
// weights_h:     [C_out, C_in, 9] FP16 in global memory (pre-converted)
// output_smem:   [C_out, 64] FP16 in shared memory
// smem_staging:  half-precision per-warp im2col workspace + FP32 acc staging.
//                Must be at least 8192 halves total. Per-warp acc-conversion
//                staging is carved from this region after the conv finishes.
// Requires blockDim.x == 256 (8 warps). FP32 accumulator inside WMMA.
__device__ void block_conv_3x3_tc(
    const __half* __restrict__ input_smem,
    const half* __restrict__ weights_h,
    __half* __restrict__ output_smem,
    half* __restrict__ smem_staging,
    int C_in, int C_out
);

// --- 3x3 convolution via shifted-copy GEMM ---
// input_smem:  [C_in, 64] FP16 in shared memory
// W_s:         9 half* pointers to [C_out, C_in] FP16 in global memory
// output_smem: [C_out, 64] FP16 in shared memory
// shifted:     [C_in*64] FP16 workspace in shared memory; reused at end
//              for per-warp FP32 acc-conversion staging.
__device__ void block_conv_3x3_shifted(
    const __half* __restrict__ input_smem,
    half* const* W_s,
    __half* __restrict__ output_smem,
    half* __restrict__ shifted,
    int C_in, int C_out
);

// --- 1x1 convolution (direct) FP16 in/out ---
__device__ void block_1x1_conv(
    const __half* __restrict__ input_smem,
    const float* __restrict__ weights,
    __half* __restrict__ output_smem,
    int C_in, int C_out
);

// --- Squeeze-and-Excitation block, in-place on [channels, 64] FP16 ---
// smem_avg:  FP32 [channels] shared workspace for global avg pool results
// smem_fc1:  FP32 [inner]    shared workspace for FC1 output
__device__ void block_se_block(
    __half* data,
    const float* fc1_w,    // [inner, channels]
    const float* fc2_w,    // [channels, inner]
    int channels,
    int inner,
    float* smem_avg,
    float* smem_fc1
);

// --- Log-softmax in-place over 1D FP32 array ---
// data lives in global memory (policy_out) — kept FP32 for precision.
// smem_reduce: FP32 [256] workspace for parallel reduction.
__device__ void block_log_softmax(
    float* data,
    int size,
    float* smem_reduce
);

// --- Board encoding: BoardState -> [17, 8, 8] FP16 planes ---
// planes lives in shared memory; output is STM-relative (flips for Black).
__device__ void block_board_to_planes(
    const BoardState* bs,
    __half* planes           // [17*64] FP16 in shared memory
);

#endif // __CUDACC__
