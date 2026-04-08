#pragma once

#include "common.cuh"
#include "transformer_weights.cuh"

#ifdef __CUDACC__

// ============================================================
// Transformer Operations — 256-thread block cooperative
//
// All functions expect blockDim.x == 256.
// Activations in shared memory, weights in global FP16.
// ============================================================

// Dense GEMM: C[M, N] = A[M, K] × B[K, N] using wmma FP16
// A: FP16 global memory (weights), row-major, ld_a
// B: FP32 shared memory (activations), row-major, ld_b = N
// C: FP32 shared memory (output), row-major, ld_c = N
// Accumulates into C (does NOT zero C first).
// Uses workspace for FP32→FP16 conversion of B rows + staging.
__device__ void tf_gemm_acc(
    const half* __restrict__ A_global,    // [M, K] FP16 weights
    const float* __restrict__ B_smem,     // [K, N] FP32 activations
    float* __restrict__ C_smem,           // [M, N] FP32 output (accumulated)
    half* __restrict__ workspace,         // FP16 workspace in smem
    int M, int N, int K, int ld_a
);

// Dense GEMM that zeros C first, then accumulates
__device__ void tf_gemm(
    const half* __restrict__ A_global,
    const float* __restrict__ B_smem,
    float* __restrict__ C_smem,
    half* __restrict__ workspace,
    int M, int N, int K, int ld_a
);

// LayerNorm: out[token, d] = gamma[d] * (x[token, d] - mean) / sqrt(var + eps) + beta[d]
// Per-token normalization. Input and output may alias.
// smem_reduce: [256] floats workspace for parallel reduction
__device__ void tf_layer_norm(
    const float* __restrict__ input,      // [num_tokens, d_model] smem
    float* __restrict__ output,           // [num_tokens, d_model] smem (may == input)
    const float* __restrict__ gamma,      // [d_model] global
    const float* __restrict__ beta,       // [d_model] global
    int num_tokens, int d_model,
    float* __restrict__ smem_reduce       // [256] smem workspace
);

// Row-wise softmax in-place on [rows, cols] in shared memory
__device__ void tf_softmax_rows(
    float* __restrict__ data,             // [rows, cols] smem
    int rows, int cols,
    float* __restrict__ smem_reduce       // [256] smem workspace
);

// Add bias in-place: data[i] += bias[i % N] for all i < M*N
__device__ void tf_add_bias(
    float* __restrict__ data,
    const float* __restrict__ bias,
    int M, int N
);

// Elementwise add: dst[i] += src[i] for i in [0, count)
__device__ void tf_residual_add(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int count
);

// ReLU in-place
__device__ void tf_relu(float* data, int count);

// GEMM with both operands in shared memory FP32:
// C[M,N] = A_smem[M,K] × B_smem[K,N], convert both to FP16 on-the-fly
__device__ void tf_gemm_smem(
    const float* __restrict__ A_smem,
    const float* __restrict__ B_smem,
    float* __restrict__ C_smem,
    half* __restrict__ workspace,
    int M, int N, int K
);

// GEMM with A×B^T, both in shared memory FP32:
// C[M,N] = A_smem[M,K] × B_smem[N,K]^T, with optional scaling
__device__ void tf_gemm_smem_abt(
    const float* __restrict__ A_smem,  // [M, K]
    const float* __restrict__ B_smem,  // [N, K] — transposed in multiply
    float* __restrict__ C_smem,        // [M, N]
    half* __restrict__ workspace,
    int M, int N, int K,
    float scale = 1.0f
);

// Linear layer: output[M, N] = input[M, K] × weight^T[K, N] + bias[N]
// weight_fp16: [N, K] row-major in global FP16 (PyTorch convention)
// input_smem: [M, K] row-major in shared FP32
// output_smem: [M, N] row-major in shared FP32 (zeroed then written)
// bias: [N] global FP32 or nullptr
// workspace: FP16 staging for wmma (needs 2 * 8 * 256 halves = 4096 halves)
// accumulate: if true, ADD to output_smem instead of overwriting
__device__ void tf_linear(
    const float* __restrict__ input_smem,
    const half* __restrict__ weight_fp16,
    float* __restrict__ output_smem,
    const float* __restrict__ bias,
    half* __restrict__ workspace,
    int M, int N, int K,
    bool accumulate = false
);

// Board encoding to [64, 17] FP32 (token-major: each token is one square's 17 features)
// Different from block_board_to_planes which outputs [17, 64] (channel-major)
__device__ void tf_board_to_tokens(
    const BoardState* bs,
    float* tokens    // [64, 17] in smem — token t has 17 features at tokens[t*17]
);

#endif // __CUDACC__
