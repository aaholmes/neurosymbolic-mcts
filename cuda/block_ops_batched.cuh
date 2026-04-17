#pragma once

#include "common.cuh"
#include <cuda_fp16.h>

#ifdef __CUDACC__

// ============================================================
// Batched shifted-copy 9-GEMM 3x3 convolution (Tensor Core)
//
// Extension of block_conv_3x3_shifted (block_ops.cu:334) to
// support batch size B > 1. Baseline design for the Phase 4
// go/no-go microbenchmark in GPU_MCTS_v2.md:
//
//   - Activations live in global memory (act_in, act_out) in
//     FP32, matching the existing kernel's activation precision.
//
//   - Weight slice for the current shift is staged through
//     shared memory once per (pass, shift) and reused across
//     all K-tiles of that shift's GEMM.
//
//   - Shifted-copy buffer [C_in, 64] FP16 lives in global
//     scratch and is rebuilt per (pass, shift). Size is
//     independent of B.
//
//   - Outer loop runs B passes, each identical in shape to the
//     existing single-batch kernel (8 warps, M-split on C_out,
//     4 accs per warp covering all 64 spatial positions).
//
// The measured speedup at B=32 vs B=1 reflects the
// amortization of kernel-launch overhead and any cross-pass
// Tensor-Core pipeline benefits. Further speedups (keeping
// multiple batches' accumulators alive, re-splitting warps
// across spatial*batch) are Phase 4 optimizations that this
// baseline does NOT implement. If the baseline clears the
// go/no-go threshold, those optimizations would improve it
// further.
//
// Caller must launch with blockDim.x == 256 (8 warps).
// Assumes C_in and C_out multiples of 16. B >= 1.
// ============================================================

__device__ void block_conv_3x3_shifted_batched(
    const float* __restrict__ act_in,   // [B, C_in, 8, 8] FP32 in global
    half* const*             W_s,       // 9 x [C_out, C_in] FP16 device ptrs
    float* __restrict__      act_out,   // [B, C_out, 8, 8] FP32 in global
    half* __restrict__       w_smem,    // shared: [C_out * C_in] FP16 staging
    float* __restrict__      acc_smem,  // shared: at least 8 warps * 256 FP32
    half* __restrict__       shifted,   // [C_in, 64*B] FP16 in global scratch
    int C_in, int C_out, int B);

// Smem staging size helpers
__host__ __device__ inline size_t batched_conv_w_smem_bytes(int C_in, int C_out) {
    return (size_t)C_in * C_out * sizeof(half);
}

// acc_smem is used for FP32 -> FP32 staging of accumulator tiles prior
// to cvt-and-store back to FP32 act_out. Holds one 16x16 tile per warp.
__host__ __device__ inline size_t batched_conv_acc_smem_bytes(int /*unused*/) {
    return (size_t)8 /*warps*/ * 256 /*16x16*/ * sizeof(float);
}

__host__ __device__ inline size_t batched_conv_shifted_bytes(int C_in, int /*B*/) {
    // Shifted buffer is rebuilt per-pass for a single batch element.
    // Size is independent of B; the simple outer-loop design reuses one
    // [C_in, 64] FP16 buffer across all B passes.
    return (size_t)C_in * 64 * sizeof(half);
}

#endif // __CUDACC__
