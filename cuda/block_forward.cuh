#pragma once

#include "common.cuh"
#include "nn_weights.cuh"
#include <cuda_fp16.h>

#ifdef __CUDACC__

// ============================================================
// Shared memory layout for oracle_net_forward_block (FP16 activations)
//
// The conv math (shifted-copy + WMMA) was already FP16 + FP32-accumulator.
// What used to be FP32 was the inter-layer activation buffers (buf1, buf2)
// which forced an FP32→FP16 conversion at every conv call. Storing the
// activation buffers in FP16 directly halves the smem footprint and
// removes those per-call conversions.
//
// All offsets below are in HALVES (sizeof(__half) == 2 bytes).
//
// buf1 (smem+0):       work buffer [128, 64] = 8192 halves = 16 KB
//   — board planes at start, policy conv output
// buf2 (smem+8192):    backbone buffer [128, 64] = 8192 halves = 16 KB
//   — start conv output; carried as backbone through policy head
// smem_reduce (+16384): 256 floats = 512 halves = 1 KB, multipurpose:
//   — SE avg pool [0:128] (FP32)
//   — SE FC1 out [128:136] (FP32)
//   — log-softmax reduction [0:256] (FP32)
//   — value FC1 out / FC2 reduction (FP32)
//   — value-head v_feat [0:64] (FP32) — routed here for end-to-end FP32 stability
// aux (+16896):        max(scalar weights, TC staging, shifted) = 8192 halves = 16 KB
//   — scalar conv path: weight tile cache [128*9 = 1152 floats = 2304 halves]
//   — TC im2col path: 8 warps × 256 halves = 2048 halves
//   — shifted-copy path: [128, 64] FP16 = 8192 halves
//   — also reused after the conv completes as per-warp FP32 staging for
//     converting WMMA accumulator → FP16 output (8 warps × 256 floats =
//     2048 floats = 4096 halves, fits in same region)
//
// Total: 25088 halves = 50,176 bytes ≈ 49 KB (was 81 KB; fits 96 KB cap with room).
// ============================================================

constexpr int BLOCK_BUF_HALVES   = NN_HIDDEN_DIM * 64;   // 8192 halves per buffer

// Per-thread element count when 256 threads cooperate on one [NN_HIDDEN_DIM, 64] buffer.
// 6×128: 32. 6×64: 16. Used by residual-save/add unrolled loops in the b2 forward.
constexpr int BLOCK_BUF_PER_THREAD = (NN_HIDDEN_DIM * 64 + 255) / 256;
constexpr int BLOCK_REDUCE_SIZE  = 256;                  // FP32 reduction workspace (in floats)
constexpr int BLOCK_AUX_HALVES   = NN_HIDDEN_DIM * 64;   // 8192 halves (largest aux user = shifted)

constexpr int BLOCK_BUF1_OFFSET_H   = 0;
constexpr int BLOCK_BUF2_OFFSET_H   = BLOCK_BUF1_OFFSET_H + BLOCK_BUF_HALVES;       // 8192
constexpr int BLOCK_REDUCE_OFFSET_H = BLOCK_BUF2_OFFSET_H + BLOCK_BUF_HALVES;       // 16384
constexpr int BLOCK_AUX_OFFSET_H    = BLOCK_REDUCE_OFFSET_H + BLOCK_REDUCE_SIZE * 2; // 16384 + 512 = 16896

constexpr int BLOCK_SMEM_HALVES = BLOCK_AUX_OFFSET_H + BLOCK_AUX_HALVES;            // 25088
constexpr int BLOCK_SMEM_BYTES  = BLOCK_SMEM_HALVES * 2;                            // 50,176

// WMMA fragment loads need 16-byte alignment for 128-bit half loads.
static_assert((BLOCK_AUX_OFFSET_H * 2) % 16 == 0, "aux offset must be 16-byte aligned for WMMA");
static_assert((BLOCK_BUF1_OFFSET_H * 2) % 16 == 0, "buf1 offset must be 16-byte aligned for WMMA");
static_assert((BLOCK_BUF2_OFFSET_H * 2) % 16 == 0, "buf2 offset must be 16-byte aligned for WMMA");

// ============================================================
// v4 batched-conv (B=2) smem layout. Two activation buffer sets,
// two shifted buffers. Tight fit at ~99 KB.
// ============================================================
constexpr int B2_BUF_HALVES_PER  = NN_HIDDEN_DIM * 64;          // 8192 per batch
constexpr int B2_BUF1_OFFSET_H   = 0;                            // [2, 128, 64], 16384 halves
constexpr int B2_BUF2_OFFSET_H   = 2 * B2_BUF_HALVES_PER;        // 16384, [2, 128, 64]
constexpr int B2_REDUCE_OFFSET_H = B2_BUF2_OFFSET_H + 2 * B2_BUF_HALVES_PER; // 32768
constexpr int B2_REDUCE_HALVES   = 512;                          // 256 floats workspace
constexpr int B2_AUX_OFFSET_H    = B2_REDUCE_OFFSET_H + B2_REDUCE_HALVES;     // 33280
// Aux holds shifted_b2: 2 × C_in × 64 halves at C_in=128 = 16384 halves = 32 KB.
// At C_in=17 (start conv) the slow-path a_staging fits in the unused tail of
// this region (start conv only fills 2*1088=2176 halves of the 16384 available).
constexpr int B2_AUX_HALVES      = 2 * NN_HIDDEN_DIM * 64;                    // 16384
constexpr int B2_SMEM_HALVES     = B2_AUX_OFFSET_H + B2_AUX_HALVES;           // 49664
constexpr int B2_SMEM_BYTES      = B2_SMEM_HALVES * 2;                        // 99,328

static_assert(B2_SMEM_BYTES <= 99 * 1024, "B2 smem must fit 99 KB sm_120 cap");

static_assert((B2_AUX_OFFSET_H * 2) % 16 == 0, "B2 aux offset must be 16-byte aligned");
static_assert((B2_BUF1_OFFSET_H * 2) % 16 == 0, "B2 buf1 offset must be 16-byte aligned");
static_assert((B2_BUF2_OFFSET_H * 2) % 16 == 0, "B2 buf2 offset must be 16-byte aligned");

// ============================================================
// Batched forward at B=2. Two BoardStates, two q_results, two outputs.
// Same caller-side smem signature pattern (`float* smem`); reinterpreted
// as `__half*` internally. Caller must supply at least B2_SMEM_BYTES.
// ============================================================
__device__ void oracle_net_forward_block_b2(
    const BoardState* bs0, const BoardState* bs1,
    float q0, float q1,
    const OracleNetWeights* weights,
    float* smem,
    float* policy_out_b2,    // [2, NN_POLICY_SIZE] FP32 in global memory
    float* value_out_b2,     // [2] FP32, written by thread 0
    float* k_out_b2,         // [2] FP32, written by thread 0
    const ConvWeightsShifted* shifted_w = nullptr
);

// ============================================================
// Full SE-ResNet forward pass using 256-thread block cooperation.
//
// All 256 threads must participate. Activations live in shared memory.
// The residual shortcut for each residual block is saved in per-thread
// FP32 registers (32 floats/thread) — registers don't share the smem
// budget and FP32 keeps the residual stream's precision intact.
//
// smem:       caller-provided, at least BLOCK_SMEM_BYTES bytes. Pointer
//             is `float*` for caller compatibility; reinterpreted as
//             `__half*` for the activation regions internally.
// policy_out: global memory [NN_POLICY_SIZE] FP32 — written as log-probs
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
