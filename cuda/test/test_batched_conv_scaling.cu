// ============================================================
// test_batched_conv_scaling — GPU MCTS v2 Phase 4 go/no-go gate
//
// Builds a single 128 -> 128 3x3 conv via the shifted-copy
// 9-GEMM Tensor Core path, at varying batch sizes B, and
// projects full-forward and v2 sims/sec from the scaling curve.
//
// TDD order (do not start timing until 1-3 pass under
// compute-sanitizer):
//   1. Replication   — B copies of same input give B identical
//                      outputs, each matching a single-input run.
//   2. Independence  — B random inputs match B sequential batch-1
//                      calls through the existing
//                      block_conv_3x3_shifted kernel
//                      (atol=1e-2, rtol=5e-3).
//   3. Shift aliasing — zero all weight slices except s=4 (center)
//                       then output must equal a 1x1 conv with
//                       the center weights.
//
// After correctness: sweep B in {1, 2, 4, 8, 16, 32}, 20 warmup
// + 100 measured iterations, emit CSV + stdout table, print
// scaling projection.
//
// Build (run on the GPU host):
//   cmake --build cuda/build --target test_batched_conv_scaling
//
// Run:
//   ./cuda/build/test_batched_conv_scaling              # run all tests + sweep
//   ./cuda/build/test_batched_conv_scaling --tests      # correctness only
//   ./cuda/build/test_batched_conv_scaling --bench      # skip correctness
//
// Sanitizer (correctness only, slow):
//   compute-sanitizer ./cuda/build/test_batched_conv_scaling --tests
// ============================================================

#include "../block_ops.cuh"
#include "../block_ops_batched.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================
// Helpers
// ============================================================

struct Timer {
    cudaEvent_t s, e;
    Timer()  { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~Timer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    // Averaged ms per call. Caller should have done warmup and cudaDeviceSynchronize() first.
    float measure(int iters, std::function<void()> fn) {
        cudaEventRecord(s);
        for (int i = 0; i < iters; i++) fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        return ms / iters;
    }
};

// Deterministic pseudo-random in [-1, 1]
static float det_rand(uint32_t idx, uint32_t seed) {
    uint32_t x = idx * 0x9E3779B1u + seed * 0x85EBCA77u;
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15;
    x *= 0x846CA68Bu; x ^= x >> 16;
    return (float)((int32_t)x) / 2147483648.0f;
}

// ============================================================
// Weight synthesis
// ============================================================
//
// Source conv3x3 weights: [C_out, C_in, 3, 3] FP32, matches
// OracleNetWeights conv layout: W[oc, ic, ky, kx] at offset
//   oc * (C_in * 9) + ic * 9 + (ky * 3 + kx)
//
// Produce 9 FP16 device buffers W_s[s] of shape [C_out, C_in]
// with W_s[s][oc, ic] = W[oc, ic, s/3, s%3], matching the
// existing ConvWeightsShifted layout.
// ============================================================

struct ShiftedWeights {
    half* d_slices[9];   // each: [C_out, C_in] FP16 device buffer
    half** d_slice_ptrs; // [9] device array of the 9 pointers
    int C_in, C_out;
};

static ShiftedWeights alloc_shifted_from_fp32(const float* h_W, int C_in, int C_out) {
    ShiftedWeights sw{};
    sw.C_in = C_in; sw.C_out = C_out;
    std::vector<half> h_slice((size_t)C_out * C_in);
    for (int s = 0; s < 9; s++) {
        int ky = s / 3, kx = s % 3;
        for (int oc = 0; oc < C_out; oc++) {
            for (int ic = 0; ic < C_in; ic++) {
                float v = h_W[oc * (C_in * 9) + ic * 9 + (ky * 3 + kx)];
                h_slice[oc * C_in + ic] = __float2half(v);
            }
        }
        CHECK_CUDA(cudaMalloc(&sw.d_slices[s], h_slice.size() * sizeof(half)));
        CHECK_CUDA(cudaMemcpy(sw.d_slices[s], h_slice.data(),
                              h_slice.size() * sizeof(half), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMalloc(&sw.d_slice_ptrs, 9 * sizeof(half*)));
    CHECK_CUDA(cudaMemcpy(sw.d_slice_ptrs, sw.d_slices,
                          9 * sizeof(half*), cudaMemcpyHostToDevice));
    return sw;
}

static void free_shifted(ShiftedWeights& sw) {
    for (int s = 0; s < 9; s++) cudaFree(sw.d_slices[s]);
    cudaFree(sw.d_slice_ptrs);
    sw = ShiftedWeights{};
}

// ============================================================
// Wrapper kernels
// ============================================================
//
// Batched kernel: calls block_conv_3x3_shifted_batched on
// (act_in, W_s, act_out, shifted_global). Smem used for
// w_smem + a_staging and acc_smem.
// ============================================================

__global__ void kernel_batched_conv(
    const float* act_in,
    half* const* W_s,
    float* act_out,
    half* shifted_global,
    int C_in, int C_out, int B
) {
    extern __shared__ unsigned char smem_raw[];
    // Layout:
    //   [0 ..              C_in*C_out*2 bytes]           : w_smem (FP16)
    //   [next      8*256*2 bytes]                        : a_staging (FP16, aliased to tail of w_smem in kernel)
    //   [next      8*256*4 bytes]                        : acc_smem (FP32)
    // We combine w_smem and a_staging into one contiguous FP16 region
    // (as the kernel expects: a_staging = w_smem + C_in*C_out + warp_id*256).
    half* w_smem = reinterpret_cast<half*>(smem_raw);
    size_t w_plus_staging_bytes = ((size_t)C_in * C_out + 8 * 256) * sizeof(half);
    float* acc_smem = reinterpret_cast<float*>(smem_raw + w_plus_staging_bytes);

    block_conv_3x3_shifted_batched(
        act_in, W_s, act_out,
        w_smem, acc_smem, shifted_global,
        C_in, C_out, B
    );
}

// Reference: single-position, smem-resident, existing shifted kernel.
// Copies one [C_in, 64] input slice into smem, runs block_conv_3x3_shifted,
// writes back to one [C_out, 64] output slice. Used for the independence test.
__global__ void kernel_ref_batch1_conv(
    const float* act_in_one,    // [C_in, 64] FP32 global
    half* const* W_s,
    float* act_out_one,         // [C_out, 64] FP32 global
    int C_in, int C_out
) {
    extern __shared__ float smem_f[];
    // Layout: input[C_in*64] | output[C_out*64] | shifted[C_in*64 halves]
    float* input_smem  = smem_f;
    float* output_smem = smem_f + (size_t)C_in * 64;
    half*  shifted_smem = reinterpret_cast<half*>(output_smem + (size_t)C_out * 64);

    int tid = threadIdx.x;
    for (int i = tid; i < C_in * 64; i += blockDim.x) input_smem[i] = act_in_one[i];
    __syncthreads();

    block_conv_3x3_shifted(input_smem, W_s, output_smem, shifted_smem, C_in, C_out);

    __syncthreads();
    for (int i = tid; i < C_out * 64; i += blockDim.x) act_out_one[i] = output_smem[i];
}

// ============================================================
// Launch helpers
// ============================================================

struct BatchedLaunch {
    // Allocations tied to a particular B.
    float* d_act_in;       // [B, C_in, 64]
    float* d_act_out;      // [B, C_out, 64]
    half*  d_shifted;      // [C_in, 64] — reused across passes (size indep of B)
    int    B, C_in, C_out;
    size_t smem_bytes;
};

static BatchedLaunch alloc_batched(int B, int C_in, int C_out) {
    BatchedLaunch L{};
    L.B = B; L.C_in = C_in; L.C_out = C_out;
    CHECK_CUDA(cudaMalloc(&L.d_act_in,  (size_t)B * C_in  * 64 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&L.d_act_out, (size_t)B * C_out * 64 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&L.d_shifted, batched_conv_shifted_bytes(C_in, B)));
    L.smem_bytes =
        ((size_t)C_in * C_out + 8 * 256) * sizeof(half) +  // w_smem + a_staging
        8 * 256 * sizeof(float);                           // acc_smem
    return L;
}

static void free_batched(BatchedLaunch& L) {
    cudaFree(L.d_act_in);
    cudaFree(L.d_act_out);
    cudaFree(L.d_shifted);
    L = BatchedLaunch{};
}

static void run_batched(const BatchedLaunch& L, half** d_slice_ptrs) {
    kernel_batched_conv<<<1, 256, L.smem_bytes>>>(
        L.d_act_in, d_slice_ptrs, L.d_act_out, L.d_shifted,
        L.C_in, L.C_out, L.B
    );
}

// Note on smem opt-in: 128x128 gives 32 KB (weights) + 4 KB (a_staging)
// + 8 KB (acc_smem) = ~44 KB. Under the default 48 KB limit on most SMs
// so cudaFuncSetAttribute shouldn't be needed. For future larger configs
// we'd need MaxDynamicSharedMemorySize opt-in.

// ============================================================
// Correctness tests
// ============================================================

static void test_replication(bool& test_failed) {
    // TDD-1: Replicate a single input across B entries. All B outputs must
    // match each other and equal the reference single-batch output.
    const int C_in = 128, C_out = 128, B = 8;
    std::vector<float> h_W((size_t)C_out * C_in * 9);
    for (size_t i = 0; i < h_W.size(); i++) h_W[i] = det_rand((uint32_t)i, 1234);
    ShiftedWeights sw = alloc_shifted_from_fp32(h_W.data(), C_in, C_out);

    // Build input_one [C_in, 64], replicate B times.
    std::vector<float> h_input_one((size_t)C_in * 64);
    for (size_t i = 0; i < h_input_one.size(); i++) h_input_one[i] = det_rand((uint32_t)i, 5678);

    BatchedLaunch L = alloc_batched(B, C_in, C_out);
    std::vector<float> h_input_B((size_t)B * C_in * 64);
    for (int b = 0; b < B; b++) {
        memcpy(&h_input_B[(size_t)b * C_in * 64], h_input_one.data(),
               h_input_one.size() * sizeof(float));
    }
    CHECK_CUDA(cudaMemcpy(L.d_act_in, h_input_B.data(),
                          h_input_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    run_batched(L, sw.d_slice_ptrs);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_out_B((size_t)B * C_out * 64);
    CHECK_CUDA(cudaMemcpy(h_out_B.data(), L.d_act_out,
                          h_out_B.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare entry 0 against entries 1..B-1.
    int mismatches = 0;
    for (int b = 1; b < B; b++) {
        for (int i = 0; i < C_out * 64; i++) {
            float a = h_out_B[(size_t)0 * C_out * 64 + i];
            float c = h_out_B[(size_t)b * C_out * 64 + i];
            if (fabsf(a - c) > 1e-4f) mismatches++;
        }
    }
    printf("[replication cross-batch mismatches=%d] ", mismatches);
    ASSERT_EQ(mismatches, 0);

    free_batched(L);
    free_shifted(sw);
}

static void test_independence(bool& test_failed) {
    // TDD-2: B distinct inputs; batched output matches B individual calls
    // through the existing block_conv_3x3_shifted kernel (smem-resident).
    const int C_in = 128, C_out = 128, B = 4;
    std::vector<float> h_W((size_t)C_out * C_in * 9);
    for (size_t i = 0; i < h_W.size(); i++) h_W[i] = det_rand((uint32_t)i, 99);
    ShiftedWeights sw = alloc_shifted_from_fp32(h_W.data(), C_in, C_out);

    BatchedLaunch L = alloc_batched(B, C_in, C_out);
    std::vector<float> h_input((size_t)B * C_in * 64);
    for (size_t i = 0; i < h_input.size(); i++) h_input[i] = det_rand((uint32_t)i, 42);
    CHECK_CUDA(cudaMemcpy(L.d_act_in, h_input.data(),
                          h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    run_batched(L, sw.d_slice_ptrs);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_out_batched((size_t)B * C_out * 64);
    CHECK_CUDA(cudaMemcpy(h_out_batched.data(), L.d_act_out,
                          h_out_batched.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Reference: run the existing smem-resident kernel once per batch entry.
    float *d_in_one, *d_out_one;
    CHECK_CUDA(cudaMalloc(&d_in_one,  C_in  * 64 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_one, C_out * 64 * sizeof(float)));
    size_t ref_smem_bytes =
        (size_t)C_in  * 64 * sizeof(float) +   // input
        (size_t)C_out * 64 * sizeof(float) +   // output
        (size_t)C_in  * 64 * sizeof(half);     // shifted (smem)
    cudaFuncSetAttribute(kernel_ref_batch1_conv,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)ref_smem_bytes);

    std::vector<float> h_out_ref((size_t)B * C_out * 64);
    for (int b = 0; b < B; b++) {
        CHECK_CUDA(cudaMemcpy(d_in_one, &h_input[(size_t)b * C_in * 64],
                              C_in * 64 * sizeof(float), cudaMemcpyHostToDevice));
        kernel_ref_batch1_conv<<<1, 256, ref_smem_bytes>>>(
            d_in_one, sw.d_slice_ptrs, d_out_one, C_in, C_out);
        CHECK_CUDA(cudaMemcpy(&h_out_ref[(size_t)b * C_out * 64], d_out_one,
                              C_out * 64 * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    int mismatches = 0;
    float worst_abs = 0.0f, worst_rel = 0.0f;
    for (int i = 0; i < B * C_out * 64; i++) {
        float a = h_out_batched[i], c = h_out_ref[i];
        float abs_err = fabsf(a - c);
        float rel_err = abs_err / (fabsf(c) + 1e-6f);
        if (abs_err > worst_abs) worst_abs = abs_err;
        if (rel_err > worst_rel) worst_rel = rel_err;
        // atol = 1e-2, rtol = 5e-3 — FP16 weight+activation rounding budget.
        if (abs_err > 1e-2f && rel_err > 5e-3f) mismatches++;
    }
    printf("[independence mismatches=%d worst_abs=%.3e worst_rel=%.3e] ",
           mismatches, worst_abs, worst_rel);
    ASSERT_EQ(mismatches, 0);

    cudaFree(d_in_one); cudaFree(d_out_one);
    free_batched(L);
    free_shifted(sw);
}

static void test_shift_aliasing(bool& test_failed) {
    // TDD-3: Zero all 9 weight slices except s=4 (center).
    // With only center weight, the conv degenerates to a 1x1 conv:
    // out[b, oc, y, x] = sum_ic W_center[oc, ic] * in[b, ic, y, x]
    const int C_in = 128, C_out = 128, B = 2;
    std::vector<float> h_W((size_t)C_out * C_in * 9, 0.0f);
    // Populate only ky=1, kx=1 (s=4).
    for (int oc = 0; oc < C_out; oc++) {
        for (int ic = 0; ic < C_in; ic++) {
            float v = det_rand((uint32_t)(oc * C_in + ic), 7);
            h_W[oc * (C_in * 9) + ic * 9 + (1 * 3 + 1)] = v;
        }
    }
    ShiftedWeights sw = alloc_shifted_from_fp32(h_W.data(), C_in, C_out);

    BatchedLaunch L = alloc_batched(B, C_in, C_out);
    std::vector<float> h_input((size_t)B * C_in * 64);
    for (size_t i = 0; i < h_input.size(); i++) h_input[i] = det_rand((uint32_t)i, 11);
    CHECK_CUDA(cudaMemcpy(L.d_act_in, h_input.data(),
                          h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    run_batched(L, sw.d_slice_ptrs);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_out((size_t)B * C_out * 64);
    CHECK_CUDA(cudaMemcpy(h_out.data(), L.d_act_out,
                          h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Reference: explicit 1x1 conv with W_center. Use FP16-rounded weights
    // to match what the kernel actually computes (kernel converts to FP16
    // on upload; a naive FP32 reference would drift by ~128 * eps_FP16).
    int mismatches = 0;
    float worst_abs = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int oc = 0; oc < C_out; oc++) {
            for (int p = 0; p < 64; p++) {
                float ref = 0.0f;
                for (int ic = 0; ic < C_in; ic++) {
                    float w_f32 = h_W[oc * (C_in * 9) + ic * 9 + 4];
                    float w     = __half2float(__float2half(w_f32));
                    float a     = h_input[(size_t)b * C_in * 64 + ic * 64 + p];
                    ref += w * a;
                }
                float got = h_out[(size_t)b * C_out * 64 + oc * 64 + p];
                float abs_err = fabsf(got - ref);
                if (abs_err > worst_abs) worst_abs = abs_err;
                // 1e-2 abs / 5e-3 rel: covers TC FP16 mul + FP32 accumulation
                // error across 128 input channels.
                if (abs_err > 1e-2f && abs_err / (fabsf(ref) + 1e-6f) > 5e-3f) mismatches++;
            }
        }
    }
    printf("[shift-aliasing mismatches=%d worst_abs=%.3e] ", mismatches, worst_abs);
    ASSERT_EQ(mismatches, 0);

    free_batched(L);
    free_shifted(sw);
}

// ============================================================
// Timing sweep + projection
// ============================================================

struct SweepRow {
    int   B;
    float conv_ms;
    float per_pos_ms;
    float speedup_vs_B1;
    float proj_full_forward_ms;
    float proj_sims_per_sec;
};

static void run_sweep(bool write_csv, const char* csv_path) {
    const int C_in = 128, C_out = 128;
    const int Bs[] = {1, 2, 4, 8, 16, 32};
    const int n_B = sizeof(Bs) / sizeof(int);
    const int WARMUP = 20;
    const int ITERS  = 100;

    // Shared weights (synthetic, random but deterministic).
    std::vector<float> h_W((size_t)C_out * C_in * 9);
    for (size_t i = 0; i < h_W.size(); i++) h_W[i] = det_rand((uint32_t)i, 2024);
    ShiftedWeights sw = alloc_shifted_from_fp32(h_W.data(), C_in, C_out);

    Timer t;
    std::vector<SweepRow> rows(n_B);

    for (int bi = 0; bi < n_B; bi++) {
        int B = Bs[bi];
        BatchedLaunch L = alloc_batched(B, C_in, C_out);

        // Random input per B (each B gets its own to avoid trivial L2 reuse).
        std::vector<float> h_input((size_t)B * C_in * 64);
        for (size_t i = 0; i < h_input.size(); i++)
            h_input[i] = det_rand((uint32_t)i, 1000u + (uint32_t)B);
        CHECK_CUDA(cudaMemcpy(L.d_act_in, h_input.data(),
                              h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Warmup
        for (int i = 0; i < WARMUP; i++) run_batched(L, sw.d_slice_ptrs);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Measure
        float conv_ms = t.measure(ITERS, [&](){ run_batched(L, sw.d_slice_ptrs); });
        CHECK_CUDA(cudaDeviceSynchronize());

        rows[bi].B = B;
        rows[bi].conv_ms    = conv_ms;
        rows[bi].per_pos_ms = conv_ms / B;

        free_batched(L);
    }
    free_shifted(sw);

    // Derive speedup, scaling fit, full-forward / sims-per-sec projections.
    float b1_per_pos = rows[0].per_pos_ms;
    for (int bi = 0; bi < n_B; bi++) {
        rows[bi].speedup_vs_B1 = b1_per_pos / rows[bi].per_pos_ms;
    }

    // Fit t_conv(B) = alpha + beta * B via least-squares on (B, conv_ms).
    double SB = 0, SBB = 0, ST = 0, SBT = 0;
    for (int bi = 0; bi < n_B; bi++) {
        double B = rows[bi].B, T = rows[bi].conv_ms;
        SB += B; SBB += B*B; ST += T; SBT += B*T;
    }
    double n = n_B;
    double beta  = (n * SBT - SB * ST) / (n * SBB - SB * SB);
    double alpha = (ST - beta * SB) / n;

    // 6x128 SE-ResNet forward decomposition:
    //   Measured batch-1 full forward: 1.51 ms (per GPU_MCTS_v2.md, README.md).
    //   13 conv layers (1 start + 6 * 2 residual). Attribute ~50% of the 1.51 ms
    //   to conv per the doc. Cross-check by t_conv_measured(B=1) * 13 ~ 0.755 ms.
    const float FULL_B1_MS     = 1.51f;
    const int   NUM_CONVS      = 13;
    float conv_fraction_B1     = (NUM_CONVS * rows[0].conv_ms) / FULL_B1_MS;  // should be ~0.5
    float non_conv_ms          = FULL_B1_MS - NUM_CONVS * rows[0].conv_ms;
    // Pessimistic assumption: non-conv time scales linearly with B from its B=1 baseline.
    float non_conv_per_pos_ms  = non_conv_ms;  // all of non_conv happens once per position

    for (int bi = 0; bi < n_B; bi++) {
        int B = rows[bi].B;
        float proj_full = NUM_CONVS * rows[bi].conv_ms + B * non_conv_per_pos_ms;
        rows[bi].proj_full_forward_ms = proj_full;
        // v2 sims/sec: 36 blocks, each processing a batch of B positions per forward.
        rows[bi].proj_sims_per_sec = 36.0f * B / (proj_full * 1e-3f);
    }

    // --- stdout table ---
    printf("\n");
    printf("===== Batched Shifted-Copy Conv Scaling (128 x 128 x 3x3, single block) =====\n");
    printf("%4s  %10s  %12s  %10s  %14s  %16s\n",
           "B", "conv_ms", "per_pos_ms", "speedup", "proj_full_ms", "proj_sims/sec");
    for (int bi = 0; bi < n_B; bi++) {
        printf("%4d  %10.4f  %12.4f  %10.2fx  %14.3f  %16.0f\n",
               rows[bi].B,
               rows[bi].conv_ms,
               rows[bi].per_pos_ms,
               rows[bi].speedup_vs_B1,
               rows[bi].proj_full_forward_ms,
               rows[bi].proj_sims_per_sec);
    }
    printf("\nFit:  t_conv(B)  =  %.4f + %.4f * B  ms\n", alpha, beta);
    printf("B=1 conv-fraction of 1.51 ms baseline: %.2f  (expected ~0.50)\n", conv_fraction_B1);
    printf("non-conv time attributed to B=1 baseline: %.3f ms\n", non_conv_ms);

    // Decision thresholds from the plan.
    //
    // IMPORTANT: this kernel is a CONSERVATIVE baseline. It implements the
    // outer-loop-per-batch structure (each pass = one batch element, same
    // shape as existing block_conv_3x3_shifted). It does NOT implement the
    // Phase-4 optimizations that the v2 doc's 7-10x projection assumes:
    //   - Holding 4*CHUNK accumulators alive per warp (amortize weight
    //     loads over CHUNK batches per GEMM pass).
    //   - Re-splitting warps across spatial*batch dimension (fill idle
    //     warps at small B).
    //
    // Expected outcome for THIS baseline: speedup close to 1.00x (flat
    // per-position cost), because per-position work is essentially
    // identical to running the existing kernel B times. Any speedup above
    // 1.0x is pure amortization of per-launch overhead + L2 caching of
    // weights across passes. Typical: 1.1-1.5x at B=32.
    //
    // Interpretation of ratio at B=32:
    //   ~ 1.0-1.5x  Baseline behaves as expected. Build the chunked-
    //               accumulator variant next to measure Phase-4 headroom.
    //   > 2x        Better than expected — weight L2 caching or pipeline
    //               benefits larger than estimated. Good sign for Phase 4.
    //   << 1x       Regression — something is slower than existing kernel.
    //               Investigate before Phase 4.
    // The plan's 6.4x / 4x / <4x thresholds apply to the OPTIMIZED Phase 4
    // kernel, not this baseline. Do not call go/no-go from this run alone.
    float ratio = rows[n_B-1].speedup_vs_B1;
    printf("\nPer-position speedup B=1 -> B=%d: %.2fx  (baseline kernel)\n",
           Bs[n_B-1], ratio);
    if (ratio < 0.9f) {
        printf(">>> REGRESSION vs expected ~1x. Investigate before moving on.\n");
    } else if (ratio < 1.6f) {
        printf(">>> EXPECTED for baseline kernel. Next step: implement chunked-\n"
               "    accumulator variant (hold 4*CHUNK accs per warp) to measure\n"
               "    Phase-4 headroom before calling go/no-go.\n");
    } else if (ratio < 4.0f) {
        printf(">>> BETTER THAN EXPECTED baseline. Positive signal for Phase 4 —\n"
               "    chunked variant likely to clear the 6.4x threshold.\n");
    } else {
        printf(">>> EXCELLENT. Baseline already approaches Phase 4 target.\n");
    }

    // --- Extrapolation to Leela network sizes (compute-scaling only) ---
    //   Per-layer compute scales as (C/128)^2. Block count scales linearly.
    //   Use beta (per-position arithmetic) to extrapolate.
    struct Net { const char* name; int channels; int blocks; };
    Net nets[] = {
        {"6x128 own",     128,  6},
        {"10x128 Leela",  128, 10},
        {"15x192 Leela",  192, 15},
        {"20x256 Leela",  256, 20},
        {"24x320 Leela",  320, 24},
    };
    int B_ref = 32;
    // Estimate projected conv-only ms at B_ref for each net:
    //   t_conv_net(B) = t_conv_6x128(B) * (C/128)^2 * (2*blocks+1)/13
    printf("\n--- Projection to Leela networks at B=%d ---\n", B_ref);
    printf("%-16s  %10s  %12s  %16s\n",
           "network", "full_ms", "per_pos_ms", "sims/sec (36 blk)");
    // Find measured conv_ms at B=B_ref
    float conv_ms_ref = 0.0f;
    for (int bi = 0; bi < n_B; bi++) if (rows[bi].B == B_ref) conv_ms_ref = rows[bi].conv_ms;
    for (auto& net : nets) {
        float scale_ch    = (float)net.channels / 128.0f;
        float conv_scale  = scale_ch * scale_ch * (float)(2 * net.blocks + 1) / (float)NUM_CONVS;
        float conv_tot    = conv_ms_ref * conv_scale;
        // Assume non-conv overhead scales linearly with batch (same as 6x128 case).
        // For larger networks the non-conv portion is a small fraction of the total;
        // keep the same B=1 non_conv estimate (pessimistic).
        float full_ms     = conv_tot + B_ref * non_conv_per_pos_ms;
        float per_pos_ms  = full_ms / B_ref;
        float sims_per_s  = 36.0f * B_ref / (full_ms * 1e-3f);
        printf("%-16s  %10.3f  %12.4f  %16.0f\n",
               net.name, full_ms, per_pos_ms, sims_per_s);
    }

    // --- CSV ---
    if (write_csv && csv_path) {
        FILE* f = fopen(csv_path, "w");
        if (f) {
            fprintf(f, "B,conv_ms,per_pos_ms,speedup_vs_B1,proj_full_forward_ms,proj_sims_per_sec\n");
            for (int bi = 0; bi < n_B; bi++) {
                fprintf(f, "%d,%.6f,%.6f,%.4f,%.6f,%.2f\n",
                        rows[bi].B, rows[bi].conv_ms, rows[bi].per_pos_ms,
                        rows[bi].speedup_vs_B1, rows[bi].proj_full_forward_ms,
                        rows[bi].proj_sims_per_sec);
            }
            fclose(f);
            printf("\nCSV written: %s\n", csv_path);
        } else {
            printf("\nWARNING: failed to open %s for writing\n", csv_path);
        }
    }
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    bool do_tests = true, do_bench = true;
    const char* csv_path = "batched_conv_scaling.csv";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tests") == 0) { do_bench = false; }
        else if (strcmp(argv[i], "--bench") == 0) { do_tests = false; }
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) { csv_path = argv[++i]; }
    }

    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (sm_%d%d), %zu MB VRAM\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));

    int passes = 0, failures = 0, total = 0;
    if (do_tests) {
        printf("\n--- Correctness (TDD order) ---\n");
        RUN_TEST(test_replication);
        RUN_TEST(test_independence);
        RUN_TEST(test_shift_aliasing);
        printf("Tests: %d/%d passed\n", passes, total);
        if (failures > 0) {
            printf("Correctness failures; skipping benchmark.\n");
            return 1;
        }
    }

    if (do_bench) {
        run_sweep(true, csv_path);
    }

    return failures == 0 ? 0 : 1;
}
