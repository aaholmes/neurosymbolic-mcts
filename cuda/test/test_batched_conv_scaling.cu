// ============================================================
// test_batched_conv_scaling — GPU MCTS v2 Phase 4 go/no-go gate
//
// Three kernels are benchmarked side-by-side on a single
// 128 -> 128 3x3 conv:
//   K1: existing smem-resident block_conv_3x3_shifted (B=1)
//   K2: new global-memory block_conv_3x3_shifted_batched
//   K3: new global-memory chunked variant (holds 4*CHUNK accs
//       per warp; amortizes weight loads across CHUNK positions
//       per GEMM pass)
//
// The K3/K1 per-position ratio at B=32 is the actual Phase 4
// go/no-go signal. K1/K2 at B=1 quantify the cost of moving
// activations from shared to global memory.
//
// TDD order (all must pass before timing):
//   1. Replication       (K2): B copies of same input → B identical outputs.
//   2. Independence      (K2): B distinct inputs match B single-batch runs of K1.
//   3. Shift aliasing    (K2): zero all weight slices except center → 1x1 conv.
//   4. Chunked sanity    (K3): K3 CHUNK=1 matches K2 at same B.
//   5. Chunked indep.    (K3): K3 CHUNK=4 matches B single-batch runs of K1.
//
// Build (run on the GPU host):
//   cmake --build cuda/build --target test_batched_conv_scaling
//
// Run:
//   ./cuda/build/test_batched_conv_scaling              # all tests + sweep
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

// Chunked kernel wrappers — one per CHUNK template instantiation.
// __launch_bounds__(256, 1) tells the compiler we'll only ever run
// one block per SM and to prefer spending registers on that block.
#define CHUNKED_WRAPPER(NAME, C)                                              \
    __global__ __launch_bounds__(256, 1)                                      \
    void NAME(const float* act_in, half* const* W_s, float* act_out,          \
              half* shifted_global, int C_in, int C_out, int B) {             \
        extern __shared__ unsigned char smem_raw[];                           \
        half* w_smem = reinterpret_cast<half*>(smem_raw);                     \
        size_t w_bytes = batched_conv_w_smem_bytes(C_in, C_out);              \
        float* acc_smem = reinterpret_cast<float*>(smem_raw + w_bytes);       \
        block_conv_3x3_shifted_batched_chunked<C>(                            \
            act_in, W_s, act_out, w_smem, acc_smem, shifted_global,           \
            C_in, C_out, B);                                                  \
    }

CHUNKED_WRAPPER(kernel_batched_conv_chunked_c1, 1)
CHUNKED_WRAPPER(kernel_batched_conv_chunked_c2, 2)
CHUNKED_WRAPPER(kernel_batched_conv_chunked_c4, 4)
#undef CHUNKED_WRAPPER

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

// ----- Chunked kernel launch helpers -----

struct ChunkedLaunch {
    float* d_act_in;
    float* d_act_out;
    half*  d_shifted;     // [C_in, 64*CHUNK]
    int    B, CHUNK, C_in, C_out;
    size_t smem_bytes;
};

static ChunkedLaunch alloc_chunked(int B, int CHUNK, int C_in, int C_out) {
    ChunkedLaunch L{};
    L.B = B; L.CHUNK = CHUNK; L.C_in = C_in; L.C_out = C_out;
    CHECK_CUDA(cudaMalloc(&L.d_act_in,  (size_t)B * C_in  * 64 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&L.d_act_out, (size_t)B * C_out * 64 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&L.d_shifted, batched_conv_shifted_bytes_chunked(C_in, CHUNK)));
    L.smem_bytes = batched_conv_w_smem_bytes(C_in, C_out) + batched_conv_acc_smem_bytes();
    return L;
}

static void free_chunked(ChunkedLaunch& L) {
    cudaFree(L.d_act_in);
    cudaFree(L.d_act_out);
    cudaFree(L.d_shifted);
    L = ChunkedLaunch{};
}

static void run_chunked(const ChunkedLaunch& L, half** d_slice_ptrs) {
    // Dispatch on compile-time CHUNK value.
    dim3 grid(1), block(256);
    switch (L.CHUNK) {
        case 1:
            kernel_batched_conv_chunked_c1<<<grid, block, L.smem_bytes>>>(
                L.d_act_in, d_slice_ptrs, L.d_act_out, L.d_shifted,
                L.C_in, L.C_out, L.B);
            break;
        case 2:
            kernel_batched_conv_chunked_c2<<<grid, block, L.smem_bytes>>>(
                L.d_act_in, d_slice_ptrs, L.d_act_out, L.d_shifted,
                L.C_in, L.C_out, L.B);
            break;
        case 4:
            kernel_batched_conv_chunked_c4<<<grid, block, L.smem_bytes>>>(
                L.d_act_in, d_slice_ptrs, L.d_act_out, L.d_shifted,
                L.C_in, L.C_out, L.B);
            break;
        default:
            fprintf(stderr, "Unsupported CHUNK=%d (only 1, 2, 4)\n", L.CHUNK);
            abort();
    }
}

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

static void test_chunked_sanity(bool& test_failed) {
    // Kernel 3 at CHUNK=1 must match Kernel 2 (baseline) at the same B.
    // Structural check on the chunked template before trusting CHUNK=2,4.
    const int C_in = 128, C_out = 128, B = 4;
    std::vector<float> h_W((size_t)C_out * C_in * 9);
    for (size_t i = 0; i < h_W.size(); i++) h_W[i] = det_rand((uint32_t)i, 333);
    ShiftedWeights sw = alloc_shifted_from_fp32(h_W.data(), C_in, C_out);

    BatchedLaunch L2 = alloc_batched(B, C_in, C_out);
    ChunkedLaunch L3 = alloc_chunked(B, /*CHUNK=*/1, C_in, C_out);

    std::vector<float> h_input((size_t)B * C_in * 64);
    for (size_t i = 0; i < h_input.size(); i++) h_input[i] = det_rand((uint32_t)i, 444);
    CHECK_CUDA(cudaMemcpy(L2.d_act_in, h_input.data(),
                          h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(L3.d_act_in, h_input.data(),
                          h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    run_batched(L2, sw.d_slice_ptrs);
    run_chunked(L3, sw.d_slice_ptrs);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_out2((size_t)B * C_out * 64), h_out3((size_t)B * C_out * 64);
    CHECK_CUDA(cudaMemcpy(h_out2.data(), L2.d_act_out, h_out2.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out3.data(), L3.d_act_out, h_out3.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int mismatches = 0;
    float worst_abs = 0.0f;
    for (size_t i = 0; i < h_out2.size(); i++) {
        float d = fabsf(h_out2[i] - h_out3[i]);
        if (d > worst_abs) worst_abs = d;
        if (d > 1e-3f) mismatches++;
    }
    printf("[chunked(CHUNK=1) vs baseline mismatches=%d worst_abs=%.3e] ",
           mismatches, worst_abs);
    ASSERT_EQ(mismatches, 0);

    free_batched(L2);
    free_chunked(L3);
    free_shifted(sw);
}

static void test_chunked_independence(bool& test_failed) {
    // Kernel 3 at CHUNK=4 on B=4 random inputs must match B individual
    // calls to the existing smem-resident kernel (Kernel 1).
    const int C_in = 128, C_out = 128, B = 4, CHUNK = 4;
    std::vector<float> h_W((size_t)C_out * C_in * 9);
    for (size_t i = 0; i < h_W.size(); i++) h_W[i] = det_rand((uint32_t)i, 555);
    ShiftedWeights sw = alloc_shifted_from_fp32(h_W.data(), C_in, C_out);

    ChunkedLaunch L = alloc_chunked(B, CHUNK, C_in, C_out);
    std::vector<float> h_input((size_t)B * C_in * 64);
    for (size_t i = 0; i < h_input.size(); i++) h_input[i] = det_rand((uint32_t)i, 666);
    CHECK_CUDA(cudaMemcpy(L.d_act_in, h_input.data(),
                          h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    run_chunked(L, sw.d_slice_ptrs);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_out_chunked((size_t)B * C_out * 64);
    CHECK_CUDA(cudaMemcpy(h_out_chunked.data(), L.d_act_out,
                          h_out_chunked.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Reference: existing smem-resident kernel, once per batch entry.
    float *d_in_one, *d_out_one;
    CHECK_CUDA(cudaMalloc(&d_in_one,  C_in  * 64 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_one, C_out * 64 * sizeof(float)));
    size_t ref_smem_bytes =
        (size_t)C_in  * 64 * sizeof(float) +
        (size_t)C_out * 64 * sizeof(float) +
        (size_t)C_in  * 64 * sizeof(half);
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
        float a = h_out_chunked[i], c = h_out_ref[i];
        float abs_err = fabsf(a - c);
        float rel_err = abs_err / (fabsf(c) + 1e-6f);
        if (abs_err > worst_abs) worst_abs = abs_err;
        if (rel_err > worst_rel) worst_rel = rel_err;
        if (abs_err > 1e-2f && rel_err > 5e-3f) mismatches++;
    }
    printf("[chunked(CHUNK=4) vs smem-ref mismatches=%d worst_abs=%.3e worst_rel=%.3e] ",
           mismatches, worst_abs, worst_rel);
    ASSERT_EQ(mismatches, 0);

    cudaFree(d_in_one); cudaFree(d_out_one);
    free_chunked(L);
    free_shifted(sw);
}

// ============================================================
// Timing sweep + projection
// ============================================================

struct SweepRow {
    const char* kernel;       // "K1 smem-ref", "K2 baseline", "K3 CHUNK=N"
    int   CHUNK;              // 0 for K1/K2, >0 for K3
    int   B;
    float conv_ms;
    float per_pos_ms;
    float speedup_vs_K1;      // ref_k1 per-pos / this per-pos
    float proj_full_ms;       // 0 if projection suppressed
    float proj_sims_per_s;    // 0 if projection suppressed
};

static void run_sweep(bool write_csv, const char* csv_path) {
    const int C_in = 128, C_out = 128;
    const int WARMUP = 20;
    const int ITERS  = 100;

    std::vector<float> h_W((size_t)C_out * C_in * 9);
    for (size_t i = 0; i < h_W.size(); i++) h_W[i] = det_rand((uint32_t)i, 2024);
    ShiftedWeights sw = alloc_shifted_from_fp32(h_W.data(), C_in, C_out);

    Timer timer;
    std::vector<SweepRow> rows;

    // ---------- Kernel 1: existing smem-resident kernel, B=1 only ----------
    float ref_k1_ms = 0.0f;
    {
        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in,  C_in  * 64 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, C_out * 64 * sizeof(float)));
        std::vector<float> h_in((size_t)C_in * 64);
        for (size_t i = 0; i < h_in.size(); i++) h_in[i] = det_rand((uint32_t)i, 99);
        CHECK_CUDA(cudaMemcpy(d_in, h_in.data(),
                              h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
        size_t ref_smem_bytes =
            (size_t)C_in  * 64 * sizeof(float) +
            (size_t)C_out * 64 * sizeof(float) +
            (size_t)C_in  * 64 * sizeof(half);
        cudaFuncSetAttribute(kernel_ref_batch1_conv,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)ref_smem_bytes);
        auto launch_k1 = [&]() {
            kernel_ref_batch1_conv<<<1, 256, ref_smem_bytes>>>(
                d_in, sw.d_slice_ptrs, d_out, C_in, C_out);
        };
        for (int i = 0; i < WARMUP; i++) launch_k1();
        CHECK_CUDA(cudaDeviceSynchronize());
        ref_k1_ms = timer.measure(ITERS, launch_k1);
        CHECK_CUDA(cudaDeviceSynchronize());
        rows.push_back({"K1 smem-ref", 0, 1, ref_k1_ms, ref_k1_ms, 1.0f, 0.0f, 0.0f});
        cudaFree(d_in); cudaFree(d_out);
    }

    // ---------- Kernel 2: baseline global-memory, B in {1,2,4,8,16,32} ----------
    for (int B : {1, 2, 4, 8, 16, 32}) {
        BatchedLaunch L = alloc_batched(B, C_in, C_out);
        std::vector<float> h_in((size_t)B * C_in * 64);
        for (size_t i = 0; i < h_in.size(); i++)
            h_in[i] = det_rand((uint32_t)i, 1000u + (uint32_t)B);
        CHECK_CUDA(cudaMemcpy(L.d_act_in, h_in.data(),
                              h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (int i = 0; i < WARMUP; i++) run_batched(L, sw.d_slice_ptrs);
        CHECK_CUDA(cudaDeviceSynchronize());
        float ms = timer.measure(ITERS, [&](){ run_batched(L, sw.d_slice_ptrs); });
        CHECK_CUDA(cudaDeviceSynchronize());
        float per_pos = ms / B;
        rows.push_back({"K2 baseline", 0, B, ms, per_pos,
                        ref_k1_ms / per_pos, 0.0f, 0.0f});
        free_batched(L);
    }

    // ---------- Kernel 3: chunked, CHUNK in {2, 4} ----------
    for (int CHUNK : {2, 4}) {
        const char* label = (CHUNK == 2) ? "K3 CHUNK=2" : "K3 CHUNK=4";
        for (int B = CHUNK; B <= 32; B *= 2) {
            ChunkedLaunch L = alloc_chunked(B, CHUNK, C_in, C_out);
            std::vector<float> h_in((size_t)B * C_in * 64);
            for (size_t i = 0; i < h_in.size(); i++)
                h_in[i] = det_rand((uint32_t)i, 2000u + (uint32_t)(100 * CHUNK + B));
            CHECK_CUDA(cudaMemcpy(L.d_act_in, h_in.data(),
                                  h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
            for (int i = 0; i < WARMUP; i++) run_chunked(L, sw.d_slice_ptrs);
            CHECK_CUDA(cudaDeviceSynchronize());
            float ms = timer.measure(ITERS, [&](){ run_chunked(L, sw.d_slice_ptrs); });
            CHECK_CUDA(cudaDeviceSynchronize());
            float per_pos = ms / B;
            rows.push_back({label, CHUNK, B, ms, per_pos,
                            ref_k1_ms / per_pos, 0.0f, 0.0f});
            free_chunked(L);
        }
    }
    free_shifted(sw);

    // ---------- Projection (uses measured Kernel 1 as per-conv reference) ----------
    const int NUM_CONVS = 13;
    const float FULL_B1_MS = 1.51f;  // production full forward at B=1 (from v2 doc)
    float ref_conv_total = NUM_CONVS * ref_k1_ms;
    float non_conv_ms = FULL_B1_MS - ref_conv_total;
    bool non_conv_trustworthy = (non_conv_ms > 0.05f);

    for (auto& r : rows) {
        if (!non_conv_trustworthy) continue;
        float conv_total = NUM_CONVS * r.conv_ms;
        r.proj_full_ms = conv_total + r.B * non_conv_ms;
        r.proj_sims_per_s = 36.0f * r.B / (r.proj_full_ms * 1e-3f);
    }

    // ---------- Print comparison table ----------
    printf("\n");
    printf("===== Conv Scaling Comparison (128 x 128 x 3x3, single block) =====\n");
    printf("Reference: Kernel 1 (smem-resident) per-conv @ B=1: %.4f ms\n\n", ref_k1_ms);
    printf("%-14s  %5s  %4s  %10s  %12s  %12s  %14s  %16s\n",
           "kernel", "CHUNK", "B", "conv_ms", "per_pos_ms",
           "speedup_K1", "proj_full_ms", "proj_sims/sec");
    for (auto& r : rows) {
        if (r.proj_full_ms > 0) {
            printf("%-14s  %5d  %4d  %10.4f  %12.4f  %11.2fx  %14.3f  %16.0f\n",
                   r.kernel, r.CHUNK, r.B, r.conv_ms, r.per_pos_ms,
                   r.speedup_vs_K1, r.proj_full_ms, r.proj_sims_per_s);
        } else {
            printf("%-14s  %5d  %4d  %10.4f  %12.4f  %11.2fx  %14s  %16s\n",
                   r.kernel, r.CHUNK, r.B, r.conv_ms, r.per_pos_ms,
                   r.speedup_vs_K1, "(suppressed)", "(suppressed)");
        }
    }

    // ---------- Key derived ratios ----------
    printf("\n--- Key ratios ---\n");
    float k2_b1 = 0, k2_b32 = 0, k3c2_b32 = 0, k3c4_b32 = 0;
    for (auto& r : rows) {
        if (strcmp(r.kernel, "K2 baseline") == 0 && r.B == 1)  k2_b1  = r.per_pos_ms;
        if (strcmp(r.kernel, "K2 baseline") == 0 && r.B == 32) k2_b32 = r.per_pos_ms;
        if (r.CHUNK == 2 && r.B == 32) k3c2_b32 = r.per_pos_ms;
        if (r.CHUNK == 4 && r.B == 32) k3c4_b32 = r.per_pos_ms;
    }
    if (k2_b1 > 0) {
        printf("Global-mem tax: K2 B=1 per-pos / K1 per-pos = %.2fx  (>1 means slower)\n",
               k2_b1 / ref_k1_ms);
    }
    if (k3c2_b32 > 0) {
        printf("K3 (CHUNK=2, B=32) per-pos vs K1: %.2fx  (>1 means faster than smem-ref)\n",
               ref_k1_ms / k3c2_b32);
    }
    if (k3c4_b32 > 0) {
        printf("K3 (CHUNK=4, B=32) per-pos vs K1: %.2fx  (>1 means faster than smem-ref)\n",
               ref_k1_ms / k3c4_b32);
        if (k2_b1 > 0) {
            printf("K3 (CHUNK=4, B=32) per-pos vs K2 B=1: %.2fx  (isolates chunking gain)\n",
                   k2_b1 / k3c4_b32);
        }
    }

    // ---------- Decision ----------
    printf("\n--- Decision ---\n");
    if (non_conv_trustworthy) {
        printf("Reference conv fraction of 1.51 ms forward: %.2f (expected ~0.5)\n",
               ref_conv_total / FULL_B1_MS);
        printf("Non-conv time attributed to B=1 forward: %.3f ms\n", non_conv_ms);
    } else {
        printf("NON-CONV COMPUTATION WOULD BE NEGATIVE (%.3f ms). The 1.51 ms full-forward\n"
               "reference from the v2 doc does not match this GPU's per-conv measurement.\n"
               "Projections suppressed. Remeasure full forward on this GPU via\n"
               "test_profile_latency with real weights to calibrate.\n", non_conv_ms);
    }
    if (k3c4_b32 > 0 && ref_k1_ms > 0) {
        float r = ref_k1_ms / k3c4_b32;
        if (r >= 3.0f) {
            printf(">>> STRONG GO. K3 (CHUNK=4, B=32) per position is %.2fx faster than\n"
                   "    the existing smem-resident kernel. Phase 4 commits its thesis.\n", r);
        } else if (r >= 1.5f) {
            printf(">>> MODEST GO. K3 (CHUNK=4, B=32) per position is %.2fx faster than\n"
                   "    the existing kernel. Phase 4 viable; revise throughput\n"
                   "    projections down before committing downstream work.\n", r);
        } else if (r >= 0.9f) {
            printf(">>> MARGINAL. K3 (CHUNK=4, B=32) roughly matches the existing kernel\n"
                   "    per position. Further kernel work (larger CHUNK, warp\n"
                   "    redistribution, Winograd) needed before Phase 4 commits.\n");
        } else {
            printf(">>> NO-GO. K3 (CHUNK=4, B=32) per position is %.2fx of the existing\n"
                   "    kernel — global-memory overhead exceeds batching benefit.\n"
                   "    Rethink Phase 4 memory layout.\n", r);
        }
    }

    // ---------- Leela extrapolation using best chunked per-pass time ----------
    if (k3c4_b32 > 0 && non_conv_trustworthy) {
        struct Net { const char* name; int channels; int blocks; };
        Net nets[] = {
            {"6x128 own",     128,  6},
            {"10x128 Leela",  128, 10},
            {"15x192 Leela",  192, 15},
            {"20x256 Leela",  256, 20},
            {"24x320 Leela",  320, 24},
        };
        float k3_conv_ms_b32 = 0;
        for (auto& r : rows)
            if (r.CHUNK == 4 && r.B == 32) k3_conv_ms_b32 = r.conv_ms;
        int B_ref = 32;
        printf("\n--- Projection to Leela nets at B=%d (using K3 CHUNK=4 per-conv scaling) ---\n",
               B_ref);
        printf("%-16s  %10s  %12s  %16s\n",
               "network", "full_ms", "per_pos_ms", "sims/sec (36 blk)");
        for (auto& net : nets) {
            float scale_ch   = (float)net.channels / 128.0f;
            float conv_scale = scale_ch * scale_ch *
                               (float)(2 * net.blocks + 1) / (float)NUM_CONVS;
            float conv_tot   = k3_conv_ms_b32 * conv_scale;
            float full_ms    = conv_tot + B_ref * non_conv_ms;
            float per_pos    = full_ms / B_ref;
            float sps        = 36.0f * B_ref / (full_ms * 1e-3f);
            printf("%-16s  %10.3f  %12.4f  %16.0f\n",
                   net.name, full_ms, per_pos, sps);
        }
    }

    // ---------- CSV ----------
    if (write_csv && csv_path) {
        FILE* f = fopen(csv_path, "w");
        if (f) {
            fprintf(f, "kernel,CHUNK,B,conv_ms,per_pos_ms,speedup_vs_K1,"
                       "proj_full_ms,proj_sims_per_s\n");
            for (auto& r : rows) {
                fprintf(f, "%s,%d,%d,%.6f,%.6f,%.4f,%.6f,%.2f\n",
                        r.kernel, r.CHUNK, r.B, r.conv_ms, r.per_pos_ms,
                        r.speedup_vs_K1, r.proj_full_ms, r.proj_sims_per_s);
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
        RUN_TEST(test_chunked_sanity);
        RUN_TEST(test_chunked_independence);
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
