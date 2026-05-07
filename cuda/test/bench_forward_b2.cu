// Microbenchmark: per-call wall time for oracle_net_forward_block (B=1)
// vs oracle_net_forward_block_b2 (B=2). Measures pure forward throughput
// in isolation — no MCTS, no allocation, no host roundtrip.
//
// Two modes are run back-to-back inside a tight loop with CUDA-event
// timing. Reports per-call and per-sim wall time and the speedup ratio.
//
// Usage: ./cuda/build/bench_forward_b2 [n_iters]
// Default: 1000 iterations (after a 50-iter warmup).

#include "../block_forward.cuh"
#include "../movegen.cuh"
#include "../nn_weights.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

__global__ void bench_forward_b1_kernel(
    const BoardState* bs, float q, const OracleNetWeights* w,
    const ConvWeightsShifted* shifted_w,
    float* policy_out, float* value_out, float* k_out,
    int n_iters
) {
    extern __shared__ __half smem[];
    for (int it = 0; it < n_iters; it++) {
        oracle_net_forward_block(bs, q, w, (float*)smem,
                                 policy_out, value_out, k_out,
                                 nullptr, shifted_w);
    }
}

__global__ void bench_forward_b2_kernel(
    const BoardState* bs0, const BoardState* bs1, float q0, float q1,
    const OracleNetWeights* w, const ConvWeightsShifted* shifted_w,
    float* policy_out_b2, float* value_out_b2, float* k_out_b2,
    int n_iters
) {
    extern __shared__ __half smem[];
    for (int it = 0; it < n_iters; it++) {
        oracle_net_forward_block_b2(bs0, bs1, q0, q1, w, (float*)smem,
                                    policy_out_b2, value_out_b2, k_out_b2,
                                    shifted_w);
    }
}

int main(int argc, char** argv) {
    init_movegen_tables();

    int n_iters = (argc > 1) ? atoi(argv[1]) : 1000;
    int n_warmup = 50;

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    ConvWeightsShifted* d_shifted = convert_weights_shifted(d_weights);

    // Two boards (we use the same starting position for both batches —
    // throughput is independent of the actual position contents).
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, 2 * sizeof(BoardState));
    cudaMemcpy(&d_bs[0], &bs, sizeof(BoardState), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_bs[1], &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float *d_pol_b1, *d_val_b1, *d_k_b1;
    float *d_pol_b2, *d_val_b2, *d_k_b2;
    cudaMalloc(&d_pol_b1, NN_POLICY_SIZE * sizeof(float));
    cudaMalloc(&d_val_b1, sizeof(float));
    cudaMalloc(&d_k_b1,   sizeof(float));
    cudaMalloc(&d_pol_b2, 2 * NN_POLICY_SIZE * sizeof(float));
    cudaMalloc(&d_val_b2, 2 * sizeof(float));
    cudaMalloc(&d_k_b2,   2 * sizeof(float));

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Mode A: B=1 ----
    cudaFuncSetAttribute(bench_forward_b1_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    // warmup
    bench_forward_b1_kernel<<<1, 256, BLOCK_SMEM_BYTES>>>(
        &d_bs[0], 0.0f, d_weights, d_shifted, d_pol_b1, d_val_b1, d_k_b1, n_warmup);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    bench_forward_b1_kernel<<<1, 256, BLOCK_SMEM_BYTES>>>(
        &d_bs[0], 0.0f, d_weights, d_shifted, d_pol_b1, d_val_b1, d_k_b1, n_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_b1 = 0.0f;
    cudaEventElapsedTime(&ms_b1, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after B=1 bench: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // ---- Mode B: B=2 ----
    cudaFuncSetAttribute(bench_forward_b2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, B2_SMEM_BYTES);
    bench_forward_b2_kernel<<<1, 256, B2_SMEM_BYTES>>>(
        &d_bs[0], &d_bs[1], 0.0f, 0.0f, d_weights, d_shifted,
        d_pol_b2, d_val_b2, d_k_b2, n_warmup);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    bench_forward_b2_kernel<<<1, 256, B2_SMEM_BYTES>>>(
        &d_bs[0], &d_bs[1], 0.0f, 0.0f, d_weights, d_shifted,
        d_pol_b2, d_val_b2, d_k_b2, n_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_b2 = 0.0f;
    cudaEventElapsedTime(&ms_b2, start, stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after B=2 bench: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Per-call and per-sim metrics. B=2 processes 2 leaves per call.
    double per_call_b1_us = (double)ms_b1 * 1000.0 / n_iters;
    double per_call_b2_us = (double)ms_b2 * 1000.0 / n_iters;
    double per_sim_b1_us  = per_call_b1_us;
    double per_sim_b2_us  = per_call_b2_us / 2.0;
    double speedup        = per_sim_b1_us / per_sim_b2_us;

    printf("=== batched forward microbenchmark ===\n");
    printf("  iterations: %d (warmup: %d)\n", n_iters, n_warmup);
    printf("  smem_b1: %d B   smem_b2: %d B\n", BLOCK_SMEM_BYTES, B2_SMEM_BYTES);
    printf("\n");
    printf("  B=1:  %.2f us/call,  %.2f us/sim   (total %.2f ms)\n",
           per_call_b1_us, per_sim_b1_us, ms_b1);
    printf("  B=2:  %.2f us/call,  %.2f us/sim   (total %.2f ms)\n",
           per_call_b2_us, per_sim_b2_us, ms_b2);
    printf("\n");
    printf("  per-sim speedup: %.3fx\n", speedup);

    cudaFree(d_bs);
    cudaFree(d_pol_b1); cudaFree(d_val_b1); cudaFree(d_k_b1);
    cudaFree(d_pol_b2); cudaFree(d_val_b2); cudaFree(d_k_b2);
    free_shifted_weights(d_shifted);
    free_nn_weights(d_weights);
    return 0;
}
