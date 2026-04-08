// Tests for the transformer forward pass.
//
// Usage:
//   ./cuda/build/test_transformer

#include "../transformer_weights.cuh"
#include "../transformer_ops.cuh"
#include "../transformer_forward.cuh"
#include "../movegen.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstring>
#include <cmath>

// ============================================================
// Test kernels
// ============================================================

__global__ void kernel_transformer_forward(
    const BoardState* bs, float q_result,
    const TransformerWeights* weights,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    transformer_forward(bs, q_result, weights, nullptr, smem, policy_out, value_out, k_out);
}

__global__ void kernel_layer_norm_test(
    float* data, const float* gamma, const float* beta,
    int num_tokens, int d_model
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    // Copy data into smem
    for (int i = tid; i < num_tokens * d_model; i += blockDim.x)
        smem[i] = data[i];
    __syncthreads();

    float* reduce = smem + num_tokens * d_model;
    tf_layer_norm(smem, smem, gamma, beta, num_tokens, d_model, reduce);

    for (int i = tid; i < num_tokens * d_model; i += blockDim.x)
        data[i] = smem[i];
}

// ============================================================
// Tests
// ============================================================

void test_transformer_weights_size(bool& test_failed) {
    size_t sz = transformer_weights_size();
    printf("[%zu bytes = %.1f MB] ", sz, sz / (1024.0 * 1024.0));
    ASSERT_TRUE(sz > 1000000);   // should be >1MB
    ASSERT_TRUE(sz < 20000000);  // should be <20MB
}

void test_layer_norm(bool& test_failed) {
    const int T = 4, D = 8;  // small for verification
    float h_data[T * D];
    float h_gamma[D], h_beta[D];

    // Simple input: token t, dim d = t * D + d
    for (int i = 0; i < T * D; i++) h_data[i] = (float)(i % 7) - 3.0f;
    for (int d = 0; d < D; d++) { h_gamma[d] = 1.0f; h_beta[d] = 0.0f; }

    float *d_data, *d_gamma, *d_beta;
    cudaMalloc(&d_data,  T * D * sizeof(float));
    cudaMalloc(&d_gamma, D * sizeof(float));
    cudaMalloc(&d_beta,  D * sizeof(float));
    cudaMemcpy(d_data,  h_data,  T * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  D * sizeof(float), cudaMemcpyHostToDevice);

    int smem_size = (T * D + 256) * sizeof(float);
    kernel_layer_norm_test<<<1, 256, smem_size>>>(d_data, d_gamma, d_beta, T, D);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, T * D * sizeof(float), cudaMemcpyDeviceToHost);

    // Each token should have mean ≈ 0 and var ≈ 1
    bool ok = true;
    for (int t = 0; t < T; t++) {
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < D; d++) mean += h_data[t * D + d];
        mean /= D;
        for (int d = 0; d < D; d++) var += (h_data[t * D + d] - mean) * (h_data[t * D + d] - mean);
        var /= D;
        if (fabsf(mean) > 0.01f || fabsf(var - 1.0f) > 0.1f) ok = false;
    }
    printf("[mean≈0, var≈1] ");
    ASSERT_TRUE(ok);

    cudaFree(d_data); cudaFree(d_gamma); cudaFree(d_beta);
}

void test_transformer_forward_zero_weights(bool& test_failed) {
    TransformerWeights* d_weights = init_transformer_weights_zeros();

    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float *d_policy, *d_value, *d_k;
    cudaMalloc(&d_policy, NN_POLICY_SIZE * sizeof(float));
    cudaMalloc(&d_value, sizeof(float));
    cudaMalloc(&d_k, sizeof(float));

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    cudaFuncSetAttribute(kernel_transformer_forward,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, TF_SMEM_BYTES);
    kernel_transformer_forward<<<1, 256, TF_SMEM_BYTES>>>(
        d_bs, 0.0f, d_weights, d_policy, d_value, d_k);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA error: %s] ", cudaGetErrorString(err));
        test_failed = true;
    } else {
        float h_value, h_k;
        cudaMemcpy(&h_value, d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k, d_k, sizeof(float), cudaMemcpyDeviceToHost);

        printf("[val=%.4f, k=%.4f] ", h_value, h_k);

        // With zero weights (except LN gamma=1): value should be near 0
        // k = 0.47 * ln(1 + exp(0)) = 0.47 * ln(2) ≈ 0.326
        ASSERT_NEAR(h_k, 0.326f, 0.01f);

        // Policy should be approximately uniform (log-softmax of zeros → -log(4672))
        float* h_policy = (float*)malloc(NN_POLICY_SIZE * sizeof(float));
        cudaMemcpy(h_policy, d_policy, NN_POLICY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        float expected_log_prob = -logf((float)NN_POLICY_SIZE);
        int bad = 0;
        for (int i = 0; i < NN_POLICY_SIZE; i++)
            if (fabsf(h_policy[i] - expected_log_prob) > 0.1f) bad++;
        printf("[policy_uniform: %d/%d bad] ", bad, NN_POLICY_SIZE);
        ASSERT_TRUE(bad < NN_POLICY_SIZE / 10);  // allow some slack

        free(h_policy);
    }

    cudaFree(d_bs); cudaFree(d_policy); cudaFree(d_value); cudaFree(d_k);
    free_transformer_weights(d_weights);
}

void test_transformer_q_sensitivity(bool& test_failed) {
    TransformerWeights* d_weights = init_transformer_weights_zeros();
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float *d_policy, *d_value, *d_k;
    cudaMalloc(&d_policy, NN_POLICY_SIZE * sizeof(float));
    cudaMalloc(&d_value, sizeof(float));
    cudaMalloc(&d_k, sizeof(float));

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    cudaFuncSetAttribute(kernel_transformer_forward,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, TF_SMEM_BYTES);

    float q_values[] = {-3.0f, 0.0f, 3.0f};
    float h_values[3];
    for (int i = 0; i < 3; i++) {
        kernel_transformer_forward<<<1, 256, TF_SMEM_BYTES>>>(
            d_bs, q_values[i], d_weights, d_policy, d_value, d_k);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_values[i], d_value, sizeof(float), cudaMemcpyDeviceToHost);
    }

    printf("[V(-3)=%.2f, V(0)=%.2f, V(3)=%.2f] ", h_values[0], h_values[1], h_values[2]);
    ASSERT_TRUE(h_values[0] < h_values[1]);
    ASSERT_TRUE(h_values[1] < h_values[2]);
    ASSERT_TRUE(h_values[2] > 0.5f);
    ASSERT_TRUE(h_values[0] < -0.5f);

    cudaFree(d_bs); cudaFree(d_policy); cudaFree(d_value); cudaFree(d_k);
    free_transformer_weights(d_weights);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== Transformer Tests ===\n");
    init_movegen_tables();
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    int total = 0, passes = 0, failures = 0;

    RUN_TEST(test_transformer_weights_size);
    RUN_TEST(test_layer_norm);
    RUN_TEST(test_transformer_forward_zero_weights);
    RUN_TEST(test_transformer_q_sensitivity);

    printf("\nResults: %d/%d passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");
    return failures > 0 ? 1 : 0;
}
