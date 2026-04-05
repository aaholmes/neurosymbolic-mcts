#include "../nn_weights.cuh"
#include "test_helpers.cuh"
#include <cstdio>

// ============================================================
// Weight struct tests
// ============================================================

void test_weights_struct_size(bool& test_failed) {
    size_t sz = nn_weights_size();
    printf("[%zu bytes = %.2f MB] ", sz, sz / (1024.0 * 1024.0));
    // Should be ~7.9 MB
    ASSERT_TRUE(sz > 7 * 1024 * 1024);
    ASSERT_TRUE(sz < 9 * 1024 * 1024);
}

void test_dummy_weights_init(bool& test_failed) {
    OracleNetWeights* d_weights = init_nn_weights_zeros();
    ASSERT_TRUE(d_weights != nullptr);

    // Read back and verify some values
    OracleNetWeights* h_weights = (OracleNetWeights*)malloc(sizeof(OracleNetWeights));
    cudaMemcpy(h_weights, d_weights, sizeof(OracleNetWeights), cudaMemcpyDeviceToHost);

    // All conv weights should be zero
    ASSERT_NEAR(h_weights->start_conv_weight[0], 0.0f, 1e-6f);
    ASSERT_NEAR(h_weights->blocks[0].conv1_weight[0], 0.0f, 1e-6f);

    // BN running_var should be 1.0 (not zero)
    ASSERT_NEAR(h_weights->start_bn.running_var[0], 1.0f, 1e-6f);
    ASSERT_NEAR(h_weights->blocks[0].bn1.running_var[0], 1.0f, 1e-6f);
    ASSERT_NEAR(h_weights->blocks[5].bn2.running_var[127], 1.0f, 1e-6f);
    ASSERT_NEAR(h_weights->v_bn.running_var[0], 1.0f, 1e-6f);

    // k_logit should be 0.0
    ASSERT_NEAR(h_weights->k_logit, 0.0f, 1e-6f);

    free(h_weights);
    free_nn_weights(d_weights);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== NN Ops Tests ===\n");

    int total = 0, passes = 0, failures = 0;

    RUN_TEST(test_weights_struct_size);
    RUN_TEST(test_dummy_weights_init);

    printf("\n%d/%d tests passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");

    return failures > 0 ? 1 : 0;
}
