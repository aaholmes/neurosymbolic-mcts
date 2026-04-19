#include "../mcts_kernel.cuh"
#include "../tree_store.cuh"
#include "../movegen.cuh"
#include "../nn_weights.cuh"
#include "../selfplay.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstring>

// ============================================================
// Test 1: validate_pool_size accepts adequate pools
// ============================================================
void test_validate_accepts_adequate(bool& test_failed) {
    ASSERT_TRUE(validate_pool_size(8192, 200, "test"));     // exactly minimum
    ASSERT_TRUE(validate_pool_size(8192, 100, "test"));     // minimum holds at low sims
    ASSERT_TRUE(validate_pool_size(20000, 500, "test"));    // 500*35 = 17500 < 20000
    ASSERT_TRUE(validate_pool_size(35000, 1000, "test"));   // 1000*35 = 35000 == 35000
}

// ============================================================
// Test 2: validate_pool_size rejects too-small pools
// ============================================================
void test_validate_rejects_too_small(bool& test_failed) {
    ASSERT_TRUE(!validate_pool_size(300, 200, "test"));     // way under
    ASSERT_TRUE(!validate_pool_size(4096, 200, "test"));    // old default, under 8192 minimum
    ASSERT_TRUE(!validate_pool_size(8000, 100, "test"));    // just under 8192
    ASSERT_TRUE(!validate_pool_size(20000, 1000, "test"));  // 1000*35 = 35000 > 20000
}

// ============================================================
// Test 3: alloc-counter watermark — adequate pool stays under 0.9 * max
//
// At pool=8192 with 200 sims and ~30 branching, a healthy search uses
// roughly sims * branching / 2 = 3000 nodes (since not every sim expands).
// Should be well under watermark = 7372.
// ============================================================
void test_watermark_adequate_pool(bool& test_failed) {
    const int NUM_TREES = 4;
    const int SIMS = 200;
    const int MAX_NODES_PER_TREE = 8192;

    BoardState positions[NUM_TREES];
    for (int i = 0; i < NUM_TREES; i++)
        positions[i] = make_starting_position();

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, NUM_TREES * NN_POLICY_SIZE * sizeof(float));

    TreeEvalResult results[NUM_TREES];
    memset(results, 0, sizeof(results));
    int  root_idxs[NUM_TREES];
    bool fresh[NUM_TREES];
    for (int i = 0; i < NUM_TREES; i++) fresh[i] = true;
    gpu_mcts_eval_trees_budget(positions, NUM_TREES, SIMS, MAX_NODES_PER_TREE,
                               false, 1.414f, d_weights, d_policy_bufs, results,
                               root_idxs, fresh);

    int watermark = (int)(POOL_WATERMARK * (float)MAX_NODES_PER_TREE);
    for (int i = 0; i < NUM_TREES; i++) {
        if (results[i].nodes_allocated >= watermark) {
            printf("FAIL: tree %d alloc=%d >= watermark=%d at pool=%d, sims=%d\n",
                   i, results[i].nodes_allocated, watermark, MAX_NODES_PER_TREE, SIMS);
            test_failed = true;
        }
    }

    cudaFree(d_policy_bufs);
    free_nn_weights(d_weights);
}

// ============================================================
// Test 4: alloc-counter watermark — undersized pool DOES exhaust
//
// Verifies the warning path is exercised when pool is too small.
// Uses max_nodes=300 (would never pass validate_pool_size, but the
// kernel itself doesn't enforce — only selfplay.cu does). The
// nodes_allocated reading should reflect saturation at/near max.
// ============================================================
void test_watermark_undersized_pool_saturates(bool& test_failed) {
    const int NUM_TREES = 1;
    const int SIMS = 200;
    const int MAX_NODES_PER_TREE = 300;

    BoardState positions[NUM_TREES];
    positions[0] = make_starting_position();

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, NUM_TREES * NN_POLICY_SIZE * sizeof(float));

    TreeEvalResult results[NUM_TREES];
    memset(results, 0, sizeof(results));
    int  root_idxs[NUM_TREES];
    bool fresh[NUM_TREES];
    for (int i = 0; i < NUM_TREES; i++) fresh[i] = true;
    fprintf(stderr, "(expect WARNING below — this test exercises the watermark path)\n");
    gpu_mcts_eval_trees_budget(positions, NUM_TREES, SIMS, MAX_NODES_PER_TREE,
                               false, 1.414f, d_weights, d_policy_bufs, results,
                               root_idxs, fresh);

    int watermark = (int)(POOL_WATERMARK * (float)MAX_NODES_PER_TREE);
    ASSERT_TRUE(results[0].nodes_allocated >= watermark);

    cudaFree(d_policy_bufs);
    free_nn_weights(d_weights);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    init_movegen_tables();

    int total = 0, passes = 0, failures = 0;

    printf("=== Pool Size & Alloc-Counter Assertions ===\n\n");

    RUN_TEST(test_validate_accepts_adequate);
    RUN_TEST(test_validate_rejects_too_small);
    RUN_TEST(test_watermark_adequate_pool);
    RUN_TEST(test_watermark_undersized_pool_saturates);

    printf("\n%d/%d tests passed, %d failed\n", passes, total, failures);
    return failures > 0 ? 1 : 0;
}
