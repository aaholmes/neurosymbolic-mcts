// Tests for GPU self-play driver.
//
// Usage:
//   ./cuda/build/test_selfplay                         # zero weights
//   ./cuda/build/test_selfplay /tmp/transformer.bin    # with trained weights

#include "../selfplay.cuh"
#include "../movegen.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstring>
#include <cmath>

// ============================================================
// Tests
// ============================================================

void test_selfplay_single_game(bool& test_failed) {
    TransformerWeights* d_weights = init_transformer_weights_zeros();

    SelfPlayConfig config = {};
    config.num_games = 1;
    config.sims_per_move = 50;
    config.max_nodes_per_tree = config.sims_per_move + 100;
    config.explore_base = 0.80f;
    config.enable_koth = false;
    config.c_puct = 1.414f;
    config.max_concurrent = 1;

    GameRecord* record = new GameRecord();
    record->samples = nullptr; record->num_samples = 0; record->result = 0;

    int total = run_selfplay_games(d_weights, config, record, 1);

    printf("[samples=%d, result=%d] ", record->num_samples, record->result);

    // Game should have terminated
    ASSERT_TRUE(record->result >= 1 && record->result <= 3);
    // Should have produced some training samples
    ASSERT_TRUE(record->num_samples >= 4);
    ASSERT_TRUE(record->num_samples <= SP_MAX_MOVES_PER_GAME);

    // Check each sample has reasonable values
    for (int s = 0; s < record->num_samples; s++) {
        float* sample = record->samples + s * SP_SAMPLE_FLOATS;

        // Board planes: should be 0 or 1 for piece planes (first 12*64)
        for (int i = 0; i < 12 * 64; i++) {
            ASSERT_TRUE(sample[i] == 0.0f || sample[i] == 1.0f);
        }

        // Value target: should be -1, 0, or 1
        float value = sample[SP_BOARD_FLOATS + 2];  // offset: board + material + qsearch_flag
        ASSERT_TRUE(value == -1.0f || value == 0.0f || value == 1.0f);

        // Policy: should sum to ~1.0
        float policy_sum = 0.0f;
        for (int i = 0; i < NN_POLICY_SIZE; i++)
            policy_sum += sample[SP_BOARD_FLOATS + 3 + i];
        ASSERT_NEAR(policy_sum, 1.0f, 0.02f);
    }

    record->free_buf();
    delete record;
    free_transformer_weights(d_weights);
}

void test_selfplay_batch(bool& test_failed) {
    TransformerWeights* d_weights = init_transformer_weights_zeros();

    SelfPlayConfig config = {};
    config.num_games = 4;  // small batch for testing
    config.sims_per_move = 30;
    config.max_nodes_per_tree = config.sims_per_move + 100;
    config.explore_base = 0.80f;
    config.enable_koth = false;
    config.c_puct = 1.414f;
    config.max_concurrent = 4;

    GameRecord* records = new GameRecord[4];
    for (int i = 0; i < 4; i++) { records[i].samples = nullptr; records[i].num_samples = 0; records[i].result = 0; }

    int total = run_selfplay_games(d_weights, config, records, 4);

    printf("[total_samples=%d] ", total);

    int total_check = 0;
    bool all_ended = true;
    for (int g = 0; g < 4; g++) {
        if (records[g].result < 1 || records[g].result > 3) all_ended = false;
        total_check += records[g].num_samples;
        printf("g%d:%d/%d ", g, records[g].num_samples, records[g].result);
    }

    ASSERT_TRUE(all_ended);
    ASSERT_EQ(total, total_check);
    ASSERT_TRUE(total >= 4 * 4);

    for (int i = 0; i < 4; i++) records[i].free_buf();
    delete[] records;
    free_transformer_weights(d_weights);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    printf("=== Self-Play Tests ===\n");
    fflush(stdout);
    init_movegen_tables();
    printf("Movegen initialized\n");
    fflush(stdout);
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    int total = 0, passes = 0, failures = 0;

    RUN_TEST(test_selfplay_single_game);
    RUN_TEST(test_selfplay_batch);

    printf("\nResults: %d/%d passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");
    return failures > 0 ? 1 : 0;
}
