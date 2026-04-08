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

// GPU_MAKE_MOVE helper (matches common.cuh encoding)
#ifndef GPU_MAKE_MOVE
#define GPU_MAKE_MOVE(from, to, promo) ((GPUMove)((from) | ((to) << 6) | ((promo) << 12)))
#endif

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
// Unit tests for selfplay internals
// ============================================================

void test_board_to_planes_starting_pos(bool& test_failed) {
    BoardState bs = make_starting_position();
    float planes[17 * 64];
    board_to_planes_host(bs, planes);

    // White pawns (STM=white, plane 0) should be on rank 2 (squares 8-15)
    int pawn_count = 0;
    for (int i = 0; i < 64; i++) if (planes[0 * 64 + i] != 0.0f) pawn_count++;
    printf("[stm_pawns=%d] ", pawn_count);
    ASSERT_EQ(pawn_count, 8);

    // Total pieces: 16 white + 16 black = 32 on piece planes (0-11)
    int total_pieces = 0;
    for (int i = 0; i < 12 * 64; i++) if (planes[i] != 0.0f) total_pieces++;
    printf("[total=%d] ", total_pieces);
    ASSERT_EQ(total_pieces, 32);

    // Castling planes (13-16) should all be set (all 64 squares = 1.0)
    int castling_ones = 0;
    for (int p = 13; p <= 16; p++)
        for (int i = 0; i < 64; i++) if (planes[p * 64 + i] == 1.0f) castling_ones++;
    printf("[castling=%d] ", castling_ones);
    ASSERT_EQ(castling_ones, 4 * 64);
}

void test_board_to_planes_black_to_move(bool& test_failed) {
    // After 1.e4, black to move — planes should be flipped
    BoardState bs = make_starting_position();
    // Manually set black to move (crude but sufficient for encoding test)
    bs.w_to_move = 0;
    float planes[17 * 64];
    board_to_planes_host(bs, planes);

    // STM is black, so plane 0 = black pawns (flipped: rank 7 → rank 2 in STM view)
    // Black pawns are on rank 7 (squares 48-55), flipped by ^56 → rank 2 (squares 8-15)
    int stm_pawns = 0;
    for (int i = 0; i < 64; i++) if (planes[0 * 64 + i] != 0.0f) stm_pawns++;
    printf("[stm_pawns=%d] ", stm_pawns);
    ASSERT_EQ(stm_pawns, 8);
}

void test_move_to_policy_index(bool& test_failed) {
    // e2e4 = from=12, to=28 (pawn push 2 squares forward)
    // As white: dr=2, dc=0 → direction 0 (N), distance 2 → plane = 0*7 + 1 = 1
    // Index = from_sq * 73 + plane = 12 * 73 + 1 = 877
    GPUMove e2e4 = GPU_MAKE_MOVE(12, 28, 0);
    int idx = move_to_policy_index_host(e2e4, 1);
    printf("[e2e4=%d] ", idx);
    ASSERT_EQ(idx, 12 * 73 + 1);

    // Knight b1c3 = from=1, to=18, dr=2, dc=1 → knight move index 1 → plane 57
    GPUMove nb1c3 = GPU_MAKE_MOVE(1, 18, 0);
    int idx2 = move_to_policy_index_host(nb1c3, 1);
    printf("[Nb1c3=%d] ", idx2);
    ASSERT_EQ(idx2, 1 * 73 + 57);  // plane 56+1=57

    // Same move as black should flip squares
    int idx3 = move_to_policy_index_host(nb1c3, 0);
    printf("[Nb1c3_black=%d] ", idx3);
    // Flipped: from=1^56=57, to=18^56=42, dr=-2, dc=1 → knight index 7 → plane 63
    ASSERT_EQ(idx3, 57 * 73 + 63);
}

void test_sprt_computation(bool& test_failed) {
    // All draws: LLR should be near 0, inconclusive
    float llr = compute_llr(0, 0, 100, 0.0f, 10.0f);
    const char* decision = check_sprt(llr, 0.05f, 0.05f);
    printf("[all_draws: llr=%.3f %s] ", llr, decision ? decision : "inconclusive");
    ASSERT_TRUE(decision == nullptr);  // inconclusive

    // Slightly better: 55% win rate with 800 games → should favor H1
    // score=0.55, p0=0.5, p1=0.514 — close to H1's expected score
    float llr2 = compute_llr(440, 360, 0, 0.0f, 10.0f);
    const char* d2 = check_sprt(llr2, 0.05f, 0.05f);
    printf("[440w360l: llr=%.3f %s] ", llr2, d2 ? d2 : "inconclusive");
    ASSERT_TRUE(llr2 > 0.0f);  // positive LLR = favors H1

    // Slightly worse: 45% win rate → should favor H0
    float llr3 = compute_llr(360, 440, 0, 0.0f, 10.0f);
    const char* d3 = check_sprt(llr3, 0.05f, 0.05f);
    printf("[360w440l: llr=%.3f %s] ", llr3, d3 ? d3 : "inconclusive");
    ASSERT_TRUE(llr3 < 0.0f);  // negative LLR = favors H0
}

void test_selfplay_determinism(bool& test_failed) {
    // Run the same config twice — should produce identical results
    TransformerWeights* d_weights = init_transformer_weights_zeros();

    SelfPlayConfig config = {};
    config.num_games = 2;
    config.sims_per_move = 30;
    config.max_nodes_per_tree = config.sims_per_move + 100;
    config.explore_base = 0.80f;
    config.enable_koth = false;
    config.c_puct = 1.414f;
    config.max_concurrent = 2;
    config.seed = 12345;

    GameRecord* r1 = new GameRecord[2];
    GameRecord* r2 = new GameRecord[2];
    for (int i = 0; i < 2; i++) {
        r1[i].samples = nullptr; r1[i].num_samples = 0; r1[i].result = 0;
        r2[i].samples = nullptr; r2[i].num_samples = 0; r2[i].result = 0;
    }

    run_selfplay_games(d_weights, config, r1, 2);
    run_selfplay_games(d_weights, config, r2, 2);

    bool match = true;
    for (int g = 0; g < 2; g++) {
        if (r1[g].num_samples != r2[g].num_samples || r1[g].result != r2[g].result) match = false;
    }
    printf("[r1: %d/%d, r2: %d/%d, match=%s] ",
           r1[0].num_samples, r1[1].num_samples,
           r2[0].num_samples, r2[1].num_samples,
           match ? "yes" : "no");
    ASSERT_TRUE(match);

    for (int i = 0; i < 2; i++) { r1[i].free_buf(); r2[i].free_buf(); }
    delete[] r1; delete[] r2;
    free_transformer_weights(d_weights);
}

// ============================================================
// Eval mode tests
// ============================================================

void test_eval_single_game(bool& test_failed) {
    TransformerWeights* d_a = init_transformer_weights_zeros();
    TransformerWeights* d_b = init_transformer_weights_zeros();

    EvalConfig config = {};
    config.num_games = 1;
    config.sims_per_move = 30;
    config.max_nodes_per_tree = config.sims_per_move + 100;
    config.explore_base = 0.90f;
    config.enable_koth = false;
    config.c_puct = 1.414f;
    config.max_concurrent = 1;

    EvalResult result = run_eval_games(d_a, d_b, config);

    printf("[A:%d B:%d D:%d] ", result.wins_a, result.wins_b, result.draws);

    // Game must complete
    ASSERT_EQ(result.wins_a + result.wins_b + result.draws, 1);

    free_transformer_weights(d_a);
    free_transformer_weights(d_b);
}

void test_eval_batch(bool& test_failed) {
    TransformerWeights* d_a = init_transformer_weights_zeros();
    TransformerWeights* d_b = init_transformer_weights_zeros();

    EvalConfig config = {};
    config.num_games = 8;
    config.sims_per_move = 30;
    config.max_nodes_per_tree = config.sims_per_move + 100;
    config.explore_base = 0.90f;
    config.enable_koth = false;
    config.c_puct = 1.414f;
    config.max_concurrent = 8;

    EvalResult result = run_eval_games(d_a, d_b, config);

    printf("[A:%d B:%d D:%d] ", result.wins_a, result.wins_b, result.draws);

    // All games must complete
    ASSERT_EQ(result.wins_a + result.wins_b + result.draws, 8);

    free_transformer_weights(d_a);
    free_transformer_weights(d_b);
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
    RUN_TEST(test_board_to_planes_starting_pos);
    RUN_TEST(test_board_to_planes_black_to_move);
    RUN_TEST(test_move_to_policy_index);
    RUN_TEST(test_sprt_computation);
    RUN_TEST(test_selfplay_determinism);
    RUN_TEST(test_eval_single_game);
    RUN_TEST(test_eval_batch);

    printf("\nResults: %d/%d passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");
    return failures > 0 ? 1 : 0;
}
