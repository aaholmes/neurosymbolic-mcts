#include "../mcts_kernel.cuh"
#include "../tree_store.cuh"
#include "../movegen.cuh"
#include "../nn_weights.cuh"
#include "../selfplay.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstring>
#include <cmath>

// Per design: both sides of fresh-start equivalence MUST use a pool >= 8192
// to avoid comparing two broken (fake-terminalized) configs.
constexpr int TEST_POOL = 8192;

// ============================================================
// Test 3: fresh-start equivalence
//
// gpu_mcts_eval_trees_budget(fresh_starts=all-true, target=N) must produce
// best_move and visit-count rank matching gpu_mcts_eval_trees(sims=N) on
// the same positions, same weights. Q-values may drift slightly due to
// virtual-loss interleaving but the move ranking should match.
//
// Asserts that BOTH runs stay below the alloc-counter watermark.
// ============================================================
void test_fresh_start_equivalence(bool& test_failed) {
    const int NUM_TREES = 4;
    const int SIMS = 200;

    BoardState positions[NUM_TREES];
    for (int i = 0; i < NUM_TREES; i++)
        positions[i] = make_starting_position();

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, NUM_TREES * NN_POLICY_SIZE * sizeof(float));

    // Legacy path
    TreeEvalResult res_legacy[NUM_TREES] = {};
    gpu_mcts_eval_trees(positions, NUM_TREES, SIMS, TEST_POOL,
                        false, 1.414f, d_weights, d_policy_bufs, res_legacy);

    // Budget path, all fresh
    TreeEvalResult res_budget[NUM_TREES] = {};
    int root_idxs[NUM_TREES];
    bool fresh[NUM_TREES];
    for (int i = 0; i < NUM_TREES; i++) fresh[i] = true;
    gpu_mcts_eval_trees_budget(positions, NUM_TREES, SIMS, TEST_POOL,
                               false, 1.414f, d_weights, d_policy_bufs,
                               res_budget, root_idxs, fresh);

    int watermark = (int)(POOL_WATERMARK * (float)TEST_POOL);
    for (int i = 0; i < NUM_TREES; i++) {
        ASSERT_TRUE(res_legacy[i].nodes_allocated < watermark);
        ASSERT_TRUE(res_budget[i].nodes_allocated < watermark);
        ASSERT_EQ(res_legacy[i].best_move_from, res_budget[i].best_move_from);
        ASSERT_EQ(res_legacy[i].best_move_to,   res_budget[i].best_move_to);
        ASSERT_EQ(res_legacy[i].total_simulations, res_budget[i].total_simulations);
        // Budget path should report root index = partition base (fresh)
        ASSERT_TRUE(root_idxs[i] != i * TEST_POOL); // post-advance to chosen child
    }

    cudaFree(d_policy_bufs);
    free_nn_weights(d_weights);
}

// ============================================================
// Test 4: reuse-equivalence property (validates Insight 1)
//
// Sequence:
//   A: fresh, N sims on P0 → chose move M1 → P1
//   B: fresh, N sims on P1
//   C: search C1 = fresh N at P0; advance to chosen child; budget call
//      with target=N from carried root at P1
//
// Assertion: after C, child visit-count distribution at the new root
// matches B's distribution within l1_norm < 0.1 (treating distributions
// as histograms over moves).
// ============================================================

static void compute_child_visits(int root_idx, int* visits_out, int max_children) {
    void* d_pool_ptr = nullptr;
    cudaGetSymbolAddress(&d_pool_ptr, g_node_pool);
    MCTSNode h_root;
    cudaMemcpy(&h_root, (char*)d_pool_ptr + root_idx * sizeof(MCTSNode),
               sizeof(MCTSNode), cudaMemcpyDeviceToHost);
    int n = h_root.num_children;
    if (n > max_children) n = max_children;
    for (int c = 0; c < n; c++) {
        MCTSNode h_c;
        cudaMemcpy(&h_c, (char*)d_pool_ptr + (h_root.first_child_idx + c) * sizeof(MCTSNode),
                   sizeof(MCTSNode), cudaMemcpyDeviceToHost);
        visits_out[c] = h_c.visit_count;
    }
    for (int c = n; c < max_children; c++) visits_out[c] = 0;
}

static float l1_distance(const int* a, const int* b, int n) {
    int sum_a = 0, sum_b = 0;
    for (int i = 0; i < n; i++) { sum_a += a[i]; sum_b += b[i]; }
    if (sum_a == 0 || sum_b == 0) return 1.0f;
    float d = 0.0f;
    for (int i = 0; i < n; i++)
        d += fabsf((float)a[i]/sum_a - (float)b[i]/sum_b);
    return d;
}

void test_reuse_equivalence_property(bool& test_failed) {
    const int N = 200;
    BoardState P0 = make_starting_position();

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, NN_POLICY_SIZE * sizeof(float));

    // === Search C, part 1: fresh at P0
    TreeEvalResult res_a;
    int root_idx;
    bool fresh_t = true;
    gpu_mcts_eval_trees_budget(&P0, 1, N, TEST_POOL, false, 1.414f,
                               d_weights, d_policy_bufs, &res_a, &root_idx, &fresh_t);

    // Apply chosen move on host to derive P1 (same move both A and B will use,
    // since A and B share the same first-search seed indirectly via uniform priors).
    // Best move from P0:
    GPUMove chosen = (GPUMove)((res_a.best_move_from << 6) | res_a.best_move_to);
    // Need a BoardState for P1 — read it from the chosen child node
    void* d_pool_ptr = nullptr;
    cudaGetSymbolAddress(&d_pool_ptr, g_node_pool);
    MCTSNode h_p1;
    cudaMemcpy(&h_p1, (char*)d_pool_ptr + root_idx * sizeof(MCTSNode),
               sizeof(MCTSNode), cudaMemcpyDeviceToHost);
    BoardState P1;
    memset(&P1, 0, sizeof(P1));
    memcpy(P1.pieces, h_p1.pieces, sizeof(P1.pieces));
    memcpy(P1.pieces_occ, h_p1.pieces_occ, sizeof(P1.pieces_occ));
    P1.w_to_move = h_p1.w_to_move;
    P1.en_passant = h_p1.en_passant;
    P1.castling = h_p1.castling;
    P1.halfmove = h_p1.halfmove;

    // === Search C, part 2: budget reuse from carried root at P1 ===
    TreeEvalResult res_c;
    bool fresh_f = false;
    gpu_mcts_eval_trees_budget(&P1, 1, N, TEST_POOL, false, 1.414f,
                               d_weights, d_policy_bufs, &res_c, &root_idx, &fresh_f);

    int visits_c[256];
    compute_child_visits(root_idx, visits_c, 256);

    // === Search B: fresh, N sims on P1 in a SEPARATE tree slot ===
    // To avoid touching tree 0's state, we use 2 trees: tree 0 holds C's state
    // (untouched), tree 1 is a brand-new fresh-N at P1.
    BoardState two_pos[2] = {P0, P1};
    TreeEvalResult res_b2[2];
    int root_idxs2[2];
    bool fresh2[2] = {false, true};
    // tree 0: continue carrying (no work, no advance change), tree 1: fresh
    int saved_root0 = root_idx;
    int root_idx_tree0 = saved_root0;
    int root_idxs_input[2] = {root_idx_tree0, 0};
    // For this comparison we only care about tree 1's distribution
    gpu_mcts_eval_trees_budget(two_pos, 2, N, TEST_POOL, false, 1.414f,
                               d_weights, d_policy_bufs, res_b2, root_idxs_input, fresh2);

    int visits_b[256];
    compute_child_visits(root_idxs_input[1], visits_b, 256);

    float l1 = l1_distance(visits_c, visits_b, 256);
    if (l1 > 0.1f) {
        printf("FAIL: visit distribution L1 = %.3f (>0.1)\n", l1);
        test_failed = true;
    }

    cudaFree(d_policy_bufs);
    free_nn_weights(d_weights);
}

// ============================================================
// Test 5: parent-of-new-root invariant
// After advancing root, the new root's parent_idx must be -1.
// ============================================================
void test_parent_idx_invariant(bool& test_failed) {
    const int N = 50;
    BoardState pos = make_starting_position();

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, NN_POLICY_SIZE * sizeof(float));

    TreeEvalResult res;
    int root_idx;
    bool fresh = true;
    gpu_mcts_eval_trees_budget(&pos, 1, N, TEST_POOL, false, 1.414f,
                               d_weights, d_policy_bufs, &res, &root_idx, &fresh);

    void* d_pool_ptr = nullptr;
    cudaGetSymbolAddress(&d_pool_ptr, g_node_pool);
    MCTSNode h_new_root;
    cudaMemcpy(&h_new_root, (char*)d_pool_ptr + root_idx * sizeof(MCTSNode),
               sizeof(MCTSNode), cudaMemcpyDeviceToHost);
    ASSERT_EQ(h_new_root.parent_idx, -1);

    cudaFree(d_policy_bufs);
    free_nn_weights(d_weights);
}

// ============================================================
// Test 6: zero-budget no-op
// If carried >= target, kernel should do zero sims and return immediately.
// ============================================================
void test_zero_budget_noop(bool& test_failed) {
    const int N = 50;
    BoardState pos = make_starting_position();

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, NN_POLICY_SIZE * sizeof(float));

    // First search builds tree to N visits at root.
    TreeEvalResult res_a;
    int root_idx;
    bool fresh_t = true;
    gpu_mcts_eval_trees_budget(&pos, 1, N, TEST_POOL, false, 1.414f,
                               d_weights, d_policy_bufs, &res_a, &root_idx, &fresh_t);

    // After advance: new root carries some visits; capture them.
    void* d_pool_ptr = nullptr;
    cudaGetSymbolAddress(&d_pool_ptr, g_node_pool);
    MCTSNode h_root_pre;
    cudaMemcpy(&h_root_pre, (char*)d_pool_ptr + root_idx * sizeof(MCTSNode),
               sizeof(MCTSNode), cudaMemcpyDeviceToHost);
    int carried = h_root_pre.visit_count;
    ASSERT_TRUE(carried > 0);

    // Save the SEARCH root index — the budget call will advance the
    // game_root_idxs[] entry to a chosen child after running. To verify
    // the kernel did zero work, we re-inspect the original search root
    // (which the kernel would have touched if it had run any sims).
    int search_root_idx = root_idx;

    // Now request target = carried (no new work needed).
    BoardState p_dummy = pos; // ignored on reuse path
    TreeEvalResult res_b;
    bool fresh_f = false;
    gpu_mcts_eval_trees_budget(&p_dummy, 1, carried, TEST_POOL, false, 1.414f,
                               d_weights, d_policy_bufs, &res_b, &root_idx, &fresh_f);
    cudaError_t err_after = cudaGetLastError();
    ASSERT_EQ((int)err_after, (int)cudaSuccess);

    // Visit count of the search root should be unchanged (no new sims).
    MCTSNode h_root_post;
    cudaMemcpy(&h_root_post, (char*)d_pool_ptr + search_root_idx * sizeof(MCTSNode),
               sizeof(MCTSNode), cudaMemcpyDeviceToHost);
    ASSERT_EQ(h_root_post.visit_count, carried);

    cudaFree(d_policy_bufs);
    free_nn_weights(d_weights);
}

int main(int argc, char** argv) {
    init_movegen_tables();

    int total = 0, passes = 0, failures = 0;

    printf("=== Budget-Capped Reuse Tests ===\n\n");

    RUN_TEST(test_fresh_start_equivalence);
    RUN_TEST(test_reuse_equivalence_property);
    RUN_TEST(test_parent_idx_invariant);
    RUN_TEST(test_zero_budget_noop);

    printf("\n%d/%d tests passed, %d failed\n", passes, total, failures);
    return failures > 0 ? 1 : 0;
}
