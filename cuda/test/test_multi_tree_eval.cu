// Test: Multi-tree eval kernel — N independent MCTS trees running in parallel.
//
// Usage:
//   ./cuda/build/test_multi_tree_eval
//   ./cuda/build/test_multi_tree_eval /tmp/weights_gen18.bin  # with NN eval

#include "../mcts_kernel.cuh"
#include "../tree_store.cuh"
#include "../movegen.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstring>

static int char_to_piece(char c, int* color) {
    *color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
    switch (c) {
        case 'P': case 'p': return PAWN;
        case 'N': case 'n': return KNIGHT;
        case 'B': case 'b': return BISHOP;
        case 'R': case 'r': return ROOK;
        case 'Q': case 'q': return QUEEN;
        case 'K': case 'k': return KING;
        default: return -1;
    }
}

static BoardState parse_fen(const char* fen) {
    BoardState bs;
    memset(&bs, 0, sizeof(bs));
    bs.en_passant = EN_PASSANT_NONE;

    int rank = 7, file = 0;
    const char* p = fen;
    while (*p && *p != ' ') {
        if (*p == '/') { rank--; file = 0; }
        else if (*p >= '1' && *p <= '8') { file += (*p - '0'); }
        else {
            int color, piece;
            piece = char_to_piece(*p, &color);
            if (piece >= 0) {
                int sq = rank * 8 + file;
                bs.pieces[color * 6 + piece] |= (1ULL << sq);
                file++;
            }
        }
        p++;
    }
    if (*p) p++;
    bs.w_to_move = (*p == 'w') ? 1 : 0;
    if (*p) p++;
    if (*p) p++;
    while (*p && *p != ' ') {
        switch (*p) {
            case 'K': bs.castling |= CASTLE_WK; break;
            case 'Q': bs.castling |= CASTLE_WQ; break;
            case 'k': bs.castling |= CASTLE_BK; break;
            case 'q': bs.castling |= CASTLE_BQ; break;
        }
        p++;
    }
    if (*p) p++;
    if (*p && *p != '-') {
        bs.en_passant = (uint8_t)((p[1] - '1') * 8 + (p[0] - 'a'));
    }
    for (int c = 0; c < 2; c++) {
        bs.pieces_occ[c] = 0;
        for (int piece = 0; piece < 6; piece++)
            bs.pieces_occ[c] |= bs.pieces[c * 6 + piece];
    }
    return bs;
}

static const char* test_fens[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/5pkp/6p1/8/8/8/5PPP/6K1 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",  // mate-in-1
};

int main(int argc, char** argv) {
    init_movegen_tables();

    const int NUM_TREES = 6;
    const int SIMS = 100;
    const int MAX_NODES_PER_TREE = 4096;

    BoardState positions[NUM_TREES];
    for (int i = 0; i < NUM_TREES; i++) positions[i] = parse_fen(test_fens[i]);

    TreeEvalResult results[NUM_TREES];
    memset(results, 0, sizeof(results));

    // Classical mode (no NN weights)
    printf("=== Multi-Tree Eval (Classical Mode) ===\n");
    printf("Trees: %d, Simulations per tree: %d, Max nodes/tree: %d\n\n",
           NUM_TREES, SIMS, MAX_NODES_PER_TREE);

    int count = gpu_mcts_eval_trees(positions, NUM_TREES, SIMS,
                                     MAX_NODES_PER_TREE, false, 1.414f,
                                     nullptr, nullptr, results);

    printf("Processed %d trees:\n", count);
    for (int i = 0; i < count; i++) {
        printf("  Tree %d: sims=%d nodes=%d value=%.3f best=%c%d-%c%d\n",
               i, results[i].total_simulations, results[i].nodes_allocated,
               results[i].root_value,
               'a' + results[i].best_move_from % 8, results[i].best_move_from / 8 + 1,
               'a' + results[i].best_move_to % 8, results[i].best_move_to / 8 + 1);
    }

    // Check all trees got simulations
    bool all_ok = true;
    for (int i = 0; i < NUM_TREES; i++) {
        if (results[i].total_simulations < SIMS * 0.9f) {
            printf("WARNING: Tree %d only got %d sims (expected ~%d)\n",
                   i, results[i].total_simulations, SIMS);
            all_ok = false;
        }
        if (results[i].best_move_from == 0 && results[i].best_move_to == 0 &&
            results[i].root_value < 0.99f) {
            printf("WARNING: Tree %d has no best move\n", i);
            all_ok = false;
        }
    }

    if (all_ok) printf("\n%d/%d trees passed sanity checks\n", NUM_TREES, NUM_TREES);

    // NN mode
    if (argc > 1) {
        printf("\n=== Multi-Tree Eval (NN Mode) ===\n");
        OracleNetWeights* d_weights = load_nn_weights(argv[1]);
        if (!d_weights) {
            printf("Failed to load weights from %s\n", argv[1]);
            return 1;
        }

        float* d_policy_bufs = nullptr;
        cudaMalloc(&d_policy_bufs, NUM_TREES * NN_POLICY_SIZE * sizeof(float));

        memset(results, 0, sizeof(results));
        count = gpu_mcts_eval_trees(positions, NUM_TREES, SIMS,
                                     MAX_NODES_PER_TREE, false, 1.414f,
                                     d_weights, d_policy_bufs, results);

        printf("Processed %d trees (NN mode):\n", count);
        for (int i = 0; i < count; i++) {
            printf("  Tree %d: sims=%d nodes=%d value=%.3f best=%c%d-%c%d\n",
                   i, results[i].total_simulations, results[i].nodes_allocated,
                   results[i].root_value,
                   'a' + results[i].best_move_from % 8, results[i].best_move_from / 8 + 1,
                   'a' + results[i].best_move_to % 8, results[i].best_move_to / 8 + 1);
        }

        cudaFree(d_policy_bufs);
        free_nn_weights(d_weights);
    }

    return all_ok ? 0 : 1;
}
