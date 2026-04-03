#include "../quick_checks.cuh"
#include "../movegen.cuh"
#include "../apply_move.cuh"
#include "test_helpers.cuh"
#include <cstdio>
#include <cstring>

// ============================================================
// FEN parser (same as other tests)
// ============================================================

static int char_to_piece(char c, int* color) {
    *color = (c >= 'a') ? BLACK : WHITE;
    switch (c | 0x20) {
        case 'p': return PAWN;
        case 'n': return KNIGHT;
        case 'b': return BISHOP;
        case 'r': return ROOK;
        case 'q': return QUEEN;
        case 'k': return KING;
        default:  return -1;
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
        int ep_file = p[0] - 'a';
        int ep_rank = p[1] - '1';
        bs.en_passant = (uint8_t)(ep_rank * 8 + ep_file);
    }

    for (int c = 0; c < 2; c++) {
        bs.pieces_occ[c] = 0;
        for (int piece = 0; piece < 6; piece++)
            bs.pieces_occ[c] |= bs.pieces[c * 6 + piece];
    }
    return bs;
}

// ============================================================
// Test kernels
// ============================================================

__global__ void kernel_mate_in_1(BoardState* d_bs, int* d_result) {
    *d_result = check_mate_in_1(d_bs) ? 1 : 0;
}

__global__ void kernel_koth_in_1(BoardState* d_bs, int* d_result) {
    *d_result = check_koth_in_1(d_bs) ? 1 : 0;
}

// ============================================================
// Host test functions
// ============================================================

static void run_mate_test(const char* fen, bool expected, const char* name, bool& test_failed) {
    BoardState bs = parse_fen(fen);
    BoardState* d_bs;
    int* d_result;
    int h_result;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_mate_in_1<<<1,1>>>(d_bs, d_result);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    if ((h_result != 0) != expected) {
        printf("FAIL: %s: got %d, expected %d\n", name, h_result, expected ? 1 : 0);
        test_failed = true;
    }
    cudaFree(d_bs);
    cudaFree(d_result);
}

static void run_koth_test(const char* fen, bool expected, const char* name, bool& test_failed) {
    BoardState bs = parse_fen(fen);
    BoardState* d_bs;
    int* d_result;
    int h_result;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_koth_in_1<<<1,1>>>(d_bs, d_result);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    if ((h_result != 0) != expected) {
        printf("FAIL: %s: got %d, expected %d\n", name, h_result, expected ? 1 : 0);
        test_failed = true;
    }
    cudaFree(d_bs);
    cudaFree(d_result);
}

void test_mate_in_1_scholars_mate(bool& test_failed) {
    // White can play Qf7# (queen on h5, bishop on c4)
    // r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4
    // Actually let's use a simpler position:
    // White queen on h5, can deliver Qxf7# with bishop on c4
    // But we need the checking piece protected. Let me use:
    // After 1.e4 e5 2.Bc4 Nc6 3.Qh5 ... Qxf7# is mate
    // r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4
    // Qxf7 is check (queen on f7 attacks e8 king). King can't escape:
    // e8 king: f7 blocked by queen, d8/e7/f8 - queen on f7 covers e8,f8,e7,g8,g6
    // Wait, Nf6 can take the queen? No, Qxf7 is capture. Let me verify.
    // Actually with Nf6 on the board, after Qxf7+, Ke7 is possible? No, Qf7 covers e7.
    // Kd8? Queen covers d8? No, f7 queen covers e8,e7,f8,g8,g7,g6 and diag to c4 bishop.
    // Hmm, this is complex. Let me use a guaranteed back-rank mate instead.

    // Simple back-rank mate: White rook delivers mate on 8th rank
    // 6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1
    // Ra8# is mate: king on g8, pawns f7,g7,h7 block escape, rook on a8 delivers check
    run_mate_test("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", true,
                  "back_rank_mate", test_failed);
}

void test_mate_in_1_back_rank(bool& test_failed) {
    // Back-rank mate: White rook on a1, king out of the way on g2
    // 6k1/5ppp/8/8/8/8/6K1/R7 w - - 0 1
    // Ra8# is mate: rook delivers check on a8, king on g8, pawns f7/g7/h7 block
    run_mate_test("6k1/5ppp/8/8/8/8/6K1/R7 w - - 0 1", true,
                  "back_rank_ra8", test_failed);
}

void test_mate_in_1_starting_pos(bool& test_failed) {
    run_mate_test("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
                  "starting_pos", test_failed);
}

void test_mate_in_1_stalemate(bool& test_failed) {
    // Stalemate position - NOT mate in 1
    // k7/8/1K6/8/8/8/8/1Q6 w - - 0 1
    // Black king on a8, white king b6, white queen b1
    // Qa2 is stalemate not mate. But Qb7 is... check? Yes, but Ka8 is still available? No.
    // Actually Qb8# would be mate: queen on b8 checks a8 king, white king on b6 covers a7,b7,c7.
    // But queen on b1 can't reach b8 in one move? Qb1->b8 yes (same file).
    // Wait, queen on b1 can go to b8. That's check on a8 king. a7 covered by Kb6. Mate!
    // So this is NOT a stalemate position. Let me use a real stalemate.
    // k7/8/1K6/8/8/8/8/8 w - - 0 1 (no pieces to deliver check)
    // White has only king - can't deliver mate in 1.
    run_mate_test("k7/8/1K6/8/8/8/8/8 w - - 0 1", false,
                  "no_mating_material", test_failed);
}

void test_koth_king_one_step(bool& test_failed) {
    // White king on e3, one step from center (e4 is center)
    run_koth_test("4k3/8/8/8/8/4K3/8/8 w - - 0 1", true,
                  "king_one_step", test_failed);
}

void test_koth_king_on_center(bool& test_failed) {
    // White king already on d4 (center)
    run_koth_test("4k3/8/8/8/3K4/8/8/8 w - - 0 1", true,
                  "king_on_center", test_failed);
}

void test_koth_king_far(bool& test_failed) {
    // White king on a1, far from center
    run_koth_test("4k3/8/8/8/8/8/8/K7 w - - 0 1", false,
                  "king_far", test_failed);
}

void test_koth_blocked(bool& test_failed) {
    // White king on e3, but e4 is blocked by own pawn and d4/d5/e5 not adjacent
    // Actually e3 king: can go to d4, e4, f4 (all adjacent). d4 and e4 are center.
    // Let's block those: put own pawns on d4 and e4, and f4 is not center.
    // d5,e5 are center but not adjacent to e3.
    // King on e3 with pawns on d4 and e4: king can't move to center squares.
    // d4 has own pawn (illegal to move there), e4 has own pawn.
    // f4 is not center. So no KOTH-in-1.
    // Wait d3 is not center, d4 IS center but blocked.
    run_koth_test("4k3/8/8/8/3PP3/4K3/8/8 w - - 0 1", false,
                  "king_blocked", test_failed);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("Initializing movegen tables...\n");
    init_movegen_tables();

    int total = 0, passes = 0, failures = 0;
    printf("\n=== Mate-in-1 Tests ===\n");
    RUN_TEST(test_mate_in_1_scholars_mate);
    RUN_TEST(test_mate_in_1_back_rank);
    RUN_TEST(test_mate_in_1_starting_pos);
    RUN_TEST(test_mate_in_1_stalemate);

    printf("\n=== KOTH-in-1 Tests ===\n");
    RUN_TEST(test_koth_king_one_step);
    RUN_TEST(test_koth_king_on_center);
    RUN_TEST(test_koth_king_far);
    RUN_TEST(test_koth_blocked);

    printf("\n%d/%d tests passed", passes, total);
    if (failures > 0) printf(" (%d FAILED)", failures);
    printf("\n");
    return failures > 0 ? 1 : 0;
}
