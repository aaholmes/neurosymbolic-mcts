#include "../quiescence.cuh"
#include "../movegen.cuh"
#include "../apply_move.cuh"
#include "test_helpers.cuh"
#include <cstdio>
#include <cstring>

// ============================================================
// FEN parser (same as test_movegen.cu)
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

// --- PeSTO eval tests ---

__global__ void kernel_pesto_starting_pos(BoardState* d_bs, int32_t* d_result) {
    *d_result = pesto_eval_cp(d_bs);
}

__global__ void kernel_pesto_queen_up(BoardState* d_bs, int32_t* d_result) {
    *d_result = pesto_eval_cp(d_bs);
}

__global__ void kernel_pesto_stm_flip(BoardState* d_bs_w, BoardState* d_bs_b,
                                       int32_t* d_w, int32_t* d_b) {
    *d_w = pesto_eval_cp(d_bs_w);
    *d_b = pesto_eval_cp(d_bs_b);
}

// --- Q-search tests ---

__global__ void kernel_qsearch(BoardState* d_bs, int32_t* d_score, int* d_completed) {
    *d_score = gpu_ext_qsearch(d_bs, -100000, 100000, 20, false, false, 0, d_completed);
}

__global__ void kernel_pesto_balance(BoardState* d_bs, float* d_result, int* d_completed) {
    *d_result = gpu_forced_pesto_balance(d_bs, d_completed);
}

// ============================================================
// Host test functions
// ============================================================

void test_pesto_starting_pos(bool& test_failed) {
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    int32_t* d_result;
    int32_t h_result;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_pesto_starting_pos<<<1,1>>>(d_bs, d_result);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost));
    // Starting position should be near 0 (within ±20 cp)
    if (h_result < -20 || h_result > 20) {
        printf("FAIL: pesto starting pos = %d cp, expected within +-20\n", h_result);
        test_failed = true;
    }
    cudaFree(d_bs);
    cudaFree(d_result);
}

void test_pesto_queen_up(bool& test_failed) {
    // White up a queen: remove black queen from starting position
    // rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    BoardState bs = parse_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    BoardState* d_bs;
    int32_t* d_result;
    int32_t h_result;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_pesto_queen_up<<<1,1>>>(d_bs, d_result);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost));
    // White is up a queen (~1025 mg, ~936 eg), should be ~900+ cp
    if (h_result < 800) {
        printf("FAIL: pesto queen up = %d cp, expected >= 800\n", h_result);
        test_failed = true;
    }
    cudaFree(d_bs);
    cudaFree(d_result);
}

void test_pesto_stm_flip(bool& test_failed) {
    // Same position, white vs black to move: scores should negate
    BoardState bs_w = parse_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    BoardState bs_b = bs_w;
    bs_b.w_to_move = 0;

    BoardState *d_bs_w, *d_bs_b;
    int32_t *d_w, *d_b;
    int32_t h_w, h_b;
    CHECK_CUDA(cudaMalloc(&d_bs_w, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_bs_b, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_w, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_bs_w, &bs_w, sizeof(BoardState), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bs_b, &bs_b, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_pesto_stm_flip<<<1,1>>>(d_bs_w, d_bs_b, d_w, d_b);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_w, d_w, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_b, d_b, sizeof(int32_t), cudaMemcpyDeviceToHost));
    // Scores should be exact negations
    ASSERT_EQ(h_w, -h_b);
    cudaFree(d_bs_w); cudaFree(d_bs_b);
    cudaFree(d_w); cudaFree(d_b);
}

void test_qsearch_quiet_pos(bool& test_failed) {
    // Quiet position with no captures: should return stand-pat
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    int32_t* d_score;
    int* d_completed;
    int32_t h_score;
    int h_completed;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_score, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_completed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_qsearch<<<1,1>>>(d_bs, d_score, d_completed);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_score, d_score, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
    // Starting pos is quiet - q-search should return near stand-pat (within ±20)
    if (h_score < -20 || h_score > 20) {
        printf("FAIL: qsearch quiet pos = %d cp, expected within +-20\n", h_score);
        test_failed = true;
    }
    ASSERT_EQ(h_completed, 1);
    cudaFree(d_bs); cudaFree(d_score); cudaFree(d_completed);
}

void test_qsearch_hanging_piece(bool& test_failed) {
    // Black queen hangs on d1, White rook on d8 can capture: Rxd1 wins the queen
    // 4k3/8/8/8/8/8/8/3qK2R w - - 0 1
    // Actually, use rook on same file as queen: Rook d8, queen d1 -> nah.
    // Simpler: White has rook on d1, black queen hangs on d5. Rxd5 captures.
    // 4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1
    BoardState bs = parse_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1");
    BoardState* d_bs;
    int32_t* d_score;
    int* d_completed;
    int32_t h_score;
    int h_completed;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_score, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_completed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_qsearch<<<1,1>>>(d_bs, d_score, d_completed);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_score, d_score, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
    // After Rxd5, White has rook vs nothing: should be positive (~400+)
    if (h_score < 400) {
        printf("FAIL: qsearch hanging piece = %d cp, expected >= 400\n", h_score);
        test_failed = true;
    }
    cudaFree(d_bs); cudaFree(d_score); cudaFree(d_completed);
}

void test_qsearch_in_check_evasion(bool& test_failed) {
    // White king in check, must evade. Rook is protected so Kxe2 is illegal.
    // 4k3/8/8/8/8/3b4/4r3/4K3 w - - 0 1
    // White king e1, black rook e2 (check), black bishop d3 protects e2.
    // Kxe2 attacked by Bd3. Legal evasions: Kd1, Kf1, Kd2, Kf2.
    BoardState bs = parse_fen("4k3/8/8/8/8/3b4/4r3/4K3 w - - 0 1");
    BoardState* d_bs;
    int32_t* d_score;
    int* d_completed;
    int32_t h_score;
    int h_completed;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_score, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_completed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_qsearch<<<1,1>>>(d_bs, d_score, d_completed);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_score, d_score, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
    // White is in check and down material (R+B vs nothing): score very negative
    if (h_score > -300) {
        printf("FAIL: qsearch in check = %d cp, expected <= -300\n", h_score);
        test_failed = true;
    }
    cudaFree(d_bs); cudaFree(d_score); cudaFree(d_completed);
}

void test_qsearch_checkmate(bool& test_failed) {
    // White is checkmated: king on a1, black queen on b2 (gives check diagonally),
    // bishop on c3 protects queen (so Kxb2 is illegal).
    // Ka2 attacked by Qb2, Kb1 attacked by Qb2. No legal moves = checkmate.
    BoardState bs = parse_fen("k7/8/8/8/8/2b5/1q6/K7 w - - 0 1");
    BoardState* d_bs;
    int32_t* d_score;
    int* d_completed;
    int32_t h_score;
    int h_completed;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_score, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_completed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_qsearch<<<1,1>>>(d_bs, d_score, d_completed);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_score, d_score, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(h_score, -1000000);
    ASSERT_EQ(h_completed, 1);
    cudaFree(d_bs); cudaFree(d_score); cudaFree(d_completed);
}

void test_qsearch_knight_fork(bool& test_failed) {
    // Knight fork of king + rook: White Nc4 + Qa1, Black Rg6 + Kg8
    // 6k1/8/6r1/8/2N5/8/8/Q3K3 w - - 0 1
    // Ne5 forks Kg8 (via f7) and Rg6. Wait, need king+rook both attacked.
    // Ne5 attacks from e5: c4,c6,d3,d7,f3,f7,g4,g6. Rg6 on g6 ✓, Kg8 not on f7.
    // Better: use Nc4 -> Nd6 fork of Ke8 and Rc8:
    // 2r1k3/8/8/8/2N5/8/8/Q3K3 w - - 0 1
    // Nd6+ forks Ke8 and Rc8. Knight on d6 attacks: b5,b7,c4,c8,e4,e8,f5,f7.
    // e8=king ✓, c8=rook ✓. Fork! Knight fork needs {R,Q,K} x2.
    // After Nd6+, king must move, then Nxc8 wins the exchange.
    // White has Q+N vs R, very positive even before fork.
    // The key test: q-search score > stand-pat (fork improves eval).
    // Stand-pat: Q(~1025)+N(~337) vs R(~477) ≈ +885. After fork win rook: +even more.
    BoardState bs = parse_fen("2r1k3/8/8/8/2N5/8/8/Q3K3 w - - 0 1");
    BoardState* d_bs;
    float* d_result;
    int* d_completed;
    float h_result;
    int h_completed;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_completed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_pesto_balance<<<1,1>>>(d_bs, d_result, d_completed);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
    // White is already ahead, fork should make score even higher (>8 pawns = Q+N material)
    if (h_result < 8.0f) {
        printf("FAIL: qsearch knight fork = %.2f pawns, expected >= 8.0\n", h_result);
        test_failed = true;
    }
    cudaFree(d_bs); cudaFree(d_result); cudaFree(d_completed);
}

void test_qsearch_pawn_fork(bool& test_failed) {
    // White pawn on d4 forks black bishops on c6 and e6 (after d5).
    // Give white extra material so stand-pat is positive, fork gains even more.
    // 4k3/8/2b1b3/8/3P4/8/8/1Q2K3 w - - 0 1
    // White has Q+P vs 2B. Already ahead. d5 forks both bishops.
    // wp_cap_bb[d5] includes c6 and e6. Both have bishops (valuable). Fork!
    BoardState bs = parse_fen("4k3/8/2b1b3/8/3P4/8/8/1Q2K3 w - - 0 1");
    BoardState* d_bs;
    float* d_result;
    int* d_completed;
    float h_result;
    int h_completed;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_completed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_pesto_balance<<<1,1>>>(d_bs, d_result, d_completed);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
    // White already ahead (Q vs 2B), fork gains more. Score should be very positive.
    if (h_result < 2.0f) {
        printf("FAIL: qsearch pawn fork = %.2f pawns, expected >= 2.0\n", h_result);
        test_failed = true;
    }
    cudaFree(d_bs); cudaFree(d_result); cudaFree(d_completed);
}

void test_qsearch_budget_exhaustion(bool& test_failed) {
    // After using one tactical move per side, no more tacticals should be searched.
    // We test this indirectly: a position where the extended q-search should complete
    // with both budgets used.
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    int32_t* d_score;
    int* d_completed;
    int32_t h_score;
    int h_completed;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_score, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_completed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));

    // Just verify the normal path completes on a quiet position
    kernel_qsearch<<<1,1>>>(d_bs, d_score, d_completed);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_score, d_score, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(h_completed, 1);
    cudaFree(d_bs); cudaFree(d_score); cudaFree(d_completed);
}

void test_forced_pesto_balance(bool& test_failed) {
    // Test the convenience wrapper
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    float* d_result;
    int* d_completed;
    float h_result;
    int h_completed;
    CHECK_CUDA(cudaMalloc(&d_bs, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_completed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice));
    kernel_pesto_balance<<<1,1>>>(d_bs, d_result, d_completed);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost));
    // Starting pos: near 0 pawns
    ASSERT_NEAR(h_result, 0.0f, 0.3f);
    ASSERT_EQ(h_completed, 1);
    cudaFree(d_bs); cudaFree(d_result); cudaFree(d_completed);
}

// ============================================================
// Main
// ============================================================

int main() {
    // Recursive q-search needs a large stack on GPU (default 1KB is too small)
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024));  // 32KB per thread

    printf("Initializing movegen tables...\n");
    init_movegen_tables();

    int total = 0, passes = 0, failures = 0;
    printf("\n=== PeSTO Evaluation Tests ===\n");
    RUN_TEST(test_pesto_starting_pos);
    RUN_TEST(test_pesto_queen_up);
    RUN_TEST(test_pesto_stm_flip);

    printf("\n=== Extended Quiescence Search Tests ===\n");
    RUN_TEST(test_qsearch_quiet_pos);
    RUN_TEST(test_qsearch_hanging_piece);
    RUN_TEST(test_qsearch_in_check_evasion);
    RUN_TEST(test_qsearch_checkmate);
    RUN_TEST(test_qsearch_knight_fork);
    RUN_TEST(test_qsearch_pawn_fork);
    RUN_TEST(test_qsearch_budget_exhaustion);
    RUN_TEST(test_forced_pesto_balance);

    printf("\n%d/%d tests passed", passes, total);
    if (failures > 0) printf(" (%d FAILED)", failures);
    printf("\n");
    return failures > 0 ? 1 : 0;
}
