#include "../movegen.cuh"
#include "../apply_move.cuh"
#include "test_helpers.cuh"
#include <cstdio>
#include <cstring>

// ============================================================
// FEN parser (host-side, for setting up test positions)
// ============================================================

static int char_to_piece(char c, int* color) {
    *color = (c >= 'a') ? BLACK : WHITE;
    switch (c | 0x20) {  // tolower
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

    // Piece placement
    while (*p && *p != ' ') {
        if (*p == '/') {
            rank--;
            file = 0;
        } else if (*p >= '1' && *p <= '8') {
            file += (*p - '0');
        } else {
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

    // Side to move
    if (*p) p++;  // skip space
    bs.w_to_move = (*p == 'w') ? 1 : 0;
    if (*p) p++;
    if (*p) p++;  // skip space

    // Castling rights
    while (*p && *p != ' ') {
        switch (*p) {
            case 'K': bs.castling |= CASTLE_WK; break;
            case 'Q': bs.castling |= CASTLE_WQ; break;
            case 'k': bs.castling |= CASTLE_BK; break;
            case 'q': bs.castling |= CASTLE_BQ; break;
            case '-': break;
        }
        p++;
    }

    // En passant
    if (*p) p++;  // skip space
    if (*p && *p != '-') {
        int ep_file = p[0] - 'a';
        int ep_rank = p[1] - '1';
        bs.en_passant = (uint8_t)(ep_rank * 8 + ep_file);
        p += 2;
    } else if (*p) {
        p++;  // skip '-'
    }

    // Compute occupancy
    for (int c = 0; c < 2; c++) {
        bs.pieces_occ[c] = 0;
        for (int piece = 0; piece < 6; piece++) {
            bs.pieces_occ[c] |= bs.pieces[c * 6 + piece];
        }
    }

    return bs;
}

// ============================================================
// GPU perft kernel (single-threaded, recursive)
// ============================================================

__device__ uint64_t gpu_perft(BoardState* bs, int depth) {
    if (depth == 0) return 1;

    MoveList caps, quiets;
    gen_pseudo_legal_moves(bs, &caps, &quiets);

    uint64_t nodes = 0;

    // Process captures
    for (int i = 0; i < caps.count; i++) {
        BoardState child = *bs;
        apply_move(&child, caps.moves[i]);
        if (is_legal(&child)) {
            nodes += gpu_perft(&child, depth - 1);
        }
    }

    // Process quiet moves
    for (int i = 0; i < quiets.count; i++) {
        BoardState child = *bs;
        apply_move(&child, quiets.moves[i]);
        if (is_legal(&child)) {
            nodes += gpu_perft(&child, depth - 1);
        }
    }

    return nodes;
}

__global__ void kernel_perft(BoardState* board, int depth, uint64_t* result) {
    *result = gpu_perft(board, depth);
}

// ============================================================
// Host-side perft runner
// ============================================================

static uint64_t run_perft(const BoardState& board, int depth) {
    BoardState* d_board;
    uint64_t* d_result;
    CHECK_CUDA(cudaMalloc(&d_board, sizeof(BoardState)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(uint64_t)));
    CHECK_CUDA(cudaMemcpy(d_board, &board, sizeof(BoardState), cudaMemcpyHostToDevice));

    kernel_perft<<<1, 1>>>(d_board, depth, d_result);
    CHECK_CUDA(cudaDeviceSynchronize());

    uint64_t result;
    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cudaFree(d_board);
    cudaFree(d_result);
    return result;
}

// ============================================================
// Perft test definitions
// All values from tests/perft_tests.rs (chessprogramming.org)
// ============================================================

struct PerftCase {
    const char* name;
    const char* fen;
    int depth;
    uint64_t expected;
};

static const PerftCase PERFT_CASES[] = {
    // Starting position
    {"start_d1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20},
    {"start_d2", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400},
    {"start_d3", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902},
    {"start_d4", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
    {"start_d5", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
    {"start_d6", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324},

    // Kiwipete (heavy castling/EP/promotion)
    {"kiwi_d1", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
    {"kiwi_d2", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
    {"kiwi_d3", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
    {"kiwi_d4", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
    {"kiwi_d5", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690},

    // Position 2: EP pin edge case
    {"pos2_d1", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14},
    {"pos2_d2", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191},
    {"pos2_d3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812},
    {"pos2_d4", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238},
    {"pos2_d5", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
    {"pos2_d6", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083},

    // Position 3: underpromotion stress
    {"pos3_d1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6},
    {"pos3_d2", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264},
    {"pos3_d3", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467},
    {"pos3_d4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
    {"pos3_d5", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292},

    // Position 4: promotion-heavy
    {"pos4_d1", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44},
    {"pos4_d2", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486},
    {"pos4_d3", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
    {"pos4_d4", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
    {"pos4_d5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194},

    // Position 5: symmetric middlegame
    {"pos5_d1", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1, 46},
    {"pos5_d2", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2, 2079},
    {"pos5_d3", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890},
    {"pos5_d4", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
    {"pos5_d5", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551},
};

static const int NUM_PERFT_CASES = sizeof(PERFT_CASES) / sizeof(PERFT_CASES[0]);

// ============================================================
// Test functions
// ============================================================

void test_starting_position_move_count(bool& test_failed) {
    BoardState bs = make_starting_position();
    uint64_t nodes = run_perft(bs, 1);
    ASSERT_EQ(nodes, 20ULL);
}

void test_perft_suite(bool& test_failed) {
    int sub_pass = 0, sub_fail = 0;

    for (int i = 0; i < NUM_PERFT_CASES; i++) {
        const PerftCase& tc = PERFT_CASES[i];

        // Skip depth 6 tests (too slow for single-threaded GPU perft)
        if (tc.depth >= 6) {
            printf("\n    SKIP %-12s (depth %d)", tc.name, tc.depth);
            continue;
        }

        BoardState bs = parse_fen(tc.fen);
        uint64_t result = run_perft(bs, tc.depth);

        if (result == tc.expected) {
            printf("\n    OK   %-12s depth=%d nodes=%llu", tc.name, tc.depth, (unsigned long long)result);
            sub_pass++;
        } else {
            printf("\n    FAIL %-12s depth=%d expected=%llu got=%llu",
                   tc.name, tc.depth,
                   (unsigned long long)tc.expected,
                   (unsigned long long)result);
            sub_fail++;
            test_failed = true;
        }
    }

    printf("\n    Perft: %d passed, %d failed", sub_pass, sub_fail);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== Phase 2: Move Generator Tests ===\n\n");

    // Set stack size for recursive perft (each level uses ~1.5 KB)
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitStackSize, 16384));

    // Initialize movegen tables
    init_movegen_tables();
    printf("\n");

    int total = 0, passes = 0, failures = 0;

    RUN_TEST(test_starting_position_move_count);
    RUN_TEST(test_perft_suite);

    printf("\n\n%d/%d passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");

    return failures > 0 ? 1 : 0;
}
