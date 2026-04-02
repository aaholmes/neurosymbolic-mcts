#pragma once

#include <cstdio>
#include <cstdlib>

// Simple test assertion macro
#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        printf("FAIL: %s:%d: %s == %lld, expected %lld\n", \
               __FILE__, __LINE__, #a, (long long)_a, (long long)_b); \
        test_failed = true; \
    } \
} while(0)

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        printf("FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        test_failed = true; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    auto _a = (a); auto _b = (b); \
    if ((_a - _b) > (eps) || (_b - _a) > (eps)) { \
        printf("FAIL: %s:%d: %s == %f, expected %f (eps=%f)\n", \
               __FILE__, __LINE__, #a, (double)_a, (double)_b, (double)(eps)); \
        test_failed = true; \
    } \
} while(0)

#define RUN_TEST(fn) do { \
    bool test_failed = false; \
    printf("  %-50s ", #fn); \
    fn(test_failed); \
    if (test_failed) { printf("FAILED\n"); failures++; } \
    else { printf("OK\n"); passes++; } \
    total++; \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Helper to set up starting position BoardState
inline BoardState make_starting_position() {
    BoardState bs;
    memset(&bs, 0, sizeof(bs));

    // White pieces (color=0)
    bs.pieces[WHITE * 6 + PAWN]   = 0x000000000000FF00ULL;  // rank 2
    bs.pieces[WHITE * 6 + KNIGHT] = 0x0000000000000042ULL;  // b1, g1
    bs.pieces[WHITE * 6 + BISHOP] = 0x0000000000000024ULL;  // c1, f1
    bs.pieces[WHITE * 6 + ROOK]   = 0x0000000000000081ULL;  // a1, h1
    bs.pieces[WHITE * 6 + QUEEN]  = 0x0000000000000008ULL;  // d1
    bs.pieces[WHITE * 6 + KING]   = 0x0000000000000010ULL;  // e1

    // Black pieces (color=1)
    bs.pieces[BLACK * 6 + PAWN]   = 0x00FF000000000000ULL;  // rank 7
    bs.pieces[BLACK * 6 + KNIGHT] = 0x4200000000000000ULL;  // b8, g8
    bs.pieces[BLACK * 6 + BISHOP] = 0x2400000000000000ULL;  // c8, f8
    bs.pieces[BLACK * 6 + ROOK]   = 0x8100000000000000ULL;  // a8, h8
    bs.pieces[BLACK * 6 + QUEEN]  = 0x0800000000000000ULL;  // d8
    bs.pieces[BLACK * 6 + KING]   = 0x1000000000000000ULL;  // e8

    // Occupancy
    bs.pieces_occ[WHITE] = 0x000000000000FFFFULL;
    bs.pieces_occ[BLACK] = 0xFFFF000000000000ULL;

    bs.w_to_move = 1;
    bs.en_passant = EN_PASSANT_NONE;
    bs.castling = CASTLE_ALL;
    bs.halfmove = 0;

    return bs;
}
