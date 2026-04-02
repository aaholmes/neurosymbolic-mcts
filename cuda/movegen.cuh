#pragma once

#include "common.cuh"

// ============================================================
// Move generation device functions
// ============================================================

#ifdef __CUDACC__

// --- Sliding piece attacks via magic bitboards ---

__device__ uint64_t bishop_attacks(int sq, uint64_t occ);
__device__ uint64_t rook_attacks(int sq, uint64_t occ);
__device__ __forceinline__ uint64_t queen_attacks(int sq, uint64_t occ) {
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ);
}

// --- Square attack detection ---

// Is the given square attacked by the specified side?
// Uses reverse lookup: checks if any piece of `by_color` attacks `sq`.
__device__ bool is_square_attacked(const BoardState* bs, int sq, int by_color);

// --- Move generation ---

// Generate all pseudo-legal moves (captures in `caps`, quiet moves in `quiets`).
__device__ void gen_pseudo_legal_moves(const BoardState* bs, MoveList* caps, MoveList* quiets);

// Generate only pseudo-legal captures and promotions.
__device__ void gen_pseudo_legal_captures(const BoardState* bs, MoveList* caps);

// --- Piece-specific generators ---

__device__ void gen_pawn_moves(const BoardState* bs, MoveList* caps, MoveList* quiets);
__device__ void gen_knight_moves(const BoardState* bs, MoveList* caps, MoveList* quiets);
__device__ void gen_bishop_moves(const BoardState* bs, MoveList* caps, MoveList* quiets);
__device__ void gen_rook_moves(const BoardState* bs, MoveList* caps, MoveList* quiets);
__device__ void gen_queen_moves(const BoardState* bs, MoveList* caps, MoveList* quiets);
__device__ void gen_king_moves(const BoardState* bs, MoveList* caps, MoveList* quiets);

#endif // __CUDACC__

// ============================================================
// Host-side API
// ============================================================

// Initialize all movegen tables on GPU (call once at startup).
// Loads magic constants into constant memory and attack tables into global memory.
void init_movegen_tables();
