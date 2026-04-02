#pragma once

#include "common.cuh"

#ifdef __CUDACC__

// Apply a move to a BoardState, producing a new BoardState.
// Handles: normal moves, captures, en passant, castling, promotion,
// double pawn push (sets en_passant), castling rights revocation.
// Does NOT update Zobrist hash (not needed for movegen correctness).
__device__ void apply_move(BoardState* bs, GPUMove move);

// Check if the position is legal: the king of the side that just moved
// (i.e., the side that is NOT to move in `bs`) is not in check.
__device__ bool is_legal(const BoardState* bs);

#endif // __CUDACC__
