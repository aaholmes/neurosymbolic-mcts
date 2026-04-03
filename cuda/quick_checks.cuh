#pragma once

#include "common.cuh"

#ifdef __CUDACC__

// KOTH center squares: d4, e4, d5, e5
constexpr uint64_t KOTH_CENTER = 0x0000001818000000ULL;

// Check if STM has a mate-in-1 available.
// Iterates all pseudo-legal moves, applies each, checks if opponent
// has zero legal moves and their king is in check.
__device__ bool check_mate_in_1(const BoardState* bs);

// Check if STM can reach a KOTH win in 1 move (or is already on center).
// Returns true if king is on center or can move to center legally.
__device__ bool check_koth_in_1(const BoardState* bs);

#endif // __CUDACC__
