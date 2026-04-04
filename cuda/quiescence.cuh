#pragma once

#include "common.cuh"

#ifdef __CUDACC__

// ============================================================
// PeSTO tapered evaluation (device-side)
// ============================================================

// Full-recompute PeSTO eval in centipawns from STM's perspective.
// Mirrors PestoEval::pst_eval_cp() from src/eval.rs.
__device__ int32_t pesto_eval_cp(const BoardState* bs);

// ============================================================
// Extended quiescence search (device-side)
// Port of ext_pesto_qsearch_counted from src/search/quiescence.rs
// ============================================================

// Extended q-search with tactical moves (checks, forks, forked retreats).
// Returns score in centipawns from STM's perspective.
// `completed` is set to the number of positions fully resolved (0 or 1).
__device__ int32_t gpu_ext_qsearch(
    const BoardState* bs,
    int32_t alpha, int32_t beta,
    int max_depth,
    bool white_tactic_used,
    bool black_tactic_used,
    uint64_t forked_pieces_bb,
    int* completed
);

// Convenience wrapper: returns PeSTO balance in pawn units (float).
// `completed` is set to 1 if q-search fully resolved, 0 otherwise.
__device__ float gpu_forced_pesto_balance(const BoardState* bs, int* completed);

// ============================================================
// Principal Exchange (PE) q-search
// Follow the single best MVV-LVA capture at each node.
// A straight line, not a tree. ~1–5 nodes. GPU-friendly.
// ============================================================

// PE search with alpha-beta bounds. Returns score in centipawns from STM perspective.
__device__ int32_t gpu_principal_exchange_search(
    const BoardState* bs,
    int32_t alpha, int32_t beta,
    int max_depth
);

// Convenience wrapper: returns PeSTO balance in pawn units (float).
__device__ float gpu_principal_exchange(const BoardState* bs);

// ============================================================
// Helper functions
// ============================================================

// Check if STM king is in check.
__device__ bool is_in_check(const BoardState* bs);

// Check if a move gives check to the opponent.
__device__ bool gives_check(const BoardState* bs, GPUMove mv);

// Compute fork targets for a pawn or knight move.
// Returns bitboard of forked enemy pieces, or 0 if no fork.
__device__ uint64_t compute_fork_targets(const BoardState* bs, GPUMove mv);

// Check if a quiet move is tactical (gives check or creates a fork).
__device__ bool is_tactical_quiet(const BoardState* bs, GPUMove mv);

#endif // __CUDACC__
