#pragma once

#include "common.cuh"
#include "nn_weights.cuh"
#include "nn_forward.cuh"
#include "transformer_weights.cuh"

// ============================================================
// GPU MCTS Kernel — Single-Explorer Classical Mode
//
// A persistent kernel running one explorer warp that performs the
// full MCTS loop: select → expand → quick checks → PE q-search →
// classical value → backup.
//
// Classical mode: V = tanh(0.326 * q_result), no neural network.
// Uniform policy priors (1/num_children for each child).
// ============================================================

#ifdef __CUDACC__

// The persistent MCTS kernel. Runs `max_simulations` iterations of
// select→expand→evaluate→backup on a single warp.
__global__ void mcts_kernel(int max_simulations, bool enable_koth, float c_puct);

#endif // __CUDACC__

// ============================================================
// Host-side API
// ============================================================

// Result of a GPU MCTS search.
struct GPUMctsResult {
    int best_move_from;
    int best_move_to;
    int best_move_promo;
    float root_value;       // Q-value of root (average of children)
    int total_simulations;
    int nodes_allocated;
};

// Run a full MCTS search on the GPU.
// Initializes the tree, uploads the root position, runs the kernel,
// and reads back the best move.
GPUMctsResult gpu_mcts_search(
    const BoardState& root_position,
    int simulations,
    bool enable_koth,
    float c_puct = 1.414f
);

// Read root children info for debugging/comparison.
// Fills arrays with visit counts and Q-values for each root child.
// Returns number of children.
int read_root_children(
    int* visit_counts,  // array of size MAX_CHILDREN
    float* q_values,    // array of size MAX_CHILDREN
    uint16_t* moves,    // array of size MAX_CHILDREN (GPUMove encoding)
    int max_children
);

// Get the board state of the best child (for playing sequential games).
// Returns true if found, false if root has no children.
bool get_best_child_board(BoardState* out_board, uint16_t* out_move);

// ============================================================
// NN-mode search (with neural network forward pass)
// ============================================================

// Run MCTS with neural network policy and value.
// Uses warp-cooperative forward pass (32 threads per warp).
// num_warps: number of explorer warps (each gets its own scratch).
GPUMctsResult gpu_mcts_search_nn(
    const BoardState& root_position,
    int simulations,
    bool enable_koth,
    float c_puct,
    OracleNetWeights* d_weights,   // GPU-resident weights (shared, read-only)
    float* d_nn_scratch,            // per-warp scratch (num_warps * SCRATCH_TOTAL_FLOATS)
    int num_warps
);

// ============================================================
// Block-mode NN search (256 threads/block, activations in shared memory)
// ============================================================

// Run MCTS with neural network policy and value.
// Uses 256-thread block-cooperative forward pass with shared memory activations.
// num_blocks: number of concurrent explorer blocks (each block = one MCTS stream).
// d_policy_bufs: per-block global memory policy buffers (num_blocks * NN_POLICY_SIZE floats).
//   Allocate with cudaMalloc(num_blocks * NN_POLICY_SIZE * sizeof(float)).
GPUMctsResult gpu_mcts_search_nn_block(
    const BoardState& root_position,
    int simulations,
    bool enable_koth,
    float c_puct,
    OracleNetWeights* d_weights,
    float* d_policy_bufs,
    int num_blocks
);

// ============================================================
// Multi-tree eval mode (N independent trees, 1 block each)
// ============================================================

// Result of one tree's evaluation.
struct TreeEvalResult {
    int best_move_from;
    int best_move_to;
    int best_move_promo;
    float root_value;
    int total_simulations;
    int nodes_allocated;
};

// Run N independent MCTS searches on N different root positions.
// Each tree gets its own block (256 threads) and partitioned node pool.
// All trees share the same NN weights and policy buffers.
//
// num_trees: number of independent trees (typically 8-16)
// max_nodes_per_tree: node pool size per tree (e.g. 4096)
// root_positions: host array of N BoardStates
// h_results: host array of N TreeEvalResult (written on return)
// d_policy_bufs: pre-allocated [num_trees × NN_POLICY_SIZE] floats
//
// Returns number of trees processed.
int gpu_mcts_eval_trees(
    const BoardState* root_positions,
    int num_trees,
    int simulations_per_tree,
    int max_nodes_per_tree,
    bool enable_koth,
    float c_puct,
    OracleNetWeights* d_weights,
    float* d_policy_bufs,
    TreeEvalResult* h_results
);

// ============================================================
// Transformer-mode search
// ============================================================

GPUMctsResult gpu_mcts_search_transformer(
    const BoardState& root_position,
    int simulations,
    bool enable_koth,
    float c_puct,
    TransformerWeights* d_weights,
    float* d_policy_bufs,
    int num_blocks
);

int gpu_mcts_eval_trees_transformer(
    const BoardState* root_positions,
    int num_trees,
    int simulations_per_tree,
    int max_nodes_per_tree,
    bool enable_koth,
    float c_puct,
    TransformerWeights* d_weights,
    float* d_policy_bufs,
    TreeEvalResult* h_results
);
