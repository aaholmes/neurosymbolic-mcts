#pragma once

#include "common.cuh"
#include "transformer_weights.cuh"
#include "mcts_kernel.cuh"

// ============================================================
// GPU Self-Play: Play complete chess games using GPU MCTS
//
// Host-side game loop calls gpu_mcts_eval_trees_transformer
// for batches of concurrent games, then applies moves and
// records training data on the host.
// ============================================================

// Training data format (must match python/train.py)
constexpr int SP_BOARD_FLOATS = 17 * 64;          // 1088
constexpr int SP_SAMPLE_FLOATS = SP_BOARD_FLOATS + 1 + 1 + 1 + NN_POLICY_SIZE;  // 5763
constexpr int SP_MAX_MOVES_PER_GAME = 200;
constexpr int SP_MAX_CONCURRENT = 36;

struct SelfPlayConfig {
    int num_games;                // total games to play
    int sims_per_move;            // MCTS simulations per move
    int max_nodes_per_tree;       // node pool per tree (default 4096)
    float explore_base;           // proportional sampling decay (default 0.80)
    bool enable_koth;             // King of the Hill
    float c_puct;                 // PUCT exploration constant
    int max_concurrent;           // max concurrent games (default 36)
};

struct GameRecord {
    float* samples;    // heap-allocated: SP_MAX_MOVES_PER_GAME * SP_SAMPLE_FLOATS floats
    int num_samples;
    int result;        // 0=ongoing, 1=white_win, 2=black_win, 3=draw

    void alloc() { samples = (float*)calloc(SP_MAX_MOVES_PER_GAME * SP_SAMPLE_FLOATS, sizeof(float)); }
    void free_buf() { if (samples) { free(samples); samples = nullptr; } }
};

// Run self-play games and write training data to output directory.
// Returns total number of training samples generated.
int run_selfplay(
    const char* weights_path,      // transformer weights binary
    const SelfPlayConfig& config,
    const char* output_dir         // directory for .bin training data files
);

// Lower-level: play games and return records in memory (for testing)
int run_selfplay_games(
    TransformerWeights* d_weights,
    const SelfPlayConfig& config,
    GameRecord* records,           // array of num_games GameRecords
    int num_games
);
