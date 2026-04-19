#pragma once

#include "common.cuh"
#include "nn_weights.cuh"
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
    int sims_per_move;            // MCTS simulations per move (target visit count)
    int max_nodes_per_tree;       // node pool per tree (>= MIN_POOL_PER_TREE)
    float explore_base;           // proportional sampling decay (default 0.80)
    bool enable_koth;             // King of the Hill
    float c_puct;                 // PUCT exploration constant
    int max_concurrent;           // max concurrent games (default 36)
    int seed;                     // base RNG seed (each game gets seed derived from game_idx + this)
    bool use_resnet;              // use SE-ResNet kernel instead of transformer
};

// Threshold above which a tree's per-call alloc counter forces fresh restart
// on the next move tick. Set below the watermark (POOL_WATERMARK = 0.9) so
// normal reuse cycling never trips the warning.
constexpr float REUSE_RESTART_THRESHOLD = 0.8f;

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
// d_weights is OracleNetWeights* when config.use_resnet, TransformerWeights* otherwise.
int run_selfplay_games(
    void* d_weights,
    const SelfPlayConfig& config,
    GameRecord* records,           // array of num_games GameRecords
    int num_games
);

// ============================================================
// Eval mode: two networks play against each other
// ============================================================

struct EvalConfig {
    int num_games;
    int sims_per_move;
    int max_nodes_per_tree;
    float explore_base;           // typically 0.90 for eval
    bool enable_koth;
    float c_puct;
    int max_concurrent;
    int seed;
    bool use_resnet;              // use SE-ResNet kernel instead of transformer
    // SPRT early stopping
    float sprt_elo0;              // H0 Elo difference (typically 0)
    float sprt_elo1;              // H1 Elo difference (typically 10)
    float sprt_alpha;             // false positive rate (typically 0.05)
    float sprt_beta;              // false negative rate (typically 0.05)
    // Training data output (nullptr = don't save)
    const char* training_data_dir;
};

struct EvalResult {
    int wins_a;       // games won by player A
    int wins_b;       // games won by player B
    int draws;
    int games_played; // may be < num_games if SPRT stopped early
    float llr;        // log-likelihood ratio at termination
    const char* sprt_result;  // "H1", "H0", or "inconclusive"
};

// ============================================================
// Pool-size discipline
//
// Required: max_nodes_per_tree >= max(8192, sims_per_move * 35).
// At branching ~30, a 200-sim search needs ~6000 nodes; pool=8192 gives
// headroom and keeps the per-tree pool below the watermark.
//
// Returns true iff config is valid. Caller should refuse to run if false.
// Reason for failure is printed to stderr.
// ============================================================
constexpr int MIN_POOL_PER_TREE = 8192;
constexpr int POOL_FACTOR_PER_SIM = 35;
constexpr float POOL_WATERMARK = 0.9f;

bool validate_pool_size(int max_nodes_per_tree, int sims_per_move,
                        const char* context);

// ============================================================
// Exposed for testing
// ============================================================

void board_to_planes_host(const BoardState& bs, float* planes);
int move_to_policy_index_host(GPUMove move, int w_to_move);
float compute_llr(int wins, int losses, int draws, float elo0, float elo1);
const char* check_sprt(float llr, float alpha, float beta);

// Play evaluation games between two networks with SPRT early stopping.
// Half the games have A=white, half have A=black.
// Optionally saves training data from both players.
// d_weights_a/b are OracleNetWeights* when config.use_resnet, TransformerWeights* otherwise.
EvalResult run_eval_games(
    void* d_weights_a,
    void* d_weights_b,
    const EvalConfig& config
);
