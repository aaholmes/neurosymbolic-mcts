#include "selfplay.cuh"
#include "movegen.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static int parse_seed(int argc, char** argv) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], "--seed") == 0) return atoi(argv[i + 1]);
    return 0;
}

static bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) return true;
    return false;
}

static int run_selfplay_cmd(int argc, char** argv) {
    // selfplay <weights.bin> <num_games> <sims> <output_dir> [--seed N] [--koth]
    if (argc < 6) {
        printf("Usage: selfplay selfplay <weights.bin> <num_games> <sims_per_move> <output_dir> [--seed N] [--koth]\n");
        return 1;
    }

    SelfPlayConfig config = {};
    config.num_games = atoi(argv[3]);
    config.sims_per_move = atoi(argv[4]);
    config.max_nodes_per_tree = config.sims_per_move + 100;
    config.explore_base = 0.80f;
    config.enable_koth = has_flag(argc, argv, "--koth");
    config.c_puct = 1.414f;
    config.max_concurrent = SP_MAX_CONCURRENT;
    config.seed = parse_seed(argc, argv);

    int actual = config.num_games < config.max_concurrent ? config.num_games : config.max_concurrent;
    printf("Self-play: %d games, %d sims/move, %d concurrent, seed=%d\n",
           config.num_games, config.sims_per_move, actual, config.seed);

    clock_t start = clock();
    int samples = run_selfplay(argv[2], config, argv[5]);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    if (samples >= 0)
        printf("Generated %d samples in %.1f seconds (%.1f samples/sec)\n",
               samples, elapsed, samples / elapsed);

    return samples < 0 ? 1 : 0;
}

static int run_eval_cmd(int argc, char** argv) {
    // eval <weights_a.bin> <weights_b.bin> <num_games> <sims> [--seed N] [--koth]
    if (argc < 6) {
        printf("Usage: selfplay eval <weights_a.bin> <weights_b.bin> <num_games> <sims_per_move> [--seed N] [--koth]\n");
        return 1;
    }

    EvalConfig config = {};
    config.num_games = atoi(argv[4]);
    config.sims_per_move = atoi(argv[5]);
    config.max_nodes_per_tree = config.sims_per_move + 100;
    config.explore_base = 0.90f;  // eval uses higher explore base
    config.enable_koth = has_flag(argc, argv, "--koth");
    config.c_puct = 1.414f;
    config.max_concurrent = SP_MAX_CONCURRENT;
    config.seed = parse_seed(argc, argv);

    int actual = config.num_games < config.max_concurrent ? config.num_games : config.max_concurrent;
    printf("Eval: %d games, %d sims/move, %d concurrent, seed=%d\n",
           config.num_games, config.sims_per_move, actual, config.seed);

    // Load both weight sets
    TransformerWeights* d_a = load_transformer_weights(argv[2]);
    if (!d_a) return 1;
    TransformerWeights* d_b = load_transformer_weights(argv[3]);
    if (!d_b) { free_transformer_weights(d_a); return 1; }

    clock_t start = clock();
    EvalResult result = run_eval_games(d_a, d_b, config);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    int total = result.wins_a + result.wins_b + result.draws;
    float winrate = total > 0 ? (float)(result.wins_a + result.draws * 0.5f) / total : 0.5f;

    // Output in format orchestrator can parse
    printf("WINS=%d LOSSES=%d DRAWS=%d WINRATE=%.4f GAMES_PLAYED=%d\n",
           result.wins_a, result.wins_b, result.draws, winrate, total);
    printf("Eval completed in %.1f seconds\n", elapsed);

    free_transformer_weights(d_a);
    free_transformer_weights(d_b);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <selfplay|eval> ...\n", argv[0]);
        printf("  selfplay <weights.bin> <num_games> <sims> <output_dir> [--seed N] [--koth]\n");
        printf("  eval <weights_a.bin> <weights_b.bin> <num_games> <sims> [--seed N] [--koth]\n");
        return 1;
    }

    init_movegen_tables();

    if (strcmp(argv[1], "selfplay") == 0) return run_selfplay_cmd(argc, argv);
    if (strcmp(argv[1], "eval") == 0) return run_eval_cmd(argc, argv);

    printf("Unknown command: %s\n", argv[1]);
    return 1;
}
