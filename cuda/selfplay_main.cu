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

static int parse_int_flag(int argc, char** argv, const char* flag, int def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
    return def;
}

static int default_pool_size(int sims_per_move) {
    int v = sims_per_move * POOL_FACTOR_PER_SIM;
    if (v < MIN_POOL_PER_TREE) v = MIN_POOL_PER_TREE;
    return v;
}

static int run_selfplay_cmd(int argc, char** argv) {
    // selfplay <weights.bin> <num_games> <sims> <output_dir> [--seed N] [--koth] [--pool-size N]
    if (argc < 6) {
        printf("Usage: selfplay selfplay <weights.bin> <num_games> <sims_per_move> <output_dir> "
               "[--seed N] [--koth] [--resnet] [--pool-size N]\n");
        return 1;
    }

    SelfPlayConfig config = {};
    config.num_games = atoi(argv[3]);
    config.sims_per_move = atoi(argv[4]);
    config.max_nodes_per_tree = parse_int_flag(argc, argv, "--pool-size",
                                               default_pool_size(config.sims_per_move));
    config.explore_base = 0.80f;
    config.enable_koth = has_flag(argc, argv, "--koth");
    config.use_resnet = has_flag(argc, argv, "--resnet");
    config.c_puct = 1.414f;
    config.max_concurrent = SP_MAX_CONCURRENT;
    config.seed = parse_seed(argc, argv);

    int actual = config.num_games < config.max_concurrent ? config.num_games : config.max_concurrent;
    printf("Self-play: %d games, %d sims/move, %d concurrent, seed=%d%s\n",
           config.num_games, config.sims_per_move, actual, config.seed,
           config.use_resnet ? ", resnet" : ", transformer");

    clock_t start = clock();
    int samples = run_selfplay(argv[2], config, argv[5]);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    if (samples >= 0)
        printf("Generated %d samples in %.1f seconds (%.1f samples/sec)\n",
               samples, elapsed, samples / elapsed);

    return samples < 0 ? 1 : 0;
}

static const char* parse_str_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}

static float parse_float_flag(int argc, char** argv, const char* flag, float def) {
    const char* v = parse_str_flag(argc, argv, flag);
    return v ? (float)atof(v) : def;
}

static int run_eval_cmd(int argc, char** argv) {
    // eval <weights_a.bin> <weights_b.bin> <num_games> <sims> [flags]
    if (argc < 6) {
        printf("Usage: selfplay eval <weights_a.bin> <weights_b.bin> <num_games> <sims_per_move>\n"
               "  [--seed N] [--koth] [--save-training-data <dir>]\n"
               "  [--sprt-elo0 F] [--sprt-elo1 F] [--sprt-alpha F] [--sprt-beta F]\n");
        return 1;
    }

    EvalConfig config = {};
    config.num_games = atoi(argv[4]);
    config.sims_per_move = atoi(argv[5]);
    config.max_nodes_per_tree = parse_int_flag(argc, argv, "--pool-size",
                                               default_pool_size(config.sims_per_move));
    config.explore_base = 0.90f;
    config.enable_koth = has_flag(argc, argv, "--koth");
    config.use_resnet = has_flag(argc, argv, "--resnet");
    config.c_puct = 1.414f;
    config.max_concurrent = SP_MAX_CONCURRENT;
    config.seed = parse_seed(argc, argv);
    config.sprt_elo0 = parse_float_flag(argc, argv, "--sprt-elo0", 0.0f);
    config.sprt_elo1 = parse_float_flag(argc, argv, "--sprt-elo1", 10.0f);
    config.sprt_alpha = parse_float_flag(argc, argv, "--sprt-alpha", 0.05f);
    config.sprt_beta = parse_float_flag(argc, argv, "--sprt-beta", 0.05f);
    config.training_data_dir = parse_str_flag(argc, argv, "--save-training-data");

    int actual = config.num_games < config.max_concurrent ? config.num_games : config.max_concurrent;
    printf("Eval: %d games, %d sims/move, %d concurrent, seed=%d\n",
           config.num_games, config.sims_per_move, actual, config.seed);

    // Load both weight sets
    void* d_a;
    void* d_b;
    if (config.use_resnet) {
        d_a = load_nn_weights(argv[2]);
        if (!d_a) return 1;
        d_b = load_nn_weights(argv[3]);
        if (!d_b) { free_nn_weights((OracleNetWeights*)d_a); return 1; }
    } else {
        d_a = load_transformer_weights(argv[2]);
        if (!d_a) return 1;
        d_b = load_transformer_weights(argv[3]);
        if (!d_b) { free_transformer_weights((TransformerWeights*)d_a); return 1; }
    }

    clock_t start = clock();
    EvalResult result = run_eval_games(d_a, d_b, config);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    int total = result.wins_a + result.wins_b + result.draws;
    float winrate = total > 0 ? (float)(result.wins_a + result.draws * 0.5f) / total : 0.5f;

    // Output in format orchestrator can parse
    printf("WINS=%d LOSSES=%d DRAWS=%d WINRATE=%.4f GAMES_PLAYED=%d LLR=%.4f SPRT_RESULT=%s\n",
           result.wins_a, result.wins_b, result.draws, winrate,
           result.games_played, result.llr, result.sprt_result);
    printf("Eval completed in %.1f seconds\n", elapsed);

    if (config.use_resnet) {
        free_nn_weights((OracleNetWeights*)d_a);
        free_nn_weights((OracleNetWeights*)d_b);
    } else {
        free_transformer_weights((TransformerWeights*)d_a);
        free_transformer_weights((TransformerWeights*)d_b);
    }
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
