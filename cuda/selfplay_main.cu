#include "selfplay.cuh"
#include "movegen.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <weights.bin> <num_games> <sims_per_move> <output_dir>\n", argv[0]);
        return 1;
    }

    init_movegen_tables();

    SelfPlayConfig config = {};
    config.num_games = atoi(argv[2]);
    config.sims_per_move = atoi(argv[3]);
    // Each sim expands at most 1 node. Tree is rebuilt each move (no subtree reuse).
    config.max_nodes_per_tree = config.sims_per_move + 100;
    config.explore_base = 0.80f;
    config.enable_koth = false;
    config.c_puct = 1.414f;
    config.max_concurrent = SP_MAX_CONCURRENT;
    config.seed = 0;  // deterministic; pass --seed N for different runs

    int actual_concurrent = config.num_games < config.max_concurrent ? config.num_games : config.max_concurrent;
    printf("Self-play: %d games, %d sims/move, %d concurrent\n",
           config.num_games, config.sims_per_move, actual_concurrent);

    clock_t start = clock();
    int samples = run_selfplay(argv[1], config, argv[4]);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    if (samples >= 0)
        printf("Generated %d samples in %.1f seconds (%.1f samples/sec)\n",
               samples, elapsed, samples / elapsed);

    return samples < 0 ? 1 : 0;
}
