// Throughput benchmark for GPU MCTS v2 (post-pool-fix).
//
// Uses zero NN weights so no saved model is needed — throughput depends
// only on architecture, not weight values. Reports wall-time
// samples/sec for a configurable games × sims × concurrent run.
//
// Usage: ./cuda/build/bench_throughput [num_games] [sims_per_move] [max_concurrent]
// Defaults: 100 games, 200 sims/move, 36 concurrent (SP_MAX_CONCURRENT).

#include "../selfplay.cuh"
#include "../movegen.cuh"
#include "../nn_weights.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static double wall_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int required_pool(int sims) {
    int v = sims * POOL_FACTOR_PER_SIM;
    return v < MIN_POOL_PER_TREE ? MIN_POOL_PER_TREE : v;
}

int main(int argc, char** argv) {
    init_movegen_tables();

    int num_games  = (argc > 1) ? atoi(argv[1]) : 100;
    int sims       = (argc > 2) ? atoi(argv[2]) : 200;
    int concurrent = (argc > 3) ? atoi(argv[3]) : SP_MAX_CONCURRENT;
    bool use_p2    = (argc > 4) && (strcmp(argv[4], "p2") == 0);
    int pool       = required_pool(sims);

    printf("=== GPU MCTS throughput benchmark ===\n");
    printf("  games=%d  sims/move=%d  concurrent=%d  pool/tree=%d  vloss_p2=%s\n",
           num_games, sims, concurrent, pool, use_p2 ? "ON" : "OFF");

    SelfPlayConfig cfg = {};
    cfg.num_games = num_games;
    cfg.sims_per_move = sims;
    cfg.max_nodes_per_tree = pool;
    cfg.explore_base = 0.80f;
    cfg.enable_koth = false;
    cfg.c_puct = 1.414f;
    cfg.max_concurrent = concurrent;
    cfg.seed = 42;
    cfg.use_resnet = true;
    cfg.use_vloss_p2 = use_p2;

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    GameRecord* records = new GameRecord[num_games];
    for (int i = 0; i < num_games; i++) {
        records[i].samples = nullptr; records[i].num_samples = 0; records[i].result = 0;
    }

    double t0 = wall_seconds();
    int total_samples = run_selfplay_games((void*)d_weights, cfg, records, num_games);
    double elapsed = wall_seconds() - t0;

    int games_done = 0;
    long long total_sims = 0;
    for (int i = 0; i < num_games; i++) {
        if (records[i].num_samples > 0) {
            games_done++;
            total_sims += (long long)records[i].num_samples * sims;
        }
        records[i].free_buf();
    }

    printf("\nResults:\n");
    printf("  wall time:    %.2f s\n", elapsed);
    printf("  games done:   %d/%d\n", games_done, num_games);
    printf("  samples:      %d  (%.1f/game avg)\n",
           total_samples, games_done ? (double)total_samples / games_done : 0.0);
    printf("  samples/sec:  %.1f\n", total_samples / elapsed);
    printf("  sims/sec:     %.0f\n", (double)total_sims / elapsed);

    delete[] records;
    free_nn_weights(d_weights);
    return 0;
}
