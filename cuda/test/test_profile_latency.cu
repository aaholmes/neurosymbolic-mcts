// Profile inference latency with real trained OracleNet weights.
// Loads weights from a binary file exported by python/export_weights_cuda.py,
// runs single forward passes and full MCTS searches, measures wall-clock time.
//
// Usage:
//   ./cuda/build/test_profile_latency /tmp/weights_gen18.bin

#include "../mcts_kernel.cuh"
#include "../tree_store.cuh"
#include "../movegen.cuh"
#include "../nn_weights.cuh"
#include "../nn_forward.cuh"
#include "../block_forward.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstring>
#include <cstdlib>

// FEN parser
static int char_to_piece(char c, int* color) {
    *color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
    switch (c) {
        case 'P': case 'p': return PAWN;
        case 'N': case 'n': return KNIGHT;
        case 'B': case 'b': return BISHOP;
        case 'R': case 'r': return ROOK;
        case 'Q': case 'q': return QUEEN;
        case 'K': case 'k': return KING;
        default: return -1;
    }
}

static BoardState parse_fen(const char* fen) {
    BoardState bs;
    memset(&bs, 0, sizeof(bs));
    bs.en_passant = EN_PASSANT_NONE;

    int rank = 7, file = 0;
    const char* p = fen;

    while (*p && *p != ' ') {
        if (*p == '/') { rank--; file = 0; }
        else if (*p >= '1' && *p <= '8') { file += (*p - '0'); }
        else {
            int color, piece;
            piece = char_to_piece(*p, &color);
            if (piece >= 0) {
                int sq = rank * 8 + file;
                bs.pieces[color * 6 + piece] |= (1ULL << sq);
                file++;
            }
        }
        p++;
    }
    if (*p) p++;
    bs.w_to_move = (*p == 'w') ? 1 : 0;
    if (*p) p++;
    if (*p) p++;

    while (*p && *p != ' ') {
        switch (*p) {
            case 'K': bs.castling |= CASTLE_WK; break;
            case 'Q': bs.castling |= CASTLE_WQ; break;
            case 'k': bs.castling |= CASTLE_BK; break;
            case 'q': bs.castling |= CASTLE_BQ; break;
        }
        p++;
    }
    if (*p) p++;
    if (*p && *p != '-') {
        int ep_file = p[0] - 'a';
        int ep_rank = p[1] - '1';
        bs.en_passant = (uint8_t)(ep_rank * 8 + ep_file);
    }
    for (int c = 0; c < 2; c++) {
        bs.pieces_occ[c] = 0;
        for (int piece = 0; piece < 6; piece++)
            bs.pieces_occ[c] |= bs.pieces[c * 6 + piece];
    }
    return bs;
}

// Wrapper kernel for forward pass (oracle_net_forward is __device__)
__global__ void kernel_forward_pass(
    const BoardState* bs, float q_result,
    const OracleNetWeights* weights, float* scratch,
    float* policy_out, float* value_out, float* k_out
) {
    oracle_net_forward(bs, q_result, weights, scratch, policy_out, value_out, k_out);
}

// Wrapper kernel for block forward pass (oracle_net_forward_block is __device__)
__global__ void kernel_block_forward_pass(
    const BoardState* bs, float q_result,
    const OracleNetWeights* weights,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, weights, smem, policy_out, value_out, k_out);
}

// CUDA event timer helper
struct Timer {
    cudaEvent_t start, stop;
    Timer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void record_start() { cudaEventRecord(start); }
    float elapsed_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Test positions covering different complexity levels
static const char* test_fens[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",   // starting
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  // italian
    "r2qk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq - 4 7",  // giuoco piano
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  // scholar's mate threat
    "8/5pkp/6p1/8/8/8/5PPP/6K1 w - - 0 1",  // endgame
    "rnbqk2r/ppp2ppp/3p1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 2 6",  // tactical
};

static const char* test_names[] = {
    "starting_pos",
    "italian_game",
    "giuoco_piano",
    "scholars_mate_threat",
    "endgame_kpp",
    "tactical_middlegame",
};

static const int NUM_POSITIONS = 6;

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <weights.bin>\n", argv[0]);
        return 1;
    }

    init_movegen_tables();

    const char* weights_path = argv[1];
    printf("=== Inference Latency Profiling ===\n");
    printf("Weights: %s\n", weights_path);

    // Load real weights
    OracleNetWeights* d_weights = load_nn_weights(weights_path);
    if (!d_weights) {
        printf("FATAL: Failed to load weights from %s\n", weights_path);
        return 1;
    }
    printf("\n");

    Timer timer;

    // -----------------------------------------------------------------------
    // Benchmark 1: Single forward pass latency (warmup + measured)
    // -----------------------------------------------------------------------
    printf("--- Single Forward Pass Latency ---\n");

    float* d_scratch = alloc_nn_scratch(1);
    BoardState* d_bs = nullptr;
    cudaMalloc(&d_bs, sizeof(BoardState));

    // Output buffers for single forward pass
    float* d_policy = nullptr;
    float* d_value = nullptr;
    float* d_k = nullptr;
    cudaMalloc(&d_policy, NN_POLICY_SIZE * sizeof(float));
    cudaMalloc(&d_value, sizeof(float));
    cudaMalloc(&d_k, sizeof(float));

    // Set stack size for forward pass (deep call stack)
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    // -----------------------------------------------------------------------
    // Benchmark 2: Block-mode (256 threads) forward pass latency
    // -----------------------------------------------------------------------
    printf("\n--- Block-Mode (256 threads) Forward Pass ---\n");

    // Set up block-mode kernel for extended shared memory
    cudaFuncSetAttribute(kernel_block_forward_pass,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);

    // Warmup
    BoardState start = make_starting_position();
    cudaMemcpy(d_bs, &start, sizeof(BoardState), cudaMemcpyHostToDevice);
    float q_result = 0.0f;
    kernel_block_forward_pass<<<1, 256, BLOCK_SMEM_BYTES>>>(
        d_bs, q_result, d_weights, d_policy, d_value, d_k);
    cudaDeviceSynchronize();

    // Measure block forward pass
    const int FP_ITERS = 100;
    timer.record_start();
    for (int i = 0; i < FP_ITERS; i++) {
        kernel_block_forward_pass<<<1, 256, BLOCK_SMEM_BYTES>>>(
            d_bs, q_result, d_weights, d_policy, d_value, d_k);
    }
    float bfp_total_ms = timer.elapsed_ms();
    float bfp_per_call = bfp_total_ms / FP_ITERS;
    printf("Single block forward pass: %.2f ms/iter (%.0f iters/sec)\n",
           bfp_per_call, 1000.0f / bfp_per_call);

    // -----------------------------------------------------------------------
    // Benchmark 3: Single warp forward pass latency (for comparison)
    // -----------------------------------------------------------------------
    printf("\n--- Warp-Mode (32 threads) Forward Pass ---\n");

    // Set stack size for warp forward pass (deep call stack)
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    // Warmup
    kernel_forward_pass<<<1, 32>>>(d_bs, q_result, d_weights, d_scratch,
                                    d_policy, d_value, d_k);
    cudaDeviceSynchronize();

    // Measure warp forward pass
    timer.record_start();
    for (int i = 0; i < FP_ITERS; i++) {
        kernel_forward_pass<<<1, 32>>>(d_bs, q_result, d_weights, d_scratch,
                                        d_policy, d_value, d_k);
    }
    float fp_total_ms = timer.elapsed_ms();
    float fp_per_call = fp_total_ms / FP_ITERS;
    printf("Single warp forward pass: %.2f ms/iter (%.0f iters/sec)\n",
           fp_per_call, 1000.0f / fp_per_call);
    printf("Block speedup vs warp: %.1fx\n", fp_per_call / bfp_per_call);

    // -----------------------------------------------------------------------
    // Benchmark 4: Block-mode MCTS search
    // -----------------------------------------------------------------------
    printf("\n--- Block-Mode MCTS Search (400 sims/move) ---\n");

    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, NN_POLICY_SIZE * sizeof(float));  // 1 block

    const int SIMS_PER_MOVE = 400;

    // Warmup
    BoardState pos = make_starting_position();
    GPUMctsResult block_warmup = gpu_mcts_search_nn_block(
        pos, 10, false, 1.414f, d_weights, d_policy_bufs, 1);

    const int SIM_COUNTS[] = {10, 50, 100, 200, 400};
    const int NUM_SIM_COUNTS = 5;

    // Per-simulation breakdown on starting position
    printf("\n--- Block-Mode Per-Simulation Breakdown (starting pos) ---\n");
    pos = make_starting_position();
    for (int si = 0; si < NUM_SIM_COUNTS; si++) {
        timer.record_start();
        GPUMctsResult result = gpu_mcts_search_nn_block(
            pos, SIM_COUNTS[si], false, 1.414f, d_weights, d_policy_bufs, 1);
        float total_ms = timer.elapsed_ms();
        float per_sim = total_ms / SIM_COUNTS[si];
        printf("  %4d sims: %6.1f ms total, %.2f ms/sim, %d nodes, v=%.3f\n",
               SIM_COUNTS[si], total_ms, per_sim,
               result.nodes_allocated, result.root_value);
    }

    // Profile several positions
    printf("\n--- Block-Mode Full MCTS (400 sims/move) ---\n");
    for (int pi = 0; pi < NUM_POSITIONS; pi++) {
        pos = parse_fen(test_fens[pi]);
        timer.record_start();
        GPUMctsResult result = gpu_mcts_search_nn_block(
            pos, SIMS_PER_MOVE, false, 1.414f, d_weights, d_policy_bufs, 1);
        float move_ms = timer.elapsed_ms();

        printf("  %-28s  %6.1f ms  [%d sims, %d nodes, v=%.3f, best=%c%d-%c%d]\n",
               test_names[pi], move_ms,
               result.total_simulations, result.nodes_allocated,
               result.root_value,
               'a' + (result.best_move_from % 8), (result.best_move_from / 8) + 1,
               'a' + (result.best_move_to % 8), (result.best_move_to / 8) + 1);
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n--- Summary ---\n");
    printf("Block forward pass:     %.2f ms\n", bfp_per_call);
    printf("Warp forward pass:      %.2f ms  (%.1fx slower than block)\n", fp_per_call, fp_per_call / bfp_per_call);
    printf("400 sims block (1 block): [see block MCTS results above]\n");
    printf("Warp 400 sims:          [skipped — takes 100+ seconds]\n");
    printf("Projected speedup vs CPU baseline (~840 ms): %.0fx\n",
           840.0f / bfp_per_call / 400.0f * 1000.0f);

    // Cleanup
    free_nn_weights(d_weights);
    cudaFree(d_bs);
    cudaFree(d_policy);
    cudaFree(d_value);
    cudaFree(d_k);
    cudaFree(d_policy_bufs);
    // d_scratch freed below

    printf("\nProfiling complete.\n");
    return 0;
}
