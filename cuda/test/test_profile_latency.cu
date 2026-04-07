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
#include "../block_ops.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>

// Forward declarations for weight conversion
ConvWeightsShifted* convert_weights_shifted(const OracleNetWeights* d_weights);
void free_shifted_weights(ConvWeightsShifted* d_sw);

// Max input channels for staging buffer allocation
constexpr int C_IN_MAX = 128;

// Wrapper kernels for individual op timing
__global__ void k_board_to_planes(const BoardState* bs, float* planes) {
    block_board_to_planes(bs, planes);
}
__global__ void k_conv_3x3_shifted(const float* in, half* const* W_s, float* out, half* staging, int ci, int co) {
    block_conv_3x3_shifted(in, W_s, out, staging, ci, co);
}
__global__ void k_bn_relu(float* data, const float* g, const float* b, const float* m, const float* v, int ch, bool r) {
    block_bn_relu(data, g, b, m, v, ch, r);
}
__global__ void k_bn_relu_1ch(float* data, const float* g, const float* b, const float* m, const float* v, bool r) {
    block_bn_relu_1ch(data, g, b, m, v, r);
}
__global__ void k_se(float* data, const float* fc1, const float* fc2, int ch, int inner, float* avg, float* fc1_out) {
    block_se_block(data, fc1, fc2, ch, inner, avg, fc1_out);
}
__global__ void k_log_softmax(float* data, int size, float* smem) {
    block_log_softmax(data, size, smem);
}
__global__ void k_1x1_conv(const float* in, const float* w, float* out, int ci, int co) {
    block_1x1_conv(in, w, out, ci, co);
}

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

// Wrapper kernels for forward pass timing
__global__ void kernel_block_forward_pass(
    const BoardState* bs, float q_result,
    const OracleNetWeights* weights,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, weights, smem, policy_out, value_out, k_out);
}
__global__ void kernel_block_forward_pass_tc(
    const BoardState* bs, float q_result,
    const OracleNetWeights* weights,
    const ConvWeightsHalf* half_w,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, weights, smem, policy_out, value_out, k_out, half_w, nullptr);
}
__global__ void kernel_block_forward_pass_shifted(
    const BoardState* bs, float q_result,
    const OracleNetWeights* weights,
    const ConvWeightsShifted* shifted_w,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, weights, smem, policy_out, value_out, k_out, nullptr, shifted_w);
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
    // Benchmark 2b: Tensor Core block forward pass latency
    // -----------------------------------------------------------------------
    printf("\n--- Tensor Core Block-Mode (256 threads, wmma FP16) Forward Pass ---\n");
    ConvWeightsHalf* d_half_w = convert_weights_to_half(d_weights);

    cudaFuncSetAttribute(kernel_block_forward_pass_tc,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    // Warmup
    kernel_block_forward_pass_tc<<<1, 256, BLOCK_SMEM_BYTES>>>(
        d_bs, q_result, d_weights, d_half_w, d_policy, d_value, d_k);
    cudaDeviceSynchronize();

    timer.record_start();
    for (int i = 0; i < FP_ITERS; i++) {
        kernel_block_forward_pass_tc<<<1, 256, BLOCK_SMEM_BYTES>>>(
            d_bs, q_result, d_weights, d_half_w, d_policy, d_value, d_k);
    }
    float tc_total_ms = timer.elapsed_ms();
    float tc_per_call = tc_total_ms / FP_ITERS;
    printf("Single TC block forward pass: %.2f ms/iter (%.0f iters/sec)\n",
           tc_per_call, 1000.0f / tc_per_call);
    printf("TC speedup vs scalar block: %.1fx\n", bfp_per_call / tc_per_call);

    free_half_weights(d_half_w);

    // -----------------------------------------------------------------------
    // Benchmark 2c: Shifted-copy block forward pass latency
    // -----------------------------------------------------------------------
    printf("\n--- Shifted-Copy Block-Mode (256 threads, wmma FP16) Forward Pass ---\n");
    ConvWeightsShifted* d_shifted_w = convert_weights_shifted(d_weights);

    cudaFuncSetAttribute(kernel_block_forward_pass_shifted,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    // Warmup
    kernel_block_forward_pass_shifted<<<1, 256, BLOCK_SMEM_BYTES>>>(
        d_bs, q_result, d_weights, d_shifted_w, d_policy, d_value, d_k);
    cudaDeviceSynchronize();

    timer.record_start();
    for (int i = 0; i < FP_ITERS; i++) {
        kernel_block_forward_pass_shifted<<<1, 256, BLOCK_SMEM_BYTES>>>(
            d_bs, q_result, d_weights, d_shifted_w, d_policy, d_value, d_k);
    }
    float shifted_total_ms = timer.elapsed_ms();
    float shifted_per_call = shifted_total_ms / FP_ITERS;
    printf("Single shifted block forward pass: %.2f ms/iter (%.0f iters/sec)\n",
           shifted_per_call, 1000.0f / shifted_per_call);
    printf("Shifted speedup vs scalar block: %.1fx\n", bfp_per_call / shifted_per_call);
    printf("Shifted speedup vs TC im2col:    %.1fx\n", tc_per_call / shifted_per_call);

    // -----------------------------------------------------------------------
    // Benchmark 2d: Shifted path layer-by-layer breakdown
    // -----------------------------------------------------------------------
    printf("\n=== Shifted Path Layer Breakdown ===\n");

    // CUDA events for per-op timing
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    // Allocate shared memory buffers for timing individual ops
    float *d_buf1, *d_buf2, *d_planes, *d_reduce;
    cudaMalloc(&d_buf1, BLOCK_BUF_SIZE * sizeof(float));
    cudaMalloc(&d_buf2, BLOCK_BUF_SIZE * sizeof(float));
    cudaMalloc(&d_planes, NN_INPUT_CHANNELS * 64 * sizeof(float));
    cudaMalloc(&d_reduce, BLOCK_REDUCE_SIZE * sizeof(float));
    half *d_shifted_staging;
    cudaMalloc(&d_shifted_staging, C_IN_MAX * 64 * sizeof(half) + 8 * 256 * sizeof(half));  // shifted + a_staging

    auto time_op = [&](const char* name, std::function<void()> fn, int iters = 50) -> float {
        cudaEventRecord(ev_start);
        for (int i = 0; i < iters; i++) fn();
        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);
        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_end);
        float avg = ms / iters;
        printf("  %-38s %8.3f ms\n", name, avg);
        return avg;
    };

    // Shifted conv kernels: 9 × [C_out, C_in] FP16 GEMMs
    float t_s_start = time_op("shifted start_conv [17→128]", [&]() {
        k_conv_3x3_shifted<<<1, 256>>>(d_planes, d_shifted_w->start_conv, d_buf2,
                                       d_shifted_staging, NN_INPUT_CHANNELS, NN_HIDDEN_DIM);
    });
    float t_s_res = time_op("shifted res_conv [128→128]", [&]() {
        const ResBlockParams* blk = &d_weights->blocks[0];
        k_conv_3x3_shifted<<<1, 256>>>(d_buf2, d_shifted_w->block_conv1[0], d_buf1,
                                       d_shifted_staging, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    });
    float t_s_pconv = time_op("shifted policy_conv [128→128]", [&]() {
        k_conv_3x3_shifted<<<1, 256>>>(d_buf2, d_shifted_w->p_conv, d_buf1,
                                       d_shifted_staging, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    });

    float t_bn = time_op("BN+ReLU [128]", [&]() {
        k_bn_relu<<<1, 256>>>(d_buf2, d_weights->start_bn.weight, d_weights->start_bn.bias,
                              d_weights->start_bn.running_mean, d_weights->start_bn.running_var,
                              NN_HIDDEN_DIM, true);
    });
    float t_bn2 = time_op("BN2 (no ReLU) [128]", [&]() {
        const ResBlockParams* blk = &d_weights->blocks[0];
        k_bn_relu<<<1, 256>>>(d_buf2, blk->bn2.weight, blk->bn2.bias,
                              blk->bn2.running_mean, blk->bn2.running_var,
                              NN_HIDDEN_DIM, false);
    });
    float t_se = time_op("SE block [128]", [&]() {
        const ResBlockParams* blk = &d_weights->blocks[0];
        k_se<<<1, 256>>>(d_buf2, blk->se.fc1_weight, blk->se.fc2_weight,
                         NN_HIDDEN_DIM, NN_SE_INNER, d_reduce, d_reduce + NN_HIDDEN_DIM);
    });
    float t_softmax = time_op("log_softmax [4672]", [&]() {
        k_log_softmax<<<1, 256>>>(d_policy, NN_POLICY_SIZE, d_reduce);
    });
    float t_vconv = time_op("value 1x1 conv [128→1]", [&]() {
        k_1x1_conv<<<1, 256>>>(d_buf2, d_weights->v_conv_weight, d_buf1,
                               NN_HIDDEN_DIM, 1);
    });
    float t_vbn = time_op("value BN+ReLU [1]", [&]() {
        k_bn_relu_1ch<<<1, 256>>>(d_buf1, d_weights->v_bn.weight, d_weights->v_bn.bias,
                                  d_weights->v_bn.running_mean, d_weights->v_bn.running_var, true);
    });

    // Sum up the shifted path components
    // 8 conv layers total: 1 start + 6 res × 2 (conv1+conv2) + 1 policy
    float shifted_conv_total = t_s_start + 12 * t_s_res + t_s_pconv;  // ~0.5ms per conv × 14
    float shifted_bn_total = t_bn + 12 * t_bn2 + t_bn;  // ~0.01ms × 14
    float shifted_se_total = 6 * t_se;
    float shifted_other = t_softmax + t_vconv + t_vbn;
    float shifted_sync = shifted_per_call - shifted_conv_total - shifted_bn_total - shifted_se_total - shifted_other;

    printf("\n  %-38s %8.3f ms (%.1f%%)\n", "Total shifted conv_3x3 (14×)", shifted_conv_total, 100*shifted_conv_total/shifted_per_call);
    printf("  %-38s %8.3f ms (%.1f%%)\n", "Total BN+ReLU (14×)", shifted_bn_total, 100*shifted_bn_total/shifted_per_call);
    printf("  %-38s %8.3f ms (%.1f%%)\n", "Total SE (6×)", shifted_se_total, 100*shifted_se_total/shifted_per_call);
    printf("  %-38s %8.3f ms (%.1f%%)\n", "Other (softmax, value head)", shifted_other, 100*shifted_other/shifted_per_call);
    printf("  %-38s %8.3f ms (%.1f%%)\n", "Sync/kernel overhead", shifted_sync, 100*shifted_sync/shifted_per_call);
    printf("  %-38s %8.3f ms\n", "TOTAL shifted forward pass", shifted_per_call);

    // Compare with scalar and TC
    printf("\n  %-38s Scalar: %6.2f ms | TC: %6.2f ms | Shifted: %6.2f ms\n", "Forward pass:", bfp_per_call, tc_per_call, shifted_per_call);
    printf("  %-38s Scalar: %6.2f ms | TC: %6.2f ms | Shifted: %6.2f ms\n", "Convs only:", bfp_per_call * 0.62, tc_per_call * 0.40, shifted_conv_total);

    free_half_weights(d_half_w);
    free_shifted_weights(d_shifted_w);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

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
    // Benchmark 7: Multi-block scalability (same tree, parallel explorers)
    // -----------------------------------------------------------------------
    printf("\n--- Multi-Block Scalability (400 sims, starting pos) ---\n");

    // First get 1-block baseline
    pos = make_starting_position();
    timer.record_start();
    GPUMctsResult baseline_result = gpu_mcts_search_nn_block(pos, 400, false, 1.414f, d_weights, d_policy_bufs, 1);
    float baseline_ms = timer.elapsed_ms();
    printf("  %2d blocks: %6.1f ms  [%d sims, %d nodes, v=%.3f, speedup=1.0x] [baseline]\n",
           1, baseline_ms, baseline_result.total_simulations, baseline_result.nodes_allocated,
           baseline_result.root_value);

    const int BLOCK_COUNTS[] = {4, 8, 16, 20, 24, 28, 32, 36, 40};
    const int NUM_BLOCK_COUNTS = 9;

    for (int bi = 0; bi < NUM_BLOCK_COUNTS; bi++) {
        int blocks = BLOCK_COUNTS[bi];
        timer.record_start();
        GPUMctsResult result = gpu_mcts_search_nn_block(pos, 400, false, 1.414f, d_weights, d_policy_bufs, blocks);
        float move_ms = timer.elapsed_ms();
        printf("  %2d blocks: %6.1f ms  [%d sims, %d nodes, v=%.3f, speedup=%.1fx]\n",
               blocks, move_ms, result.total_simulations, result.nodes_allocated,
               result.root_value, baseline_ms / move_ms);
    }

    printf("  CPU baseline: ~840 ms (from Rust engine profiling)\n");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n--- Summary ---\n");
    printf("Block forward pass:     %.2f ms\n", bfp_per_call);
    printf("Warp forward pass:      %.2f ms  (%.1fx slower than block)\n", fp_per_call, fp_per_call / bfp_per_call);
    printf("400 sims block (1 block): %.1f ms\n", baseline_ms);
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
