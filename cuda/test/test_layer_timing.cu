// Layer-by-layer profiling using CUDA events around individual operations.
// Each operation is run 100 times and averaged.
//
// Usage:
//   ./cuda/build/test_layer_timing /tmp/weights_gen18.bin

#include "../mcts_kernel.cuh"
#include "../tree_store.cuh"
#include "../movegen.cuh"
#include "../nn_weights.cuh"
#include "../block_ops.cuh"
#include "../block_forward.cuh"
#include "../nn_ops.cuh"

#include <cstdio>
#include <cstring>
#include <functional>

static const char* test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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
                bs.pieces[color * 6 + piece] |= (1ULL << (rank * 8 + file));
                file++;
            }
        }
        p++;
    }
    if (*p) p++;
    bs.w_to_move = (*p == 'w') ? 1 : 0;
    for (int c = 0; c < 2; c++) {
        bs.pieces_occ[c] = 0;
        for (int piece = 0; piece < 6; piece++)
            bs.pieces_occ[c] |= bs.pieces[c * 6 + piece];
    }
    return bs;
}

// Timing helper
struct Timer {
    cudaEvent_t s, e;
    Timer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~Timer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    float measure(int iters, std::function<void()> fn) {
        cudaEventRecord(s);
        for (int i = 0; i < iters; i++) fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        return ms / iters;
    }
};

// Wrapper kernel for full forward pass timing
__global__ void kernel_block_forward_pass(
    const BoardState* bs, float q_result,
    const OracleNetWeights* weights,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, weights, smem, policy_out, value_out, k_out);
}

// Individual kernels for timing each operation
__global__ void k_board_to_planes(const BoardState* bs, float* planes) {
    block_board_to_planes(bs, planes);
}

__global__ void k_conv_3x3(const float* in, const float* w, float* out, int ci, int co) {
    block_conv_3x3(in, w, out, ci, co);
}

__global__ void k_conv_3x3_smem(const float* in, const float* w, float* out, float* sw, int ci, int co) {
    block_conv_3x3_smem_w(in, w, out, sw, ci, co);
}

__global__ void k_conv_3x3_tc(const float* in, const half* w, float* out, int ci, int co) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    // Load input into smem
    for (int i = tid; i < ci * 64; i += blockDim.x) smem[i] = in[i];
    __syncthreads();
    half* staging = (half*)(smem + ci * 64 + co * 64);
    block_conv_3x3_tc(smem, w, smem + ci * 64, staging, ci, co);
    // Write output
    for (int i = tid; i < co * 64; i += blockDim.x) out[i] = smem[ci * 64 + i];
    __syncthreads();
}

__global__ void kernel_block_forward_tc(
    const BoardState* bs, float q_result,
    const OracleNetWeights* weights, const ConvWeightsHalf* half_w,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, weights, smem, policy_out, value_out, k_out, half_w);
}

__global__ void k_bn_relu(float* data, const float* g, const float* b,
                          const float* m, const float* v, int ch, bool r) {
    block_bn_relu(data, g, b, m, v, ch, r);
}

__global__ void k_bn_relu_1ch(float* data, const float* g, const float* b,
                              const float* m, const float* v, bool r) {
    block_bn_relu_1ch(data, g, b, m, v, r);
}

__global__ void k_se(float* data, const float* fc1, const float* fc2,
                     int ch, int inner, float* avg, float* fc1_out) {
    block_se_block(data, fc1, fc2, ch, inner, avg, fc1_out);
}

__global__ void k_log_softmax(float* data, int size, float* smem) {
    block_log_softmax(data, size, smem);
}

__global__ void k_1x1_conv(const float* in, const float* w, float* out, int ci, int co) {
    block_1x1_conv(in, w, out, ci, co);
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s <weights.bin>\n", argv[0]); return 1; }
    init_movegen_tables();

    OracleNetWeights* d_weights = load_nn_weights(argv[1]);
    if (!d_weights) return 1;

    // Allocate shared memory sized buffers
    float *d_buf1, *d_buf2, *d_planes, *d_reduce;
    cudaMalloc(&d_buf1, BLOCK_BUF_SIZE * sizeof(float));
    cudaMalloc(&d_buf2, BLOCK_BUF_SIZE * sizeof(float));
    cudaMalloc(&d_planes, NN_INPUT_CHANNELS * 64 * sizeof(float));
    cudaMalloc(&d_reduce, BLOCK_REDUCE_SIZE * sizeof(float));

    // Shared memory weight buffer (needs dynamic smem kernel)
    float *d_weights_smem_host;
    cudaHostAlloc(&d_weights_smem_host, BLOCK_WEIGHTS_SIZE * sizeof(float), 0);

    BoardState bs = parse_fen(test_fen);
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    // Policy and value output buffers
    float *d_policy; cudaMalloc(&d_policy, NN_POLICY_SIZE * sizeof(float));
    float *d_value;  cudaMalloc(&d_value, sizeof(float));
    float *d_k;      cudaMalloc(&d_k, sizeof(float));

    Timer t;
    const int ITERS = 50;
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    printf("=== Layer-by-Layer Timing Profile (block-mode, 256 threads) ===\n");
    printf("Averaged over %d iterations\n\n", ITERS);

    // 1. Board to planes
    float ms_board = t.measure(ITERS, [&]() {
        k_board_to_planes<<<1, 256>>>(d_bs, d_planes);
    });
    printf("  %-30s %8.2f ms\n", "board_to_planes [17,64]", ms_board);

    // 2. Start conv 3x3 (shared memory weight tiling version)
    float* d_conv_smem;
    cudaMalloc(&d_conv_smem, NN_HIDDEN_DIM * 9 * sizeof(float));  // weight tile buffer

    float ms_start_conv = t.measure(ITERS, [&]() {
        k_conv_3x3_smem<<<1, 256>>>(d_planes, d_weights->start_conv_weight, d_buf2,
                                    d_conv_smem, NN_INPUT_CHANNELS, NN_HIDDEN_DIM);
    });
    printf("  %-30s %8.2f ms\n", "start_conv_3x3 [17→128]", ms_start_conv);

    // 3. Start BN+ReLU
    float ms_start_bn = t.measure(ITERS, [&]() {
        k_bn_relu<<<1, 256>>>(d_buf2, d_weights->start_bn.weight, d_weights->start_bn.bias,
                              d_weights->start_bn.running_mean, d_weights->start_bn.running_var,
                              NN_HIDDEN_DIM, true);
    });
    printf("  %-30s %8.2f ms\n", "start_BN+ReLU [128]", ms_start_bn);

    // 4. One residual block (all 4 sub-steps)
    const ResBlockParams* blk = &d_weights->blocks[0];

    float ms_conv1 = t.measure(ITERS, [&]() {
        k_conv_3x3_smem<<<1, 256>>>(d_buf2, blk->conv1_weight, d_buf1,
                                    d_conv_smem, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    });
    printf("  %-30s %8.2f ms\n", "res_conv1 [128→128] (smem_w)", ms_conv1);

    float ms_bn1 = t.measure(ITERS, [&]() {
        k_bn_relu<<<1, 256>>>(d_buf1, blk->bn1.weight, blk->bn1.bias,
                              blk->bn1.running_mean, blk->bn1.running_var,
                              NN_HIDDEN_DIM, true);
    });
    printf("  %-30s %8.2f ms\n", "res_BN1+ReLU [128]", ms_bn1);

    float ms_conv2 = t.measure(ITERS, [&]() {
        k_conv_3x3_smem<<<1, 256>>>(d_buf1, blk->conv2_weight, d_buf2,
                                    d_conv_smem, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    });
    printf("  %-30s %8.2f ms\n", "res_conv2 [128→128] (smem_w)", ms_conv2);

    float ms_bn2_se = t.measure(ITERS, [&]() {
        k_bn_relu<<<1, 256>>>(d_buf2, blk->bn2.weight, blk->bn2.bias,
                              blk->bn2.running_mean, blk->bn2.running_var,
                              NN_HIDDEN_DIM, false);
        k_se<<<1, 256>>>(d_buf2, blk->se.fc1_weight, blk->se.fc2_weight,
                         NN_HIDDEN_DIM, NN_SE_INNER, d_reduce, d_reduce + NN_HIDDEN_DIM);
    });
    printf("  %-30s %8.2f ms\n", "res_BN2+SE [128]", ms_bn2_se);

    // Policy head
    float ms_pconv = t.measure(ITERS, [&]() {
        k_conv_3x3_smem<<<1, 256>>>(d_buf2, d_weights->p_conv_weight, d_buf1,
                                    d_conv_smem, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    });
    printf("  %-30s %8.2f ms\n", "policy_conv_3x3 [128→128] (smem_w)", ms_pconv);

    float ms_pbn = t.measure(ITERS, [&]() {
        k_bn_relu<<<1, 256>>>(d_buf1, d_weights->p_bn.weight, d_weights->p_bn.bias,
                              d_weights->p_bn.running_mean, d_weights->p_bn.running_var,
                              NN_HIDDEN_DIM, true);
    });
    printf("  %-30s %8.2f ms\n", "policy_BN+ReLU [128]", ms_pbn);

    // Value head
    float ms_vconv = t.measure(ITERS, [&]() {
        // 1x1 conv is inlined in block_forward.cu, just time the loop directly
        // We'll use the same loop as in block_forward.cu
        k_1x1_conv<<<1, 256>>>(d_buf2, d_weights->v_conv_weight, d_buf1,
                               NN_HIDDEN_DIM, 1);
    });
    printf("  %-30s %8.2f ms\n", "value_conv_1x1 [128→1]", ms_vconv);

    float ms_vbn = t.measure(ITERS, [&]() {
        k_bn_relu_1ch<<<1, 256>>>(d_buf1, d_weights->v_bn.weight, d_weights->v_bn.bias,
                                  d_weights->v_bn.running_mean, d_weights->v_bn.running_var, true);
    });
    printf("  %-30s %8.2f ms\n", "value_BN+ReLU [1]", ms_vbn);

    // log_softmax
    float ms_softmax = t.measure(ITERS, [&]() {
        k_log_softmax<<<1, 256>>>(d_policy, NN_POLICY_SIZE, d_reduce);
    });
    printf("  %-30s %8.2f ms\n", "log_softmax [4672]", ms_softmax);

    // Full forward pass (for comparison)
    float ms_full = t.measure(ITERS, [&]() {
        cudaFuncSetAttribute(kernel_block_forward_pass,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
        kernel_block_forward_pass<<<1, 256, BLOCK_SMEM_BYTES>>>(
            d_bs, 0.0f, d_weights, d_policy, d_value, d_k);
        cudaDeviceSynchronize();
    });
    printf("\n  %-30s %8.2f ms\n", "FULL FORWARD PASS", ms_full);

    // Sum of individual layers
    float sum_conv = ms_start_conv + 6*(ms_conv1 + ms_conv2) + ms_pconv + ms_vconv;
    float sum_bn   = ms_start_bn + 6*(ms_bn1 + ms_bn2_se) + ms_pbn + ms_vbn;
    float sum_misc = ms_board + ms_softmax;
    printf("\n  %-30s %8.2f ms (%.1f%%)\n", "Total conv_3x3_smem_w (8x)",
           sum_conv - ms_vconv, 100*(sum_conv-ms_vconv)/ms_full);
    printf("  %-30s %8.2f ms (%.1f%%)\n", "Total BN/SE/residual",
           sum_bn, 100*sum_bn/ms_full);
    printf("  %-30s %8.2f ms (%.1f%%)\n", "Other (board, 1x1v, softmax)",
           sum_misc + ms_vconv, 100*(sum_misc+ms_vconv)/ms_full);
    printf("  %-30s %8.2f ms (%.1f%%)\n", "Sync/overhead",
           ms_full - sum_conv - sum_bn - sum_misc,
           100*(ms_full - sum_conv - sum_bn - sum_misc)/ms_full);

    // ================================================================
    // Tensor Core layer-by-layer profiling
    // ================================================================
    printf("\n\n=== Layer-by-Layer Timing Profile (TC mode, wmma FP16) ===\n");
    printf("Averaged over %d iterations\n\n", ITERS);

    ConvWeightsHalf* d_half = convert_weights_to_half(d_weights);

    // Read half weight pointers back from device
    ConvWeightsHalf h_half;
    cudaMemcpy(&h_half, d_half, sizeof(ConvWeightsHalf), cudaMemcpyDeviceToHost);

    // TC start conv (17→128)
    // smem: input[17*64] + output[128*64] + staging[1024]
    size_t tc_start_smem = (NN_INPUT_CHANNELS * 64 + NN_HIDDEN_DIM * 64 + 1024) * sizeof(float);
    cudaFuncSetAttribute(k_conv_3x3_tc,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tc_start_smem);
    float ms_tc_start_conv = t.measure(ITERS, [&]() {
        k_conv_3x3_tc<<<1, 256, tc_start_smem>>>(d_planes,
            h_half.start_conv, d_buf2, NN_INPUT_CHANNELS, NN_HIDDEN_DIM);
    });
    printf("  %-40s %8.2f ms\n", "start_conv_3x3 [17→128] (TC)", ms_tc_start_conv);

    // BN+ReLU (same as scalar — not TC-accelerated)
    printf("  %-40s %8.2f ms\n", "start_BN+ReLU [128]", ms_start_bn);

    // TC residual conv (128→128)
    size_t tc_res_smem = (NN_HIDDEN_DIM * 64 * 2 + 1024) * sizeof(float);
    cudaFuncSetAttribute(k_conv_3x3_tc,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tc_res_smem);

    float ms_tc_conv1 = t.measure(ITERS, [&]() {
        k_conv_3x3_tc<<<1, 256, tc_res_smem>>>(d_buf2,
            h_half.block_conv1[0], d_buf1, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    });
    printf("  %-40s %8.2f ms\n", "res_conv1 [128→128] (TC)", ms_tc_conv1);
    printf("  %-40s %8.2f ms\n", "res_BN1+ReLU [128]", ms_bn1);

    float ms_tc_conv2 = t.measure(ITERS, [&]() {
        k_conv_3x3_tc<<<1, 256, tc_res_smem>>>(d_buf1,
            h_half.block_conv2[0], d_buf2, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    });
    printf("  %-40s %8.2f ms\n", "res_conv2 [128→128] (TC)", ms_tc_conv2);
    printf("  %-40s %8.2f ms\n", "res_BN2+SE [128]", ms_bn2_se);

    // Policy conv TC
    float ms_tc_pconv = t.measure(ITERS, [&]() {
        k_conv_3x3_tc<<<1, 256, tc_res_smem>>>(d_buf2,
            h_half.p_conv, d_buf1, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    });
    printf("  %-40s %8.2f ms\n", "policy_conv_3x3 [128→128] (TC)", ms_tc_pconv);
    printf("  %-40s %8.2f ms\n", "policy_BN+ReLU [128]", ms_pbn);

    // Value head (unchanged)
    printf("  %-40s %8.2f ms\n", "value_conv_1x1 [128→1]", ms_vconv);
    printf("  %-40s %8.2f ms\n", "value_BN+ReLU [1]", ms_vbn);
    printf("  %-40s %8.2f ms\n", "log_softmax [4672]", ms_softmax);

    // Full TC forward pass
    cudaFuncSetAttribute(kernel_block_forward_tc,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    float ms_tc_full = t.measure(ITERS, [&]() {
        kernel_block_forward_tc<<<1, 256, BLOCK_SMEM_BYTES>>>(
            d_bs, 0.0f, d_weights, d_half, d_policy, d_value, d_k);
        cudaDeviceSynchronize();
    });
    printf("\n  %-40s %8.2f ms\n", "FULL FORWARD PASS (TC)", ms_tc_full);

    // Breakdown
    float tc_conv_total = ms_tc_start_conv + 6*(ms_tc_conv1 + ms_tc_conv2) + ms_tc_pconv;
    float tc_bn_total   = ms_start_bn + 6*(ms_bn1 + ms_bn2_se) + ms_pbn + ms_vbn;
    float tc_misc       = ms_board + ms_softmax + ms_vconv;
    float tc_overhead    = ms_tc_full - tc_conv_total - tc_bn_total - tc_misc;
    printf("\n  %-40s %8.2f ms (%5.1f%%)\n", "Total conv_3x3 TC (14 convs)",
           tc_conv_total, 100*tc_conv_total/ms_tc_full);
    printf("  %-40s %8.2f ms (%5.1f%%)\n", "Total BN/SE/residual",
           tc_bn_total, 100*tc_bn_total/ms_tc_full);
    printf("  %-40s %8.2f ms (%5.1f%%)\n", "Other (board, 1x1, softmax)",
           tc_misc, 100*tc_misc/ms_tc_full);
    printf("  %-40s %8.2f ms (%5.1f%%)\n", "Sync/overhead/residual-add",
           tc_overhead, 100*tc_overhead/ms_tc_full);

    // Comparison
    printf("\n  --- Speedup vs scalar ---\n");
    printf("  %-40s %.1fx (%.2f ms → %.2f ms)\n", "conv_3x3 (per call, 128→128)",
           ms_conv1 / ms_tc_conv1, ms_conv1, ms_tc_conv1);
    printf("  %-40s %.1fx (%.2f ms → %.2f ms)\n", "Full forward pass",
           ms_full / ms_tc_full, ms_full, ms_tc_full);

    free_half_weights(d_half);

    cudaFree(d_bs); cudaFree(d_buf1); cudaFree(d_buf2); cudaFree(d_planes);
    cudaFree(d_reduce); cudaFree(d_policy); cudaFree(d_value); cudaFree(d_k);
    cudaFree(d_conv_smem);
    cudaFreeHost(d_weights_smem_host);
    free_nn_weights(d_weights);

    return 0;
}
