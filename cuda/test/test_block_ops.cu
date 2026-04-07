#include "../block_ops.cuh"
#include "../block_forward.cuh"
#include "../nn_ops.cuh"
#include "../nn_forward.cuh"
#include "../nn_weights.cuh"
#include "../mcts_kernel.cuh"
#include "../movegen.cuh"
#include "test_helpers.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>

// ============================================================
// FEN parser (shared with test_nn_ops.cu)
// ============================================================

static int char_to_piece_blk(char c, int* color) {
    *color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
    switch (c) {
        case 'P': case 'p': return PAWN;   case 'N': case 'n': return KNIGHT;
        case 'B': case 'b': return BISHOP; case 'R': case 'r': return ROOK;
        case 'Q': case 'q': return QUEEN;  case 'K': case 'k': return KING;
        default: return -1;
    }
}
static BoardState parse_fen_blk(const char* fen) {
    BoardState bs; memset(&bs, 0, sizeof(bs)); bs.en_passant = EN_PASSANT_NONE;
    int rank = 7, file = 0; const char* p = fen;
    while (*p && *p != ' ') {
        if (*p == '/') { rank--; file = 0; }
        else if (*p >= '1' && *p <= '8') { file += (*p - '0'); }
        else { int color, piece = char_to_piece_blk(*p, &color);
            if (piece >= 0) { bs.pieces[color*6+piece] |= (1ULL << (rank*8+file)); file++; } }
        p++;
    }
    if (*p) p++; bs.w_to_move = (*p == 'w') ? 1 : 0; if (*p) p++; if (*p) p++;
    while (*p && *p != ' ') { switch(*p) { case 'K': bs.castling |= CASTLE_WK; break;
        case 'Q': bs.castling |= CASTLE_WQ; break; case 'k': bs.castling |= CASTLE_BK; break;
        case 'q': bs.castling |= CASTLE_BQ; break; } p++; }
    for (int c = 0; c < 2; c++) { bs.pieces_occ[c] = 0;
        for (int piece = 0; piece < 6; piece++) bs.pieces_occ[c] |= bs.pieces[c*6+piece]; }
    return bs;
}

// ============================================================
// block_conv_3x3: known-value test (1 channel, uniform weights)
// ============================================================

__global__ void kernel_block_conv_1ch(
    const float* weights, const float* input, float* output
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    // Copy input (64 floats) into smem[0:64]
    for (int i = tid; i < 64; i += blockDim.x) smem[i] = input[i];
    __syncthreads();

    block_conv_3x3(smem, weights, smem + 64, 1, 1);

    for (int i = tid; i < 64; i += blockDim.x) output[i] = smem[64 + i];
    __syncthreads();
}

void test_block_conv_known_values(bool& test_failed) {
    // 1 channel, 8x8, all-ones input, all-ones weights (3x3 kernel)
    // Interior pixels: sum of 9 neighbors = 9
    // Edge pixels: sum of 6 or 4 neighbors
    float h_weights[9]; for (int i = 0; i < 9; i++) h_weights[i] = 1.0f;
    float h_input[64];  for (int i = 0; i < 64; i++) h_input[i]  = 1.0f;
    float h_output[64];

    float *d_w, *d_in, *d_out;
    cudaMalloc(&d_w,   9 * 4);
    cudaMalloc(&d_in, 64 * 4);
    cudaMalloc(&d_out, 64 * 4);
    cudaMemcpy(d_w,  h_weights, 9 * 4,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, h_input,   64 * 4, cudaMemcpyHostToDevice);

    // smem: input[64] + output[64] = 128 floats
    kernel_block_conv_1ch<<<1, 256, 128 * 4>>>(d_w, d_in, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_out, 64 * 4, cudaMemcpyDeviceToHost);

    // Interior pixel (3,3) → spatial index 3*8+3=27: sum of 9 ones = 9
    ASSERT_NEAR(h_output[27], 9.0f, 1e-4f);
    // Corner pixel (0,0) → spatial 0: only 4 valid neighbors (padding zeros)
    ASSERT_NEAR(h_output[0],  4.0f, 1e-4f);
    // Edge pixel (0,3) → spatial 3: 6 valid neighbors
    ASSERT_NEAR(h_output[3],  6.0f, 1e-4f);

    printf("[interior=9, corner=4, edge=6] ");
    cudaFree(d_w); cudaFree(d_in); cudaFree(d_out);
}

// ============================================================
// block_conv_3x3 vs warp_im2col + warp_gemm (128ch, [8,8])
// ============================================================

// Warp-version conv (for reference)
__global__ void kernel_warp_conv_ref(
    const float* weights, const float* input, float* output, float* col_buf
) {
    warp_im2col_3x3(input, col_buf, NN_HIDDEN_DIM);
    __syncwarp();
    warp_gemm(weights, col_buf, output, NN_HIDDEN_DIM, 64, NN_HIDDEN_DIM * 9);
}

// Block-version conv in shared memory
__global__ void kernel_block_conv_128ch(
    const float* weights, const float* input_global, float* output_global
) {
    extern __shared__ float smem[];  // [128*64] input + [128*64] output = 16384 floats
    int tid = threadIdx.x;

    // Load input into smem
    for (int i = tid; i < NN_HIDDEN_DIM * 64; i += blockDim.x)
        smem[i] = input_global[i];
    __syncthreads();

    block_conv_3x3(smem, weights, smem + NN_HIDDEN_DIM * 64, NN_HIDDEN_DIM, NN_HIDDEN_DIM);

    // Write output to global memory
    for (int i = tid; i < NN_HIDDEN_DIM * 64; i += blockDim.x)
        output_global[i] = smem[NN_HIDDEN_DIM * 64 + i];
    __syncthreads();
}

void test_block_conv_vs_warp(bool& test_failed) {
    int N = NN_HIDDEN_DIM * 64;        // 8192
    int W = NN_HIDDEN_DIM * NN_HIDDEN_DIM * 9; // 147456

    float* h_input   = (float*)malloc(N * 4);
    float* h_weights = (float*)malloc(W * 4);
    float* h_out_warp  = (float*)malloc(N * 4);
    float* h_out_block = (float*)malloc(N * 4);

    // Random-ish input and weights (deterministic)
    for (int i = 0; i < N; i++) h_input[i]   = ((i * 7 + 3) % 17) / 8.0f - 1.0f;
    for (int i = 0; i < W; i++) h_weights[i] = ((i * 11 + 5) % 13) / 6.0f - 1.0f;

    float *d_input, *d_weights, *d_out_warp, *d_out_block, *d_col;
    cudaMalloc(&d_input,    N * 4);
    cudaMalloc(&d_weights,  W * 4);
    cudaMalloc(&d_out_warp, N * 4);
    cudaMalloc(&d_out_block,N * 4);
    cudaMalloc(&d_col,      NN_HIDDEN_DIM * 9 * 64 * 4);  // im2col buffer

    cudaMemcpy(d_input,   h_input,   N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, W * 4, cudaMemcpyHostToDevice);

    // Warp reference
    kernel_warp_conv_ref<<<1, 32>>>(d_weights, d_input, d_out_warp, d_col);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_warp, d_out_warp, N * 4, cudaMemcpyDeviceToHost);

    // Block version (shared memory: 2 × N floats = 64 KB, needs opt-in on sm_89+)
    cudaFuncSetAttribute(kernel_block_conv_128ch,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * N * 4);
    kernel_block_conv_128ch<<<1, 256, 2 * N * 4>>>(d_weights, d_input, d_out_block);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_block, d_out_block, N * 4, cudaMemcpyDeviceToHost);

    // Compare (allow small FP32 rounding differences)
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_out_warp[i] - h_out_block[i]);
        if (diff > 0.05f) mismatches++;
    }
    printf("[mismatches=%d/%d] ", mismatches, N);
    ASSERT_EQ(mismatches, 0);

    free(h_input); free(h_weights); free(h_out_warp); free(h_out_block);
    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_out_warp);
    cudaFree(d_out_block); cudaFree(d_col);
}

// ============================================================
// block_bn_relu matches warp_bn_relu
// ============================================================

__global__ void kernel_block_bn_relu(float* data, const float* g, const float* b,
                                      const float* m, const float* v, int ch, bool relu) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    for (int i = tid; i < ch * 64; i += blockDim.x) smem[i] = data[i];
    __syncthreads();
    block_bn_relu(smem, g, b, m, v, ch, relu);
    for (int i = tid; i < ch * 64; i += blockDim.x) data[i] = smem[i];
    __syncthreads();
}

__global__ void kernel_warp_bn_relu_ref(float* data, const float* g, const float* b,
                                         const float* m, const float* v, int ch, bool relu) {
    warp_bn_relu(data, g, b, m, v, ch, relu);
}

void test_block_bn_relu_matches_warp(bool& test_failed) {
    const int ch = NN_HIDDEN_DIM;
    const int N = ch * 64;
    float* h_data = (float*)malloc(N * 4);
    float* h_gamma = (float*)malloc(ch * 4);
    float* h_beta  = (float*)malloc(ch * 4);
    float* h_mean  = (float*)malloc(ch * 4);
    float* h_var   = (float*)malloc(ch * 4);

    for (int i = 0; i < N;  i++) h_data[i]  = ((i * 13 + 7) % 19) / 9.0f - 1.0f;
    for (int i = 0; i < ch; i++) { h_gamma[i] = 0.8f + (i % 5) * 0.1f;
        h_beta[i] = 0.1f * (i % 3); h_mean[i] = 0.01f * i;
        h_var[i]  = 0.5f + 0.5f * ((i * 3) % 7) / 7.0f; }

    float *d_warp, *d_block, *d_g, *d_b, *d_m, *d_v;
    cudaMalloc(&d_warp,  N * 4); cudaMalloc(&d_block, N * 4);
    cudaMalloc(&d_g, ch*4); cudaMalloc(&d_b, ch*4);
    cudaMalloc(&d_m, ch*4); cudaMalloc(&d_v, ch*4);
    cudaMemcpy(d_warp,  h_data, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block, h_data, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_gamma, ch*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_beta,  ch*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_mean,  ch*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_var,   ch*4, cudaMemcpyHostToDevice);

    kernel_warp_bn_relu_ref<<<1, 32>>>(d_warp, d_g, d_b, d_m, d_v, ch, true);
    cudaDeviceSynchronize();

    kernel_block_bn_relu<<<1, 256, N * 4>>>(d_block, d_g, d_b, d_m, d_v, ch, true);
    cudaDeviceSynchronize();

    float* h_warp  = (float*)malloc(N * 4);
    float* h_block = (float*)malloc(N * 4);
    cudaMemcpy(h_warp,  d_warp,  N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_block, d_block, N*4, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_warp[i] - h_block[i]) > 1e-5f) mismatches++;
    printf("[mismatches=%d/%d] ", mismatches, N);
    ASSERT_EQ(mismatches, 0);

    free(h_data); free(h_gamma); free(h_beta); free(h_mean); free(h_var);
    free(h_warp); free(h_block);
    cudaFree(d_warp); cudaFree(d_block); cudaFree(d_g); cudaFree(d_b);
    cudaFree(d_m); cudaFree(d_v);
}

// ============================================================
// oracle_net_forward_block vs oracle_net_forward (zero weights)
// ============================================================

__global__ void kernel_warp_forward(
    const BoardState* bs, float q_result, const OracleNetWeights* w,
    float* scratch, float* policy_out, float* value_out, float* k_out
) {
    oracle_net_forward(bs, q_result, w, scratch, policy_out, value_out, k_out);
}

__global__ void kernel_block_forward(
    const BoardState* bs, float q_result, const OracleNetWeights* w,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, w, smem, policy_out, value_out, k_out);
}

void test_block_forward_zero_weights(bool& test_failed) {
    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_scratch = alloc_nn_scratch(1);
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float *d_policy_w, *d_value_w, *d_k_w;
    float *d_policy_b, *d_value_b, *d_k_b;
    cudaMalloc(&d_policy_w, NN_POLICY_SIZE * 4); cudaMalloc(&d_value_w, 4); cudaMalloc(&d_k_w, 4);
    cudaMalloc(&d_policy_b, NN_POLICY_SIZE * 4); cudaMalloc(&d_value_b, 4); cudaMalloc(&d_k_b, 4);

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    // Warp version
    kernel_warp_forward<<<1, 32>>>(d_bs, 0.0f, d_weights, d_scratch,
                                    d_policy_w, d_value_w, d_k_w);
    cudaDeviceSynchronize();

    // Block version (needs opt-in to extended smem)
    cudaFuncSetAttribute(kernel_block_forward,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    kernel_block_forward<<<1, 256, BLOCK_SMEM_BYTES>>>(
        d_bs, 0.0f, d_weights, d_policy_b, d_value_b, d_k_b);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ((int)err, (int)cudaSuccess);

    float h_val_w, h_val_b, h_k_w, h_k_b;
    float* h_pol_w = (float*)malloc(NN_POLICY_SIZE * 4);
    float* h_pol_b = (float*)malloc(NN_POLICY_SIZE * 4);

    cudaMemcpy(&h_val_w, d_value_w, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_val_b, d_value_b, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_k_w,   d_k_w,     4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_k_b,   d_k_b,     4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pol_w, d_policy_w, NN_POLICY_SIZE * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pol_b, d_policy_b, NN_POLICY_SIZE * 4, cudaMemcpyDeviceToHost);

    printf("[val: warp=%.4f block=%.4f, k: %.4f vs %.4f] ",
           h_val_w, h_val_b, h_k_w, h_k_b);

    // Values and k should match
    ASSERT_NEAR(h_val_b, h_val_w, 0.05f);
    ASSERT_NEAR(h_k_b,   h_k_w,   0.01f);

    // Policy should be uniform (log-softmax of zeros → -log(4672) ≈ -8.45)
    // Both should be close to each other
    int pol_mismatch = 0;
    for (int i = 0; i < NN_POLICY_SIZE; i++)
        if (fabsf(h_pol_w[i] - h_pol_b[i]) > 0.05f) pol_mismatch++;
    printf("[pol_mismatches=%d] ", pol_mismatch);
    ASSERT_EQ(pol_mismatch, 0);

    free(h_pol_w); free(h_pol_b);
    cudaFree(d_bs); cudaFree(d_policy_w); cudaFree(d_value_w); cudaFree(d_k_w);
    cudaFree(d_policy_b); cudaFree(d_value_b); cudaFree(d_k_b);
    free_nn_scratch(d_scratch); free_nn_weights(d_weights);
}

void test_block_forward_q_sensitivity(bool& test_failed) {
    // With zero weights: V = tanh(k * q_result), same formula as warp version.
    // Block version should also show V(-3) < V(0) < V(3).
    OracleNetWeights* d_weights = init_nn_weights_zeros();
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float *d_policy, *d_value, *d_k;
    cudaMalloc(&d_policy, NN_POLICY_SIZE * 4);
    cudaMalloc(&d_value, 4); cudaMalloc(&d_k, 4);
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    float q_values[] = {-3.0f, 0.0f, 3.0f};
    float h_values[3];

    cudaFuncSetAttribute(kernel_block_forward,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    for (int i = 0; i < 3; i++) {
        kernel_block_forward<<<1, 256, BLOCK_SMEM_BYTES>>>(
            d_bs, q_values[i], d_weights, d_policy, d_value, d_k);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_values[i], d_value, 4, cudaMemcpyDeviceToHost);
    }

    printf("[V(-3)=%.2f, V(0)=%.2f, V(3)=%.2f] ",
           h_values[0], h_values[1], h_values[2]);
    ASSERT_TRUE(h_values[0] < h_values[1]);
    ASSERT_TRUE(h_values[1] < h_values[2]);
    ASSERT_NEAR(h_values[1], 0.0f, 0.05f);
    ASSERT_TRUE(h_values[2] > 0.5f);
    ASSERT_TRUE(h_values[0] < -0.5f);

    cudaFree(d_bs); cudaFree(d_policy); cudaFree(d_value); cudaFree(d_k);
    free_nn_weights(d_weights);
}

// ============================================================
// Shared memory size check
// ============================================================

void test_block_smem_size(bool& test_failed) {
    int sz = block_smem_bytes();
    printf("[%d bytes = %.1f KB] ", sz, sz / 1024.0f);
    // Must fit in 96 KB per SM (standard limit)
    ASSERT_TRUE(sz > 60 * 1024);   // sanity lower bound
    ASSERT_TRUE(sz < 96 * 1024);   // must fit
}

// ============================================================
// Block-mode MCTS produces a legal move (zero weights)
// ============================================================

void test_block_mcts_produces_moves(bool& test_failed) {
    BoardState start = make_starting_position();
    OracleNetWeights* d_weights = init_nn_weights_zeros();

    // Policy buffers: 1 block × 4672 floats
    float* d_policy_bufs;
    cudaMalloc(&d_policy_bufs, 1 * NN_POLICY_SIZE * sizeof(float));
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    GPUMctsResult result = gpu_mcts_search_nn_block(
        start, 200, false, 1.414f, d_weights, d_policy_bufs, 1
    );

    printf("[sims=%d, move %d->%d] ", result.total_simulations,
           result.best_move_from, result.best_move_to);

    // Should have done some simulations
    ASSERT_TRUE(result.total_simulations > 0);
    // Move should be on the board (valid square indices)
    ASSERT_TRUE(result.best_move_from >= 0 && result.best_move_from < 64);
    ASSERT_TRUE(result.best_move_to   >= 0 && result.best_move_to   < 64);
    // From starting position, moves must be from rank 1 or 2 (white pawns/knights)
    ASSERT_TRUE(result.best_move_from < 16);

    cudaFree(d_policy_bufs);
    free_nn_weights(d_weights);
}

// ============================================================
// Block-mode MCTS: find checkmate from mate-in-1 position
// ============================================================

void test_block_mcts_mate_detection(bool& test_failed) {
    // White to move, Qh5 is checkmate
    // FEN: r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4
    BoardState bs = parse_fen_blk("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4");

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_policy_bufs;
    cudaMalloc(&d_policy_bufs, 1 * NN_POLICY_SIZE * sizeof(float));
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    GPUMctsResult result = gpu_mcts_search_nn_block(
        bs, 200, false, 1.414f, d_weights, d_policy_bufs, 1
    );

    printf("[sims=%d, move %d->%d] ", result.total_simulations,
           result.best_move_from, result.best_move_to);

    // Qh5xe8 or Qh5f7 is the winning move — just check a move is returned
    ASSERT_TRUE(result.best_move_from >= 0 && result.best_move_from < 64);
    ASSERT_TRUE(result.best_move_to   >= 0 && result.best_move_to   < 64);
    // The root value should be strongly positive (forced win found)
    printf("[val=%.2f] ", result.root_value);
    ASSERT_TRUE(result.root_value > 0.5f);

    cudaFree(d_policy_bufs);
    free_nn_weights(d_weights);
}

// ============================================================
// block_log_softmax outputs sum to 1 (probability mass test)
// ============================================================

__global__ void kernel_block_log_softmax(float* data, int size) {
    extern __shared__ float smem[];
    block_log_softmax(data, size, smem);
}

void test_block_log_softmax(bool& test_failed) {
    const int N = NN_POLICY_SIZE;  // 4672
    float* h_data = (float*)malloc(N * 4);
    // Non-uniform input
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 23) / 23.0f;

    float* d_data;
    cudaMalloc(&d_data, N * 4);
    cudaMemcpy(d_data, h_data, N * 4, cudaMemcpyHostToDevice);

    kernel_block_log_softmax<<<1, 256, 256 * 4>>>(d_data, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, N * 4, cudaMemcpyDeviceToHost);

    // Sum of exp(log_softmax) should ≈ 1.0
    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += exp((double)h_data[i]);
    printf("[sum_prob=%.6f] ", (float)sum);
    ASSERT_NEAR((float)sum, 1.0f, 0.001f);

    // All values should be ≤ 0 (log probabilities)
    for (int i = 0; i < N; i++) ASSERT_TRUE(h_data[i] <= 0.0f);

    free(h_data);
    cudaFree(d_data);
}

// ============================================================
// Tensor Core conv3x3 tests
// ============================================================

__global__ void test_fp32_to_fp16(const float* src, half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// TC conv kernel: loads input into smem, runs TC conv, writes output
__global__ void kernel_tc_conv_128ch(
    const float* input_global, const half* weights_h, float* output_global
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    // smem layout: input[128*64] + output[128*64] + staging[1024]
    float* input_smem  = smem;
    float* output_smem = smem + NN_HIDDEN_DIM * 64;
    half*  staging     = (half*)(smem + 2 * NN_HIDDEN_DIM * 64);

    // Load input into smem
    for (int i = tid; i < NN_HIDDEN_DIM * 64; i += blockDim.x)
        input_smem[i] = input_global[i];
    __syncthreads();

    block_conv_3x3_tc(input_smem, weights_h, output_smem, staging,
                      NN_HIDDEN_DIM, NN_HIDDEN_DIM);

    // Write output to global memory
    for (int i = tid; i < NN_HIDDEN_DIM * 64; i += blockDim.x)
        output_global[i] = output_smem[NN_HIDDEN_DIM * 64 * 0 + i];
    // Note: output_smem starts at smem + 8192, but output_smem var already points there
}

// TC conv for start conv: 17->128 channels
__global__ void kernel_tc_conv_start(
    const float* input_global, const half* weights_h, float* output_global
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float* input_smem  = smem;                       // [17, 64]
    float* output_smem = smem + NN_INPUT_CHANNELS * 64;  // [128, 64]
    half*  staging     = (half*)(smem + NN_INPUT_CHANNELS * 64 + NN_HIDDEN_DIM * 64);

    for (int i = tid; i < NN_INPUT_CHANNELS * 64; i += blockDim.x)
        input_smem[i] = input_global[i];
    __syncthreads();

    block_conv_3x3_tc(input_smem, weights_h, output_smem, staging,
                      NN_INPUT_CHANNELS, NN_HIDDEN_DIM);

    for (int i = tid; i < NN_HIDDEN_DIM * 64; i += blockDim.x)
        output_global[i] = output_smem[i];
}

void test_tc_conv_vs_scalar(bool& test_failed) {
    int N = NN_HIDDEN_DIM * 64;        // 8192
    int W = NN_HIDDEN_DIM * NN_HIDDEN_DIM * 9; // 147456

    float* h_input   = (float*)malloc(N * 4);
    float* h_weights = (float*)malloc(W * 4);
    float* h_out_scalar = (float*)malloc(N * 4);
    float* h_out_tc     = (float*)malloc(N * 4);

    // Deterministic input and weights (small values to avoid FP16 overflow)
    for (int i = 0; i < N; i++) h_input[i]   = ((i * 7 + 3) % 17) / 17.0f - 0.5f;
    for (int i = 0; i < W; i++) h_weights[i] = ((i * 11 + 5) % 13) / 13.0f - 0.5f;

    float *d_input, *d_weights, *d_out_scalar, *d_out_tc;
    cudaMalloc(&d_input,      N * 4);
    cudaMalloc(&d_weights,    W * 4);
    cudaMalloc(&d_out_scalar, N * 4);
    cudaMalloc(&d_out_tc,     N * 4);
    cudaMemcpy(d_input,   h_input,   N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, W * 4, cudaMemcpyHostToDevice);

    // Scalar reference
    size_t smem_scalar = 2 * N * 4;
    cudaFuncSetAttribute(kernel_block_conv_128ch,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_scalar);
    kernel_block_conv_128ch<<<1, 256, smem_scalar>>>(d_weights, d_input, d_out_scalar);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_scalar, d_out_scalar, N * 4, cudaMemcpyDeviceToHost);

    // Convert weights to FP16
    half* d_weights_h;
    cudaMalloc(&d_weights_h, W * sizeof(half));
    test_fp32_to_fp16<<<(W + 255) / 256, 256>>>(d_weights, d_weights_h, W);
    cudaDeviceSynchronize();

    // TC version: smem = input[8192] + output[8192] + staging[1024] = 17408 floats
    size_t smem_tc = (2 * N + 1024) * 4;
    cudaFuncSetAttribute(kernel_tc_conv_128ch,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_tc);
    kernel_tc_conv_128ch<<<1, 256, smem_tc>>>(d_input, d_weights_h, d_out_tc);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA error: %s] ", cudaGetErrorString(err));
        test_failed = true;
        free(h_input); free(h_weights); free(h_out_scalar); free(h_out_tc);
        cudaFree(d_input); cudaFree(d_weights); cudaFree(d_out_scalar);
        cudaFree(d_out_tc); cudaFree(d_weights_h);
        return;
    }

    cudaMemcpy(h_out_tc, d_out_tc, N * 4, cudaMemcpyDeviceToHost);

    // Compare: FP16 inputs mean ~1e-3 relative error per multiply,
    // accumulated over 128*9=1152 terms. Allow generous tolerance.
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_out_scalar[i] - h_out_tc[i]);
        if (diff > max_diff) max_diff = diff;
        // Tolerance: relative or absolute
        float tol = fmaxf(0.5f, 0.02f * fabsf(h_out_scalar[i]));
        if (diff > tol) mismatches++;
    }
    printf("[mismatches=%d/%d, max_diff=%.3f] ", mismatches, N, max_diff);
    ASSERT_EQ(mismatches, 0);

    free(h_input); free(h_weights); free(h_out_scalar); free(h_out_tc);
    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_out_scalar);
    cudaFree(d_out_tc); cudaFree(d_weights_h);
}

// Scalar GPU conv for arbitrary C_in -> C_out
__global__ void kernel_block_conv_generic(
    const float* weights, const float* input_global, float* output_global,
    int C_in, int C_out
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int N_in  = C_in * 64;
    int N_out = C_out * 64;

    for (int i = tid; i < N_in; i += blockDim.x)
        smem[i] = input_global[i];
    __syncthreads();

    block_conv_3x3(smem, weights, smem + N_in, C_in, C_out);

    for (int i = tid; i < N_out; i += blockDim.x)
        output_global[i] = smem[N_in + i];
    __syncthreads();
}

void test_tc_conv_start(bool& test_failed) {
    int N_in = NN_INPUT_CHANNELS * 64;   // 17*64 = 1088
    int N_out = NN_HIDDEN_DIM * 64;      // 128*64 = 8192
    int W = NN_HIDDEN_DIM * NN_INPUT_CHANNELS * 9;  // 128*17*9 = 19584

    float* h_input   = (float*)malloc(N_in * 4);
    float* h_weights = (float*)malloc(W * 4);
    float* h_out_scalar = (float*)malloc(N_out * 4);
    float* h_out_tc     = (float*)malloc(N_out * 4);

    for (int i = 0; i < N_in; i++) h_input[i]   = ((i * 7 + 3) % 17) / 17.0f - 0.5f;
    for (int i = 0; i < W;    i++) h_weights[i]  = ((i * 11 + 5) % 13) / 13.0f - 0.5f;

    float *d_input, *d_weights, *d_out_scalar, *d_out_tc;
    cudaMalloc(&d_input,      N_in * 4);
    cudaMalloc(&d_weights,    W * 4);
    cudaMalloc(&d_out_scalar, N_out * 4);
    cudaMalloc(&d_out_tc,     N_out * 4);
    cudaMemcpy(d_input,   h_input,   N_in * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, W * 4,    cudaMemcpyHostToDevice);

    // Scalar GPU reference
    size_t smem_scalar = (N_in + N_out) * 4;
    cudaFuncSetAttribute(kernel_block_conv_generic,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_scalar);
    kernel_block_conv_generic<<<1, 256, smem_scalar>>>(
        d_weights, d_input, d_out_scalar, NN_INPUT_CHANNELS, NN_HIDDEN_DIM);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_scalar, d_out_scalar, N_out * 4, cudaMemcpyDeviceToHost);

    // TC version
    half* d_weights_h;
    cudaMalloc(&d_weights_h, W * sizeof(half));
    test_fp32_to_fp16<<<(W + 255) / 256, 256>>>(d_weights, d_weights_h, W);
    cudaDeviceSynchronize();

    // smem = input[17*64] + output[128*64] + staging[1024]
    size_t smem_tc = (N_in + N_out + 1024) * 4;
    cudaFuncSetAttribute(kernel_tc_conv_start,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_tc);
    kernel_tc_conv_start<<<1, 256, smem_tc>>>(d_input, d_weights_h, d_out_tc);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA error: %s] ", cudaGetErrorString(err));
        test_failed = true;
    } else {
        cudaMemcpy(h_out_tc, d_out_tc, N_out * 4, cudaMemcpyDeviceToHost);

        int mismatches = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < N_out; i++) {
            float diff = fabsf(h_out_scalar[i] - h_out_tc[i]);
            if (diff > max_diff) max_diff = diff;
            float tol = fmaxf(0.5f, 0.02f * fabsf(h_out_scalar[i]));
            if (diff > tol) mismatches++;
        }
        printf("[mismatches=%d/%d, max_diff=%.3f] ", mismatches, N_out, max_diff);
        ASSERT_EQ(mismatches, 0);
    }

    free(h_input); free(h_weights); free(h_out_scalar); free(h_out_tc);
    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_out_scalar);
    cudaFree(d_out_tc); cudaFree(d_weights_h);
}

// TC forward pass test: compare scalar vs TC forward pass with zero weights
__global__ void kernel_block_forward_tc(
    const BoardState* bs, float q_result, const OracleNetWeights* w,
    const ConvWeightsHalf* half_w,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, w, smem, policy_out, value_out, k_out, half_w);
}

void test_tc_forward_zero_weights(bool& test_failed) {
    OracleNetWeights* d_weights = init_nn_weights_zeros();
    ConvWeightsHalf* d_half = convert_weights_to_half(d_weights);

    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float *d_policy_s, *d_value_s, *d_k_s;  // scalar
    float *d_policy_t, *d_value_t, *d_k_t;  // TC
    cudaMalloc(&d_policy_s, NN_POLICY_SIZE * 4); cudaMalloc(&d_value_s, 4); cudaMalloc(&d_k_s, 4);
    cudaMalloc(&d_policy_t, NN_POLICY_SIZE * 4); cudaMalloc(&d_value_t, 4); cudaMalloc(&d_k_t, 4);

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    // Scalar forward pass
    cudaFuncSetAttribute(kernel_block_forward,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    kernel_block_forward<<<1, 256, BLOCK_SMEM_BYTES>>>(
        d_bs, 0.0f, d_weights, d_policy_s, d_value_s, d_k_s);
    cudaDeviceSynchronize();

    // TC forward pass
    cudaFuncSetAttribute(kernel_block_forward_tc,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    kernel_block_forward_tc<<<1, 256, BLOCK_SMEM_BYTES>>>(
        d_bs, 0.0f, d_weights, d_half, d_policy_t, d_value_t, d_k_t);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA error: %s] ", cudaGetErrorString(err));
        test_failed = true;
    } else {
        float h_val_s, h_val_t, h_k_s, h_k_t;
        cudaMemcpy(&h_val_s, d_value_s, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_val_t, d_value_t, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k_s,   d_k_s,     4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k_t,   d_k_t,     4, cudaMemcpyDeviceToHost);

        printf("[val: scalar=%.4f tc=%.4f, k: %.4f vs %.4f] ",
               h_val_s, h_val_t, h_k_s, h_k_t);

        // With zero weights, both should produce value ≈ 0
        ASSERT_NEAR(h_val_t, h_val_s, 0.1f);
        ASSERT_NEAR(h_k_t,   h_k_s,   0.01f);

        // Compare policies
        float* h_pol_s = (float*)malloc(NN_POLICY_SIZE * 4);
        float* h_pol_t = (float*)malloc(NN_POLICY_SIZE * 4);
        cudaMemcpy(h_pol_s, d_policy_s, NN_POLICY_SIZE * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pol_t, d_policy_t, NN_POLICY_SIZE * 4, cudaMemcpyDeviceToHost);

        int pol_mismatch = 0;
        for (int i = 0; i < NN_POLICY_SIZE; i++)
            if (fabsf(h_pol_s[i] - h_pol_t[i]) > 0.1f) pol_mismatch++;
        printf("[pol_mismatches=%d] ", pol_mismatch);
        ASSERT_EQ(pol_mismatch, 0);

        free(h_pol_s); free(h_pol_t);
    }

    cudaFree(d_bs);
    cudaFree(d_policy_s); cudaFree(d_value_s); cudaFree(d_k_s);
    cudaFree(d_policy_t); cudaFree(d_value_t); cudaFree(d_k_t);
    free_half_weights(d_half);
    free_nn_weights(d_weights);
}

// ============================================================
// Shifted-copy conv3x3 tests
// ============================================================

__global__ void kernel_shifted_conv_128ch(
    const float* input_global, half* const* W_s, float* output_global
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float* input_smem = smem;
    float* output_smem = smem + NN_HIDDEN_DIM * 64;
    half*  shifted = (half*)(smem + 2 * NN_HIDDEN_DIM * 64);

    for (int i = tid; i < NN_HIDDEN_DIM * 64; i += blockDim.x)
        input_smem[i] = input_global[i];
    __syncthreads();

    block_conv_3x3_shifted(input_smem, W_s, output_smem, shifted,
                           NN_HIDDEN_DIM, NN_HIDDEN_DIM);

    for (int i = tid; i < NN_HIDDEN_DIM * 64; i += blockDim.x)
        output_global[i] = output_smem[i];
}

void test_shifted_conv_vs_scalar(bool& test_failed) {
    int N = NN_HIDDEN_DIM * 64;
    int W = NN_HIDDEN_DIM * NN_HIDDEN_DIM * 9;

    float* h_input   = (float*)malloc(N * 4);
    float* h_weights = (float*)malloc(W * 4);
    float* h_out_scalar = (float*)malloc(N * 4);
    float* h_out_shifted = (float*)malloc(N * 4);

    for (int i = 0; i < N; i++) h_input[i]   = ((i * 7 + 3) % 17) / 17.0f - 0.5f;
    for (int i = 0; i < W; i++) h_weights[i] = ((i * 11 + 5) % 13) / 13.0f - 0.5f;

    float *d_input, *d_weights, *d_out_scalar, *d_out_shifted;
    cudaMalloc(&d_input,      N * 4);
    cudaMalloc(&d_weights,    W * 4);
    cudaMalloc(&d_out_scalar, N * 4);
    cudaMalloc(&d_out_shifted, N * 4);
    cudaMemcpy(d_input,   h_input,   N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, W * 4, cudaMemcpyHostToDevice);

    // Scalar reference
    size_t smem_scalar = 2 * N * 4;
    cudaFuncSetAttribute(kernel_block_conv_128ch,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_scalar);
    kernel_block_conv_128ch<<<1, 256, smem_scalar>>>(d_weights, d_input, d_out_scalar);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_scalar, d_out_scalar, N * 4, cudaMemcpyDeviceToHost);

    // Convert weights to shifted layout (9 × [128, 128])
    // Build on host, copy to device
    half* h_slices[9];
    half* d_slices[9];
    int slice_n = NN_HIDDEN_DIM * NN_HIDDEN_DIM;
    for (int s = 0; s < 9; s++) {
        h_slices[s] = (half*)malloc(slice_n * sizeof(half));
        for (int oc = 0; oc < NN_HIDDEN_DIM; oc++)
            for (int c = 0; c < NN_HIDDEN_DIM; c++)
                h_slices[s][oc * NN_HIDDEN_DIM + c] =
                    __float2half(h_weights[oc * NN_HIDDEN_DIM * 9 + c * 9 + s]);
        cudaMalloc(&d_slices[s], slice_n * sizeof(half));
        cudaMemcpy(d_slices[s], h_slices[s], slice_n * sizeof(half), cudaMemcpyHostToDevice);
        free(h_slices[s]);
    }

    // Copy pointer array to device
    half** d_W_s;
    cudaMalloc(&d_W_s, 9 * sizeof(half*));
    cudaMemcpy(d_W_s, d_slices, 9 * sizeof(half*), cudaMemcpyHostToDevice);

    // Shifted conv: smem = input[8192] + output[8192] + shifted[4096] = 20480 floats
    size_t smem_shifted = (2 * N + NN_HIDDEN_DIM * 64 / 2) * 4;
    cudaFuncSetAttribute(kernel_shifted_conv_128ch,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_shifted);
    kernel_shifted_conv_128ch<<<1, 256, smem_shifted>>>(d_input, d_W_s, d_out_shifted);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA error: %s] ", cudaGetErrorString(err));
        test_failed = true;
    } else {
        cudaMemcpy(h_out_shifted, d_out_shifted, N * 4, cudaMemcpyDeviceToHost);

        int mismatches = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = fabsf(h_out_scalar[i] - h_out_shifted[i]);
            if (diff > max_diff) max_diff = diff;
            float tol = fmaxf(0.5f, 0.02f * fabsf(h_out_scalar[i]));
            if (diff > tol) mismatches++;
        }
        printf("[mismatches=%d/%d, max_diff=%.3f] ", mismatches, N, max_diff);
        ASSERT_EQ(mismatches, 0);
    }

    free(h_input); free(h_weights); free(h_out_scalar); free(h_out_shifted);
    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_out_scalar); cudaFree(d_out_shifted);
    for (int s = 0; s < 9; s++) cudaFree(d_slices[s]);
    cudaFree(d_W_s);
}

// Shifted forward pass test with zero weights
__global__ void kernel_block_forward_shifted(
    const BoardState* bs, float q_result, const OracleNetWeights* w,
    const ConvWeightsShifted* shifted_w,
    float* policy_out, float* value_out, float* k_out
) {
    extern __shared__ float smem[];
    oracle_net_forward_block(bs, q_result, w, smem, policy_out, value_out, k_out,
                             nullptr, shifted_w);
}

// Generic shifted conv kernel for arbitrary C_in/C_out
__global__ void kernel_shifted_conv_generic(
    const float* input_global, half* const* W_s, float* output_global,
    int C_in, int C_out
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int N_in = C_in * 64;
    int N_out = C_out * 64;
    float* input_smem  = smem;
    float* output_smem = smem + N_in;
    half*  shifted     = (half*)(smem + N_in + N_out);

    for (int i = tid; i < N_in; i += blockDim.x) input_smem[i] = input_global[i];
    __syncthreads();

    block_conv_3x3_shifted(input_smem, W_s, output_smem, shifted, C_in, C_out);

    for (int i = tid; i < N_out; i += blockDim.x) output_global[i] = output_smem[i];
}

// Shifted conv for start conv dimensions (C_in=17, C_out=128) — exercises a_staging boundary path
void test_shifted_conv_start(bool& test_failed) {
    int C_in = NN_INPUT_CHANNELS;  // 17
    int C_out = NN_HIDDEN_DIM;     // 128
    int N_in = C_in * 64;
    int N_out = C_out * 64;
    int W_total = C_out * C_in * 9;

    float* h_input   = (float*)malloc(N_in * 4);
    float* h_weights = (float*)malloc(W_total * 4);
    float* h_out_scalar = (float*)malloc(N_out * 4);
    float* h_out_shifted = (float*)malloc(N_out * 4);

    for (int i = 0; i < N_in;    i++) h_input[i]   = ((i * 7 + 3) % 17) / 17.0f - 0.5f;
    for (int i = 0; i < W_total; i++) h_weights[i]  = ((i * 11 + 5) % 13) / 13.0f - 0.5f;

    float *d_input, *d_weights, *d_out_scalar, *d_out_shifted;
    cudaMalloc(&d_input,       N_in * 4);
    cudaMalloc(&d_weights,     W_total * 4);
    cudaMalloc(&d_out_scalar,  N_out * 4);
    cudaMalloc(&d_out_shifted, N_out * 4);
    cudaMemcpy(d_input,   h_input,   N_in * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, W_total * 4, cudaMemcpyHostToDevice);

    // Scalar GPU reference
    size_t smem_scalar = (N_in + N_out) * 4;
    cudaFuncSetAttribute(kernel_block_conv_generic,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_scalar);
    kernel_block_conv_generic<<<1, 256, smem_scalar>>>(
        d_weights, d_input, d_out_scalar, C_in, C_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_scalar, d_out_scalar, N_out * 4, cudaMemcpyDeviceToHost);

    // Build shifted weight slices: 9 × [128, 17]
    int slice_n = C_out * C_in;
    half* d_slices[9];
    for (int s = 0; s < 9; s++) {
        half* h_slice = (half*)malloc(slice_n * sizeof(half));
        for (int oc = 0; oc < C_out; oc++)
            for (int c = 0; c < C_in; c++)
                h_slice[oc * C_in + c] =
                    __float2half(h_weights[oc * C_in * 9 + c * 9 + s]);
        cudaMalloc(&d_slices[s], slice_n * sizeof(half));
        cudaMemcpy(d_slices[s], h_slice, slice_n * sizeof(half), cudaMemcpyHostToDevice);
        free(h_slice);
    }
    half** d_W_s;
    cudaMalloc(&d_W_s, 9 * sizeof(half*));
    cudaMemcpy(d_W_s, d_slices, 9 * sizeof(half*), cudaMemcpyHostToDevice);

    // Shifted conv: smem = input[17*64] + output[128*64] + shifted[4096 floats]
    size_t smem_shifted = (N_in + N_out + BLOCK_SHIFTED_SIZE) * 4;
    cudaFuncSetAttribute(kernel_shifted_conv_generic,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_shifted);
    kernel_shifted_conv_generic<<<1, 256, smem_shifted>>>(
        d_input, d_W_s, d_out_shifted, C_in, C_out);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA error: %s] ", cudaGetErrorString(err));
        test_failed = true;
    } else {
        cudaMemcpy(h_out_shifted, d_out_shifted, N_out * 4, cudaMemcpyDeviceToHost);

        int mismatches = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < N_out; i++) {
            float diff = fabsf(h_out_scalar[i] - h_out_shifted[i]);
            if (diff > max_diff) max_diff = diff;
            float tol = fmaxf(0.5f, 0.02f * fabsf(h_out_scalar[i]));
            if (diff > tol) mismatches++;
        }
        printf("[mismatches=%d/%d, max_diff=%.3f] ", mismatches, N_out, max_diff);
        ASSERT_EQ(mismatches, 0);
    }

    free(h_input); free(h_weights); free(h_out_scalar); free(h_out_shifted);
    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_out_scalar); cudaFree(d_out_shifted);
    for (int s = 0; s < 9; s++) cudaFree(d_slices[s]);
    cudaFree(d_W_s);
}

void test_shifted_forward_zero_weights(bool& test_failed) {
    OracleNetWeights* d_weights = init_nn_weights_zeros();
    ConvWeightsShifted* d_shifted = convert_weights_shifted(d_weights);

    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float *d_policy_s, *d_value_s, *d_k_s;
    float *d_policy_sh, *d_value_sh, *d_k_sh;
    cudaMalloc(&d_policy_s,  NN_POLICY_SIZE * 4); cudaMalloc(&d_value_s, 4); cudaMalloc(&d_k_s, 4);
    cudaMalloc(&d_policy_sh, NN_POLICY_SIZE * 4); cudaMalloc(&d_value_sh, 4); cudaMalloc(&d_k_sh, 4);

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    // Scalar forward pass
    cudaFuncSetAttribute(kernel_block_forward,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    kernel_block_forward<<<1, 256, BLOCK_SMEM_BYTES>>>(
        d_bs, 0.0f, d_weights, d_policy_s, d_value_s, d_k_s);
    cudaDeviceSynchronize();

    // Shifted forward pass
    cudaFuncSetAttribute(kernel_block_forward_shifted,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BLOCK_SMEM_BYTES);
    kernel_block_forward_shifted<<<1, 256, BLOCK_SMEM_BYTES>>>(
        d_bs, 0.0f, d_weights, d_shifted, d_policy_sh, d_value_sh, d_k_sh);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA error: %s] ", cudaGetErrorString(err));
        test_failed = true;
    } else {
        float h_val_s, h_val_sh, h_k_s, h_k_sh;
        cudaMemcpy(&h_val_s,  d_value_s,  4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_val_sh, d_value_sh, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k_s,    d_k_s,      4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k_sh,   d_k_sh,     4, cudaMemcpyDeviceToHost);

        printf("[val: scalar=%.4f shifted=%.4f, k: %.4f vs %.4f] ",
               h_val_s, h_val_sh, h_k_s, h_k_sh);

        ASSERT_NEAR(h_val_sh, h_val_s, 0.1f);
        ASSERT_NEAR(h_k_sh,   h_k_s,   0.01f);

        float* h_pol_s  = (float*)malloc(NN_POLICY_SIZE * 4);
        float* h_pol_sh = (float*)malloc(NN_POLICY_SIZE * 4);
        cudaMemcpy(h_pol_s,  d_policy_s,  NN_POLICY_SIZE * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pol_sh, d_policy_sh, NN_POLICY_SIZE * 4, cudaMemcpyDeviceToHost);

        int pol_mismatch = 0;
        for (int i = 0; i < NN_POLICY_SIZE; i++)
            if (fabsf(h_pol_s[i] - h_pol_sh[i]) > 0.1f) pol_mismatch++;
        printf("[pol_mismatches=%d] ", pol_mismatch);
        ASSERT_EQ(pol_mismatch, 0);

        free(h_pol_s); free(h_pol_sh);
    }

    cudaFree(d_bs);
    cudaFree(d_policy_s); cudaFree(d_value_s); cudaFree(d_k_s);
    cudaFree(d_policy_sh); cudaFree(d_value_sh); cudaFree(d_k_sh);
    free_shifted_weights(d_shifted);
    free_nn_weights(d_weights);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== Block Ops Tests ===\n");
    init_movegen_tables();
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    int total = 0, passes = 0, failures = 0;

    RUN_TEST(test_block_smem_size);
    RUN_TEST(test_block_conv_known_values);
    RUN_TEST(test_block_conv_vs_warp);
    RUN_TEST(test_block_bn_relu_matches_warp);
    RUN_TEST(test_block_log_softmax);
    RUN_TEST(test_block_forward_zero_weights);
    RUN_TEST(test_block_forward_q_sensitivity);
    RUN_TEST(test_block_mcts_produces_moves);
    RUN_TEST(test_block_mcts_mate_detection);
    RUN_TEST(test_tc_conv_vs_scalar);
    RUN_TEST(test_tc_conv_start);
    RUN_TEST(test_tc_forward_zero_weights);
    RUN_TEST(test_shifted_conv_vs_scalar);
    RUN_TEST(test_shifted_conv_start);
    RUN_TEST(test_shifted_forward_zero_weights);

    printf("\nResults: %d/%d passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");
    return failures > 0 ? 1 : 0;
}
