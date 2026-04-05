#include "../nn_weights.cuh"
#include "../nn_ops.cuh"
#include "../nn_forward.cuh"
#include "../quiescence.cuh"
#include "../movegen.cuh"
#include "test_helpers.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>

// FEN parser for test positions
static int char_to_piece_nn(char c, int* color) {
    *color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
    switch (c) {
        case 'P': case 'p': return PAWN; case 'N': case 'n': return KNIGHT;
        case 'B': case 'b': return BISHOP; case 'R': case 'r': return ROOK;
        case 'Q': case 'q': return QUEEN; case 'K': case 'k': return KING;
        default: return -1;
    }
}
static BoardState parse_fen(const char* fen) {
    BoardState bs; memset(&bs, 0, sizeof(bs)); bs.en_passant = EN_PASSANT_NONE;
    int rank = 7, file = 0; const char* p = fen;
    while (*p && *p != ' ') {
        if (*p == '/') { rank--; file = 0; }
        else if (*p >= '1' && *p <= '8') { file += (*p - '0'); }
        else { int color, piece; piece = char_to_piece_nn(*p, &color);
            if (piece >= 0) { bs.pieces[color * 6 + piece] |= (1ULL << (rank * 8 + file)); file++; } }
        p++;
    }
    if (*p) p++; bs.w_to_move = (*p == 'w') ? 1 : 0;
    if (*p) p++; if (*p) p++;
    while (*p && *p != ' ') { switch (*p) { case 'K': bs.castling |= CASTLE_WK; break; case 'Q': bs.castling |= CASTLE_WQ; break; case 'k': bs.castling |= CASTLE_BK; break; case 'q': bs.castling |= CASTLE_BQ; break; } p++; }
    for (int c = 0; c < 2; c++) { bs.pieces_occ[c] = 0; for (int piece = 0; piece < 6; piece++) bs.pieces_occ[c] |= bs.pieces[c * 6 + piece]; }
    return bs;
}

// ============================================================
// Weight struct tests
// ============================================================

void test_weights_struct_size(bool& test_failed) {
    size_t sz = nn_weights_size();
    printf("[%zu bytes = %.2f MB] ", sz, sz / (1024.0 * 1024.0));
    // Should be ~7.9 MB
    ASSERT_TRUE(sz > 7 * 1024 * 1024);
    ASSERT_TRUE(sz < 9 * 1024 * 1024);
}

void test_dummy_weights_init(bool& test_failed) {
    OracleNetWeights* d_weights = init_nn_weights_zeros();
    ASSERT_TRUE(d_weights != nullptr);

    // Read back and verify some values
    OracleNetWeights* h_weights = (OracleNetWeights*)malloc(sizeof(OracleNetWeights));
    cudaMemcpy(h_weights, d_weights, sizeof(OracleNetWeights), cudaMemcpyDeviceToHost);

    // All conv weights should be zero
    ASSERT_NEAR(h_weights->start_conv_weight[0], 0.0f, 1e-6f);
    ASSERT_NEAR(h_weights->blocks[0].conv1_weight[0], 0.0f, 1e-6f);

    // BN running_var should be 1.0 (not zero)
    ASSERT_NEAR(h_weights->start_bn.running_var[0], 1.0f, 1e-6f);
    ASSERT_NEAR(h_weights->blocks[0].bn1.running_var[0], 1.0f, 1e-6f);
    ASSERT_NEAR(h_weights->blocks[5].bn2.running_var[127], 1.0f, 1e-6f);
    ASSERT_NEAR(h_weights->v_bn.running_var[0], 1.0f, 1e-6f);

    // k_logit should be 0.0
    ASSERT_NEAR(h_weights->k_logit, 0.0f, 1e-6f);

    free(h_weights);
    free_nn_weights(d_weights);
}

// ============================================================
// GEMM tests
// ============================================================

// Test kernel: runs warp_gemm on device
__global__ void kernel_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    warp_gemm(A, B, C, M, N, K);
}

void test_gemm_identity(bool& test_failed) {
    // 4x4 identity × 4x4 input → should equal input
    const int N = 4;
    float h_A[16], h_B[16], h_C[16];
    memset(h_A, 0, sizeof(h_A));
    for (int i = 0; i < N; i++) h_A[i * N + i] = 1.0f; // identity
    for (int i = 0; i < 16; i++) h_B[i] = (float)(i + 1); // 1..16

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 64); cudaMalloc(&d_B, 64); cudaMalloc(&d_C, 64);
    cudaMemcpy(d_A, h_A, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 64, cudaMemcpyHostToDevice);

    kernel_gemm<<<1, 32>>>(d_A, d_B, d_C, N, N, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, 64, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) {
        ASSERT_NEAR(h_C[i], h_B[i], 1e-4f);
    }
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void test_gemm_small(bool& test_failed) {
    // [2,3] × [3,2] = [2,2]
    // A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]
    // C = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //   = [[58, 64], [139, 154]]
    float h_A[] = {1,2,3, 4,5,6};
    float h_B[] = {7,8, 9,10, 11,12};
    float h_C[4];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 24); cudaMalloc(&d_B, 24); cudaMalloc(&d_C, 16);
    cudaMemcpy(d_A, h_A, 24, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 24, cudaMemcpyHostToDevice);

    kernel_gemm<<<1, 32>>>(d_A, d_B, d_C, 2, 2, 3);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, 16, cudaMemcpyDeviceToHost);

    ASSERT_NEAR(h_C[0], 58.0f, 1e-4f);
    ASSERT_NEAR(h_C[1], 64.0f, 1e-4f);
    ASSERT_NEAR(h_C[2], 139.0f, 1e-4f);
    ASSERT_NEAR(h_C[3], 154.0f, 1e-4f);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void test_gemm_conv_size(bool& test_failed) {
    // Test with actual conv dimensions: [128, 1152] × [1152, 64] → [128, 64]
    int M = 128, N = 64, K = 1152;
    int szA = M * K, szB = K * N, szC = M * N;

    // Use all-ones matrices: each output element should equal K
    float* h_A = (float*)malloc(szA * 4);
    float* h_B = (float*)malloc(szB * 4);
    float* h_C = (float*)malloc(szC * 4);
    for (int i = 0; i < szA; i++) h_A[i] = 1.0f;
    for (int i = 0; i < szB; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, szA * 4); cudaMalloc(&d_B, szB * 4); cudaMalloc(&d_C, szC * 4);
    cudaMemcpy(d_A, h_A, szA * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, szB * 4, cudaMemcpyHostToDevice);

    kernel_gemm<<<1, 32>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, szC * 4, cudaMemcpyDeviceToHost);

    // Spot check: each element should be K=1152
    ASSERT_NEAR(h_C[0], (float)K, 1.0f);
    ASSERT_NEAR(h_C[M * N - 1], (float)K, 1.0f);
    ASSERT_NEAR(h_C[M * N / 2], (float)K, 1.0f);
    printf("[128x1152 × 1152x64 OK] ");

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ============================================================
// im2col test
// ============================================================

__global__ void kernel_im2col(const float* input, float* col, int C_in) {
    warp_im2col_3x3(input, col, C_in);
}

void test_im2col_single_channel(bool& test_failed) {
    // 1 channel, 8x8 input filled with sequential values 1..64
    float h_input[64];
    for (int i = 0; i < 64; i++) h_input[i] = (float)(i + 1);

    int col_size = 1 * 9 * 64; // C_in * 9 * 64
    float* h_col = (float*)calloc(col_size, sizeof(float));

    float *d_input, *d_col;
    cudaMalloc(&d_input, 64 * 4);
    cudaMalloc(&d_col, col_size * 4);
    cudaMemcpy(d_input, h_input, 64 * 4, cudaMemcpyHostToDevice);

    kernel_im2col<<<1, 32>>>(d_input, d_col, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_col, d_col, col_size * 4, cudaMemcpyDeviceToHost);

    // Check center element (3,3): kernel position (1,1)=center should equal input[3,3]=28
    // col layout: [kernel_idx * 64 + spatial_idx]
    // center kernel = position 4 (row 1, col 1 in 3x3)
    // spatial (3,3) = index 3*8+3 = 27
    ASSERT_NEAR(h_col[4 * 64 + 27], 28.0f, 1e-6f); // input[3*8+3]=28

    // Check corner (0,0): top-left kernel (0,0) = position (-1,-1) → zero padding
    // kernel position 0 (ky=-1, kx=-1), spatial (0,0) = index 0
    ASSERT_NEAR(h_col[0 * 64 + 0], 0.0f, 1e-6f); // padded

    // Check (0,0) center kernel: should be input[0,0]=1
    ASSERT_NEAR(h_col[4 * 64 + 0], 1.0f, 1e-6f);

    printf("[1ch im2col OK] ");

    free(h_col);
    cudaFree(d_input); cudaFree(d_col);
}

// ============================================================
// Board encoding test
// ============================================================

__global__ void kernel_board_to_planes(const BoardState* bs, float* planes) {
    warp_board_to_planes(bs, planes);
}

void test_board_encoding_starting_pos(bool& test_failed) {
    BoardState bs = make_starting_position();

    BoardState* d_bs;
    float* d_planes;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMalloc(&d_planes, 17 * 64 * sizeof(float));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    kernel_board_to_planes<<<1, 32>>>(d_bs, d_planes);
    cudaDeviceSynchronize();

    float h_planes[17 * 64];
    cudaMemcpy(h_planes, d_planes, 17 * 64 * sizeof(float), cudaMemcpyDeviceToHost);

    // White to move: STM=White, no flip
    // Plane 0 = STM pawns (White) = rank 2
    // e2 = sq 12, mapped = 12 (no flip)
    ASSERT_NEAR(h_planes[0 * 64 + 12], 1.0f, 1e-6f); // White pawn on e2

    // Plane 6 = opponent pawns (Black) = rank 7
    // e7 = sq 52, mapped = 52 (no flip)
    ASSERT_NEAR(h_planes[6 * 64 + 52], 1.0f, 1e-6f); // Black pawn on e7

    // Plane 5 = STM king (White) = e1 = sq 4
    ASSERT_NEAR(h_planes[5 * 64 + 4], 1.0f, 1e-6f);

    // Plane 13 = STM kingside castling (should be all 1s)
    ASSERT_NEAR(h_planes[13 * 64 + 0], 1.0f, 1e-6f);
    ASSERT_NEAR(h_planes[13 * 64 + 63], 1.0f, 1e-6f);

    printf("[start pos encoding OK] ");

    cudaFree(d_bs); cudaFree(d_planes);
}

// ============================================================
// Log-softmax test
// ============================================================

__global__ void kernel_log_softmax(const float* input, float* output, int size) {
    warp_log_softmax(input, output, size);
}

void test_log_softmax(bool& test_failed) {
    // Small test: [1, 2, 3]
    // softmax = [e^1, e^2, e^3] / sum = [0.0900, 0.2447, 0.6652]
    // log_softmax = [ln(0.0900), ln(0.2447), ln(0.6652)] = [-2.4076, -1.4076, -0.4076]
    float h_input[] = {1.0f, 2.0f, 3.0f};
    float h_output[3];

    float *d_in, *d_out;
    cudaMalloc(&d_in, 12); cudaMalloc(&d_out, 12);
    cudaMemcpy(d_in, h_input, 12, cudaMemcpyHostToDevice);

    kernel_log_softmax<<<1, 32>>>(d_in, d_out, 3);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_out, 12, cudaMemcpyDeviceToHost);

    // log_softmax(x) = x - max - log(sum(exp(x - max)))
    // max = 3, sum(exp) = exp(-2)+exp(-1)+exp(0) = 0.1353+0.3679+1.0 = 1.5032
    // log(sum) = 0.4076
    ASSERT_NEAR(h_output[0], 1.0f - 3.0f - logf(1.5032f), 0.01f);
    ASSERT_NEAR(h_output[2], 3.0f - 3.0f - logf(1.5032f), 0.01f);

    // Verify exp(log_softmax) sums to 1
    float sum = expf(h_output[0]) + expf(h_output[1]) + expf(h_output[2]);
    ASSERT_NEAR(sum, 1.0f, 0.01f);

    cudaFree(d_in); cudaFree(d_out);
}

// ============================================================
// Full forward pass test
// ============================================================

__global__ void kernel_forward_pass(
    const BoardState* bs, float q_result,
    const OracleNetWeights* weights, float* scratch,
    float* policy_out, float* value_out, float* k_out
) {
    oracle_net_forward(bs, q_result, weights, scratch, policy_out, value_out, k_out);
}

void test_full_forward_dummy_weights(bool& test_failed) {
    // With zero weights: all convolutions output 0, BN normalizes to 0 (gamma=0),
    // policy logits are all 0 → log_softmax gives uniform: log(1/4672)
    // v_logit = 0, k = 0.47 * ln(2) ≈ 0.326
    // V = tanh(0 + 0.326 * q_result)

    OracleNetWeights* d_weights = init_nn_weights_zeros();
    ASSERT_TRUE(d_weights != nullptr);

    float* d_scratch = alloc_nn_scratch(1);
    ASSERT_TRUE(d_scratch != nullptr);

    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float* d_policy;
    float* d_value;
    float* d_k;
    cudaMalloc(&d_policy, NN_POLICY_SIZE * sizeof(float));
    cudaMalloc(&d_value, sizeof(float));
    cudaMalloc(&d_k, sizeof(float));

    // Set stack size for forward pass (deep call stack)
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    float q_result = 1.5f; // +1.5 pawns
    kernel_forward_pass<<<1, 32>>>(d_bs, q_result, d_weights, d_scratch,
                                    d_policy, d_value, d_k);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s ", cudaGetErrorString(err));
        test_failed = true;
        cudaFree(d_bs); cudaFree(d_policy); cudaFree(d_value); cudaFree(d_k);
        free_nn_scratch(d_scratch); free_nn_weights(d_weights);
        return;
    }

    // Read back results
    float h_policy[NN_POLICY_SIZE];
    float h_value, h_k;
    cudaMemcpy(h_policy, d_policy, NN_POLICY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_value, d_value, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_k, d_k, sizeof(float), cudaMemcpyDeviceToHost);

    // k should be 0.47 * ln(2) ≈ 0.326
    float expected_k = 0.47f * logf(2.0f);
    ASSERT_NEAR(h_k, expected_k, 0.01f);

    // Value should be tanh(0 + 0.326 * 1.5) = tanh(0.489) ≈ 0.453
    float expected_v = tanhf(expected_k * q_result);
    ASSERT_NEAR(h_value, expected_v, 0.05f);

    // Policy should be approximately uniform: log(1/4672) ≈ -8.449
    float expected_log_prob = logf(1.0f / NN_POLICY_SIZE);
    ASSERT_NEAR(h_policy[0], expected_log_prob, 0.1f);
    ASSERT_NEAR(h_policy[NN_POLICY_SIZE - 1], expected_log_prob, 0.1f);

    // Verify softmax sums to 1
    float prob_sum = 0.0f;
    for (int i = 0; i < NN_POLICY_SIZE; i++) {
        prob_sum += expf(h_policy[i]);
    }
    ASSERT_NEAR(prob_sum, 1.0f, 0.01f);

    printf("[k=%.3f, v=%.3f, policy_uniform=%.2f] ",
           h_k, h_value, h_policy[0]);

    cudaFree(d_bs); cudaFree(d_policy); cudaFree(d_value); cudaFree(d_k);
    free_nn_scratch(d_scratch);
    free_nn_weights(d_weights);
}

// ============================================================
// BatchNorm + ReLU test
// ============================================================

__global__ void kernel_bn_relu(float* data, const float* g, const float* b,
                                const float* m, const float* v, int ch, bool relu) {
    warp_bn_relu(data, g, b, m, v, ch, relu);
}

void test_bn_relu_identity(bool& test_failed) {
    // BN with gamma=1, beta=0, mean=0, var=1 → output = input (when positive, with ReLU)
    const int ch = 2;
    const int total = ch * 64;
    float h_data[total];
    for (int i = 0; i < total; i++) h_data[i] = (float)(i % 7) - 3.0f; // -3 to 3

    float h_gamma[] = {1.0f, 1.0f};
    float h_beta[] = {0.0f, 0.0f};
    float h_mean[] = {0.0f, 0.0f};
    float h_var[] = {1.0f, 1.0f};

    float *d_data, *d_g, *d_b, *d_m, *d_v;
    cudaMalloc(&d_data, total * 4); cudaMalloc(&d_g, 8); cudaMalloc(&d_b, 8);
    cudaMalloc(&d_m, 8); cudaMalloc(&d_v, 8);
    cudaMemcpy(d_data, h_data, total * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_gamma, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_beta, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_mean, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_var, 8, cudaMemcpyHostToDevice);

    kernel_bn_relu<<<1, 32>>>(d_data, d_g, d_b, d_m, d_v, ch, true);
    cudaDeviceSynchronize();

    float h_out[total];
    cudaMemcpy(h_out, d_data, total * 4, cudaMemcpyDeviceToHost);

    // With identity BN + ReLU: output = max(0, input)
    for (int i = 0; i < 8; i++) {
        float expected = h_data[i] > 0 ? h_data[i] : 0.0f;
        ASSERT_NEAR(h_out[i], expected, 0.01f);
    }

    cudaFree(d_data); cudaFree(d_g); cudaFree(d_b); cudaFree(d_m); cudaFree(d_v);
}

// ============================================================
// Add+ReLU test
// ============================================================

__global__ void kernel_add_relu(const float* a, const float* b, float* out, int size) {
    warp_add_relu(a, b, out, size);
}

void test_add_relu(bool& test_failed) {
    float h_a[] = {1.0f, -2.0f, 3.0f, -4.0f};
    float h_b[] = {-0.5f, 1.0f, -5.0f, 3.0f};
    // expected: max(0, a+b) = [0.5, 0, 0, 0]
    float h_out[4];

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, 16); cudaMalloc(&d_b, 16); cudaMalloc(&d_out, 16);
    cudaMemcpy(d_a, h_a, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 16, cudaMemcpyHostToDevice);

    kernel_add_relu<<<1, 32>>>(d_a, d_b, d_out, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 16, cudaMemcpyDeviceToHost);

    ASSERT_NEAR(h_out[0], 0.5f, 1e-6f);
    ASSERT_NEAR(h_out[1], 0.0f, 1e-6f); // -2+1=-1 → relu → 0
    ASSERT_NEAR(h_out[2], 0.0f, 1e-6f); // 3-5=-2 → relu → 0
    ASSERT_NEAR(h_out[3], 0.0f, 1e-6f); // -4+3=-1 → relu → 0

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

// ============================================================
// GPU principal exchange test
// ============================================================

__global__ void kernel_pe(const BoardState* bs, float* result) {
    *result = gpu_principal_exchange(bs);
}

void test_gpu_pe_starting_pos(bool& test_failed) {
    BoardState bs = make_starting_position();
    BoardState* d_bs; float* d_result; float h_result;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMalloc(&d_result, 4);
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    kernel_pe<<<1, 1>>>(d_bs, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);

    ASSERT_NEAR(h_result, 0.0f, 0.1f); // starting pos ≈ 0 pawns
    cudaFree(d_bs); cudaFree(d_result);
}

void test_gpu_pe_queen_advantage(bool& test_failed) {
    // White has an extra queen → PE should show large advantage
    BoardState bs = parse_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    BoardState* d_bs; float* d_result; float h_result;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMalloc(&d_result, 4);
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    kernel_pe<<<1, 1>>>(d_bs, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);

    printf("[pe=%.2f] ", h_result);
    ASSERT_TRUE(h_result > 5.0f); // queen ≈ 9 pawns
    cudaFree(d_bs); cudaFree(d_result);
}

void test_gpu_pe_hanging_piece(bool& test_failed) {
    // White queen can capture undefended rook
    BoardState bs = parse_fen("4k3/8/8/3r4/8/8/8/3QK3 w - - 0 1");
    BoardState* d_bs; float* d_result; float h_result;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMalloc(&d_result, 4);
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    kernel_pe<<<1, 1>>>(d_bs, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);

    printf("[pe=%.2f] ", h_result);
    ASSERT_TRUE(h_result > 3.0f); // should capture rook
    cudaFree(d_bs); cudaFree(d_result);
}

// ============================================================
// Forward pass with different q_result values
// ============================================================

void test_forward_pass_q_result_sensitivity(bool& test_failed) {
    // With dummy weights, value should scale with q_result:
    // V = tanh(0.326 * q_result)
    OracleNetWeights* d_weights = init_nn_weights_zeros();
    float* d_scratch = alloc_nn_scratch(1);
    BoardState bs = make_starting_position();
    BoardState* d_bs;
    cudaMalloc(&d_bs, sizeof(BoardState));
    cudaMemcpy(d_bs, &bs, sizeof(BoardState), cudaMemcpyHostToDevice);

    float* d_policy; float* d_value; float* d_k;
    cudaMalloc(&d_policy, NN_POLICY_SIZE * 4);
    cudaMalloc(&d_value, 4);
    cudaMalloc(&d_k, 4);
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    float q_values[] = {-3.0f, 0.0f, 3.0f};
    float h_values[3];

    for (int i = 0; i < 3; i++) {
        kernel_forward_pass<<<1, 32>>>(d_bs, q_values[i], d_weights, d_scratch,
                                        d_policy, d_value, d_k);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_values[i], d_value, 4, cudaMemcpyDeviceToHost);
    }

    // V(-3) < V(0) < V(3)
    printf("[V(-3)=%.2f, V(0)=%.2f, V(3)=%.2f] ", h_values[0], h_values[1], h_values[2]);
    ASSERT_TRUE(h_values[0] < h_values[1]);
    ASSERT_TRUE(h_values[1] < h_values[2]);
    ASSERT_NEAR(h_values[1], 0.0f, 0.05f); // q=0 → V≈0
    ASSERT_TRUE(h_values[2] > 0.5f);  // q=3 → V > 0.5
    ASSERT_TRUE(h_values[0] < -0.5f); // q=-3 → V < -0.5

    cudaFree(d_bs); cudaFree(d_policy); cudaFree(d_value); cudaFree(d_k);
    free_nn_scratch(d_scratch); free_nn_weights(d_weights);
}

// ============================================================
// Scratch allocation test
// ============================================================

void test_scratch_allocation(bool& test_failed) {
    float* scratch = alloc_nn_scratch(16);
    ASSERT_TRUE(scratch != nullptr);
    printf("[16 warps = %.1f MB] ", 16.0f * SCRATCH_TOTAL_BYTES / (1024.0f * 1024.0f));
    free_nn_scratch(scratch);

    // Test large allocation
    float* scratch_big = alloc_nn_scratch(144);
    ASSERT_TRUE(scratch_big != nullptr);
    printf("[144 warps = %.1f MB] ", 144.0f * SCRATCH_TOTAL_BYTES / (1024.0f * 1024.0f));
    free_nn_scratch(scratch_big);
}

// ============================================================
// Move encoding tests (AlphaZero 73-plane encoding)
// ============================================================

__global__ void kernel_move_to_index(GPUMove mv, bool w_to_move, int* result) {
    *result = move_to_policy_index(mv, w_to_move);
}

void test_move_encoding_queen_slide(bool& test_failed) {
    int* d_result; int h_result;
    cudaMalloc(&d_result, 4);

    // e4(28)→e5(36): N, dist 1. Plane = 0*7+0 = 0. Index = 28*73+0 = 2044
    kernel_move_to_index<<<1,1>>>(MAKE_GPU_MOVE(28, 36, 0), true, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);
    ASSERT_EQ(h_result, 28 * 73 + 0);

    // e4(28)→h4(31): E, dist 3. Plane = 2*7+2 = 16. Index = 28*73+16
    kernel_move_to_index<<<1,1>>>(MAKE_GPU_MOVE(28, 31, 0), true, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);
    ASSERT_EQ(h_result, 28 * 73 + 16);

    cudaFree(d_result);
}

void test_move_encoding_knight(bool& test_failed) {
    int* d_result; int h_result;
    cudaMalloc(&d_result, 4);

    // e4(28)→f6(45): dx=1,dy=2 → knight idx 0. Plane = 56. Index = 28*73+56
    kernel_move_to_index<<<1,1>>>(MAKE_GPU_MOVE(28, 45, 0), true, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);
    ASSERT_EQ(h_result, 28 * 73 + 56);

    cudaFree(d_result);
}

void test_move_encoding_underpromotion(bool& test_failed) {
    int* d_result; int h_result;
    cudaMalloc(&d_result, 4);

    // a7(48)→a8(56) promote to Rook(3). dx=0, Plane = 64+0+6 = 70. Index = 48*73+70
    kernel_move_to_index<<<1,1>>>(MAKE_GPU_MOVE(48, 56, ROOK), true, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);
    ASSERT_EQ(h_result, 48 * 73 + 70);

    cudaFree(d_result);
}

void test_move_encoding_black_flip(bool& test_failed) {
    int* d_result; int h_result;
    cudaMalloc(&d_result, 4);

    // Black plays e7(52)→e5(36): flip → e2(12)→e4(28). N, dist 2. Plane = 0*7+1 = 1
    // Index = 12*73+1 = 877
    kernel_move_to_index<<<1,1>>>(MAKE_GPU_MOVE(52, 36, 0), false, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);
    ASSERT_EQ(h_result, 12 * 73 + 1);

    cudaFree(d_result);
}

void test_move_encoding_castling(bool& test_failed) {
    int* d_result; int h_result;
    cudaMalloc(&d_result, 4);

    // White O-O: e1(4)→g1(6): E, dist 2. Plane = 2*7+1 = 15. Index = 4*73+15 = 307
    kernel_move_to_index<<<1,1>>>(MAKE_GPU_MOVE(4, 6, 0), true, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, 4, cudaMemcpyDeviceToHost);
    ASSERT_EQ(h_result, 4 * 73 + 15);

    cudaFree(d_result);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== NN Ops Tests ===\n");

    // Load movegen tables for PE tests
    init_movegen_tables();

    int total = 0, passes = 0, failures = 0;

    RUN_TEST(test_weights_struct_size);
    RUN_TEST(test_dummy_weights_init);
    RUN_TEST(test_gemm_identity);
    RUN_TEST(test_gemm_small);
    RUN_TEST(test_gemm_conv_size);
    RUN_TEST(test_im2col_single_channel);
    RUN_TEST(test_board_encoding_starting_pos);
    RUN_TEST(test_log_softmax);
    RUN_TEST(test_bn_relu_identity);
    RUN_TEST(test_add_relu);
    RUN_TEST(test_gpu_pe_starting_pos);
    RUN_TEST(test_gpu_pe_queen_advantage);
    RUN_TEST(test_gpu_pe_hanging_piece);
    RUN_TEST(test_full_forward_dummy_weights);
    RUN_TEST(test_forward_pass_q_result_sensitivity);
    RUN_TEST(test_scratch_allocation);
    RUN_TEST(test_move_encoding_queen_slide);
    RUN_TEST(test_move_encoding_knight);
    RUN_TEST(test_move_encoding_underpromotion);
    RUN_TEST(test_move_encoding_black_flip);
    RUN_TEST(test_move_encoding_castling);

    printf("\n%d/%d tests passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");

    return failures > 0 ? 1 : 0;
}
