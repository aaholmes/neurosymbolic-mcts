#include "../nn_weights.cuh"
#include "../nn_ops.cuh"
#include "test_helpers.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>

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
// Main
// ============================================================

int main() {
    printf("=== NN Ops Tests ===\n");

    int total = 0, passes = 0, failures = 0;

    RUN_TEST(test_weights_struct_size);
    RUN_TEST(test_dummy_weights_init);
    RUN_TEST(test_gemm_identity);
    RUN_TEST(test_gemm_small);
    RUN_TEST(test_gemm_conv_size);
    RUN_TEST(test_im2col_single_channel);
    RUN_TEST(test_board_encoding_starting_pos);
    RUN_TEST(test_log_softmax);

    printf("\n%d/%d tests passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");

    return failures > 0 ? 1 : 0;
}
