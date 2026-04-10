// Tests for selfplay encoding functions and transformer element-wise ops.
//
// Selfplay (host-side):
//   - board_to_planes_host: board → [17, 64] planes
//   - move_to_policy_index_host: GPUMove → policy index in [0, 4672)
//   - compute_llr: SPRT log-likelihood ratio
//   - check_sprt: SPRT decision boundaries
//
// Transformer ops (device-side, tested via thin kernels):
//   - tf_softmax_rows: row-wise softmax
//   - tf_relu: elementwise ReLU
//   - tf_add_bias: broadcast bias addition

#include "../selfplay.cuh"
#include "../transformer_ops.cuh"
#include "../movegen.cuh"
#include "test_helpers.cuh"

#include <cstdio>
#include <cstring>
#include <cmath>

// ============================================================
// Thin GPU kernels wrapping __device__ ops for testing
// ============================================================

__global__ void kernel_test_softmax(float* data, int rows, int cols) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    // Copy global → smem
    for (int i = tid; i < rows * cols; i += blockDim.x)
        smem[i] = data[i];
    __syncthreads();

    float* reduce = smem + rows * cols;
    tf_softmax_rows(smem, rows, cols, reduce);

    // Copy smem → global
    for (int i = tid; i < rows * cols; i += blockDim.x)
        data[i] = smem[i];
}

__global__ void kernel_test_relu(float* data, int count) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += blockDim.x)
        smem[i] = data[i];
    __syncthreads();

    tf_relu(smem, count);

    for (int i = tid; i < count; i += blockDim.x)
        data[i] = smem[i];
}

__global__ void kernel_test_add_bias(float* data, const float* bias, int M, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    for (int i = tid; i < M * N; i += blockDim.x)
        smem[i] = data[i];
    __syncthreads();

    tf_add_bias(smem, bias, M, N);

    for (int i = tid; i < M * N; i += blockDim.x)
        data[i] = smem[i];
}

// ============================================================
// board_to_planes_host tests
// ============================================================

void test_planes_starting_white_pawns(bool& test_failed) {
    // Starting position, white to move: STM=white, plane 0 = STM pawns
    BoardState bs = make_starting_position();
    float planes[17 * 64];
    board_to_planes_host(bs, planes);

    // White pawns on rank 2 (squares 8-15). STM pawns = plane 0.
    // No flip because white to move.
    for (int sq = 0; sq < 64; sq++) {
        float expected = (sq >= 8 && sq <= 15) ? 1.0f : 0.0f;
        ASSERT_NEAR(planes[0 * 64 + sq], expected, 1e-6f);
    }
}

void test_planes_starting_black_pawns(bool& test_failed) {
    // Starting position, white to move: opponent = black, plane 6 = OPP pawns
    BoardState bs = make_starting_position();
    float planes[17 * 64];
    board_to_planes_host(bs, planes);

    // Black pawns on rank 7 (squares 48-55), no flip
    for (int sq = 0; sq < 64; sq++) {
        float expected = (sq >= 48 && sq <= 55) ? 1.0f : 0.0f;
        ASSERT_NEAR(planes[6 * 64 + sq], expected, 1e-6f);
    }
}

void test_planes_castling_all(bool& test_failed) {
    // All castling rights → planes 13-16 all ones
    BoardState bs = make_starting_position();
    float planes[17 * 64];
    board_to_planes_host(bs, planes);

    for (int p = 13; p <= 16; p++) {
        for (int sq = 0; sq < 64; sq++) {
            ASSERT_NEAR(planes[p * 64 + sq], 1.0f, 1e-6f);
        }
    }
}

void test_planes_no_ep(bool& test_failed) {
    // No en passant → plane 12 all zeros
    BoardState bs = make_starting_position();
    float planes[17 * 64];
    board_to_planes_host(bs, planes);

    for (int sq = 0; sq < 64; sq++) {
        ASSERT_NEAR(planes[12 * 64 + sq], 0.0f, 1e-6f);
    }
}

void test_planes_black_to_move_flip(bool& test_failed) {
    // When black to move, STM=black → plane 0 = black pawns (flipped)
    // Black pawns are on rank 7 (sq 48-55), after flip ^56 → rank 2 (sq 8-15)
    BoardState bs = make_starting_position();
    bs.w_to_move = 0;  // black to move
    float planes[17 * 64];
    board_to_planes_host(bs, planes);

    // STM pawns (black) should appear at rank 2 after flip
    for (int sq = 0; sq < 64; sq++) {
        float expected = (sq >= 8 && sq <= 15) ? 1.0f : 0.0f;
        ASSERT_NEAR(planes[0 * 64 + sq], expected, 1e-6f);
    }

    // OPP pawns (white, rank 2 → after flip → rank 7, sq 48-55)
    for (int sq = 0; sq < 64; sq++) {
        float expected = (sq >= 48 && sq <= 55) ? 1.0f : 0.0f;
        ASSERT_NEAR(planes[6 * 64 + sq], expected, 1e-6f);
    }
}

void test_planes_piece_count(bool& test_failed) {
    // Total set bits across piece planes 0-11 should equal 32 (all pieces)
    BoardState bs = make_starting_position();
    float planes[17 * 64];
    board_to_planes_host(bs, planes);

    int total = 0;
    for (int p = 0; p < 12; p++)
        for (int sq = 0; sq < 64; sq++)
            if (planes[p * 64 + sq] > 0.5f) total++;

    ASSERT_EQ(total, 32);
}

// ============================================================
// move_to_policy_index_host tests
// ============================================================

void test_policy_index_e2e4(bool& test_failed) {
    // e2=12, e4=28, no promo → pawn push
    GPUMove move = MAKE_GPU_MOVE(12, 28, 0);
    int idx = move_to_policy_index_host(move, 1);  // white to move
    ASSERT_TRUE(idx >= 0);
    ASSERT_TRUE(idx < 4672);
    // e2=sq12, direction N (dr=+2, dc=0) → dir=0, dist=2, plane = 0*7 + 1 = 1
    // index = 12 * 73 + 1 = 877
    ASSERT_EQ(idx, 12 * 73 + 1);
}

void test_policy_index_nf3(bool& test_failed) {
    // Ng1-f3: g1=6, f3=21, no promo
    // Knight move: dr = 21/8 - 6/8 = 2-0 = 2, dc = 21%8 - 6%8 = 5-6 = -1
    // dr=2, dc=-1 → ki=0, plane = 56
    GPUMove move = MAKE_GPU_MOVE(6, 21, 0);
    int idx = move_to_policy_index_host(move, 1);
    ASSERT_TRUE(idx >= 0);
    ASSERT_TRUE(idx < 4672);
    ASSERT_EQ(idx, 6 * 73 + 56);
}

void test_policy_index_promotion_queen(bool& test_failed) {
    // e7-e8=Q: e7=52, e8=60, promo=4 (queen) → treated as normal slide (queen default)
    // dr=1, dc=0 → dir=0, dist=1, plane=0
    GPUMove move = MAKE_GPU_MOVE(52, 60, 4);
    int idx = move_to_policy_index_host(move, 1);
    ASSERT_TRUE(idx >= 0);
    ASSERT_TRUE(idx < 4672);
    // Queen promotion uses the queen-move encoding (plane 0 = N dist 1)
    ASSERT_EQ(idx, 52 * 73 + 0);
}

void test_policy_index_underpromotion_knight(bool& test_failed) {
    // e7-e8=N: e7=52, e8=60, promo=1 (knight)
    // Underpromotion: promo=1, dc=0, plane = 64 + (1-1)*3 + 1 = 65
    GPUMove move = MAKE_GPU_MOVE(52, 60, 1);
    int idx = move_to_policy_index_host(move, 1);
    ASSERT_TRUE(idx >= 0);
    ASSERT_TRUE(idx < 4672);
    ASSERT_EQ(idx, 52 * 73 + 65);
}

void test_policy_index_black_flip(bool& test_failed) {
    // Black to move: e7-e5 (e7=52, e5=36)
    // After flip: from=52^56=12, to=36^56=28
    // Same geometry as white e2-e4: dr=+2, dc=0 → dir=0, dist=2, plane=1
    GPUMove move = MAKE_GPU_MOVE(52, 36, 0);
    int idx = move_to_policy_index_host(move, 0);  // black to move
    ASSERT_TRUE(idx >= 0);
    ASSERT_TRUE(idx < 4672);
    // After flip: from_sq=12, plane=1 → index = 12 * 73 + 1 = 877
    ASSERT_EQ(idx, 12 * 73 + 1);
}

// ============================================================
// compute_llr tests
// ============================================================

void test_llr_even_score(bool& test_failed) {
    // 50% score with elo_diff=50: should be near 0
    // 50-50 with 100 games: wins=50, losses=50, draws=0
    // score = 0.5 = expected under H0 (elo0=0)
    // Under elo0=0, p0=0.5; under elo1=50, p1>0.5
    // LLR = N * [0.5*log(p1/p0) + 0.5*log((1-p1)/(1-p0))]
    // = N * [0.5*log(p1/0.5) + 0.5*log((1-p1)/0.5)]
    // = N * 0.5 * log(p1*(1-p1)/(0.25))
    // Since p1*(1-p1) < 0.25 for p1 != 0.5, LLR should be slightly negative
    float llr = compute_llr(50, 50, 0, 0.0f, 50.0f);
    // With score=0.5 and elo1=50, LLR should be negative (data favors H0)
    ASSERT_TRUE(llr < 0.0f);
    // But not extremely negative
    ASSERT_TRUE(llr > -20.0f);
}

void test_llr_high_score(bool& test_failed) {
    // 75% score with elo0=0, elo1=50 → positive LLR (data supports H1)
    float llr = compute_llr(75, 25, 0, 0.0f, 50.0f);
    ASSERT_TRUE(llr > 0.0f);
}

void test_llr_low_score(bool& test_failed) {
    // 25% score → negative LLR (data rejects H1)
    float llr = compute_llr(25, 75, 0, 0.0f, 50.0f);
    ASSERT_TRUE(llr < 0.0f);
}

void test_llr_zero_games(bool& test_failed) {
    float llr = compute_llr(0, 0, 0, 0.0f, 50.0f);
    ASSERT_NEAR(llr, 0.0f, 1e-6f);
}

// ============================================================
// check_sprt tests
// ============================================================

void test_sprt_accept(bool& test_failed) {
    // Upper bound = log((1-beta)/alpha) = log(0.95/0.05) ≈ 2.944
    // LLR >> upper → H1 accepted
    const char* result = check_sprt(5.0f, 0.05f, 0.05f);
    ASSERT_TRUE(result != nullptr);
    ASSERT_EQ(strcmp(result, "H1"), 0);
}

void test_sprt_reject(bool& test_failed) {
    // Lower bound = log(beta/(1-alpha)) = log(0.05/0.95) ≈ -2.944
    // LLR << lower → H0 (reject)
    const char* result = check_sprt(-5.0f, 0.05f, 0.05f);
    ASSERT_TRUE(result != nullptr);
    ASSERT_EQ(strcmp(result, "H0"), 0);
}

void test_sprt_inconclusive(bool& test_failed) {
    // LLR between bounds → continue (nullptr)
    const char* result = check_sprt(0.0f, 0.05f, 0.05f);
    ASSERT_TRUE(result == nullptr);
}

// ============================================================
// tf_softmax_rows tests
// ============================================================

void test_softmax_uniform(bool& test_failed) {
    // Uniform input → uniform output (1/N each)
    // Use 64 rows, 64 cols to match transformer's expected dimensions
    const int rows = 64, cols = 64;
    float h_data[rows * cols];
    for (int i = 0; i < rows * cols; i++) h_data[i] = 1.0f;

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h_data)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));

    int smem_size = (rows * cols + 256) * sizeof(float);
    kernel_test_softmax<<<1, 256, smem_size>>>(d_data, rows, cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_data, d_data, sizeof(h_data), cudaMemcpyDeviceToHost));

    float expected = 1.0f / cols;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            ASSERT_NEAR(h_data[r * cols + c], expected, 1e-5f);

    cudaFree(d_data);
}

void test_softmax_one_hot(bool& test_failed) {
    // One large value → output ≈ 1.0 for that element
    const int rows = 64, cols = 64;
    float h_data[rows * cols];
    for (int i = 0; i < rows * cols; i++) h_data[i] = 0.0f;
    // Set element [0, 3] to a large value
    h_data[3] = 100.0f;

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h_data)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));

    int smem_size = (rows * cols + 256) * sizeof(float);
    kernel_test_softmax<<<1, 256, smem_size>>>(d_data, rows, cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_data, d_data, sizeof(h_data), cudaMemcpyDeviceToHost));

    // Row 0: element 3 should be ~1.0, rest ~0.0
    ASSERT_NEAR(h_data[3], 1.0f, 1e-5f);
    for (int c = 0; c < cols; c++) {
        if (c != 3) ASSERT_NEAR(h_data[c], 0.0f, 1e-5f);
    }

    cudaFree(d_data);
}

void test_softmax_rows_sum_to_one(bool& test_failed) {
    // Varying input, check rows sum to 1.0
    const int rows = 64, cols = 64;
    float h_data[rows * cols];
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            h_data[r * cols + c] = (float)(r * cols + c) * 0.01f - 2.0f;

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h_data)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));

    int smem_size = (rows * cols + 256) * sizeof(float);
    kernel_test_softmax<<<1, 256, smem_size>>>(d_data, rows, cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_data, d_data, sizeof(h_data), cudaMemcpyDeviceToHost));

    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += h_data[r * cols + c];
            ASSERT_TRUE(h_data[r * cols + c] >= 0.0f);
        }
        ASSERT_NEAR(sum, 1.0f, 1e-4f);
    }

    cudaFree(d_data);
}

// ============================================================
// tf_relu tests
// ============================================================

void test_relu_positive_unchanged(bool& test_failed) {
    const int N = 256;
    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)(i + 1) * 0.5f;

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h_data)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));

    kernel_test_relu<<<1, 256, N * sizeof(float)>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_out[N];
    CHECK_CUDA(cudaMemcpy(h_out, d_data, sizeof(h_out), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        ASSERT_NEAR(h_out[i], h_data[i], 1e-6f);

    cudaFree(d_data);
}

void test_relu_negative_zeroed(bool& test_failed) {
    const int N = 256;
    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = -(float)(i + 1) * 0.5f;

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h_data)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));

    kernel_test_relu<<<1, 256, N * sizeof(float)>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_out[N];
    CHECK_CUDA(cudaMemcpy(h_out, d_data, sizeof(h_out), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        ASSERT_NEAR(h_out[i], 0.0f, 1e-6f);

    cudaFree(d_data);
}

void test_relu_mixed(bool& test_failed) {
    const int N = 256;
    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)i - 128.0f;

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h_data)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));

    kernel_test_relu<<<1, 256, N * sizeof(float)>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_out[N];
    CHECK_CUDA(cudaMemcpy(h_out, d_data, sizeof(h_out), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        float expected = h_data[i] > 0.0f ? h_data[i] : 0.0f;
        ASSERT_NEAR(h_out[i], expected, 1e-6f);
    }

    cudaFree(d_data);
}

// ============================================================
// tf_add_bias tests
// ============================================================

void test_add_bias_correct(bool& test_failed) {
    // M=4 rows, N=8 cols: bias[j] added to each row
    const int M = 4, N = 8;
    float h_data[M * N];
    float h_bias[N];

    for (int i = 0; i < M * N; i++) h_data[i] = (float)i;
    for (int j = 0; j < N; j++) h_bias[j] = 100.0f + (float)j;

    float* d_data;
    float* d_bias;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h_data)));
    CHECK_CUDA(cudaMalloc(&d_bias, sizeof(h_bias)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, sizeof(h_bias), cudaMemcpyHostToDevice));

    // tf_add_bias reads bias from global memory, data from smem
    // But our kernel wrapper copies data to smem and uses global bias
    // Need smem for M*N floats
    kernel_test_add_bias<<<1, 256, M * N * sizeof(float)>>>(d_data, d_bias, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_out[M * N];
    CHECK_CUDA(cudaMemcpy(h_out, d_data, sizeof(h_out), cudaMemcpyDeviceToHost));

    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            float expected = (float)(r * N + c) + 100.0f + (float)c;
            ASSERT_NEAR(h_out[r * N + c], expected, 1e-5f);
        }

    cudaFree(d_data);
    cudaFree(d_bias);
}

void test_add_bias_zero_bias(bool& test_failed) {
    // Zero bias should leave data unchanged
    const int M = 4, N = 8;
    float h_data[M * N];
    float h_bias[N];

    for (int i = 0; i < M * N; i++) h_data[i] = (float)i * 0.3f;
    for (int j = 0; j < N; j++) h_bias[j] = 0.0f;

    float* d_data;
    float* d_bias;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(h_data)));
    CHECK_CUDA(cudaMalloc(&d_bias, sizeof(h_bias)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, sizeof(h_bias), cudaMemcpyHostToDevice));

    kernel_test_add_bias<<<1, 256, M * N * sizeof(float)>>>(d_data, d_bias, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_out[M * N];
    CHECK_CUDA(cudaMemcpy(h_out, d_data, sizeof(h_out), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; i++)
        ASSERT_NEAR(h_out[i], h_data[i], 1e-6f);

    cudaFree(d_data);
    cudaFree(d_bias);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== Selfplay Encoding & Transformer Ops Tests ===\n");
    init_movegen_tables();

    int total = 0, passes = 0, failures = 0;

    printf("\n--- board_to_planes_host ---\n");
    RUN_TEST(test_planes_starting_white_pawns);
    RUN_TEST(test_planes_starting_black_pawns);
    RUN_TEST(test_planes_castling_all);
    RUN_TEST(test_planes_no_ep);
    RUN_TEST(test_planes_black_to_move_flip);
    RUN_TEST(test_planes_piece_count);

    printf("\n--- move_to_policy_index_host ---\n");
    RUN_TEST(test_policy_index_e2e4);
    RUN_TEST(test_policy_index_nf3);
    RUN_TEST(test_policy_index_promotion_queen);
    RUN_TEST(test_policy_index_underpromotion_knight);
    RUN_TEST(test_policy_index_black_flip);

    printf("\n--- compute_llr ---\n");
    RUN_TEST(test_llr_even_score);
    RUN_TEST(test_llr_high_score);
    RUN_TEST(test_llr_low_score);
    RUN_TEST(test_llr_zero_games);

    printf("\n--- check_sprt ---\n");
    RUN_TEST(test_sprt_accept);
    RUN_TEST(test_sprt_reject);
    RUN_TEST(test_sprt_inconclusive);

    printf("\n--- tf_softmax_rows ---\n");
    RUN_TEST(test_softmax_uniform);
    RUN_TEST(test_softmax_one_hot);
    RUN_TEST(test_softmax_rows_sum_to_one);

    printf("\n--- tf_relu ---\n");
    RUN_TEST(test_relu_positive_unchanged);
    RUN_TEST(test_relu_negative_zeroed);
    RUN_TEST(test_relu_mixed);

    printf("\n--- tf_add_bias ---\n");
    RUN_TEST(test_add_bias_correct);
    RUN_TEST(test_add_bias_zero_bias);

    printf("\n%d/%d tests passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");
    return failures > 0 ? 1 : 0;
}
