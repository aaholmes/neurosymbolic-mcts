#include "nn_ops.cuh"
#include <cfloat>

// All functions assume they're called by a full warp (32 threads).
// Work is striped across lanes: thread i processes elements i, i+32, i+64, ...

__device__ void warp_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int lane = threadIdx.x & 31;
    int total = M * N;
    for (int idx = lane; idx < total; idx += 32) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[idx] = sum;
    }
}

__device__ void warp_im2col_3x3(
    const float* input,
    float* col,
    int C_in
) {
    int lane = threadIdx.x & 31;
    int total = C_in * 9 * 64;
    for (int idx = lane; idx < total; idx += 32) {
        int spatial = idx % 64;     // output spatial index (0-63)
        int k = idx / 64;           // kernel index (0 to C_in*9-1)
        int ci = k / 9;            // input channel
        int ki = k % 9;            // kernel position (0-8)
        int ky = ki / 3 - 1;       // kernel y offset (-1, 0, 1)
        int kx = ki % 3 - 1;       // kernel x offset
        int oy = spatial / 8;      // output y
        int ox = spatial % 8;      // output x
        int iy = oy + ky;
        int ix = ox + kx;
        if (iy >= 0 && iy < 8 && ix >= 0 && ix < 8)
            col[idx] = input[ci * 64 + iy * 8 + ix];
        else
            col[idx] = 0.0f;       // zero padding
    }
}

__device__ void warp_bn_relu(
    float* data,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int channels,
    bool relu
) {
    int lane = threadIdx.x & 31;
    int total = channels * 64;
    for (int idx = lane; idx < total; idx += 32) {
        int ch = idx / 64;
        float x = data[idx];
        float norm = (x - running_mean[ch]) * rsqrtf(running_var[ch] + 1e-5f);
        float result = norm * gamma[ch] + beta[ch];
        if (relu && result < 0.0f) result = 0.0f;
        data[idx] = result;
    }
}

__device__ void warp_bn_relu_1ch(
    float* data,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    bool relu
) {
    int lane = threadIdx.x & 31;
    float g = gamma[0], b = beta[0], m = running_mean[0];
    float inv_std = rsqrtf(running_var[0] + 1e-5f);
    for (int idx = lane; idx < 64; idx += 32) {
        float x = data[idx];
        float result = (x - m) * inv_std * g + b;
        if (relu && result < 0.0f) result = 0.0f;
        data[idx] = result;
    }
}

__device__ void warp_add_relu(
    const float* a, const float* b, float* out,
    int size
) {
    int lane = threadIdx.x & 31;
    for (int idx = lane; idx < size; idx += 32) {
        float val = a[idx] + b[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

__device__ void warp_se_block(
    float* data,
    const float* fc1_w,  // [inner, channels]
    const float* fc2_w,  // [channels, inner]
    int channels,
    int inner
) {
    // 1. Global average pool: [channels, 64] → [channels]
    // Use warp reduction for efficiency
    // Each thread accumulates partial sums, then we do a warp shuffle reduce
    int lane = threadIdx.x & 31;

    // Shared via registers — each thread computes avg for some channels
    // For channels=128 and 32 lanes: 4 channels per thread
    float channel_avg[4]; // max 128/32 = 4 channels per thread
    for (int c = lane; c < channels; c += 32) {
        float sum = 0.0f;
        for (int i = 0; i < 64; i++) {
            sum += data[c * 64 + i];
        }
        channel_avg[c / 32] = sum / 64.0f;
    }

    // We need all channel averages accessible to all threads for the FC layers.
    // Store in a small buffer (reuse part of data? No, we need data intact).
    // Use warp shuffle to broadcast: impractical for 128 values.
    // Instead, write to a small temp array in the caller's scratch space.
    // For simplicity: use a shared temp in global scratch.
    // Actually, for small inner dim (8), we can compute FC1 output per-thread:

    // Alternative: compute SE directly without materializing the avg vector.
    // FC1[j] = sum_c(fc1_w[j,c] * avg[c]) for j=0..inner-1
    // FC2[c] = sum_j(fc2_w[c,j] * relu(FC1[j])) for c=0..channels-1

    // Each thread computes some FC1 outputs
    float fc1_out[1]; // inner=8, with 32 threads: some threads compute 0 FC1 values
    // Actually, let's just have each thread compute ALL FC1 outputs (inner=8, cheap)

    float fc1[8]; // small enough for registers
    for (int j = 0; j < inner; j++) {
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            // Need avg[c] — compute it inline
            float avg = 0.0f;
            for (int i = 0; i < 64; i++) {
                avg += data[c * 64 + i];
            }
            avg /= 64.0f;
            sum += fc1_w[j * channels + c] * avg;
        }
        fc1[j] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    // FC2: scale[c] = sigmoid(sum_j(fc2_w[c,j] * fc1[j]))
    // Apply scale to data
    for (int c = lane; c < channels; c += 32) {
        float sum = 0.0f;
        for (int j = 0; j < inner; j++) {
            sum += fc2_w[c * inner + j] * fc1[j];
        }
        float scale = 1.0f / (1.0f + expf(-sum)); // sigmoid
        for (int i = 0; i < 64; i++) {
            data[c * 64 + i] *= scale;
        }
    }
}

__device__ void warp_log_softmax(
    const float* input,
    float* output,
    int size
) {
    int lane = threadIdx.x & 31;

    // 1. Find max (for numerical stability)
    float local_max = -FLT_MAX;
    for (int i = lane; i < size; i += 32) {
        if (input[i] > local_max) local_max = input[i];
    }
    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    local_max = __shfl_sync(0xFFFFFFFF, local_max, 0); // broadcast

    // 2. Compute sum(exp(x - max))
    float local_sum = 0.0f;
    for (int i = lane; i < size; i += 32) {
        local_sum += expf(input[i] - local_max);
    }
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    float total = __shfl_sync(0xFFFFFFFF, local_sum, 0);
    float log_total = logf(total);

    // 3. log_softmax[i] = (x[i] - max) - log(sum)
    for (int i = lane; i < size; i += 32) {
        output[i] = (input[i] - local_max) - log_total;
    }
}

__device__ void warp_board_to_planes(
    const BoardState* bs,
    float* planes
) {
    int lane = threadIdx.x & 31;
    int stm = bs->w_to_move ? 0 : 1; // WHITE=0, BLACK=1
    int opp = 1 - stm;
    bool flip = !bs->w_to_move; // flip ranks for Black STM

    // Zero all planes first
    for (int i = lane; i < 17 * 64; i += 32) {
        planes[i] = 0.0f;
    }
    __syncwarp();

    // Planes 0-5: STM pieces, planes 6-11: opponent pieces
    // Each thread handles some squares
    for (int piece = 0; piece < 6; piece++) {
        uint64_t stm_bb = bs->pieces[stm * 6 + piece];
        uint64_t opp_bb = bs->pieces[opp * 6 + piece];

        for (int sq = lane; sq < 64; sq += 32) {
            int mapped = flip ? (sq ^ 56) : sq; // flip ranks for Black
            if ((stm_bb >> sq) & 1) planes[piece * 64 + mapped] = 1.0f;
            if ((opp_bb >> sq) & 1) planes[(6 + piece) * 64 + mapped] = 1.0f;
        }
    }

    // Plane 12: en passant
    if (bs->en_passant != EN_PASSANT_NONE && lane == 0) {
        int ep_sq = bs->en_passant;
        int mapped = flip ? (ep_sq ^ 56) : ep_sq;
        planes[12 * 64 + mapped] = 1.0f;
    }

    // Planes 13-16: castling rights (STM-relative)
    if (lane == 0) {
        uint8_t stm_ks, stm_qs, opp_ks, opp_qs;
        if (bs->w_to_move) {
            stm_ks = bs->castling & CASTLE_WK;
            stm_qs = bs->castling & CASTLE_WQ;
            opp_ks = bs->castling & CASTLE_BK;
            opp_qs = bs->castling & CASTLE_BQ;
        } else {
            stm_ks = bs->castling & CASTLE_BK;
            stm_qs = bs->castling & CASTLE_BQ;
            opp_ks = bs->castling & CASTLE_WK;
            opp_qs = bs->castling & CASTLE_WQ;
        }
        if (stm_ks) for (int i = 0; i < 64; i++) planes[13 * 64 + i] = 1.0f;
        if (stm_qs) for (int i = 0; i < 64; i++) planes[14 * 64 + i] = 1.0f;
        if (opp_ks) for (int i = 0; i < 64; i++) planes[15 * 64 + i] = 1.0f;
        if (opp_qs) for (int i = 0; i < 64; i++) planes[16 * 64 + i] = 1.0f;
    }
    __syncwarp();
}
