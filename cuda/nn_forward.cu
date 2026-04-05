#include "nn_forward.cuh"
#include "nn_ops.cuh"
#include <cstdio>

// ============================================================
// Full forward pass implementation
// ============================================================

__device__ void oracle_net_forward(
    const BoardState* bs,
    float q_result,
    const OracleNetWeights* weights,
    float* scratch,
    float* policy_out,
    float* value_out,
    float* k_out
) {
    // === Scratch buffer partitioning ===
    float* buf1   = scratch;                                    // [128, 64]
    float* buf2   = scratch + SCRATCH_BUF_SIZE;                 // [128, 64]
    float* col    = scratch + SCRATCH_BUF_SIZE * 2;             // [C_in*9, 64]
    float* planes = scratch + SCRATCH_BUF_SIZE * 2 + SCRATCH_COL_SIZE; // [17, 64]
    float* policy = scratch + SCRATCH_BUF_SIZE * 2 + SCRATCH_COL_SIZE + SCRATCH_PLANES_SIZE; // [4672]

    int lane = threadIdx.x & 31;

    // === 1. Board encoding → [17, 8, 8] ===
    warp_board_to_planes(bs, planes);
    __syncwarp();

    // === 2. Input conv(17→128, k=3, p=1) + BN + ReLU ===
    warp_im2col_3x3(planes, col, NN_INPUT_CHANNELS);  // col: [17*9, 64] = [153, 64]
    __syncwarp();
    warp_gemm(weights->start_conv_weight, col, buf1, NN_HIDDEN_DIM, 64, NN_INPUT_CHANNELS * 9);
    __syncwarp();
    warp_bn_relu(buf1, weights->start_bn.weight, weights->start_bn.bias,
                 weights->start_bn.running_mean, weights->start_bn.running_var,
                 NN_HIDDEN_DIM, true);
    __syncwarp();

    // === 3. Residual blocks ===
    for (int b = 0; b < NN_NUM_BLOCKS; b++) {
        const ResBlockParams& block = weights->blocks[b];

        // conv1 + BN + ReLU
        warp_im2col_3x3(buf1, col, NN_HIDDEN_DIM);
        __syncwarp();
        warp_gemm(block.conv1_weight, col, buf2, NN_HIDDEN_DIM, 64, NN_HIDDEN_DIM * 9);
        __syncwarp();
        warp_bn_relu(buf2, block.bn1.weight, block.bn1.bias,
                     block.bn1.running_mean, block.bn1.running_var,
                     NN_HIDDEN_DIM, true);
        __syncwarp();

        // conv2 + BN (no ReLU yet)
        warp_im2col_3x3(buf2, col, NN_HIDDEN_DIM);
        __syncwarp();
        warp_gemm(block.conv2_weight, col, buf2, NN_HIDDEN_DIM, 64, NN_HIDDEN_DIM * 9);
        __syncwarp();
        warp_bn_relu(buf2, block.bn2.weight, block.bn2.bias,
                     block.bn2.running_mean, block.bn2.running_var,
                     NN_HIDDEN_DIM, false);  // no ReLU
        __syncwarp();

        // SE block
        warp_se_block(buf2, block.se.fc1_weight, block.se.fc2_weight,
                      NN_HIDDEN_DIM, NN_SE_INNER);
        __syncwarp();

        // Residual + ReLU: buf1 = relu(buf1 + buf2)
        warp_add_relu(buf1, buf2, buf1, NN_HIDDEN_DIM * 64);
        __syncwarp();
    }

    // buf1 now contains the backbone output: [128, 8, 8]

    // === 4. Policy head ===
    // Policy conv1: Conv(128→128, k=3, p=1) + BN + ReLU
    warp_im2col_3x3(buf1, col, NN_HIDDEN_DIM);
    __syncwarp();
    warp_gemm(weights->p_conv_weight, col, buf2, NN_HIDDEN_DIM, 64, NN_HIDDEN_DIM * 9);
    __syncwarp();
    warp_bn_relu(buf2, weights->p_bn.weight, weights->p_bn.bias,
                 weights->p_bn.running_mean, weights->p_bn.running_var,
                 NN_HIDDEN_DIM, true);
    __syncwarp();

    // Policy conv2: Conv(128→73, k=1) + bias → [73, 64]
    // For 1x1 conv: no im2col needed, just GEMM
    // weight: [73, 128], input: [128, 64] → output: [73, 64]
    warp_gemm(weights->p_head_weight, buf2, col, NN_POLICY_PLANES, 64, NN_HIDDEN_DIM);
    __syncwarp();

    // Add bias
    for (int idx = lane; idx < NN_POLICY_PLANES * 64; idx += 32) {
        int ch = idx / 64;
        col[idx] += weights->p_head_bias[ch];
    }
    __syncwarp();

    // Permute [73, 8, 8] → [8, 8, 73] → flatten [4672]
    // policy[y*8*73 + x*73 + plane] = col[plane*64 + y*8 + x]
    for (int idx = lane; idx < NN_POLICY_SIZE; idx += 32) {
        int plane = idx % NN_POLICY_PLANES;       // 0-72
        int spatial = idx / NN_POLICY_PLANES;      // 0-63
        policy[idx] = col[plane * 64 + spatial];
    }
    __syncwarp();

    // Log-softmax
    warp_log_softmax(policy, policy_out, NN_POLICY_SIZE);
    __syncwarp();

    // === 5. Value head ===
    // Value conv: Conv(128→1, k=1) + bias + BN + ReLU → [1, 64]
    // 1x1 conv: weight [1, 128], input [128, 64] → output [1, 64]
    // Reuse buf2 as temp (only need 64 floats)
    {
        // GEMM for 1x1 conv
        float* v_feat = buf2; // [64] (reuse start of buf2)
        for (int idx = lane; idx < 64; idx += 32) {
            float sum = 0.0f;
            for (int c = 0; c < NN_HIDDEN_DIM; c++) {
                sum += weights->v_conv_weight[c] * buf1[c * 64 + idx];
            }
            v_feat[idx] = sum + weights->v_conv_bias[0];
        }
        __syncwarp();

        // BN + ReLU on [1, 64]
        warp_bn_relu_1ch(v_feat, weights->v_bn.weight, weights->v_bn.bias,
                         weights->v_bn.running_mean, weights->v_bn.running_var, true);
        __syncwarp();

        // Flatten [64] + concat q_result → [65]
        // FC1: [256, 65] × [65] → [256]
        float* fc1_out = buf2 + 64; // [256] (after v_feat in buf2)
        for (int idx = lane; idx < NN_VALUE_FC1_OUT; idx += 32) {
            float sum = 0.0f;
            for (int j = 0; j < 64; j++) {
                sum += weights->v_fc_weight[idx * NN_VALUE_FC1_IN + j] * v_feat[j];
            }
            // Last input is q_result (index 64)
            sum += weights->v_fc_weight[idx * NN_VALUE_FC1_IN + 64] * q_result;
            sum += weights->v_fc_bias[idx];
            fc1_out[idx] = sum > 0.0f ? sum : 0.0f; // ReLU
        }
        __syncwarp();

        // FC2: [1, 256] × [256] → [1] = v_logit
        if (lane == 0) {
            float v_logit = 0.0f;
            for (int j = 0; j < NN_VALUE_FC1_OUT; j++) {
                v_logit += weights->v_out_weight[j] * fc1_out[j];
            }
            v_logit += weights->v_out_bias[0];

            // k = 0.47 * softplus(k_logit)
            float k = 0.47f * logf(1.0f + expf(weights->k_logit));

            // V = tanh(v_logit + k * q_result)
            *value_out = tanhf(v_logit + k * q_result);
            *k_out = k;
        }
    }
    __syncwarp();
}

// ============================================================
// Host-side API
// ============================================================

float* alloc_nn_scratch(int num_warps) {
    float* d_scratch = nullptr;
    size_t total_bytes = (size_t)num_warps * SCRATCH_TOTAL_FLOATS * sizeof(float);
    cudaError_t err = cudaMalloc(&d_scratch, total_bytes);
    if (err != cudaSuccess) {
        printf("Failed to allocate NN scratch (%zu MB): %s\n",
               total_bytes / (1024 * 1024), cudaGetErrorString(err));
        return nullptr;
    }
    cudaMemset(d_scratch, 0, total_bytes);
    printf("NN scratch allocated: %d warps × %d KB = %.1f MB\n",
           num_warps, SCRATCH_TOTAL_BYTES / 1024, total_bytes / (1024.0 * 1024.0));
    return d_scratch;
}

void free_nn_scratch(float* d_scratch) {
    if (d_scratch) cudaFree(d_scratch);
}
