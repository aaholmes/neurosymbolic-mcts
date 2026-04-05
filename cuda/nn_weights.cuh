#pragma once

#include "common.cuh"

// ============================================================
// OracleNet Weight Storage for GPU-Resident Inference
//
// All weights stored in GPU global memory as flat float arrays.
// One shared copy (~7.6 MB) read by all explorer warps.
// Layout matches PyTorch's OracleNet (model.py) exactly.
//
// Architecture: SE-ResNet with 6 blocks, 128 channels
//   Input:  [B, 17, 8, 8]
//   Output: policy [B, 4672], v_logit [B, 1], k scalar
// ============================================================

// Fixed architecture constants
constexpr int NN_HIDDEN_DIM = 128;
constexpr int NN_INPUT_CHANNELS = 17;
constexpr int NN_NUM_BLOCKS = 6;
constexpr int NN_SE_REDUCTION = 16;  // 128 / 16 = 8
constexpr int NN_SE_INNER = NN_HIDDEN_DIM / NN_SE_REDUCTION;  // 8
constexpr int NN_POLICY_PLANES = 73;
constexpr int NN_POLICY_SIZE = 4672;  // 73 * 64
constexpr int NN_VALUE_FC1_IN = 65;   // 64 + 1 (q_result)
constexpr int NN_VALUE_FC1_OUT = 256;

// BatchNorm parameters (4 arrays: weight/gamma, bias/beta, running_mean, running_var)
struct BNParams {
    float weight[NN_HIDDEN_DIM];       // gamma
    float bias[NN_HIDDEN_DIM];         // beta
    float running_mean[NN_HIDDEN_DIM]; // running mean
    float running_var[NN_HIDDEN_DIM];  // running var
};

// BatchNorm for single-channel (value head)
struct BNParams1 {
    float weight[1];
    float bias[1];
    float running_mean[1];
    float running_var[1];
};

// SE block: FC(128→8, no bias) + ReLU + FC(8→128, no bias) + Sigmoid
struct SEParams {
    float fc1_weight[NN_SE_INNER * NN_HIDDEN_DIM];  // [8, 128]
    float fc2_weight[NN_HIDDEN_DIM * NN_SE_INNER];  // [128, 8]
};

// One residual block: 2 convs + 2 BNs + SE
struct ResBlockParams {
    float conv1_weight[NN_HIDDEN_DIM * NN_HIDDEN_DIM * 3 * 3];  // [128, 128, 3, 3]
    BNParams bn1;
    float conv2_weight[NN_HIDDEN_DIM * NN_HIDDEN_DIM * 3 * 3];  // [128, 128, 3, 3]
    BNParams bn2;
    SEParams se;
};

// Complete OracleNet weights
struct OracleNetWeights {
    // === Input conv + BN ===
    float start_conv_weight[NN_HIDDEN_DIM * NN_INPUT_CHANNELS * 3 * 3]; // [128, 17, 3, 3]
    BNParams start_bn;

    // === 6 Residual blocks ===
    ResBlockParams blocks[NN_NUM_BLOCKS];

    // === Policy head ===
    float p_conv_weight[NN_HIDDEN_DIM * NN_HIDDEN_DIM * 3 * 3];  // [128, 128, 3, 3]
    BNParams p_bn;
    float p_head_weight[NN_POLICY_PLANES * NN_HIDDEN_DIM];        // [73, 128, 1, 1] (1x1 conv)
    float p_head_bias[NN_POLICY_PLANES];                           // [73]

    // === Value head ===
    float v_conv_weight[NN_HIDDEN_DIM];    // [1, 128, 1, 1] (1x1 conv)
    float v_conv_bias[1];                   // [1]
    BNParams1 v_bn;
    float v_fc_weight[NN_VALUE_FC1_OUT * NN_VALUE_FC1_IN];  // [256, 65]
    float v_fc_bias[NN_VALUE_FC1_OUT];                       // [256]
    float v_out_weight[NN_VALUE_FC1_OUT];   // [1, 256]
    float v_out_bias[1];                     // [1]

    // === K scalar ===
    float k_logit;  // 0.0 at init → k = 0.47 * ln(2) ≈ 0.326
};

// ============================================================
// Host-side API
// ============================================================

// Allocate weights on GPU, initialize to zeros (classical-equivalent behavior:
// uniform policy, v_logit=0, k=0.326).
// Returns pointer to GPU-resident weights.
OracleNetWeights* init_nn_weights_zeros();

// Load weights from a flat binary file (exported by export_weights_cuda.py).
// Returns pointer to GPU-resident weights, or nullptr on failure.
OracleNetWeights* load_nn_weights(const char* path);

// Free GPU-resident weights.
void free_nn_weights(OracleNetWeights* d_weights);

// Hot-swap: update weights from a new binary file without reallocating.
// Returns true on success.
bool update_nn_weights(OracleNetWeights* d_weights, const char* path);

// Get the size of the weights struct in bytes.
size_t nn_weights_size();
