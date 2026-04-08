#pragma once

#include "common.cuh"
#include "nn_weights.cuh"  // for NN_HIDDEN_DIM, NN_INPUT_CHANNELS, NN_POLICY_*
#include <cuda_fp16.h>

// ============================================================
// Transformer Weight Storage for GPU-Resident Inference
//
// Architecture: Transformer-128×6 (pre-LayerNorm)
//   - 64 tokens (one per square), d_model=128
//   - 4 attention heads, head_dim=32
//   - FFN: 128→512→128 with ReLU
//   - Input: [B, 17, 8, 8] → linear projection → [B, 64, 128]
//   - Output: policy [B, 4672], v_logit [B, 1], k scalar
// ============================================================

// Transformer architecture constants
constexpr int TF_NUM_LAYERS   = 6;
constexpr int TF_NUM_HEADS    = 4;
constexpr int TF_HEAD_DIM     = 32;   // NN_HIDDEN_DIM / TF_NUM_HEADS
constexpr int TF_FFN_DIM      = 512;  // 4× expansion
constexpr int TF_FFN_TILE     = 64;   // tile size for FFN (fits in workspace)
constexpr int TF_VALUE_FC1_OUT = 256;
constexpr int TF_NUM_TOKENS   = 64;   // 8×8 board

// LayerNorm parameters
struct TFLayerNorm {
    float weight[NN_HIDDEN_DIM];  // gamma
    float bias[NN_HIDDEN_DIM];    // beta
};

// One transformer block
struct TransformerBlock {
    // Pre-attention LayerNorm
    TFLayerNorm ln1;
    // QKV projection (fused: [d_model, 3*d_model])
    float qkv_weight[NN_HIDDEN_DIM * 3 * NN_HIDDEN_DIM];  // [128, 384]
    float qkv_bias[3 * NN_HIDDEN_DIM];                      // [384]
    // Output projection
    float out_proj_weight[NN_HIDDEN_DIM * NN_HIDDEN_DIM];   // [128, 128]
    float out_proj_bias[NN_HIDDEN_DIM];                      // [128]
    // Pre-FFN LayerNorm
    TFLayerNorm ln2;
    // FFN
    float ffn1_weight[NN_HIDDEN_DIM * TF_FFN_DIM];          // [128, 512]
    float ffn1_bias[TF_FFN_DIM];                              // [512]
    float ffn2_weight[TF_FFN_DIM * NN_HIDDEN_DIM];          // [512, 128]
    float ffn2_bias[NN_HIDDEN_DIM];                           // [128]
};

// Complete Transformer weights
struct TransformerWeights {
    // === Input projection ===
    float input_proj_weight[NN_HIDDEN_DIM * NN_INPUT_CHANNELS];  // [128, 17]
    float input_proj_bias[NN_HIDDEN_DIM];                         // [128]
    float pos_embedding[TF_NUM_TOKENS * NN_HIDDEN_DIM];          // [64, 128]

    // === Transformer blocks ===
    TransformerBlock blocks[TF_NUM_LAYERS];

    // === Policy head ===
    TFLayerNorm p_ln;
    float p_head_weight[NN_POLICY_PLANES * NN_HIDDEN_DIM];       // [73, 128]
    float p_head_bias[NN_POLICY_PLANES];                          // [73]

    // === Value head ===
    TFLayerNorm v_ln;
    float v_fc1_weight[TF_VALUE_FC1_OUT * (NN_HIDDEN_DIM + 1)];  // [256, 129]
    float v_fc1_bias[TF_VALUE_FC1_OUT];                            // [256]
    float v_fc2_weight[TF_VALUE_FC1_OUT];                          // [1, 256]
    float v_fc2_bias[1];                                           // [1]

    // === K scalar ===
    float k_logit;
};

// Pre-converted FP16 weights for Tensor Core path
struct TransformerWeightsHalf {
    struct BlockHalf {
        half* qkv;          // [128, 384] (fused Q/K/V)
        half* q_head[TF_NUM_HEADS];  // per-head [32, 128]
        half* k_head[TF_NUM_HEADS];  // per-head [32, 128]
        half* v_head[TF_NUM_HEADS];  // per-head [32, 128]
        half* qkv_head[TF_NUM_HEADS]; // per-head fused [96, 128] (Q|K|V concat)
        half* out_proj;     // [128, 128]
        half* ffn1;         // [128, 512] (full, kept for reference)
        half* ffn2;         // [512, 128] (full, kept for reference)
        half* ffn1_tile[TF_FFN_DIM / TF_FFN_TILE];  // 8 × [64, 128] pre-split tiles
        half* ffn2_tile[TF_FFN_DIM / TF_FFN_TILE];  // 8 × [128, 64] pre-split tiles
    } blocks[TF_NUM_LAYERS];
    half* input_proj;       // [128, 17]
    half* p_head;           // [73, 128]
};

// ============================================================
// Host-side API
// ============================================================

TransformerWeights* init_transformer_weights_zeros();
TransformerWeights* load_transformer_weights(const char* path);
void free_transformer_weights(TransformerWeights* d_weights);
size_t transformer_weights_size();

TransformerWeightsHalf* convert_transformer_to_half(const TransformerWeights* d_weights);
void free_transformer_half(TransformerWeightsHalf* d_half);
