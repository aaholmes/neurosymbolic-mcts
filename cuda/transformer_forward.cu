#include "transformer_forward.cuh"
#include "transformer_ops.cuh"
#include "block_ops.cuh"  // for block_log_softmax

int transformer_smem_bytes() { return TF_SMEM_BYTES; }

// ============================================================
// Transformer forward pass — 256-thread block cooperative
//
// buf_out is FP16 to free 16 KB for a dedicated TC staging region.
// TC accelerates Q/K/V projections, FFN1, and FFN2.
// QK^T, softmax, attn×V, out_proj stay scalar (workspace aliasing).
// ============================================================

__device__ void transformer_forward(
    const BoardState* bs,
    float q_result,
    const TransformerWeights* weights,
    const TransformerWeightsHalf* half_w,
    float* smem,
    float* policy_out,
    float* value_out,
    float* k_out
) {
    int tid = threadIdx.x;

    float* buf_x     = smem + TF_BUF_X_OFFSET;            // [64, 128] FP32 residual
    half*  buf_out   = (half*)(smem + TF_BUF_OUT_OFFSET);  // [64, 128] FP16 layer output
    float* workspace = smem + TF_WORKSPACE_OFFSET;         // [6144] FP32 Q/K/V/attn/FFN
    half*  staging   = (half*)(smem + TF_STAGING_OFFSET);  // [4096 floats] TC staging
    float* reduce    = smem + TF_REDUCE_OFFSET;            // [256] FP32 reductions

    const int D = NN_HIDDEN_DIM;  // 128
    const int T = TF_NUM_TOKENS;  // 64

    // === 1. Board encoding → tokens[64, 17] in workspace (temporary) ===
    float* tokens = workspace;
    tf_board_to_tokens(bs, tokens);

    // === 2. Input projection: tokens[64, 17] × W^T → buf_out[64, 128] FP16 ===
    for (int i = tid; i < T * D; i += blockDim.x) {
        int tok = i / D; int d = i % D;
        float sum = 0.0f;
        for (int c = 0; c < NN_INPUT_CHANNELS; c++)
            sum += tokens[tok * NN_INPUT_CHANNELS + c] * weights->input_proj_weight[d * NN_INPUT_CHANNELS + c];
        buf_out[i] = __float2half(sum + weights->input_proj_bias[d]);
    }
    __syncthreads();

    // Add positional embeddings: buf_x = buf_out(FP16) + pos_embedding(FP32) → buf_x(FP32)
    for (int i = tid; i < T * D; i += blockDim.x)
        buf_x[i] = __half2float(buf_out[i]) + weights->pos_embedding[i];
    __syncthreads();

    // === 3. Transformer blocks ===
    for (int blk = 0; blk < TF_NUM_LAYERS; blk++) {
        const TransformerBlock& block = weights->blocks[blk];

        // --- Multi-Head Self-Attention ---
        // LayerNorm(buf_x FP32) → buf_out(FP16)
        tf_layer_norm_f16out(buf_x, buf_out, block.ln1.weight, block.ln1.bias, T, D, reduce);

        // Save residual to registers, zero buf_x for accumulation
        float res[32];
        for (int k = 0; k < 32; k++) res[k] = buf_x[tid + k * 256];
        for (int i = tid; i < T * D; i += blockDim.x) buf_x[i] = 0.0f;
        __syncthreads();

        float* ws_q    = workspace;
        float* ws_k    = workspace + T * TF_HEAD_DIM;
        float* ws_attn = workspace;
        float* ws_v    = workspace + T * T;  // +4096

        for (int h = 0; h < TF_NUM_HEADS; h++) {
            int qkv_offset = h * TF_HEAD_DIM;

            // Q/K/V projections — TC when half_w available
            if (half_w) {
                tf_linear_f16in(buf_out, half_w->blocks[blk].q_head[h], ws_q,
                                block.qkv_bias + qkv_offset, staging, T, TF_HEAD_DIM, D);
                tf_linear_f16in(buf_out, half_w->blocks[blk].k_head[h], ws_k,
                                block.qkv_bias + D + qkv_offset, staging, T, TF_HEAD_DIM, D);
            } else {
                // Scalar fallback (no half_w)
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += __half2float(buf_out[tok * D + k]) * block.qkv_weight[(qkv_offset + d) * D + k];
                    ws_q[i] = sum + block.qkv_bias[qkv_offset + d];
                }
                __syncthreads();
                int k_offset = D + qkv_offset;
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += __half2float(buf_out[tok * D + k]) * block.qkv_weight[(k_offset + d) * D + k];
                    ws_k[i] = sum + block.qkv_bias[k_offset + d];
                }
                __syncthreads();
            }

            // QK^T — scalar with local_results pattern (workspace aliasing)
            float scale = rsqrtf((float)TF_HEAD_DIM);
            {
                float local_attn[16];
                int n_attn = (T * T + blockDim.x - 1) / blockDim.x;
                for (int e = 0; e < n_attn; e++) {
                    int i = tid + e * blockDim.x;
                    if (i >= T * T) break;
                    int row = i / T; int col = i % T;
                    float sum = 0.0f;
                    for (int d = 0; d < TF_HEAD_DIM; d++)
                        sum += ws_q[row * TF_HEAD_DIM + d] * ws_k[col * TF_HEAD_DIM + d];
                    local_attn[e] = sum * scale;
                }
                __syncthreads();
                for (int e = 0; e < n_attn; e++) {
                    int i = tid + e * blockDim.x;
                    if (i >= T * T) break;
                    ws_attn[i] = local_attn[e];
                }
                __syncthreads();
            }

            tf_softmax_rows(ws_attn, T, T, reduce);

            // V projection — TC when half_w available
            if (half_w) {
                tf_linear_f16in(buf_out, half_w->blocks[blk].v_head[h], ws_v,
                                block.qkv_bias + 2 * D + qkv_offset, staging, T, TF_HEAD_DIM, D);
            } else {
                int v_offset = 2 * D + qkv_offset;
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += __half2float(buf_out[tok * D + k]) * block.qkv_weight[(v_offset + d) * D + k];
                    ws_v[i] = sum + block.qkv_bias[v_offset + d];
                }
                __syncthreads();
            }

            // attn×V — scalar with local_results (workspace aliasing)
            {
                float local_results[8];
                int n_elems = (T * TF_HEAD_DIM + blockDim.x - 1) / blockDim.x;
                for (int e = 0; e < n_elems; e++) {
                    int i = tid + e * blockDim.x;
                    if (i >= T * TF_HEAD_DIM) break;
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < T; k++)
                        sum += ws_attn[tok * T + k] * ws_v[k * TF_HEAD_DIM + d];
                    local_results[e] = sum;
                }
                __syncthreads();
                for (int e = 0; e < n_elems; e++) {
                    int i = tid + e * blockDim.x;
                    if (i >= T * TF_HEAD_DIM) break;
                    ws_q[i] = local_results[e];
                }
                __syncthreads();
            }

            // Output projection — scalar (weight not pre-split per-head)
            for (int i = tid; i < T * D; i += blockDim.x) {
                int tok = i / D; int d = i % D;
                float sum = 0.0f;
                for (int j = 0; j < TF_HEAD_DIM; j++)
                    sum += ws_q[tok * TF_HEAD_DIM + j] * block.out_proj_weight[d * D + h * TF_HEAD_DIM + j];
                buf_x[i] += sum;
            }
            __syncthreads();
        }

        tf_add_bias(buf_x, block.out_proj_bias, T, D);
        for (int k = 0; k < 32; k++) buf_x[tid + k * 256] += res[k];
        __syncthreads();

        // --- FFN ---
        for (int k = 0; k < 32; k++) res[k] = buf_x[tid + k * 256];
        tf_layer_norm_f16out(buf_x, buf_out, block.ln2.weight, block.ln2.bias, T, D, reduce);

        for (int i = tid; i < T * D; i += blockDim.x) buf_x[i] = 0.0f;
        __syncthreads();

        float* ws_tile = workspace;

        for (int tile = 0; tile < TF_FFN_DIM / TF_FFN_TILE; tile++) {
            int tile_start = tile * TF_FFN_TILE;

            if (half_w) {
                // FFN1: buf_out(FP16) → ws_tile(FP32) via TC
                tf_linear_f16in(buf_out, half_w->blocks[blk].ffn1_tile[tile], ws_tile,
                                block.ffn1_bias + tile_start, staging, T, TF_FFN_TILE, D);
                tf_relu(ws_tile, T * TF_FFN_TILE);
                // FFN2: ws_tile(FP32) → buf_x(FP32) via TC, accumulate
                tf_linear(ws_tile, half_w->blocks[blk].ffn2_tile[tile], buf_x,
                          nullptr, staging, T, D, TF_FFN_TILE, true);
            } else {
                for (int i = tid; i < T * TF_FFN_TILE; i += blockDim.x) {
                    int tok = i / TF_FFN_TILE; int d = i % TF_FFN_TILE;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += __half2float(buf_out[tok * D + k]) * block.ffn1_weight[(tile_start + d) * D + k];
                    float val = sum + block.ffn1_bias[tile_start + d];
                    ws_tile[i] = val > 0.0f ? val : 0.0f;
                }
                __syncthreads();

                for (int i = tid; i < T * D; i += blockDim.x) {
                    int tok = i / D; int d = i % D;
                    float sum = 0.0f;
                    for (int j = 0; j < TF_FFN_TILE; j++)
                        sum += ws_tile[tok * TF_FFN_TILE + j] * block.ffn2_weight[d * TF_FFN_DIM + tile_start + j];
                    buf_x[i] += sum;
                }
                __syncthreads();
            }
        }

        tf_add_bias(buf_x, block.ffn2_bias, T, D);
        for (int k = 0; k < 32; k++) buf_x[tid + k * 256] += res[k];
        __syncthreads();
    }

    // === 4. Policy head ===
    tf_layer_norm_f16out(buf_x, buf_out, weights->p_ln.weight, weights->p_ln.bias, T, D, reduce);

    for (int i = tid; i < NN_POLICY_SIZE; i += blockDim.x) {
        int tok = i / NN_POLICY_PLANES;
        int plane = i % NN_POLICY_PLANES;
        float sum = 0.0f;
        for (int d = 0; d < D; d++)
            sum += __half2float(buf_out[tok * D + d]) * weights->p_head_weight[plane * D + d];
        policy_out[i] = sum + weights->p_head_bias[plane];
    }
    __syncthreads();

    block_log_softmax(policy_out, NN_POLICY_SIZE, reduce);

    // === 5. Value head ===
    tf_layer_norm_f16out(buf_x, buf_out, weights->v_ln.weight, weights->v_ln.bias, T, D, reduce);

    float* avg_pool = reduce;
    for (int d = tid; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (int tok = 0; tok < T; tok++) sum += __half2float(buf_out[tok * D + d]);
        avg_pool[d] = sum / (float)T;
    }
    __syncthreads();

    float* fc1_out = reduce;
    for (int j = tid; j < TF_VALUE_FC1_OUT; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < D; i++)
            sum += weights->v_fc1_weight[j * (D + 1) + i] * avg_pool[i];
        sum += weights->v_fc1_weight[j * (D + 1) + D] * q_result;
        sum += weights->v_fc1_bias[j];
        fc1_out[j] = sum > 0.0f ? sum : 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < TF_VALUE_FC1_OUT; j += blockDim.x)
        local_sum += weights->v_fc2_weight[j] * fc1_out[j];
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float v_logit = reduce[0] + weights->v_fc2_bias[0];
        float k = 0.47f * logf(1.0f + expf(weights->k_logit));
        *value_out = tanhf(v_logit + k * q_result);
        *k_out = k;
    }
    __syncthreads();
}
