#include "transformer_forward.cuh"
#include "transformer_ops.cuh"
#include "block_ops.cuh"  // for block_log_softmax

int transformer_smem_bytes() { return TF_SMEM_BYTES; }

// ============================================================
// Transformer forward pass — 256-thread block cooperative
//
// Data flow:
//   board → tokens[64, 17]
//   input_proj: tokens × W_proj → buf_out[64, 128]
//   buf_x = buf_out + pos_embedding
//   6× transformer block:
//     LN(buf_x) → buf_out
//     multi_head_attention(buf_out) → buf_out
//     buf_x += buf_out
//     LN(buf_x) → buf_out
//     FFN(buf_out) → buf_out
//     buf_x += buf_out
//   Policy: LN → per-token linear → flatten → log_softmax
//   Value:  LN → avg pool → FC → tanh(v + k*q)
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

    float* buf_x     = smem + TF_BUF_X_OFFSET;       // [64, 128] residual/input
    float* buf_out   = smem + TF_BUF_OUT_OFFSET;      // [64, 128] output
    float* workspace = smem + TF_WORKSPACE_OFFSET;     // [6144] attention/FFN/GEMM
    float* reduce    = smem + TF_REDUCE_OFFSET;        // [256] LayerNorm/softmax
    half*  ws_half   = (half*)workspace;               // workspace as half* for GEMM staging

    const int D = NN_HIDDEN_DIM;  // 128
    const int T = TF_NUM_TOKENS;  // 64

    // === 1. Board encoding → tokens[64, 17] in workspace (temporary) ===
    // Workspace has 6144 floats = 24 KB; tokens need 64×17 = 1088 floats
    float* tokens = workspace;
    tf_board_to_tokens(bs, tokens);

    // === 2. Input projection: tokens[64, 17] × W^T → buf_out[64, 128] ===
    if (half_w) {
        tf_linear(tokens, half_w->input_proj, buf_out, weights->input_proj_bias,
                  ws_half, T, D, NN_INPUT_CHANNELS);
    } else {
        for (int i = tid; i < T * D; i += blockDim.x) {
            int tok = i / D; int d = i % D;
            float sum = 0.0f;
            for (int c = 0; c < NN_INPUT_CHANNELS; c++)
                sum += tokens[tok * NN_INPUT_CHANNELS + c] * weights->input_proj_weight[d * NN_INPUT_CHANNELS + c];
            buf_out[i] = sum + weights->input_proj_bias[d];
        }
        __syncthreads();
    }

    // Add positional embeddings: buf_x = buf_out + pos_embedding
    for (int i = tid; i < T * D; i += blockDim.x)
        buf_x[i] = buf_out[i] + weights->pos_embedding[i];
    __syncthreads();

    // === 3. Transformer blocks ===
    for (int blk = 0; blk < TF_NUM_LAYERS; blk++) {
        const TransformerBlock& block = weights->blocks[blk];

        // --- Multi-Head Self-Attention ---
        // LayerNorm(buf_x) → buf_out (buf_x preserved for residual)
        tf_layer_norm(buf_x, buf_out, block.ln1.weight, block.ln1.bias, T, D, reduce);

        // Compute attention output into workspace, then project to buf_out
        // We'll accumulate the projected head outputs directly into a temp area,
        // then overwrite buf_out.

        // First, zero an accumulator for the multi-head output
        // We'll use the first T*D floats of workspace as attn_accum[64, 128]
        // But workspace is also used for Q/K/V staging... need to be careful.
        //
        // Actually, we can accumulate directly into buf_x's position since we
        // already copied LN'd data to buf_out. But buf_x holds the residual.
        //
        // Strategy: process heads sequentially, accumulate output projection
        // results into a temp buffer, then copy to buf_out, then residual add.
        //
        // The workspace layout for attention:
        //   ws[0:2048]     = Q_h[64, 32] (8 KB)
        //   ws[2048:4096]  = K_h[64, 32] (8 KB)
        //   ws[0:4096]     = attn[64, 64] (16 KB, reuses Q/K after they're consumed)
        //   ws[4096:6144]  = V_h[64, 32] (8 KB)
        //   ws[4096+2048..] = head_out[64, 32] (8 KB, after V consumed for attn×V)
        //
        // After attention, we need to project head_out[64, 32] × W_o[32, 128] and
        // accumulate into some output buffer. Let's use buf_out for accumulation
        // (we already have LN'd input there, but we'll overwrite it per-head).

        // Zero buf_out for accumulating projected heads (attn output replaces LN'd input)
        // But we need LN'd input as source for Q/K/V projections!
        // Solution: save LN'd input to a temp, or compute Q/K/V directly from buf_out
        // before zeroing it.
        //
        // Better approach:
        // 1. buf_out has LN(buf_x)
        // 2. For each head, compute Q/K/V from buf_out (read-only)
        // 3. Compute attention, project, accumulate into workspace[0:T*D]
        // 4. After all heads: copy workspace[0:T*D] to buf_out
        // But workspace is only 6144 floats, and T*D = 8192. Doesn't fit!
        //
        // Alternative: accumulate projected head outputs directly into buf_out.
        // But buf_out holds the LN'd input needed for Q/K/V projection of later heads.
        //
        // Cleanest solution: compute all Q/K/V projections from buf_out (LN'd input),
        // and accumulate the output projection into buf_x temporarily (it'll be
        // overwritten but we need it for residual... no, residual is buf_x += attn_output).
        //
        // Actually: the output projection accumulation can happen in buf_out itself
        // if we process heads carefully. For head 0, overwrite buf_out with projected
        // output. For heads 1-3, ADD to buf_out. But head 1 still needs to read
        // LN'd data from buf_out for Q/K/V... which we just overwrote.
        //
        // Real solution: use the fused QKV projection. Compute Q[64,128], K[64,128],
        // V[64,128] all at once using the fused QKV weight [128, 384]. But that
        // needs 3 × 8192 = 24576 floats = 96 KB. Way too much.
        //
        // Practical solution: compute one head at a time. For each head:
        // 1. Q_h = buf_out × W_q_h (scalar, small: [64,128]×[128,32])
        // 2. K_h = buf_out × W_k_h
        // 3. attn = Q_h × K_h^T, softmax
        // 4. V_h = buf_out × W_v_h
        // 5. head_out = attn × V_h
        // 6. For head 0: zero a section of buf_x (we'll use residual_save pattern)
        //
        // Wait — I'm overcomplicating this. Let me use a different strategy:
        // Save residual (buf_x) in registers (32 per thread, same as SE-ResNet),
        // then we have buf_x free for accumulation.

        // Save buf_x to registers for residual (same pattern as SE-ResNet)
        float res[32];
        for (int k = 0; k < 32; k++) res[k] = buf_x[tid + k * 256];

        // Now buf_x is free. Use it for attention output accumulation.
        // Zero buf_x for accumulating projected head outputs.
        for (int i = tid; i < T * D; i += blockDim.x) buf_x[i] = 0.0f;
        __syncthreads();

        // buf_out = LN(original buf_x), still valid for Q/K/V source
        // buf_x = accumulator for attention output (zeroed)

        float* ws_q    = workspace;           // [64, 32] = 2048 floats
        float* ws_k    = workspace + 2048;    // [64, 32] = 2048 floats
        float* ws_attn = workspace;           // [64, 64] = 4096 floats (reuses Q/K)
        float* ws_v    = workspace + 4096;    // [64, 32] = 2048 floats
        float* ws_head = workspace + 4096;    // [64, 32] = 2048 floats (reuses V after attn×V)

        for (int h = 0; h < TF_NUM_HEADS; h++) {
            int qkv_offset = h * TF_HEAD_DIM;  // column offset within Q/K/V sections

            // Q_h[64, 32] = buf_out[64, 128] × W_q_h^T
            if (half_w) {
                tf_linear(buf_out, half_w->blocks[blk].q_head[h], ws_q,
                          block.qkv_bias + qkv_offset, ws_half, T, TF_HEAD_DIM, D);
            } else {
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += buf_out[tok * D + k] * block.qkv_weight[k * (3 * D) + qkv_offset + d];
                    ws_q[i] = sum + block.qkv_bias[qkv_offset + d];
                }
                __syncthreads();
            }

            // K_h[64, 32]
            int k_offset = D + qkv_offset;
            if (half_w) {
                tf_linear(buf_out, half_w->blocks[blk].k_head[h], ws_k,
                          block.qkv_bias + k_offset, ws_half, T, TF_HEAD_DIM, D);
            } else {
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += buf_out[tok * D + k] * block.qkv_weight[k * (3 * D) + k_offset + d];
                    ws_k[i] = sum + block.qkv_bias[k_offset + d];
                }
                __syncthreads();
            }

            // attn[64, 64] = Q_h × K_h^T / sqrt(head_dim)
            float scale = rsqrtf((float)TF_HEAD_DIM);
            if (half_w) {
                tf_gemm_smem_abt(ws_q, ws_k, ws_attn, ws_half, T, T, TF_HEAD_DIM, scale);
            } else {
                for (int i = tid; i < T * T; i += blockDim.x) {
                    int row = i / T; int col = i % T;
                    float sum = 0.0f;
                    for (int d = 0; d < TF_HEAD_DIM; d++)
                        sum += ws_q[row * TF_HEAD_DIM + d] * ws_k[col * TF_HEAD_DIM + d];
                    ws_attn[i] = sum * scale;
                }
                __syncthreads();
            }

            // Softmax per row
            tf_softmax_rows(ws_attn, T, T, reduce);

            // V_h[64, 32]
            int v_offset = 2 * D + qkv_offset;
            if (half_w) {
                tf_linear(buf_out, half_w->blocks[blk].v_head[h], ws_v,
                          block.qkv_bias + v_offset, ws_half, T, TF_HEAD_DIM, D);
            } else {
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += buf_out[tok * D + k] * block.qkv_weight[k * (3 * D) + v_offset + d];
                    ws_v[i] = sum + block.qkv_bias[v_offset + d];
                }
                __syncthreads();
            }

            // head_out[64, 32] = attn[64, 64] × V_h[64, 32]
            if (half_w) {
                tf_gemm_smem(ws_attn, ws_v, ws_head, ws_half, T, TF_HEAD_DIM, T);
            } else {
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < T; k++)
                        sum += ws_attn[tok * T + k] * ws_v[k * TF_HEAD_DIM + d];
                    ws_head[i] = sum;
                }
                __syncthreads();
            }

            // Output projection: buf_x += head_out[64, 32] × W_o_h^T[32, 128]
            // W_o rows [h*32:(h+1)*32] of [128, 128] = contiguous [32, 128]
            if (half_w) {
                // out_proj_weight is [128, 128], per-head slice at row h*32 = [32, 128]
                // out_proj FP16 pointer + h*32*128 halves
                tf_linear(ws_head, half_w->blocks[blk].out_proj + h * TF_HEAD_DIM * D,
                          buf_x, nullptr, ws_half, T, D, TF_HEAD_DIM, true);
            } else {
                for (int i = tid; i < T * D; i += blockDim.x) {
                    int tok = i / D; int d = i % D;
                    float sum = 0.0f;
                    for (int j = 0; j < TF_HEAD_DIM; j++)
                        sum += ws_head[tok * TF_HEAD_DIM + j] * block.out_proj_weight[(h * TF_HEAD_DIM + j) * D + d];
                    buf_x[i] += sum;
                }
                __syncthreads();
            }
        }

        // Add output projection bias
        tf_add_bias(buf_x, block.out_proj_bias, T, D);

        // Residual add: buf_x += saved residual
        for (int k = 0; k < 32; k++) buf_x[tid + k * 256] += res[k];
        __syncthreads();

        // --- FFN ---
        // Save residual
        for (int k = 0; k < 32; k++) res[k] = buf_x[tid + k * 256];

        // LayerNorm(buf_x) → buf_out
        tf_layer_norm(buf_x, buf_out, block.ln2.weight, block.ln2.bias, T, D, reduce);

        // FFN: buf_x = W2 × ReLU(W1 × buf_out + b1) + b2
        // Tiled: process FFN_DIM in tiles of TF_FFN_TILE=64
        for (int i = tid; i < T * D; i += blockDim.x) buf_x[i] = 0.0f;
        __syncthreads();

        float* ws_tile = workspace;  // [64, 64] = 4096 floats

        for (int tile = 0; tile < TF_FFN_DIM / TF_FFN_TILE; tile++) {
            int tile_start = tile * TF_FFN_TILE;

            if (half_w) {
                // FFN1 tile: ws_tile[64,64] = buf_out[64,128] × W1_tile[64,128]^T + bias
                tf_linear(buf_out, half_w->blocks[blk].ffn1_tile[tile], ws_tile,
                          block.ffn1_bias + tile_start, ws_half, T, TF_FFN_TILE, D);
                tf_relu(ws_tile, T * TF_FFN_TILE);
                // FFN2 tile: buf_x += ws_tile[64,64] × W2_tile[128,64]^T
                tf_linear(ws_tile, half_w->blocks[blk].ffn2_tile[tile], buf_x,
                          nullptr, ws_half, T, D, TF_FFN_TILE, true);
            } else {
                for (int i = tid; i < T * TF_FFN_TILE; i += blockDim.x) {
                    int tok = i / TF_FFN_TILE; int d = i % TF_FFN_TILE;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += buf_out[tok * D + k] * block.ffn1_weight[k * TF_FFN_DIM + tile_start + d];
                    float val = sum + block.ffn1_bias[tile_start + d];
                    ws_tile[i] = val > 0.0f ? val : 0.0f;
                }
                __syncthreads();

                for (int i = tid; i < T * D; i += blockDim.x) {
                    int tok = i / D; int d = i % D;
                    float sum = 0.0f;
                    for (int j = 0; j < TF_FFN_TILE; j++)
                        sum += ws_tile[tok * TF_FFN_TILE + j] * block.ffn2_weight[(tile_start + j) * D + d];
                    buf_x[i] += sum;
                }
                __syncthreads();
            }
        }

        // Add FFN bias
        tf_add_bias(buf_x, block.ffn2_bias, T, D);

        // Residual add
        for (int k = 0; k < 32; k++) buf_x[tid + k * 256] += res[k];
        __syncthreads();
    }

    // buf_x = final transformer output [64, 128]

    // === 4. Policy head ===
    // LayerNorm → per-token linear [128 → 73] → flatten to [4672] → log_softmax
    tf_layer_norm(buf_x, buf_out, weights->p_ln.weight, weights->p_ln.bias, T, D, reduce);

    // Per-token linear: policy_out = buf_out × W_p^T + bias
    // policy_out is [64, 73] flattened to [4672] in global memory
    // tf_linear writes to shared memory, but policy_out is global. Use scalar for now.
    for (int i = tid; i < NN_POLICY_SIZE; i += blockDim.x) {
        int tok = i / NN_POLICY_PLANES;
        int plane = i % NN_POLICY_PLANES;
        float sum = 0.0f;
        for (int d = 0; d < D; d++)
            sum += buf_out[tok * D + d] * weights->p_head_weight[plane * D + d];
        policy_out[i] = sum + weights->p_head_bias[plane];
    }
    __syncthreads();

    // Log-softmax on policy (global memory)
    block_log_softmax(policy_out, NN_POLICY_SIZE, reduce);

    // === 5. Value head ===
    // LayerNorm → global avg pool → concat q_result → FC(129→256) → ReLU → FC(256→1)
    tf_layer_norm(buf_x, buf_out, weights->v_ln.weight, weights->v_ln.bias, T, D, reduce);

    // Global average pool: avg_pool[d] = mean over 64 tokens of buf_out[:, d]
    // Store in reduce[0:128] (reuse reduce buffer, which is 256 floats)
    float* avg_pool = reduce;
    for (int d = tid; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (int tok = 0; tok < T; tok++) sum += buf_out[tok * D + d];
        avg_pool[d] = sum / (float)T;
    }
    __syncthreads();

    // FC1: [256, 129] × [avg_pool[128]; q_result] → reduce[0:256]
    // Reuse reduce for FC1 output (avg_pool already consumed)
    float* fc1_out = reduce;
    for (int j = tid; j < TF_VALUE_FC1_OUT; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < D; i++)
            sum += weights->v_fc1_weight[j * (D + 1) + i] * avg_pool[i];
        sum += weights->v_fc1_weight[j * (D + 1) + D] * q_result;
        sum += weights->v_fc1_bias[j];
        fc1_out[j] = sum > 0.0f ? sum : 0.0f;  // ReLU
    }
    __syncthreads();

    // FC2: [1, 256] × fc1_out → v_logit
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
