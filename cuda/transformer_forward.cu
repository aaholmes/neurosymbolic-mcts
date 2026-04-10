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
    // Note: TC (Tensor Core) path is disabled — staging memory in workspace overlaps
    // with workspace I/O buffers. Needs dedicated staging region to fix.

    const int D = NN_HIDDEN_DIM;  // 128
    const int T = TF_NUM_TOKENS;  // 64

    // === 1. Board encoding → tokens[64, 17] in workspace (temporary) ===
    // Workspace has 6144 floats = 24 KB; tokens need 64×17 = 1088 floats
    float* tokens = workspace;
    tf_board_to_tokens(bs, tokens);

    // === 2. Input projection: tokens[64, 17] × W^T → buf_out[64, 128] ===
    // Always scalar: TC staging in workspace would corrupt tokens also in workspace.
    for (int i = tid; i < T * D; i += blockDim.x) {
        int tok = i / D; int d = i % D;
        float sum = 0.0f;
        for (int c = 0; c < NN_INPUT_CHANNELS; c++)
            sum += tokens[tok * NN_INPUT_CHANNELS + c] * weights->input_proj_weight[d * NN_INPUT_CHANNELS + c];
        buf_out[i] = sum + weights->input_proj_bias[d];
    }
    __syncthreads();

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

        // Workspace: ws_q[64,32] at +0, ws_k[64,32] at +2048
        // ws_attn[64,64] at +0 (reuses Q/K), ws_v[64,32] at +4096
        float* ws_q    = workspace;
        float* ws_k    = workspace + T * TF_HEAD_DIM;
        float* ws_attn = workspace;
        float* ws_v    = workspace + T * T;  // +4096

        for (int h = 0; h < TF_NUM_HEADS; h++) {
            int qkv_offset = h * TF_HEAD_DIM;

            {
                // Scalar attention: Q/K/V projections + QK^T + attn×V
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += buf_out[tok * D + k] * block.qkv_weight[(qkv_offset + d) * D + k];
                    ws_q[i] = sum + block.qkv_bias[qkv_offset + d];
                }
                __syncthreads();

                int k_offset = D + qkv_offset;
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += buf_out[tok * D + k] * block.qkv_weight[(k_offset + d) * D + k];
                    ws_k[i] = sum + block.qkv_bias[k_offset + d];
                }
                __syncthreads();

                float scale = rsqrtf((float)TF_HEAD_DIM);
                // QK^T — local_results pattern to avoid ws_q/ws_attn aliasing race
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

                int v_offset = 2 * D + qkv_offset;
                for (int i = tid; i < T * TF_HEAD_DIM; i += blockDim.x) {
                    int tok = i / TF_HEAD_DIM; int d = i % TF_HEAD_DIM;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += buf_out[tok * D + k] * block.qkv_weight[(v_offset + d) * D + k];
                    ws_v[i] = sum + block.qkv_bias[v_offset + d];
                }
                __syncthreads();

                // attn×V: read ws_attn and ws_v, write to ws_q (safe: ws_q at workspace+0,
                // ws_attn also at workspace+0 but attn is read per-row before ws_q overwrite)
                // Actually ws_q and ws_attn alias too. Need a truly separate destination.
                // Use buf_x temporarily (it was zeroed for accumulation, head outputs will be
                // projected and added back to buf_x for each head)
                // For this head, compute attn×V into ws_q (workspace+0) which is safe because
                // ws_attn spans workspace[0:4096] but each thread reads one full row then writes.
                // Wait — multiple threads write to ws_q[0..2047] while others read ws_attn[0..4095].
                // ws_q[i] = workspace[i], ws_attn[tok*64 + k] = workspace[tok*64 + k].
                // Thread writing ws_q[0] affects ws_attn[0,0]. Thread reading ws_attn[1,0] reads
                // workspace[64] which is unaffected. But thread reading ws_attn[0, k] for k>0
                // reads workspace[k] which might be overwritten by ws_q[k].
                //
                // SAFE APPROACH: compute into a temp on the stack per thread, then write.
                // Each thread handles ~8 elements (2048/256). Accumulate into local array.
                {
                    // Each thread computes its assigned elements of head_out[64, 32]
                    // Store results to ws_q (which is safe after attn is fully consumed per-element)
                    float local_results[8];  // max 2048/256 = 8
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
                    __syncthreads();  // ensure all reads from ws_attn/ws_v are done
                    // Now safe to write to workspace
                    for (int e = 0; e < n_elems; e++) {
                        int i = tid + e * blockDim.x;
                        if (i >= T * TF_HEAD_DIM) break;
                        ws_q[i] = local_results[e];  // ws_q = workspace+0, safe now
                    }
                    __syncthreads();
                }
            }

            // Output projection: ws_q[64,32] × out_proj → buf_x[64,128] (accumulate)
            {
                for (int i = tid; i < T * D; i += blockDim.x) {
                    int tok = i / D; int d = i % D;
                    float sum = 0.0f;
                    for (int j = 0; j < TF_HEAD_DIM; j++)
                        sum += ws_q[tok * TF_HEAD_DIM + j] * block.out_proj_weight[d * D + h * TF_HEAD_DIM + j];
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

            {
                for (int i = tid; i < T * TF_FFN_TILE; i += blockDim.x) {
                    int tok = i / TF_FFN_TILE; int d = i % TF_FFN_TILE;
                    float sum = 0.0f;
                    for (int k = 0; k < D; k++)
                        sum += buf_out[tok * D + k] * block.ffn1_weight[(tile_start + d) * D + k];
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
