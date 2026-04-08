#include "transformer_weights.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

TransformerWeights* init_transformer_weights_zeros() {
    TransformerWeights* d_w = nullptr;
    cudaMalloc(&d_w, sizeof(TransformerWeights));
    cudaMemset(d_w, 0, sizeof(TransformerWeights));

    // Set LayerNorm weights to 1.0 (identity transform when bias=0)
    TransformerWeights h_w;
    memset(&h_w, 0, sizeof(h_w));
    for (int i = 0; i < NN_HIDDEN_DIM; i++) {
        for (int b = 0; b < TF_NUM_LAYERS; b++) {
            h_w.blocks[b].ln1.weight[i] = 1.0f;
            h_w.blocks[b].ln2.weight[i] = 1.0f;
        }
        h_w.p_ln.weight[i] = 1.0f;
        h_w.v_ln.weight[i] = 1.0f;
    }
    cudaMemcpy(d_w, &h_w, sizeof(TransformerWeights), cudaMemcpyHostToDevice);
    return d_w;
}

TransformerWeights* load_transformer_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Failed to open transformer weights: %s\n", path); return nullptr; }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if ((size_t)file_size != sizeof(TransformerWeights)) {
        printf("Transformer weight file size mismatch: expected %zu, got %ld\n",
               sizeof(TransformerWeights), file_size);
        fclose(f);
        return nullptr;
    }

    TransformerWeights* h_w = (TransformerWeights*)malloc(sizeof(TransformerWeights));
    fread(h_w, 1, sizeof(TransformerWeights), f);
    fclose(f);

    TransformerWeights* d_w = nullptr;
    cudaMalloc(&d_w, sizeof(TransformerWeights));
    cudaMemcpy(d_w, h_w, sizeof(TransformerWeights), cudaMemcpyHostToDevice);
    free(h_w);

    printf("Transformer weights loaded: %zu bytes (%.1f MB)\n",
           sizeof(TransformerWeights), sizeof(TransformerWeights) / (1024.0 * 1024.0));
    return d_w;
}

void free_transformer_weights(TransformerWeights* d_w) {
    if (d_w) cudaFree(d_w);
}

size_t transformer_weights_size() {
    return sizeof(TransformerWeights);
}

// ============================================================
// FP16 weight conversion
// ============================================================

__global__ void kernel_fp32_to_fp16_tf(const float* src, half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

static void convert_tf(const float* d_src, half* d_dst, int count) {
    kernel_fp32_to_fp16_tf<<<(count + 255) / 256, 256>>>(d_src, d_dst, count);
}

// Extract per-head QKV slice: dst[head_dim, d_model] from src[d_model, 3*d_model]
// dst[d * d_model + k] = __float2half(src[k * (3*d_model) + col_offset + d])
__global__ void kernel_extract_qkv_head(
    const float* src, half* dst, int d_model, int head_dim, int col_offset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = head_dim * d_model;
    if (i >= total) return;
    int d = i / d_model;        // output row [0, head_dim)
    int k = i % d_model;        // output col [0, d_model)
    dst[i] = __float2half(src[k * (3 * d_model) + col_offset + d]);
}

TransformerWeightsHalf* convert_transformer_to_half(const TransformerWeights* d_w) {
    TransformerWeightsHalf h_half;

    // Input projection: [128, 17]
    int inp_n = NN_HIDDEN_DIM * NN_INPUT_CHANNELS;
    cudaMalloc(&h_half.input_proj, inp_n * sizeof(half));
    convert_tf((const float*)d_w + offsetof(TransformerWeights, input_proj_weight) / sizeof(float),
               h_half.input_proj, inp_n);

    // Per block
    for (int b = 0; b < TF_NUM_LAYERS; b++) {
        int qkv_n = NN_HIDDEN_DIM * 3 * NN_HIDDEN_DIM;
        cudaMalloc(&h_half.blocks[b].qkv, qkv_n * sizeof(half));
        convert_tf((const float*)d_w +
            (offsetof(TransformerWeights, blocks) + b * sizeof(TransformerBlock) +
             offsetof(TransformerBlock, qkv_weight)) / sizeof(float),
            h_half.blocks[b].qkv, qkv_n);

        // Extract per-head Q/K/V slices: [head_dim, d_model] = [32, 128] each
        const float* qkv_src = (const float*)d_w +
            (offsetof(TransformerWeights, blocks) + b * sizeof(TransformerBlock) +
             offsetof(TransformerBlock, qkv_weight)) / sizeof(float);
        int head_slice_n = TF_HEAD_DIM * NN_HIDDEN_DIM;  // 32 * 128 = 4096
        int head_blocks = (head_slice_n + 255) / 256;
        for (int h = 0; h < TF_NUM_HEADS; h++) {
            cudaMalloc(&h_half.blocks[b].q_head[h], head_slice_n * sizeof(half));
            kernel_extract_qkv_head<<<head_blocks, 256>>>(
                qkv_src, h_half.blocks[b].q_head[h], NN_HIDDEN_DIM, TF_HEAD_DIM, h * TF_HEAD_DIM);

            cudaMalloc(&h_half.blocks[b].k_head[h], head_slice_n * sizeof(half));
            kernel_extract_qkv_head<<<head_blocks, 256>>>(
                qkv_src, h_half.blocks[b].k_head[h], NN_HIDDEN_DIM, TF_HEAD_DIM, NN_HIDDEN_DIM + h * TF_HEAD_DIM);

            cudaMalloc(&h_half.blocks[b].v_head[h], head_slice_n * sizeof(half));
            kernel_extract_qkv_head<<<head_blocks, 256>>>(
                qkv_src, h_half.blocks[b].v_head[h], NN_HIDDEN_DIM, TF_HEAD_DIM, 2 * NN_HIDDEN_DIM + h * TF_HEAD_DIM);
        }

        int op_n = NN_HIDDEN_DIM * NN_HIDDEN_DIM;
        cudaMalloc(&h_half.blocks[b].out_proj, op_n * sizeof(half));
        convert_tf((const float*)d_w +
            (offsetof(TransformerWeights, blocks) + b * sizeof(TransformerBlock) +
             offsetof(TransformerBlock, out_proj_weight)) / sizeof(float),
            h_half.blocks[b].out_proj, op_n);

        int f1_n = NN_HIDDEN_DIM * TF_FFN_DIM;
        cudaMalloc(&h_half.blocks[b].ffn1, f1_n * sizeof(half));
        convert_tf((const float*)d_w +
            (offsetof(TransformerWeights, blocks) + b * sizeof(TransformerBlock) +
             offsetof(TransformerBlock, ffn1_weight)) / sizeof(float),
            h_half.blocks[b].ffn1, f1_n);

        int f2_n = TF_FFN_DIM * NN_HIDDEN_DIM;
        cudaMalloc(&h_half.blocks[b].ffn2, f2_n * sizeof(half));
        convert_tf((const float*)d_w +
            (offsetof(TransformerWeights, blocks) + b * sizeof(TransformerBlock) +
             offsetof(TransformerBlock, ffn2_weight)) / sizeof(float),
            h_half.blocks[b].ffn2, f2_n);
    }

    // Policy head: [73, 128]
    int ph_n = NN_POLICY_PLANES * NN_HIDDEN_DIM;
    cudaMalloc(&h_half.p_head, ph_n * sizeof(half));
    convert_tf((const float*)d_w + offsetof(TransformerWeights, p_head_weight) / sizeof(float),
               h_half.p_head, ph_n);

    cudaDeviceSynchronize();

    TransformerWeightsHalf* d_half = nullptr;
    cudaMalloc(&d_half, sizeof(TransformerWeightsHalf));
    cudaMemcpy(d_half, &h_half, sizeof(TransformerWeightsHalf), cudaMemcpyHostToDevice);
    return d_half;
}

void free_transformer_half(TransformerWeightsHalf* d_half) {
    if (!d_half) return;
    TransformerWeightsHalf h;
    cudaMemcpy(&h, d_half, sizeof(TransformerWeightsHalf), cudaMemcpyDeviceToHost);
    cudaFree(h.input_proj);
    cudaFree(h.p_head);
    for (int b = 0; b < TF_NUM_LAYERS; b++) {
        cudaFree(h.blocks[b].qkv);
        for (int hh = 0; hh < TF_NUM_HEADS; hh++) {
            cudaFree(h.blocks[b].q_head[hh]);
            cudaFree(h.blocks[b].k_head[hh]);
            cudaFree(h.blocks[b].v_head[hh]);
        }
        cudaFree(h.blocks[b].out_proj);
        cudaFree(h.blocks[b].ffn1);
        cudaFree(h.blocks[b].ffn2);
    }
    cudaFree(d_half);
}
