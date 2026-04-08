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
// Extract per-head QKV slice: dst[head_dim, d_model] from src[3*d_model, d_model]
// PyTorch nn.Linear(d_model, 3*d_model) stores weight as [3*d_model, d_model] (out, in)
// For head h: Q = rows [h*head_dim : (h+1)*head_dim], K = rows [d_model + h*hd : ...], etc.
// dst[d, k] = src[(col_offset + d) * d_model + k]
__global__ void kernel_extract_qkv_head(
    const float* src, half* dst, int d_model, int head_dim, int col_offset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = head_dim * d_model;
    if (i >= total) return;
    int d = i / d_model;        // output row [0, head_dim) = which output neuron
    int k = i % d_model;        // output col [0, d_model) = which input neuron
    dst[i] = __float2half(src[(col_offset + d) * d_model + k]);
}

// Extract FFN1 tile: dst[tile_size, d_model] from src[ffn_dim, d_model]
// PyTorch nn.Linear(d_model, ffn_dim) stores weight as [ffn_dim, d_model] (out, in)
// Tile t = rows [tile_start : tile_start + tile_size] of the [ffn_dim, d_model] matrix
// dst[n, k] = src[(tile_start + n) * d_model + k]
__global__ void kernel_extract_ffn1_tile(
    const float* src, half* dst, int d_model, int ffn_dim, int tile_start, int tile_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tile_size * d_model) return;
    int n = i / d_model;  // [0, tile_size) = output neuron within tile
    int k = i % d_model;  // [0, d_model) = input neuron
    dst[i] = __float2half(src[(tile_start + n) * d_model + k]);
}

// Extract FFN2 tile: dst[d_model, tile_size] from src[d_model, ffn_dim]
// PyTorch nn.Linear(ffn_dim, d_model) stores weight as [d_model, ffn_dim] (out, in)
// Tile t = columns [tile_start : tile_start + tile_size] of the [d_model, ffn_dim] matrix
// For tf_linear: dst should be [N=d_model, K=tile_size]
// dst[n, k] = src[n * ffn_dim + tile_start + k]
__global__ void kernel_extract_ffn2_tile(
    const float* src, half* dst, int d_model, int ffn_dim, int tile_start, int tile_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_model * tile_size) return;
    int n = i / tile_size;  // [0, d_model) = output neuron
    int k = i % tile_size;  // [0, tile_size) = input neuron within tile
    dst[i] = __float2half(src[n * ffn_dim + tile_start + k]);
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

            // Fused qkv_head[h]: [96, 128] = concat of q_head, k_head, v_head
            int fused_n = 3 * TF_HEAD_DIM * NN_HIDDEN_DIM;  // 96 * 128 = 12288
            cudaMalloc(&h_half.blocks[b].qkv_head[h], fused_n * sizeof(half));
            // Copy q_head[32,128] to offset 0, k_head to offset 32*128, v_head to offset 64*128
            cudaMemcpy(h_half.blocks[b].qkv_head[h],
                       h_half.blocks[b].q_head[h], head_slice_n * sizeof(half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(h_half.blocks[b].qkv_head[h] + head_slice_n,
                       h_half.blocks[b].k_head[h], head_slice_n * sizeof(half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(h_half.blocks[b].qkv_head[h] + 2 * head_slice_n,
                       h_half.blocks[b].v_head[h], head_slice_n * sizeof(half), cudaMemcpyDeviceToDevice);
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

        // Pre-split FFN tiles
        const float* ffn1_src = (const float*)d_w +
            (offsetof(TransformerWeights, blocks) + b * sizeof(TransformerBlock) +
             offsetof(TransformerBlock, ffn1_weight)) / sizeof(float);
        const float* ffn2_src = (const float*)d_w +
            (offsetof(TransformerWeights, blocks) + b * sizeof(TransformerBlock) +
             offsetof(TransformerBlock, ffn2_weight)) / sizeof(float);

        int num_tiles = TF_FFN_DIM / TF_FFN_TILE;
        for (int t = 0; t < num_tiles; t++) {
            // FFN1 tile: [TF_FFN_TILE, NN_HIDDEN_DIM] = [64, 128]
            int tile_n1 = TF_FFN_TILE * NN_HIDDEN_DIM;
            cudaMalloc(&h_half.blocks[b].ffn1_tile[t], tile_n1 * sizeof(half));
            kernel_extract_ffn1_tile<<<(tile_n1 + 255) / 256, 256>>>(
                ffn1_src, h_half.blocks[b].ffn1_tile[t],
                NN_HIDDEN_DIM, TF_FFN_DIM, t * TF_FFN_TILE, TF_FFN_TILE);

            // FFN2 tile: [NN_HIDDEN_DIM, TF_FFN_TILE] = [128, 64]
            int tile_n2 = NN_HIDDEN_DIM * TF_FFN_TILE;
            cudaMalloc(&h_half.blocks[b].ffn2_tile[t], tile_n2 * sizeof(half));
            kernel_extract_ffn2_tile<<<(tile_n2 + 255) / 256, 256>>>(
                ffn2_src, h_half.blocks[b].ffn2_tile[t],
                NN_HIDDEN_DIM, TF_FFN_DIM, t * TF_FFN_TILE, TF_FFN_TILE);
        }
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
            cudaFree(h.blocks[b].qkv_head[hh]);
        }
        cudaFree(h.blocks[b].out_proj);
        cudaFree(h.blocks[b].ffn1);
        cudaFree(h.blocks[b].ffn2);
        for (int t = 0; t < TF_FFN_DIM / TF_FFN_TILE; t++) {
            cudaFree(h.blocks[b].ffn1_tile[t]);
            cudaFree(h.blocks[b].ffn2_tile[t]);
        }
    }
    cudaFree(d_half);
}
