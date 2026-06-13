// Standalone debug test: isolate exactly which head count breaks attention
#include "../transformer_weights.cuh"
#include "../transformer_ops.cuh"
#include "../transformer_forward.cuh"
#include "../movegen.cuh"
#include "test_helpers.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>

__global__ void kernel_attn_nheads(
    const float* input, const TransformerWeights* weights,
    float* output, int num_heads_to_run
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    const int T = 64, D = 128, HD = 32;

    float* buf_x = smem;
    float* buf_out = smem + T * D;
    float* workspace = smem + 2 * T * D;
    float* reduce = smem + 2 * T * D + 6144;

    for (int i = tid; i < T * D; i += blockDim.x) buf_x[i] = input[i];
    __syncthreads();

    const TransformerBlock& block = weights->blocks[0];
    tf_layer_norm(buf_x, buf_out, block.ln1.weight, block.ln1.bias, T, D, reduce);

    float res[32];
    for (int k = 0; k < 32; k++) res[k] = buf_x[tid + k * 256];
    for (int i = tid; i < T * D; i += blockDim.x) buf_x[i] = 0.0f;
    __syncthreads();

    float* ws_q = workspace;
    float* ws_k = workspace + T * HD;
    float* ws_attn = workspace;
    float* ws_v = workspace + T * T;

    for (int h = 0; h < num_heads_to_run; h++) {
        int qkv_offset = h * HD;

        // Q
        for (int i = tid; i < T * HD; i += blockDim.x) {
            int tok = i / HD; int d = i % HD;
            float sum = 0.0f;
            for (int k = 0; k < D; k++)
                sum += buf_out[tok * D + k] * block.qkv_weight[(qkv_offset + d) * D + k];
            ws_q[i] = sum + block.qkv_bias[qkv_offset + d];
        }
        __syncthreads();

        // K
        int k_off = D + qkv_offset;
        for (int i = tid; i < T * HD; i += blockDim.x) {
            int tok = i / HD; int d = i % HD;
            float sum = 0.0f;
            for (int k = 0; k < D; k++)
                sum += buf_out[tok * D + k] * block.qkv_weight[(k_off + d) * D + k];
            ws_k[i] = sum + block.qkv_bias[k_off + d];
        }
        __syncthreads();

        // QK^T — compute into local storage first to avoid ws_q/ws_attn aliasing race
        float scale = rsqrtf((float)HD);
        {
            float local_attn[16];  // T*T/256 = 4096/256 = 16 per thread
            int n_elems = (T * T + 255) / 256;
            for (int e = 0; e < n_elems; e++) {
                int i = tid + e * 256;
                if (i >= T * T) break;
                int row = i / T; int col = i % T;
                float sum = 0.0f;
                for (int d = 0; d < HD; d++)
                    sum += ws_q[row * HD + d] * ws_k[col * HD + d];
                local_attn[e] = sum * scale;
            }
            __syncthreads();  // all reads from ws_q/ws_k done
            for (int e = 0; e < n_elems; e++) {
                int i = tid + e * 256;
                if (i >= T * T) break;
                ws_attn[i] = local_attn[e];
            }
            __syncthreads();
        }

        // Softmax
        tf_softmax_rows(ws_attn, T, T, reduce);

        // V
        int v_off = 2 * D + qkv_offset;
        for (int i = tid; i < T * HD; i += blockDim.x) {
            int tok = i / HD; int d = i % HD;
            float sum = 0.0f;
            for (int k = 0; k < D; k++)
                sum += buf_out[tok * D + k] * block.qkv_weight[(v_off + d) * D + k];
            ws_v[i] = sum + block.qkv_bias[v_off + d];
        }
        __syncthreads();

        // attn×V with race fix
        {
            float local_results[8];
            int n_elems = (T * HD + 255) / 256;
            for (int e = 0; e < n_elems; e++) {
                int i = tid + e * 256;
                if (i >= T * HD) break;
                int tok = i / HD; int d = i % HD;
                float sum = 0.0f;
                for (int k = 0; k < T; k++)
                    sum += ws_attn[tok * T + k] * ws_v[k * HD + d];
                local_results[e] = sum;
            }
            __syncthreads();
            for (int e = 0; e < n_elems; e++) {
                int i = tid + e * 256;
                if (i >= T * HD) break;
                ws_q[i] = local_results[e];
            }
            __syncthreads();
        }

        // Output proj
        for (int i = tid; i < T * D; i += blockDim.x) {
            int tok = i / D; int d = i % D;
            float sum = 0.0f;
            for (int j = 0; j < HD; j++)
                sum += ws_q[tok * HD + j] * block.out_proj_weight[d * D + h * HD + j];
            buf_x[i] += sum;
        }
        __syncthreads();
    }

    tf_add_bias(buf_x, block.out_proj_bias, T, D);
    for (int k = 0; k < 32; k++) buf_x[tid + k * 256] += res[k];
    __syncthreads();

    for (int i = tid; i < T * D; i += blockDim.x) output[i] = buf_x[i];
}

int main() {
    init_movegen_tables();
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    TransformerWeights* d_w = load_transformer_weights("weights/transformer4/candidate_1.bin");
    if (!d_w) { printf("No weights\n"); return 1; }

    // Load input (PyTorch block0 input = h after pos_emb)
    float h_input[64*128];
    FILE* f = fopen("/tmp/pt_block0_input.bin", "rb");
    if (!f) { printf("No input ref\n"); return 1; }
    fread(h_input, 4, 64*128, f); fclose(f);

    float *d_input, *d_output;
    cudaMalloc(&d_input, 64*128*4);
    cudaMalloc(&d_output, 64*128*4);
    cudaMemcpy(d_input, h_input, 64*128*4, cudaMemcpyHostToDevice);

    int smem = TF_SMEM_BYTES;
    cudaFuncSetAttribute(kernel_attn_nheads, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    float h_output[64*128];

    for (int nh = 1; nh <= 4; nh++) {
        kernel_attn_nheads<<<1, 256, smem>>>(d_input, d_w, d_output, nh);
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, 64*128*4, cudaMemcpyDeviceToHost);

        // Save for Python comparison
        char path[64];
        snprintf(path, sizeof(path), "/tmp/cuda_attn_%dhead.bin", nh);
        FILE* fo = fopen(path, "wb");
        fwrite(h_output, 4, 64*128, fo);
        fclose(fo);

        printf("%d heads: out[0,:3]=[%.4f,%.4f,%.4f] out[4,:3]=[%.4f,%.4f,%.4f]\n",
               nh, h_output[0], h_output[1], h_output[2],
               h_output[4*128], h_output[4*128+1], h_output[4*128+2]);
    }

    cudaFree(d_input); cudaFree(d_output);
    free_transformer_weights(d_w);
    return 0;
}
