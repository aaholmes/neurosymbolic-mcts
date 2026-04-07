#include "nn_weights.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

OracleNetWeights* init_nn_weights_zeros() {
    OracleNetWeights* d_weights = nullptr;
    cudaError_t err = cudaMalloc(&d_weights, sizeof(OracleNetWeights));
    if (err != cudaSuccess) {
        printf("Failed to allocate NN weights: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaMemset(d_weights, 0, sizeof(OracleNetWeights));
    if (err != cudaSuccess) {
        printf("Failed to zero NN weights: %s\n", cudaGetErrorString(err));
        cudaFree(d_weights);
        return nullptr;
    }

    // Set BatchNorm running_var to 1.0 (not 0.0) to avoid division by zero.
    // With zero weights: conv outputs are 0, BN normalizes to (-mean)/sqrt(var+eps) * gamma + beta.
    // With gamma=0, beta=0, mean=0, var=1: output = 0. Clean zeros throughout.
    // But we need var != 0 for the BN formula to not produce NaN.
    OracleNetWeights h_weights;
    memset(&h_weights, 0, sizeof(h_weights));

    // Set all BN running_var to 1.0
    for (int i = 0; i < NN_HIDDEN_DIM; i++) {
        h_weights.start_bn.running_var[i] = 1.0f;
    }
    for (int b = 0; b < NN_NUM_BLOCKS; b++) {
        for (int i = 0; i < NN_HIDDEN_DIM; i++) {
            h_weights.blocks[b].bn1.running_var[i] = 1.0f;
            h_weights.blocks[b].bn2.running_var[i] = 1.0f;
        }
    }
    for (int i = 0; i < NN_HIDDEN_DIM; i++) {
        h_weights.p_bn.running_var[i] = 1.0f;
    }
    h_weights.v_bn.running_var[0] = 1.0f;

    cudaMemcpy(d_weights, &h_weights, sizeof(OracleNetWeights), cudaMemcpyHostToDevice);
    return d_weights;
}

OracleNetWeights* load_nn_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open weights file: %s\n", path);
        return nullptr;
    }

    // Check file size matches struct
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if ((size_t)file_size != sizeof(OracleNetWeights)) {
        printf("Weight file size mismatch: expected %zu bytes, got %ld\n",
               sizeof(OracleNetWeights), file_size);
        fclose(f);
        return nullptr;
    }

    // Read into host buffer
    OracleNetWeights* h_weights = (OracleNetWeights*)malloc(sizeof(OracleNetWeights));
    if (!h_weights) {
        fclose(f);
        return nullptr;
    }

    size_t read = fread(h_weights, 1, sizeof(OracleNetWeights), f);
    fclose(f);
    if (read != sizeof(OracleNetWeights)) {
        printf("Failed to read weights: got %zu of %zu bytes\n", read, sizeof(OracleNetWeights));
        free(h_weights);
        return nullptr;
    }

    // Copy to GPU
    OracleNetWeights* d_weights = nullptr;
    cudaError_t err = cudaMalloc(&d_weights, sizeof(OracleNetWeights));
    if (err != cudaSuccess) {
        free(h_weights);
        return nullptr;
    }
    cudaMemcpy(d_weights, h_weights, sizeof(OracleNetWeights), cudaMemcpyHostToDevice);
    free(h_weights);

    printf("NN weights loaded: %zu bytes (%.1f MB)\n",
           sizeof(OracleNetWeights), sizeof(OracleNetWeights) / (1024.0 * 1024.0));
    return d_weights;
}

void free_nn_weights(OracleNetWeights* d_weights) {
    if (d_weights) cudaFree(d_weights);
}

bool update_nn_weights(OracleNetWeights* d_weights, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if ((size_t)file_size != sizeof(OracleNetWeights)) {
        fclose(f);
        return false;
    }

    OracleNetWeights* h_weights = (OracleNetWeights*)malloc(sizeof(OracleNetWeights));
    size_t read = fread(h_weights, 1, sizeof(OracleNetWeights), f);
    fclose(f);
    if (read != sizeof(OracleNetWeights)) {
        free(h_weights);
        return false;
    }

    cudaMemcpy(d_weights, h_weights, sizeof(OracleNetWeights), cudaMemcpyHostToDevice);
    free(h_weights);
    return true;
}

size_t nn_weights_size() {
    return sizeof(OracleNetWeights);
}

// ============================================================
// FP16 weight conversion for Tensor Core conv3x3
// ============================================================

// Guard against struct layout changes (must match export_weights_cuda.py)
static_assert(sizeof(OracleNetWeights) == 7930688,
              "OracleNetWeights size changed — update weight export and this assert");

__global__ void kernel_fp32_to_fp16(const float* src, half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

static void convert_array(const float* d_src, half* d_dst, int count) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_fp32_to_fp16<<<blocks, threads>>>(d_src, d_dst, count);
}

ConvWeightsHalf* convert_weights_to_half(const OracleNetWeights* d_weights) {
    ConvWeightsHalf h_hw;  // host copy with device pointers

    // start_conv: [128, 17, 9]
    int start_n = NN_HIDDEN_DIM * NN_INPUT_CHANNELS * 9;
    cudaMalloc(&h_hw.start_conv, start_n * sizeof(half));
    const float* start_ptr = (const float*)d_weights +
        offsetof(OracleNetWeights, start_conv_weight) / sizeof(float);
    convert_array(start_ptr, h_hw.start_conv, start_n);

    // block conv weights: [128, 128, 9] each
    int block_n = NN_HIDDEN_DIM * NN_HIDDEN_DIM * 9;
    for (int b = 0; b < NN_NUM_BLOCKS; b++) {
        cudaMalloc(&h_hw.block_conv1[b], block_n * sizeof(half));
        const float* c1 = (const float*)d_weights +
            (offsetof(OracleNetWeights, blocks) +
             b * sizeof(ResBlockParams) +
             offsetof(ResBlockParams, conv1_weight)) / sizeof(float);
        convert_array(c1, h_hw.block_conv1[b], block_n);

        cudaMalloc(&h_hw.block_conv2[b], block_n * sizeof(half));
        const float* c2 = (const float*)d_weights +
            (offsetof(OracleNetWeights, blocks) +
             b * sizeof(ResBlockParams) +
             offsetof(ResBlockParams, conv2_weight)) / sizeof(float);
        convert_array(c2, h_hw.block_conv2[b], block_n);
    }

    // policy conv: [128, 128, 9]
    cudaMalloc(&h_hw.p_conv, block_n * sizeof(half));
    const float* p_ptr = (const float*)d_weights +
        offsetof(OracleNetWeights, p_conv_weight) / sizeof(float);
    convert_array(p_ptr, h_hw.p_conv, block_n);

    cudaDeviceSynchronize();

    // Copy the struct (which contains device pointers) to device memory
    ConvWeightsHalf* d_hw = nullptr;
    cudaMalloc(&d_hw, sizeof(ConvWeightsHalf));
    cudaMemcpy(d_hw, &h_hw, sizeof(ConvWeightsHalf), cudaMemcpyHostToDevice);
    return d_hw;
}

void free_half_weights(ConvWeightsHalf* d_hw) {
    if (!d_hw) return;
    // Copy struct back to host to read the device pointers
    ConvWeightsHalf h_hw;
    cudaMemcpy(&h_hw, d_hw, sizeof(ConvWeightsHalf), cudaMemcpyDeviceToHost);
    cudaFree(h_hw.start_conv);
    for (int b = 0; b < NN_NUM_BLOCKS; b++) {
        cudaFree(h_hw.block_conv1[b]);
        cudaFree(h_hw.block_conv2[b]);
    }
    cudaFree(h_hw.p_conv);
    cudaFree(d_hw);
}
