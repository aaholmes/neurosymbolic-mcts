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
