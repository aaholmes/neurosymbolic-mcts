#include "block_forward.cuh"
#include "block_ops.cuh"

int block_smem_bytes() { return BLOCK_SMEM_BYTES; }

// ============================================================
// Full SE-ResNet forward pass — 256-thread block cooperative
// ============================================================
//
// Buffer assignment (no pointer swap):
//   buf1 = smem + BLOCK_BUF1_OFFSET (smem+0)     — work buffer
//   buf2 = smem + BLOCK_BUF2_OFFSET (smem+8192)  — backbone buffer
//
// Data flow:
//   board_to_planes   -> buf1[0:1088]
//   start_conv(buf1)  -> buf2           (buf2 = backbone from here)
//   6x res blocks:
//     conv1(buf2) -> buf1               (work)
//     conv2(buf1) -> buf2               (residual saved in regs)
//     SE + add_relu -> buf2             (backbone output of block)
//   policy_conv(buf2) -> buf1
//   1x1+bias+permute(buf1) -> policy_out (global)
//   log_softmax(policy_out)
//   v_conv(buf2) -> buf1[0:64]          (v_feat, reuse work buf)
//   FC1 -> smem_reduce[0:256]
//   FC2 reduction -> v_logit (thread 0)

__device__ void oracle_net_forward_block(
    const BoardState* bs,
    float q_result,
    const OracleNetWeights* weights,
    float* smem,
    float* policy_out,
    float* value_out,
    float* k_out
) {
    int tid = threadIdx.x;

    float* buf1        = smem + BLOCK_BUF1_OFFSET;   // work buffer [128, 64]
    float* buf2        = smem + BLOCK_BUF2_OFFSET;   // backbone buffer [128, 64]
    float* smem_reduce = smem + BLOCK_REDUCE_OFFSET; // [256] multipurpose
    float* smem_weights = smem + BLOCK_WEIGHTS_OFFSET; // [1152] weight tile cache

    // SE workspace (within smem_reduce, reused each block)
    float* smem_se_avg = smem_reduce;                // [channels = 128]
    float* smem_se_fc1 = smem_reduce + NN_HIDDEN_DIM; // [inner = 8]

    // === 1. Board encoding -> [17, 64] in buf1 ===
    block_board_to_planes(bs, buf1);

    // === 2. Input conv(17->128, k=3) + BN + ReLU -> buf2 ===
    block_conv_3x3_smem_w(buf1, weights->start_conv_weight, buf2, smem_weights,
                   NN_INPUT_CHANNELS, NN_HIDDEN_DIM);
    block_bn_relu(buf2, weights->start_bn.weight, weights->start_bn.bias,
                  weights->start_bn.running_mean, weights->start_bn.running_var,
                  NN_HIDDEN_DIM, true);
    // buf2 = start conv output; backbone lives here from now on.

    // === 3. Six residual blocks ===
    for (int b = 0; b < NN_NUM_BLOCKS; b++) {
        const ResBlockParams& blk = weights->blocks[b];

        // Save residual (buf2) in registers: 8192 floats / 256 threads = 32 per thread.
        // #pragma unroll ensures the compiler allocates 32 separate register variables
        // (not a local memory array) because indices become compile-time constants.
        float res0, res1, res2, res3, res4, res5, res6, res7;
        float res8, res9, res10, res11, res12, res13, res14, res15;
        float res16, res17, res18, res19, res20, res21, res22, res23;
        float res24, res25, res26, res27, res28, res29, res30, res31;
        // Reads happen before conv1 writes to buf1 (different buffer), no hazard.
        res0  = buf2[tid +  0*256]; res1  = buf2[tid +  1*256];
        res2  = buf2[tid +  2*256]; res3  = buf2[tid +  3*256];
        res4  = buf2[tid +  4*256]; res5  = buf2[tid +  5*256];
        res6  = buf2[tid +  6*256]; res7  = buf2[tid +  7*256];
        res8  = buf2[tid +  8*256]; res9  = buf2[tid +  9*256];
        res10 = buf2[tid + 10*256]; res11 = buf2[tid + 11*256];
        res12 = buf2[tid + 12*256]; res13 = buf2[tid + 13*256];
        res14 = buf2[tid + 14*256]; res15 = buf2[tid + 15*256];
        res16 = buf2[tid + 16*256]; res17 = buf2[tid + 17*256];
        res18 = buf2[tid + 18*256]; res19 = buf2[tid + 19*256];
        res20 = buf2[tid + 20*256]; res21 = buf2[tid + 21*256];
        res22 = buf2[tid + 22*256]; res23 = buf2[tid + 23*256];
        res24 = buf2[tid + 24*256]; res25 = buf2[tid + 25*256];
        res26 = buf2[tid + 26*256]; res27 = buf2[tid + 27*256];
        res28 = buf2[tid + 28*256]; res29 = buf2[tid + 29*256];
        res30 = buf2[tid + 30*256]; res31 = buf2[tid + 31*256];

        // conv1: buf2 -> buf1, BN + ReLU
        block_conv_3x3_smem_w(buf2, blk.conv1_weight, buf1, smem_weights, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
        block_bn_relu(buf1, blk.bn1.weight, blk.bn1.bias,
                      blk.bn1.running_mean, blk.bn1.running_var, NN_HIDDEN_DIM, true);

        // conv2: buf1 -> buf2, BN (no ReLU)
        // buf2 is overwritten here; the residual lives in registers above.
        block_conv_3x3_smem_w(buf1, blk.conv2_weight, buf2, smem_weights, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
        block_bn_relu(buf2, blk.bn2.weight, blk.bn2.bias,
                      blk.bn2.running_mean, blk.bn2.running_var, NN_HIDDEN_DIM, false);

        // SE block in-place on buf2
        block_se_block(buf2, blk.se.fc1_weight, blk.se.fc2_weight,
                       NN_HIDDEN_DIM, NN_SE_INNER, smem_se_avg, smem_se_fc1);

        // Residual add + ReLU: buf2 = relu(saved_registers + buf2)
        buf2[tid +  0*256] = fmaxf(0.0f, res0  + buf2[tid +  0*256]);
        buf2[tid +  1*256] = fmaxf(0.0f, res1  + buf2[tid +  1*256]);
        buf2[tid +  2*256] = fmaxf(0.0f, res2  + buf2[tid +  2*256]);
        buf2[tid +  3*256] = fmaxf(0.0f, res3  + buf2[tid +  3*256]);
        buf2[tid +  4*256] = fmaxf(0.0f, res4  + buf2[tid +  4*256]);
        buf2[tid +  5*256] = fmaxf(0.0f, res5  + buf2[tid +  5*256]);
        buf2[tid +  6*256] = fmaxf(0.0f, res6  + buf2[tid +  6*256]);
        buf2[tid +  7*256] = fmaxf(0.0f, res7  + buf2[tid +  7*256]);
        buf2[tid +  8*256] = fmaxf(0.0f, res8  + buf2[tid +  8*256]);
        buf2[tid +  9*256] = fmaxf(0.0f, res9  + buf2[tid +  9*256]);
        buf2[tid + 10*256] = fmaxf(0.0f, res10 + buf2[tid + 10*256]);
        buf2[tid + 11*256] = fmaxf(0.0f, res11 + buf2[tid + 11*256]);
        buf2[tid + 12*256] = fmaxf(0.0f, res12 + buf2[tid + 12*256]);
        buf2[tid + 13*256] = fmaxf(0.0f, res13 + buf2[tid + 13*256]);
        buf2[tid + 14*256] = fmaxf(0.0f, res14 + buf2[tid + 14*256]);
        buf2[tid + 15*256] = fmaxf(0.0f, res15 + buf2[tid + 15*256]);
        buf2[tid + 16*256] = fmaxf(0.0f, res16 + buf2[tid + 16*256]);
        buf2[tid + 17*256] = fmaxf(0.0f, res17 + buf2[tid + 17*256]);
        buf2[tid + 18*256] = fmaxf(0.0f, res18 + buf2[tid + 18*256]);
        buf2[tid + 19*256] = fmaxf(0.0f, res19 + buf2[tid + 19*256]);
        buf2[tid + 20*256] = fmaxf(0.0f, res20 + buf2[tid + 20*256]);
        buf2[tid + 21*256] = fmaxf(0.0f, res21 + buf2[tid + 21*256]);
        buf2[tid + 22*256] = fmaxf(0.0f, res22 + buf2[tid + 22*256]);
        buf2[tid + 23*256] = fmaxf(0.0f, res23 + buf2[tid + 23*256]);
        buf2[tid + 24*256] = fmaxf(0.0f, res24 + buf2[tid + 24*256]);
        buf2[tid + 25*256] = fmaxf(0.0f, res25 + buf2[tid + 25*256]);
        buf2[tid + 26*256] = fmaxf(0.0f, res26 + buf2[tid + 26*256]);
        buf2[tid + 27*256] = fmaxf(0.0f, res27 + buf2[tid + 27*256]);
        buf2[tid + 28*256] = fmaxf(0.0f, res28 + buf2[tid + 28*256]);
        buf2[tid + 29*256] = fmaxf(0.0f, res29 + buf2[tid + 29*256]);
        buf2[tid + 30*256] = fmaxf(0.0f, res30 + buf2[tid + 30*256]);
        buf2[tid + 31*256] = fmaxf(0.0f, res31 + buf2[tid + 31*256]);
        __syncthreads();
        // buf2 = block output; buf1 = stale (conv1 result)
    }

    // buf2 = backbone output [128, 64]

    // === 4. Policy head ===

    // Policy conv(128->128, k=3) + BN + ReLU: buf2 -> buf1
    block_conv_3x3_smem_w(buf2, weights->p_conv_weight, buf1, smem_weights, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    block_bn_relu(buf1, weights->p_bn.weight, weights->p_bn.bias,
                  weights->p_bn.running_mean, weights->p_bn.running_var,
                  NN_HIDDEN_DIM, true);
    // buf1 = policy conv BN+ReLU output; buf2 = backbone (still valid for value head)

    // Fused 1x1 conv + bias + HWC permute -> policy_out (global memory).
    // policy_out[spatial*73 + plane] = sum_c(p_head_weight[plane,c] * buf1[c,spatial]) + bias[plane]
    for (int idx = tid; idx < NN_POLICY_SIZE; idx += blockDim.x) {
        int plane   = idx % NN_POLICY_PLANES;
        int spatial = idx / NN_POLICY_PLANES;
        float acc   = 0.0f;
        const float* w_row = weights->p_head_weight + plane * NN_HIDDEN_DIM;
        for (int c = 0; c < NN_HIDDEN_DIM; c++) {
            acc += w_row[c] * buf1[c * 64 + spatial];
        }
        policy_out[idx] = acc + weights->p_head_bias[plane];
    }
    __syncthreads();

    // Log-softmax in-place on policy_out (global memory)
    block_log_softmax(policy_out, NN_POLICY_SIZE, smem_reduce);

    // === 5. Value head ===

    // 1x1 conv (128->1) + bias: read buf2 (backbone), write buf1[0:64] (v_feat)
    for (int idx = tid; idx < 64; idx += blockDim.x) {
        float sum = 0.0f;
        for (int c = 0; c < NN_HIDDEN_DIM; c++) {
            sum += weights->v_conv_weight[c] * buf2[c * 64 + idx];
        }
        buf1[idx] = sum + weights->v_conv_bias[0];
    }
    __syncthreads();

    // BN + ReLU on v_feat [1, 64]
    block_bn_relu_1ch(buf1, weights->v_bn.weight, weights->v_bn.bias,
                      weights->v_bn.running_mean, weights->v_bn.running_var, true);

    // FC1: [256, 65] x [v_feat[64]; q_result] -> smem_reduce[0:256]
    // 256 threads, 256 outputs: exactly one output per thread.
    float* smem_fc1_out = smem_reduce;  // reuse smem_reduce for FC1 output
    for (int j = tid; j < NN_VALUE_FC1_OUT; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < 64; i++) {
            sum += weights->v_fc_weight[j * NN_VALUE_FC1_IN + i] * buf1[i];
        }
        sum += weights->v_fc_weight[j * NN_VALUE_FC1_IN + 64] * q_result;
        sum += weights->v_fc_bias[j];
        smem_fc1_out[j] = sum > 0.0f ? sum : 0.0f;  // ReLU
    }
    __syncthreads();

    // FC2: [1, 256] x smem_fc1_out -> v_logit (parallel reduction into smem_reduce)
    // Reuse smem_reduce for the reduction (FC1 values already consumed above).
    float local_sum = 0.0f;
    for (int j = tid; j < NN_VALUE_FC1_OUT; j += blockDim.x) {
        local_sum += weights->v_out_weight[j] * smem_fc1_out[j];
    }
    smem_reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem_reduce[tid] += smem_reduce[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        float v_logit = smem_reduce[0] + weights->v_out_bias[0];
        float k = 0.47f * logf(1.0f + expf(weights->k_logit));
        *value_out = tanhf(v_logit + k * q_result);
        *k_out = k;
    }
    __syncthreads();
}
