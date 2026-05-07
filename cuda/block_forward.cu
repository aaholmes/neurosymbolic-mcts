#include "block_forward.cuh"
#include "block_ops.cuh"
#include <cuda_fp16.h>

int block_smem_bytes() { return BLOCK_SMEM_BYTES; }

// ============================================================
// Full SE-ResNet forward pass — 256-thread block cooperative
// (FP16 inter-layer activation storage)
// ============================================================
//
// Buffer assignment in dynamic smem (interpreted as __half*):
//   buf1 = smem + BLOCK_BUF1_OFFSET_H (smem+0 halves)         work buffer
//   buf2 = smem + BLOCK_BUF2_OFFSET_H (smem+8192 halves)      backbone buffer
//   smem_reduce: float[256] starting at BLOCK_REDUCE_OFFSET_H (in halves)
//   aux:         halves at BLOCK_AUX_OFFSET_H, sized BLOCK_AUX_HALVES
//
// Data flow (FP16 unless noted):
//   board_to_planes   -> buf1 [17*64]
//   start_conv(buf1)  -> buf2  (backbone from here)
//   6× res blocks:
//     conv1(buf2) -> buf1
//     conv2(buf1) -> buf2 (residual stays FP32 in registers)
//     SE + add_relu -> buf2
//   policy_conv(buf2) -> buf1
//   1×1+bias+permute(buf1) -> policy_out (FP32 global)
//   log_softmax(policy_out)
//   v_conv(buf2) -> smem_reduce[0:64]  (FP32 — value head stays FP32)
//   FC1 -> smem_reduce[0:256]  (FP32)
//   FC2 reduction -> v_logit (thread 0)

__device__ void oracle_net_forward_block(
    const BoardState* bs,
    float q_result,
    const OracleNetWeights* weights,
    float* smem,
    float* policy_out,
    float* value_out,
    float* k_out,
    const ConvWeightsHalf* half_w,
    const ConvWeightsShifted* shifted_w
) {
    int tid = threadIdx.x;

    // Reinterpret raw smem as __half* and place sub-buffers by half-offsets.
    __half* smem_h    = (__half*)smem;
    __half* buf1      = smem_h + BLOCK_BUF1_OFFSET_H;
    __half* buf2      = smem_h + BLOCK_BUF2_OFFSET_H;
    float*  smem_reduce  = (float*)(smem_h + BLOCK_REDUCE_OFFSET_H);
    half*   aux_half     = smem_h + BLOCK_AUX_OFFSET_H;     // shared by all conv paths
    float*  aux_float    = (float*)aux_half;

    // Dispatch priority: shifted > TC > scalar
    int conv_mode = shifted_w ? 2 : (half_w ? 1 : 0);

    // SE workspace lives at the start of smem_reduce (FP32, reused each block)
    float* smem_se_avg = smem_reduce;                       // [128]
    float* smem_se_fc1 = smem_reduce + NN_HIDDEN_DIM;       // [8]

    // === 1. Board encoding -> [17, 64] in buf1 (FP16) ===
    block_board_to_planes(bs, buf1);

    // === 2. Input conv(17->128, k=3) + BN + ReLU -> buf2 ===
    if (conv_mode == 2)
        block_conv_3x3_shifted(buf1, shifted_w->start_conv, buf2, aux_half,
                               NN_INPUT_CHANNELS, NN_HIDDEN_DIM);
    else if (conv_mode == 1)
        block_conv_3x3_tc(buf1, half_w->start_conv, buf2, aux_half,
                          NN_INPUT_CHANNELS, NN_HIDDEN_DIM);
    else
        block_conv_3x3_smem_w(buf1, weights->start_conv_weight, buf2, aux_float,
                       NN_INPUT_CHANNELS, NN_HIDDEN_DIM);
    block_bn_relu(buf2, weights->start_bn.weight, weights->start_bn.bias,
                  weights->start_bn.running_mean, weights->start_bn.running_var,
                  NN_HIDDEN_DIM, true);

    // === 3. Six residual blocks ===
    for (int b = 0; b < NN_NUM_BLOCKS; b++) {
        const ResBlockParams& blk = weights->blocks[b];

        // Save residual (buf2) in FP32 registers: 8192 / 256 = 32 per thread.
        // FP32 keeps the residual stream's precision; the load converts FP16→FP32.
        float res0, res1, res2, res3, res4, res5, res6, res7;
        float res8, res9, res10, res11, res12, res13, res14, res15;
        float res16, res17, res18, res19, res20, res21, res22, res23;
        float res24, res25, res26, res27, res28, res29, res30, res31;
        res0  = __half2float(buf2[tid +  0*256]); res1  = __half2float(buf2[tid +  1*256]);
        res2  = __half2float(buf2[tid +  2*256]); res3  = __half2float(buf2[tid +  3*256]);
        res4  = __half2float(buf2[tid +  4*256]); res5  = __half2float(buf2[tid +  5*256]);
        res6  = __half2float(buf2[tid +  6*256]); res7  = __half2float(buf2[tid +  7*256]);
        res8  = __half2float(buf2[tid +  8*256]); res9  = __half2float(buf2[tid +  9*256]);
        res10 = __half2float(buf2[tid + 10*256]); res11 = __half2float(buf2[tid + 11*256]);
        res12 = __half2float(buf2[tid + 12*256]); res13 = __half2float(buf2[tid + 13*256]);
        res14 = __half2float(buf2[tid + 14*256]); res15 = __half2float(buf2[tid + 15*256]);
        res16 = __half2float(buf2[tid + 16*256]); res17 = __half2float(buf2[tid + 17*256]);
        res18 = __half2float(buf2[tid + 18*256]); res19 = __half2float(buf2[tid + 19*256]);
        res20 = __half2float(buf2[tid + 20*256]); res21 = __half2float(buf2[tid + 21*256]);
        res22 = __half2float(buf2[tid + 22*256]); res23 = __half2float(buf2[tid + 23*256]);
        res24 = __half2float(buf2[tid + 24*256]); res25 = __half2float(buf2[tid + 25*256]);
        res26 = __half2float(buf2[tid + 26*256]); res27 = __half2float(buf2[tid + 27*256]);
        res28 = __half2float(buf2[tid + 28*256]); res29 = __half2float(buf2[tid + 29*256]);
        res30 = __half2float(buf2[tid + 30*256]); res31 = __half2float(buf2[tid + 31*256]);

        // conv1: buf2 -> buf1, BN + ReLU
        if (conv_mode == 2)
            block_conv_3x3_shifted(buf2, shifted_w->block_conv1[b], buf1, aux_half, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
        else if (conv_mode == 1)
            block_conv_3x3_tc(buf2, half_w->block_conv1[b], buf1, aux_half, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
        else
            block_conv_3x3_smem_w(buf2, blk.conv1_weight, buf1, aux_float, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
        block_bn_relu(buf1, blk.bn1.weight, blk.bn1.bias,
                      blk.bn1.running_mean, blk.bn1.running_var, NN_HIDDEN_DIM, true);

        // conv2: buf1 -> buf2, BN (no ReLU)
        if (conv_mode == 2)
            block_conv_3x3_shifted(buf1, shifted_w->block_conv2[b], buf2, aux_half, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
        else if (conv_mode == 1)
            block_conv_3x3_tc(buf1, half_w->block_conv2[b], buf2, aux_half, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
        else
            block_conv_3x3_smem_w(buf1, blk.conv2_weight, buf2, aux_float, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
        block_bn_relu(buf2, blk.bn2.weight, blk.bn2.bias,
                      blk.bn2.running_mean, blk.bn2.running_var, NN_HIDDEN_DIM, false);

        // SE block in-place on buf2 (FP16 in/out, FP32 internal)
        block_se_block(buf2, blk.se.fc1_weight, blk.se.fc2_weight,
                       NN_HIDDEN_DIM, NN_SE_INNER, smem_se_avg, smem_se_fc1);

        // Residual add + ReLU: load FP16 buf2, add FP32 residual, ReLU, store FP16
        #define RES_ADD(IDX, REG) \
            buf2[tid + (IDX)*256] = __float2half( \
                fmaxf(0.0f, (REG) + __half2float(buf2[tid + (IDX)*256])))
        RES_ADD( 0, res0 );  RES_ADD( 1, res1 );  RES_ADD( 2, res2 );  RES_ADD( 3, res3 );
        RES_ADD( 4, res4 );  RES_ADD( 5, res5 );  RES_ADD( 6, res6 );  RES_ADD( 7, res7 );
        RES_ADD( 8, res8 );  RES_ADD( 9, res9 );  RES_ADD(10, res10); RES_ADD(11, res11);
        RES_ADD(12, res12); RES_ADD(13, res13); RES_ADD(14, res14); RES_ADD(15, res15);
        RES_ADD(16, res16); RES_ADD(17, res17); RES_ADD(18, res18); RES_ADD(19, res19);
        RES_ADD(20, res20); RES_ADD(21, res21); RES_ADD(22, res22); RES_ADD(23, res23);
        RES_ADD(24, res24); RES_ADD(25, res25); RES_ADD(26, res26); RES_ADD(27, res27);
        RES_ADD(28, res28); RES_ADD(29, res29); RES_ADD(30, res30); RES_ADD(31, res31);
        #undef RES_ADD
        __syncthreads();
    }

    // buf2 = backbone output [128, 64] FP16

    // === 4. Policy head ===

    // Policy conv(128->128, k=3) + BN + ReLU: buf2 -> buf1
    if (conv_mode == 2)
        block_conv_3x3_shifted(buf2, shifted_w->p_conv, buf1, aux_half, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    else if (conv_mode == 1)
        block_conv_3x3_tc(buf2, half_w->p_conv, buf1, aux_half, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    else
        block_conv_3x3_smem_w(buf2, weights->p_conv_weight, buf1, aux_float, NN_HIDDEN_DIM, NN_HIDDEN_DIM);
    block_bn_relu(buf1, weights->p_bn.weight, weights->p_bn.bias,
                  weights->p_bn.running_mean, weights->p_bn.running_var,
                  NN_HIDDEN_DIM, true);

    // Fused 1x1 conv + bias + HWC permute -> policy_out (FP32 global memory).
    // Read FP16 buf1, accumulate FP32, write FP32 output.
    for (int idx = tid; idx < NN_POLICY_SIZE; idx += blockDim.x) {
        int plane   = idx % NN_POLICY_PLANES;
        int spatial = idx / NN_POLICY_PLANES;
        float acc   = 0.0f;
        const float* w_row = weights->p_head_weight + plane * NN_HIDDEN_DIM;
        for (int c = 0; c < NN_HIDDEN_DIM; c++) {
            acc += w_row[c] * __half2float(buf1[c * 64 + spatial]);
        }
        policy_out[idx] = acc + weights->p_head_bias[plane];
    }
    __syncthreads();

    // Log-softmax in-place on policy_out (FP32 global, unchanged)
    block_log_softmax(policy_out, NN_POLICY_SIZE, smem_reduce);

    // === 5. Value head (FP32 end-to-end for precision) ===
    //
    // The value head was the most precision-sensitive scalar path even at FP32
    // baseline. Route v_feat through smem_reduce (FP32) instead of buf1 to
    // avoid two FP16 round-trips. smem_reduce[0:64] is unused at this point.

    // 1x1 conv (128->1) + bias: read FP16 buf2, write FP32 to smem_reduce[0:64]
    float* v_feat = smem_reduce;
    for (int idx = tid; idx < 64; idx += blockDim.x) {
        float sum = 0.0f;
        for (int c = 0; c < NN_HIDDEN_DIM; c++) {
            sum += weights->v_conv_weight[c] * __half2float(buf2[c * 64 + idx]);
        }
        v_feat[idx] = sum + weights->v_conv_bias[0];
    }
    __syncthreads();

    // BN + ReLU on v_feat [1, 64] (FP32 path, unchanged)
    block_bn_relu_1ch(v_feat, weights->v_bn.weight, weights->v_bn.bias,
                      weights->v_bn.running_mean, weights->v_bn.running_var, true);

    // FC1: [256, 65] x [v_feat[64]; q_result] -> smem_reduce[0:256]
    // smem_reduce holds v_feat[0:64] right now; FC1 writes start at index 0
    // and consume v_feat values that have already been read into registers.
    // We must NOT overwrite v_feat before it's fully consumed.
    //
    // Strategy: each thread reads its 64 v_feat values into registers first
    // (no writes to smem_reduce yet), then compute FC1 and write result.
    // Since 256 threads each compute one of 256 outputs and each output reads
    // all 64 v_feat values, we need v_feat to remain valid across all reads.
    // Solution: split into two phases with __syncthreads in between, or use
    // a separate region. The simplest is to keep v_feat live until after the
    // sum is complete per-thread, then write once — same memory location is
    // safe because each thread writes to a different index.
    float fc1_local[1];  // each thread computes ≤1 output (256 threads, 256 outputs)
    int my_j = tid;  // exactly one output per thread (NN_VALUE_FC1_OUT == 256)
    float fc1_sum = 0.0f;
    if (my_j < NN_VALUE_FC1_OUT) {
        for (int i = 0; i < 64; i++) {
            fc1_sum += weights->v_fc_weight[my_j * NN_VALUE_FC1_IN + i] * v_feat[i];
        }
        fc1_sum += weights->v_fc_weight[my_j * NN_VALUE_FC1_IN + 64] * q_result;
        fc1_sum += weights->v_fc_bias[my_j];
        fc1_local[0] = fc1_sum > 0.0f ? fc1_sum : 0.0f;  // ReLU
    }
    __syncthreads();
    // Now v_feat has been fully consumed; safe to overwrite smem_reduce.
    if (my_j < NN_VALUE_FC1_OUT) {
        smem_reduce[my_j] = fc1_local[0];
    }
    __syncthreads();

    // FC2: [1, 256] × smem_reduce -> v_logit (parallel reduction)
    float local_sum = 0.0f;
    for (int j = tid; j < NN_VALUE_FC1_OUT; j += blockDim.x) {
        local_sum += weights->v_out_weight[j] * smem_reduce[j];
    }
    __syncthreads();
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
