//! OracleNet (SE-ResNet) inference, a layer-for-layer port of
//! `python/model.py::OracleNet` in eval mode.

use super::{fused, NetOutput, PolicyValueNet};
use candle_core::conv::CudnnFwdAlgo;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};
use std::collections::HashMap;

const INPUT_CHANNELS: usize = 17;
const POLICY_PLANES: usize = 73;
const POLICY_SIZE: usize = 64 * POLICY_PLANES;
const SE_REDUCTION: usize = 16;
const VALUE_HIDDEN: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleNetConfig {
    pub num_blocks: usize,
    pub hidden_dim: usize,
}

impl OracleNetConfig {
    /// Read the architecture off the state dict, so no config file is needed.
    pub fn infer(state_dict: &HashMap<String, Tensor>) -> Result<Self> {
        let start = state_dict
            .get("start_conv.weight")
            .ok_or_else(|| candle_core::Error::Msg("missing start_conv.weight".into()))?;
        let hidden_dim = start.dim(0)?;
        let num_blocks = (0..)
            .take_while(|i| state_dict.contains_key(&format!("res_blocks.{i}.conv1.weight")))
            .count();
        Ok(Self {
            num_blocks,
            hidden_dim,
        })
    }
}

const BN_EPS: f64 = 1e-5;

/// Batches at least this large use cuDNN's precomputed implicit GEMM; smaller ones
/// use plain implicit GEMM, which has a lower fixed cost per call. Measured on an
/// RTX 5060 Ti with a 6x128 net: the crossover lies between batch 32 and 128.
const LARGE_BATCH: usize = 64;

/// cuDNN forward algorithm for a batch (ignored without the `cudnn` feature).
/// Fixing it skips candle's per-call algorithm search, which cost ~0.3 ms per conv.
/// KF_CUDNN_ALGO=auto|implicit|precomp overrides the choice for experiments.
fn fwd_algo(batch: usize) -> Option<CudnnFwdAlgo> {
    match std::env::var("KF_CUDNN_ALGO").as_deref() {
        Ok("auto") => None,
        Ok("implicit") => Some(CudnnFwdAlgo::ImplicitGemm),
        Ok("precomp") => Some(CudnnFwdAlgo::ImplicitPrecompGemm),
        _ if batch >= LARGE_BATCH => Some(CudnnFwdAlgo::ImplicitPrecompGemm),
        _ => Some(CudnnFwdAlgo::ImplicitGemm),
    }
}

/// A square, stride-1, "same"-padded convolution with bias.
struct Conv {
    weight: Tensor,
    bias: Tensor,
    padding: usize,
}

impl Conv {
    fn plain(in_c: usize, out_c: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get((out_c, in_c, kernel, kernel), "weight")?,
            bias: vb.get(out_c, "bias")?,
            padding: kernel / 2,
        })
    }

    /// A convolution followed by eval-mode BatchNorm, folded into one convolution.
    ///
    /// BN computes `(y - mean) * s + beta` with `s = gamma / sqrt(var + eps)`, so
    /// `conv(x; W) -> BN` equals `conv(x; W * s) + (b - mean) * s + beta`.
    fn with_bn(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        conv_has_bias: bool,
        conv_vb: VarBuilder,
        bn_vb: VarBuilder,
    ) -> Result<Self> {
        let w = conv_vb.get((out_c, in_c, kernel, kernel), "weight")?;
        let gamma = bn_vb.get(out_c, "weight")?;
        let beta = bn_vb.get(out_c, "bias")?;
        let mean = bn_vb.get(out_c, "running_mean")?;
        let var = bn_vb.get(out_c, "running_var")?;
        let scale = gamma.div(&(var + BN_EPS)?.sqrt()?)?;
        let b = if conv_has_bias {
            conv_vb.get(out_c, "bias")?.sub(&mean)?
        } else {
            mean.neg()?
        };
        Ok(Self {
            weight: w.broadcast_mul(&scale.reshape((out_c, 1, 1, 1))?)?,
            bias: b.mul(&scale)?.add(&beta)?,
            padding: kernel / 2,
        })
    }

    /// The convolution without its bias.
    fn conv(&self, x: &Tensor) -> Result<Tensor> {
        let algo = fwd_algo(x.dim(0)?);
        x.conv2d_with_algo(&self.weight, self.padding, 1, 1, 1, algo)
    }

    /// Convolution plus bias, optionally followed by ReLU.
    fn forward(&self, x: &Tensor, relu: bool) -> Result<Tensor> {
        fused::bias_act(&self.conv(x)?, &self.bias, relu)
    }
}

/// Squeeze-and-excitation: rescales each channel by a weight in (0, 1)
/// computed from the global average of all channels.
struct SeBlock {
    fc1: Linear,
    fc2: Linear,
}

impl SeBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let inner = channels / SE_REDUCTION;
        Ok(Self {
            fc1: linear_no_bias(channels, inner, vb.pp("fc.0"))?,
            fc2: linear_no_bias(inner, channels, vb.pp("fc.2"))?,
        })
    }

    /// `relu(se(conv + bias) + residual)`, where `conv` has not had its bias added yet.
    fn forward_with_residual(
        &self,
        conv: &Tensor,
        bias: &Tensor,
        residual: &Tensor,
    ) -> Result<Tensor> {
        let y = fused::channel_mean_bias(conv, bias)?;
        let y = self.fc1.forward(&y)?.relu()?;
        let gate = candle_nn::ops::sigmoid(&self.fc2.forward(&y)?)?;
        fused::se_tail(conv, bias, &gate, residual)
    }
}

struct ResBlock {
    conv1: Conv,
    conv2: Conv,
    se: SeBlock,
}

impl ResBlock {
    fn new(hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: Conv::with_bn(hidden, hidden, 3, false, vb.pp("conv1"), vb.pp("bn1"))?,
            conv2: Conv::with_bn(hidden, hidden, 3, false, vb.pp("conv2"), vb.pp("bn2"))?,
            se: SeBlock::new(hidden, vb.pp("se"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.conv1.forward(x, true)?;
        let out = self.conv2.conv(&out)?;
        self.se.forward_with_residual(&out, &self.conv2.bias, x)
    }
}

pub struct OracleNet {
    config: OracleNetConfig,
    start_conv: Conv,
    blocks: Vec<ResBlock>,
    p_conv: Conv,
    p_head: Conv,
    v_conv: Conv,
    v_fc: Linear,
    v_out: Linear,
    k: f32,
}

impl OracleNet {
    /// Build the network from a PyTorch state dict (names as in `model.state_dict()`).
    pub fn from_state_dict(state_dict: HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let config = OracleNetConfig::infer(&state_dict)?;
        let k_logit = state_dict
            .get("k_logit")
            .ok_or_else(|| candle_core::Error::Msg("missing k_logit".into()))?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?[0];
        let vb = VarBuilder::from_tensors(state_dict, DType::F32, device);
        Self::new(config, k_logit, vb)
    }

    fn new(config: OracleNetConfig, k_logit: f64, vb: VarBuilder) -> Result<Self> {
        let h = config.hidden_dim;
        let blocks = (0..config.num_blocks)
            .map(|i| ResBlock::new(h, vb.pp(format!("res_blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            start_conv: Conv::with_bn(
                INPUT_CHANNELS,
                h,
                3,
                false,
                vb.pp("start_conv"),
                vb.pp("start_bn"),
            )?,
            blocks,
            p_conv: Conv::with_bn(h, h, 3, false, vb.pp("p_conv"), vb.pp("p_bn"))?,
            p_head: Conv::plain(h, POLICY_PLANES, 1, vb.pp("p_head"))?,
            v_conv: Conv::with_bn(h, 1, 1, true, vb.pp("v_conv"), vb.pp("v_bn"))?,
            v_fc: linear(65, VALUE_HIDDEN, vb.pp("v_fc"))?,
            v_out: linear(VALUE_HIDDEN, 1, vb.pp("v_out"))?,
            k: (0.47 * softplus(k_logit)) as f32,
        })
    }

    pub fn config(&self) -> OracleNetConfig {
        self.config
    }
}

/// PyTorch's softplus (beta 1, threshold 20).
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

impl PolicyValueNet for OracleNet {
    fn forward(&self, planes: &Tensor, scalars: &Tensor) -> Result<NetOutput> {
        let b = planes.dim(0)?;
        let q_result = scalars.narrow(1, 0, 1)?;

        let mut x = self.start_conv.forward(planes, true)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Policy: [B, 73, 8, 8] -> [B, 8, 8, 73] -> [B, 4672] (AlphaZero move index order)
        let p = self.p_conv.forward(&x, true)?;
        let p = self.p_head.forward(&p, false)?.permute((0, 2, 3, 1))?;
        let p = p.contiguous()?.reshape((b, POLICY_SIZE))?;
        let log_policy = candle_nn::ops::log_softmax(&p, D::Minus1)?;

        // Value: 1x1 conv -> [B, 64], append q_result -> [B, 65] -> FC -> v_logit
        let v = self.v_conv.forward(&x, true)?.reshape((b, 64))?;
        let v = Tensor::cat(&[&v, &q_result], 1)?;
        let v = self.v_fc.forward(&v)?.relu()?;
        let v_logit = self.v_out.forward(&v)?;

        Ok(NetOutput {
            log_policy,
            v_logit,
            k: self.k,
        })
    }
}
