//! SeresNet (SE-ResNet) inference, a layer-for-layer port of
//! `python/model.py::SeresNet` in eval mode.

use super::{fused, NetOutput, PolicyValueNet, Precision};
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
pub struct SeresNetConfig {
    pub num_blocks: usize,
    pub hidden_dim: usize,
}

impl SeresNetConfig {
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

/// KF_CANDLE_CONV=1 uses candle's own cuDNN convolution in half precision too.
#[cfg(feature = "cudnn")]
fn candle_conv() -> bool {
    static CANDLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CANDLE.get_or_init(|| std::env::var("KF_CANDLE_CONV").as_deref() == Ok("1"))
}

/// A square, stride-1, "same"-padded convolution with bias.
struct Conv {
    weight: Tensor,
    bias: Tensor,
    padding: usize,
}

impl Conv {
    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_dtype(dtype)?,
            bias: self.bias.to_dtype(dtype)?,
            padding: self.padding,
        })
    }

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
        // Half precision uses the cached convolution; f32 keeps candle's, whose
        // numerics and speed are the reference.
        #[cfg(feature = "cudnn")]
        if x.device().is_cuda() && x.dtype() != DType::F32 && !candle_conv() {
            return super::cudnn_conv::conv2d(x, &self.weight, self.padding);
        }
        let algo = fwd_algo(x.dim(0)?);
        x.conv2d_with_algo(&self.weight, self.padding, 1, 1, 1, algo)
    }

    /// Convolution plus bias, optionally followed by ReLU: one fused cuDNN call in
    /// half precision, a separate bias kernel otherwise.
    fn forward(&self, x: &Tensor, relu: bool) -> Result<Tensor> {
        #[cfg(feature = "cudnn")]
        if x.device().is_cuda() && x.dtype() != DType::F32 && !candle_conv() {
            return super::cudnn_conv::conv2d_bias_act(
                x,
                &self.weight,
                &self.bias,
                self.padding,
                relu,
            );
        }
        fused::bias_act(&self.conv(x)?, &self.bias, relu)
    }
}

/// Squeeze-and-excitation: rescales each channel by a weight in (0, 1)
/// computed from the global average of all channels.
struct SeBlock {
    fc1: Linear,
    fc2: Linear,
}

fn linear_to_dtype(l: &Linear, dtype: DType) -> Result<Linear> {
    Ok(Linear::new(
        l.weight().to_dtype(dtype)?,
        l.bias().map(|b| b.to_dtype(dtype)).transpose()?,
    ))
}

impl SeBlock {
    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            fc1: linear_to_dtype(&self.fc1, dtype)?,
            fc2: linear_to_dtype(&self.fc2, dtype)?,
        })
    }

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
    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            conv1: self.conv1.to_dtype(dtype)?,
            conv2: self.conv2.to_dtype(dtype)?,
            se: self.se.to_dtype(dtype)?,
        })
    }

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

pub struct SeresNet {
    config: SeresNetConfig,
    start_conv: Conv,
    blocks: Vec<ResBlock>,
    p_conv: Conv,
    p_head: Conv,
    v_conv: Conv,
    v_fc: Linear,
    v_out: Linear,
    k: f32,
    dtype: DType,
}

impl SeresNet {
    /// Build the network from a PyTorch state dict (names as in `model.state_dict()`).
    pub fn from_state_dict(state_dict: HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        Self::from_state_dict_with_precision(state_dict, device, Precision::F32)
    }

    /// As `from_state_dict`, computing in `precision`. The net is built in f32
    /// (so BatchNorm folds exactly) and then cast; inputs and outputs stay f32.
    pub fn from_state_dict_with_precision(
        state_dict: HashMap<String, Tensor>,
        device: &Device,
        precision: Precision,
    ) -> Result<Self> {
        if precision != Precision::F32 && !device.is_cuda() {
            candle_core::bail!("{} inference needs CUDA", precision.name());
        }
        let net = Self::build_f32(state_dict, device)?;
        if precision == Precision::F32 {
            return Ok(net);
        }
        net.to_dtype(precision.dtype())
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            config: self.config,
            start_conv: self.start_conv.to_dtype(dtype)?,
            blocks: self
                .blocks
                .iter()
                .map(|b| b.to_dtype(dtype))
                .collect::<Result<Vec<_>>>()?,
            p_conv: self.p_conv.to_dtype(dtype)?,
            p_head: self.p_head.to_dtype(dtype)?,
            v_conv: self.v_conv.to_dtype(dtype)?,
            v_fc: linear_to_dtype(&self.v_fc, dtype)?,
            v_out: linear_to_dtype(&self.v_out, dtype)?,
            k: self.k,
            dtype,
        })
    }

    fn build_f32(state_dict: HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let config = SeresNetConfig::infer(&state_dict)?;
        let k_logit = state_dict
            .get("k_logit")
            .ok_or_else(|| candle_core::Error::Msg("missing k_logit".into()))?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?[0];
        let vb = VarBuilder::from_tensors(state_dict, DType::F32, device);
        Self::new(config, k_logit, vb)
    }

    fn new(config: SeresNetConfig, k_logit: f64, vb: VarBuilder) -> Result<Self> {
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
            dtype: DType::F32,
        })
    }

    pub fn config(&self) -> SeresNetConfig {
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

impl SeresNet {
    /// The trunk and both heads, with the policy left as raw logits in the
    /// network's own layout: `[B, 4672]` indexed by `plane * 64 + square`.
    /// Returns `(policy_logits, v_logit)`.
    pub fn forward_logits(&self, planes: &Tensor, scalars: &Tensor) -> Result<(Tensor, Tensor)> {
        let b = planes.dim(0)?;
        let planes = &planes.to_dtype(self.dtype)?;
        let q_result = scalars.narrow(1, 0, 1)?.to_dtype(self.dtype)?;

        let mut x = self.start_conv.forward(planes, true)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        let p = self.p_conv.forward(&x, true)?;
        let logits = self.p_head.forward(&p, false)?.reshape((b, POLICY_SIZE))?;

        // Value: 1x1 conv -> [B, 64], append q_result -> [B, 65] -> FC -> v_logit
        let v = self.v_conv.forward(&x, true)?.reshape((b, 64))?;
        let v = Tensor::cat(&[&v, &q_result], 1)?;
        let v = self.v_fc.forward(&v)?.relu()?;
        let v_logit = self.v_out.forward(&v)?;
        Ok((logits.to_dtype(DType::F32)?, v_logit.to_dtype(DType::F32)?))
    }

    /// Material confidence scalar (position-independent).
    pub fn k(&self) -> f32 {
        self.k
    }
}

impl PolicyValueNet for SeresNet {
    fn forward(&self, planes: &Tensor, scalars: &Tensor) -> Result<NetOutput> {
        let b = planes.dim(0)?;
        let (logits, v_logit) = self.forward_logits(planes, scalars)?;
        // [B, 73, 8, 8] -> [B, 8, 8, 73] -> [B, 4672] (AlphaZero move index order)
        let p = logits
            .reshape((b, POLICY_PLANES, 8, 8))?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((b, POLICY_SIZE))?;
        Ok(NetOutput {
            log_policy: candle_nn::ops::log_softmax(&p, D::Minus1)?,
            v_logit,
            k: self.k,
        })
    }
}
