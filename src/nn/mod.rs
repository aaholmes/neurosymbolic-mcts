//! Neural network inference with candle (pure Rust, no LibTorch).
//!
//! Networks are trained in PyTorch (`python/model.py`) and re-implemented here
//! layer for layer, loading the PyTorch state dict by its parameter names.

#[cfg(feature = "cudnn")]
pub mod cudnn_conv;
pub mod fused;
pub mod seres_net;
pub mod weights;

use candle_core::{Result, Tensor};

/// Outputs of one forward pass, matching the PyTorch model in eval mode.
pub struct NetOutput {
    /// Log-probabilities over the 4672 AlphaZero move slots, `[B, 4672]`.
    pub log_policy: Tensor,
    /// Unbounded positional logit; the caller computes `tanh(v_logit + k * ΔM)`. `[B, 1]`.
    pub v_logit: Tensor,
    /// Material confidence scalar (position-independent).
    pub k: f32,
}

/// A policy/value network: board planes `[B, 17, 8, 8]` and scalars
/// `[B, 2]` (`q_result`, `qsearch_completed`) in, `NetOutput` out.
pub trait PolicyValueNet: Send + Sync {
    fn forward(&self, planes: &Tensor, scalars: &Tensor) -> Result<NetOutput>;
}

/// Floating-point format the network computes in. Half precision runs on CUDA only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    F32,
    F16,
    Bf16,
}

impl Precision {
    /// "f32", "f16" or "bf16".
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    pub fn dtype(self) -> candle_core::DType {
        match self {
            Self::F32 => candle_core::DType::F32,
            Self::F16 => candle_core::DType::F16,
            Self::Bf16 => candle_core::DType::BF16,
        }
    }

    /// From the KF_PRECISION environment variable; f32 when unset.
    /// Panics on an unknown value, so a typo cannot silently fall back to f32.
    pub fn from_env() -> Self {
        match std::env::var("KF_PRECISION") {
            Ok(s) => Self::parse(&s)
                .unwrap_or_else(|| panic!("KF_PRECISION must be f32, f16 or bf16, not {s:?}")),
            Err(_) => Self::F32,
        }
    }
}
