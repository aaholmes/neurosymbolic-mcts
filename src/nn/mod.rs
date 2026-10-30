//! Neural network inference with candle (pure Rust, no LibTorch).
//!
//! Networks are trained in PyTorch (`python/model.py`) and re-implemented here
//! layer for layer, loading the PyTorch state dict by its parameter names.

pub mod fused;
pub mod oracle_net;
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
