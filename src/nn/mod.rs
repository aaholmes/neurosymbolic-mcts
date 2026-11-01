//! Neural network inference with candle (pure Rust, no LibTorch).
//!
//! Networks are trained in PyTorch (`python/model.py`) and re-implemented here
//! layer for layer, loading the PyTorch state dict by its parameter names.

#[cfg(feature = "cudnn")]
pub mod cudnn_conv;
pub mod fused;
pub mod graph_cache;
pub mod seres_net;
pub mod weights;

/// Held while recording a CUDA graph, creating a CUDA stream, or creating a
/// cuDNN handle. Each of the latter two, done on one thread while another
/// thread records a graph, makes the recording fail, so they never overlap.
pub fn cuda_setup_lock() -> CudaSetupGuard {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // Recording a graph creates cuDNN handles on the same thread, so a thread
    // that already holds the lock takes it again without waiting.
    let guard = if SETUP_DEPTH.with(|d| d.get()) == 0 {
        Some(LOCK.lock().unwrap_or_else(|e| e.into_inner()))
    } else {
        None
    };
    SETUP_DEPTH.with(|d| d.set(d.get() + 1));
    CudaSetupGuard { _guard: guard }
}

thread_local! {
    static SETUP_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Holds [`cuda_setup_lock`] until dropped.
pub struct CudaSetupGuard {
    _guard: Option<std::sync::MutexGuard<'static, ()>>,
}

impl Drop for CudaSetupGuard {
    fn drop(&mut self) {
        SETUP_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

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

    /// The precision used when none is named: f16 on a CUDA device when cuDNN
    /// is compiled in (the fastest measured setting), f32 otherwise.
    pub fn default_for(device: &candle_core::Device) -> Self {
        if cfg!(feature = "cudnn") && device.is_cuda() {
            Self::F16
        } else {
            Self::F32
        }
    }

    /// The precision `name`d (as KF_PRECISION would give it), or `default`.
    /// Panics on an unknown name, so a typo cannot silently fall back.
    pub fn from_name_or(name: Option<&str>, default: Self) -> Self {
        match name {
            Some(s) => Self::parse(s)
                .unwrap_or_else(|| panic!("KF_PRECISION must be f32, f16 or bf16, not {s:?}")),
            None => default,
        }
    }

    /// From the KF_PRECISION environment variable, or `default` when unset.
    pub fn from_env_or(default: Self) -> Self {
        Self::from_name_or(std::env::var("KF_PRECISION").ok().as_deref(), default)
    }
}

/// CUDA device `ordinal` on its default stream. Created under
/// [`cuda_setup_lock`], since creating its cuBLAS handle while another thread
/// records a CUDA graph makes the recording fail.
#[cfg(feature = "cuda")]
pub fn cuda_device(ordinal: usize) -> Result<candle_core::Device> {
    let _setup = cuda_setup_lock();
    candle_core::Device::new_cuda(ordinal)
}

/// CUDA device `ordinal` on a new stream of its own, with candle's cross-stream
/// event tracking off.
///
/// A second stream switches candle to recording and waiting on CUDA events
/// around every tensor access (hundreds of calls per batch), which also makes
/// any CUDA graph being recorded at that moment fail. Callers keep all their
/// tensors on this stream, so no events are needed. The setting is per CUDA
/// context, i.e. process-wide, so it is switched off again under the lock that
/// graph recording holds.
#[cfg(feature = "cuda")]
pub fn cuda_device_on_own_stream(ordinal: usize) -> Result<candle_core::Device> {
    let _setup = cuda_setup_lock();
    let device = candle_core::Device::new_cuda_with_stream(ordinal)?;
    if let candle_core::Device::Cuda(d) = &device {
        // SAFETY: the caller keeps every tensor of this device on its stream.
        unsafe { d.disable_event_tracking() };
    }
    Ok(device)
}
