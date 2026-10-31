//! CUDA graph replay of the half-precision forward pass.
//!
//! An eager forward pass launches about 120 kernels and makes about 160 GPU
//! allocations; with several batching threads each launch takes ~9 µs, mostly
//! spent in the CUDA driver. A CUDA graph records the whole pass once and replays
//! it with one launch. A graph is fixed to one batch size, so batches are padded to
//! a few sizes ([`bucket`]) and each size gets its own graph, recorded the first
//! time it is needed.

/// The batch size a batch of `b` positions is padded to: 8 for up to 8
/// positions, then the next multiple of 16, never above `cap` (or `b`, if larger).
pub fn bucket(b: usize, cap: usize) -> usize {
    let size = if b <= 8 { 8 } else { b.div_ceil(16) * 16 };
    size.min(cap.max(b))
}

#[cfg(feature = "cudnn")]
pub use gpu::GraphCache;

#[cfg(feature = "cudnn")]
mod gpu {
    use super::bucket;
    use crate::nn::seres_net::SeresNet;
    use crate::tensor::NHWC_CHANNELS;
    use candle_core::cuda::cudarc::driver::{sys, CudaGraph};
    use candle_core::{DType, Device, Result, Tensor};
    use std::collections::HashMap;

    /// One recorded forward pass and the buffers it reads and writes.
    struct Entry {
        graph: CudaGraph,
        planes: Tensor,
        scalars: Tensor,
        logits: Tensor,
        value: Tensor,
    }

    /// Recorded forward passes of one channels-last network, one per batch size,
    /// on one CUDA stream. Each batching thread owns one.
    pub struct GraphCache {
        cap: usize,
        entries: HashMap<usize, Entry>,
        /// Whether the eager pass at `cap` has run, which sizes cuDNN's workspace
        /// for every smaller batch before any graph records a pointer to it.
        warmed: bool,
    }

    // SAFETY: a CudaGraph holds raw driver handles. Each cache is created, filled
    // and replayed by the one thread that owns its network; it is only moved, never
    // shared.
    unsafe impl Send for GraphCache {}

    impl GraphCache {
        pub fn new(cap: usize) -> Self {
            Self {
                cap,
                entries: HashMap::new(),
                warmed: false,
            }
        }

        /// The network's `forward_logits` on `planes: [B, 8, 8, NHWC_CHANNELS]` and
        /// `scalars: [B, 2]`, by replaying the graph for B's [`bucket`].
        ///
        /// The logits returned are a view of the graph's output buffer, which the
        /// next replay overwrites; read or copy them before then. The value is a
        /// copy.
        pub fn forward(
            &mut self,
            net: &SeresNet,
            planes: &Tensor,
            scalars: &Tensor,
        ) -> Result<(Tensor, Tensor)> {
            let b = planes.dim(0)?;
            let size = bucket(b, self.cap);
            if !self.entries.contains_key(&size) {
                if !self.warmed {
                    let top = bucket(self.cap, self.cap);
                    let (p, s) = zeros(top, planes.device())?;
                    net.forward_logits(&p, &s)?;
                    self.warmed = true;
                }
                let entry = record(net, size, planes.device())?;
                self.entries.insert(size, entry);
            }
            let e = &self.entries[&size];
            e.planes.slice_set(planes, 0, 0)?;
            e.scalars.slice_set(scalars, 0, 0)?;
            e.graph.launch().map_err(candle_core::Error::wrap)?;
            Ok((e.logits.narrow(0, 0, b)?, e.value.narrow(0, 0, b)?.copy()?))
        }
    }

    fn zeros(b: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        Ok((
            Tensor::zeros((b, 8, 8, NHWC_CHANNELS), DType::F32, dev)?,
            Tensor::zeros((b, 2), DType::F32, dev)?,
        ))
    }

    /// Record `net`'s forward pass at batch size `size`.
    fn record(net: &SeresNet, size: usize, dev: &Device) -> Result<Entry> {
        let Device::Cuda(cuda) = dev else {
            candle_core::bail!("CUDA graphs need a CUDA device");
        };
        let stream = cuda.cuda_stream();
        let (planes, scalars) = zeros(size, dev)?;
        let logits = Tensor::zeros((size, net.policy_width()), DType::F32, dev)?;
        let value = Tensor::zeros((size, 1), DType::F32, dev)?;
        // While recording, small host-to-device copies (shapes and strides) must
        // come from device-side copies made during warm-up; the cache is kept only
        // this long, since it keeps every distinct upload.
        let _cache = cuda.enable_cuda_graph_htod_cache();
        let run = || -> Result<()> {
            let (l, v) = net.forward_logits(&planes, &scalars)?;
            logits.slice_set(&l, 0, 0)?;
            value.slice_set(&v, 0, 0)
        };
        // Warm-up creates cuDNN plans, loads kernels and fills the cache.
        for _ in 0..2 {
            run()?;
        }
        stream.synchronize().map_err(candle_core::Error::wrap)?;
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(candle_core::Error::wrap)?;
        let captured = run();
        let graph = stream.end_capture(sys::CUgraphInstantiate_flags(0));
        captured?;
        let graph = graph
            .map_err(candle_core::Error::wrap)?
            .ok_or_else(|| candle_core::Error::Msg("CUDA graph capture recorded nothing".into()))?;
        graph.upload().map_err(candle_core::Error::wrap)?;
        Ok(Entry {
            graph,
            planes,
            scalars,
            logits,
            value,
        })
    }
}
