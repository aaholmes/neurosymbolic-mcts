//! Fused elementwise kernels for the OracleNet forward pass.
//!
//! candle runs bias adds, ReLUs and the squeeze-and-excitation (SE) scaling as
//! separate broadcast kernels, each of which re-reads the whole activation and
//! computes strided indices per element. On a 6x128 net at self-play batch sizes
//! those kernels took about 40% of GPU time. Each function here does the same
//! arithmetic in one pass over contiguous memory.

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Storage, Tensor};

/// `act(x + bias[c])` for `x: [B, C, H, W]`, `bias: [C]`; act is ReLU or identity.
pub fn bias_act(x: &Tensor, bias: &Tensor, relu: bool) -> Result<Tensor> {
    let dims = Dims::of(x, bias)?;
    x.contiguous()?
        .apply_op2_no_bwd(&bias.contiguous()?, &BiasAct { dims, relu })
}

/// `mean_{h,w}(x) + bias[c]` for `x: [B, C, H, W]`, giving `[B, C]`.
pub fn channel_mean_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dims = Dims::of(x, bias)?;
    x.contiguous()?
        .apply_op2_no_bwd(&bias.contiguous()?, &ChannelMeanBias { dims })
}

/// Squeeze-and-excitation tail: `relu((x + bias[c]) * gate[b, c] + residual)`,
/// for `x, residual: [B, C, H, W]` and `gate: [B, C]`.
pub fn se_tail(x: &Tensor, bias: &Tensor, gate: &Tensor, residual: &Tensor) -> Result<Tensor> {
    let dims = Dims::of(x, bias)?;
    if gate.dims() != [dims.b, dims.c] || residual.dims() != x.dims() {
        candle_core::bail!(
            "se_tail: gate {:?} / residual {:?} do not fit x {:?}",
            gate.dims(),
            residual.dims(),
            x.dims()
        );
    }
    x.contiguous()?.apply_op3_no_bwd(
        &gate.contiguous()?,
        &residual.contiguous()?,
        &SeTail {
            dims,
            bias: bias.contiguous()?,
        },
    )
}

#[derive(Clone, Copy)]
struct Dims {
    b: usize,
    c: usize,
    hw: usize,
}

impl Dims {
    fn of(x: &Tensor, bias: &Tensor) -> Result<Self> {
        let (b, c, h, w) = x.dims4()?;
        if bias.dims() != [c] {
            candle_core::bail!("bias {:?} does not match {c} channels", bias.dims());
        }
        Ok(Self { b, c, hw: h * w })
    }
    #[cfg(feature = "cuda")]
    fn n(&self) -> usize {
        self.b * self.c * self.hw
    }
}

fn cpu_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    match s {
        CpuStorage::F32(v) => Ok(&v[l.start_offset()..l.start_offset() + l.shape().elem_count()]),
        _ => candle_core::bail!("fused ops support f32 only"),
    }
}

struct BiasAct {
    dims: Dims,
    relu: bool,
}

impl CustomOp2 for BiasAct {
    fn name(&self) -> &'static str {
        "kf-bias-act"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, bias) = (cpu_f32(s1, l1)?, cpu_f32(s2, l2)?);
        let Dims { c, hw, .. } = self.dims;
        let y = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let v = v + bias[(i / hw) % c];
                if self.relu {
                    v.max(0.0)
                } else {
                    v
                }
            })
            .collect();
        Ok((CpuStorage::F32(y), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let Dims { c, hw, .. } = self.dims;
        let n = self.dims.n();
        let out = gpu::launch_elementwise(
            "kf_bias_act",
            &s1.device,
            n,
            &[gpu::view(s1, l1)?, gpu::view(s2, l2)?],
            &[n as u32, c as u32, hw as u32, self.relu as u32],
        )?;
        Ok((gpu::wrap(out, s1), l1.shape().clone()))
    }
}

struct ChannelMeanBias {
    dims: Dims,
}

impl CustomOp2 for ChannelMeanBias {
    fn name(&self) -> &'static str {
        "kf-channel-mean-bias"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, bias) = (cpu_f32(s1, l1)?, cpu_f32(s2, l2)?);
        let Dims { b, c, hw } = self.dims;
        let m = x
            .chunks_exact(hw)
            .enumerate()
            .map(|(row, chunk)| chunk.iter().sum::<f32>() / hw as f32 + bias[row % c])
            .collect();
        Ok((CpuStorage::F32(m), Shape::from((b, c))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let Dims { b, c, hw } = self.dims;
        let out = gpu::launch_rows(
            "kf_channel_mean_bias",
            &s1.device,
            b * c,
            &[gpu::view(s1, l1)?, gpu::view(s2, l2)?],
            &[(b * c) as u32, c as u32, hw as u32],
        )?;
        Ok((gpu::wrap(out, s1), Shape::from((b, c))))
    }
}

struct SeTail {
    dims: Dims,
    bias: Tensor,
}

impl CustomOp3 for SeTail {
    fn name(&self) -> &'static str {
        "kf-se-tail"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, gate, res) = (cpu_f32(s1, l1)?, cpu_f32(s2, l2)?, cpu_f32(s3, l3)?);
        let (bias_storage, bias_layout) = self.bias.storage_and_layout();
        let Storage::Cpu(bias_storage) = &*bias_storage else {
            candle_core::bail!("se_tail: bias is not on the CPU");
        };
        let bias = cpu_f32(bias_storage, bias_layout)?;
        let Dims { c, hw, .. } = self.dims;
        let y = x
            .iter()
            .zip(res)
            .enumerate()
            .map(|(i, (&v, &r))| {
                let row = i / hw;
                ((v + bias[row % c]) * gate[row] + r).max(0.0)
            })
            .collect();
        Ok((CpuStorage::F32(y), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let (bias_storage, bias_layout) = self.bias.storage_and_layout();
        let Storage::Cuda(bias_storage) = &*bias_storage else {
            candle_core::bail!("se_tail: bias is not on the GPU");
        };
        let Dims { c, hw, .. } = self.dims;
        let n = self.dims.n();
        let out = gpu::launch_elementwise(
            "kf_se_tail",
            &s1.device,
            n,
            &[
                gpu::view(s1, l1)?,
                gpu::view(bias_storage, bias_layout)?,
                gpu::view(s2, l2)?,
                gpu::view(s3, l3)?,
            ],
            &[n as u32, c as u32, hw as u32],
        )?;
        Ok((gpu::wrap(out, s1), l1.shape().clone()))
    }
}

#[cfg(feature = "cuda")]
mod gpu {
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::cudarc::nvrtc;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::{CudaDevice, CudaStorage, Layout, Result};
    use std::sync::OnceLock;

    const MODULE: &str = "kingfisher_fused";

    const SOURCE: &str = r#"
extern "C" __global__ void kf_bias_act(const float* x, const float* b, float* y,
                                       unsigned int n, unsigned int c, unsigned int hw,
                                       unsigned int relu) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] + b[(i / hw) % c];
    y[i] = (relu && v < 0.0f) ? 0.0f : v;
}

// One warp per (batch, channel) row of hw contiguous values.
extern "C" __global__ void kf_channel_mean_bias(const float* x, const float* b, float* m,
                                                unsigned int rows, unsigned int c,
                                                unsigned int hw) {
    unsigned int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* p = x + (size_t)row * hw;
    float s = 0.0f;
    for (unsigned int j = lane; j < hw; j += 32) s += p[j];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) m[row] = s / (float)hw + b[row % c];
}

extern "C" __global__ void kf_se_tail(const float* x, const float* b, const float* g,
                                      const float* r, float* y, unsigned int n,
                                      unsigned int c, unsigned int hw) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int row = i / hw;
    float v = (x[i] + b[row % c]) * g[row] + r[i];
    y[i] = v < 0.0f ? 0.0f : v;
}
"#;

    fn ptx() -> Result<&'static str> {
        static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
        PTX.get_or_init(|| {
            nvrtc::compile_ptx(SOURCE)
                .map(|p| p.to_src())
                .map_err(|e| format!("{e:?}"))
        })
        .as_deref()
        .map_err(|e| candle_core::Error::Msg(format!("compiling fused kernels: {e}")))
    }

    pub(super) fn view<'a>(s: &'a CudaStorage, l: &Layout) -> Result<CudaView<'a, f32>> {
        let start = l.start_offset();
        Ok(s.as_cuda_slice::<f32>()?
            .slice(start..start + l.shape().elem_count()))
    }

    pub(super) fn wrap(out: CudaSlice<f32>, like: &CudaStorage) -> CudaStorage {
        CudaStorage::wrap_cuda_slice(out, like.device.clone())
    }

    fn launch(
        name: &str,
        dev: &CudaDevice,
        out_len: usize,
        cfg: LaunchConfig,
        inputs: &[CudaView<'_, f32>],
        params: &[u32],
    ) -> Result<CudaSlice<f32>> {
        let func = dev.get_or_load_custom_func(name, MODULE, ptx()?)?;
        // SAFETY: every element is written by the kernel.
        let out = unsafe { dev.alloc::<f32>(out_len)? };
        let mut builder = func.builder();
        for input in inputs {
            builder.arg(input);
        }
        builder.arg(&out);
        for p in params {
            builder.arg(p);
        }
        // SAFETY: the argument list matches the kernel signature in SOURCE.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }

    /// One thread per output element.
    pub(super) fn launch_elementwise(
        name: &str,
        dev: &CudaDevice,
        n: usize,
        inputs: &[CudaView<'_, f32>],
        params: &[u32],
    ) -> Result<CudaSlice<f32>> {
        launch(
            name,
            dev,
            n,
            LaunchConfig::for_num_elems(n as u32),
            inputs,
            params,
        )
    }

    /// One warp per output row.
    pub(super) fn launch_rows(
        name: &str,
        dev: &CudaDevice,
        rows: usize,
        inputs: &[CudaView<'_, f32>],
        params: &[u32],
    ) -> Result<CudaSlice<f32>> {
        const THREADS: u32 = 256;
        let blocks = (rows as u32 * 32).div_ceil(THREADS);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        launch(name, dev, rows, cfg, inputs, params)
    }
}
