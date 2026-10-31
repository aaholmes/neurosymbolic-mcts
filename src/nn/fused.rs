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
        _ => candle_core::bail!("fused ops on the CPU support f32 only; half precision needs CUDA"),
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
        let params = [n as u32, c as u32, hw as u32, self.relu as u32];
        let out = gpu::dispatch!(s1, T, suffix => {
            let out = gpu::launch_elementwise::<T>(
                &format!("kf_bias_act{suffix}"),
                &s1.device,
                n,
                &[gpu::view::<T>(s1, l1)?, gpu::view::<T>(s2, l2)?],
                &params,
            )?;
            gpu::wrap(out, s1)
        });
        Ok((out, l1.shape().clone()))
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
        let params = [(b * c) as u32, c as u32, hw as u32];
        let out = gpu::dispatch!(s1, T, suffix => {
            let out = gpu::launch_rows::<T>(
                &format!("kf_channel_mean_bias{suffix}"),
                &s1.device,
                b * c,
                &[gpu::view::<T>(s1, l1)?, gpu::view::<T>(s2, l2)?],
                &params,
            )?;
            gpu::wrap(out, s1)
        });
        Ok((out, Shape::from((b, c))))
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
        let params = [n as u32, c as u32, hw as u32];
        let out = gpu::dispatch!(s1, T, suffix => {
            let out = gpu::launch_elementwise::<T>(
                &format!("kf_se_tail{suffix}"),
                &s1.device,
                n,
                &[
                    gpu::view::<T>(s1, l1)?,
                    gpu::view::<T>(bias_storage, bias_layout)?,
                    gpu::view::<T>(s2, l2)?,
                    gpu::view::<T>(s3, l3)?,
                ],
                &params,
            )?;
            gpu::wrap(out, s1)
        });
        Ok((out, l1.shape().clone()))
    }
}

#[cfg(feature = "cuda")]
mod gpu {
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
    };
    use candle_core::cuda_backend::cudarc::nvrtc;
    use candle_core::cuda_backend::{CudaDType, WrapErr};
    use candle_core::{CudaDevice, CudaStorage, Layout, Result};
    use std::sync::OnceLock;

    const MODULE: &str = "kingfisher_fused";

    /// Element types the kernels are compiled for.
    pub(super) trait Elem: CudaDType + DeviceRepr + ValidAsZeroBits + Copy {}
    impl Elem for f32 {}
    impl Elem for half::f16 {}
    impl Elem for half::bf16 {}

    /// Run `$body` with `$t` bound to the element type of `$storage` and `$suffix`
    /// to its kernel-name suffix ("", "_f16" or "_bf16").
    macro_rules! dispatch {
        ($storage:expr, $t:ident, $suffix:ident => $body:block) => {{
            use candle_core::backend::BackendStorage;
            match $storage.dtype() {
                candle_core::DType::F32 => {
                    type $t = f32;
                    let $suffix = "";
                    $body
                }
                candle_core::DType::F16 => {
                    type $t = half::f16;
                    let $suffix = "_f16";
                    $body
                }
                candle_core::DType::BF16 => {
                    type $t = half::bf16;
                    let $suffix = "_bf16";
                    $body
                }
                d => candle_core::bail!("fused ops do not support {d:?}"),
            }
        }};
    }
    pub(super) use dispatch;

    // Each kernel is written once as a template computing in f32, and compiled for
    // f32, f16 and bf16 storage. Half values are converted with plain PTX and bit
    // operations, so NVRTC needs no CUDA headers.
    const SOURCE: &str = r#"
struct h16 { unsigned short x; };
struct b16 { unsigned short x; };

__device__ __forceinline__ float tof(float v) { return v; }
__device__ __forceinline__ float tof(h16 v) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(v.x)); return f;
}
__device__ __forceinline__ float tof(b16 v) { return __int_as_float(((unsigned int)v.x) << 16); }

__device__ __forceinline__ void put(float* p, float f) { *p = f; }
__device__ __forceinline__ void put(h16* p, float f) {
    unsigned short r; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f)); p->x = r;
}
__device__ __forceinline__ void put(b16* p, float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) { p->x = 0x7fc0; return; }  // NaN
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest, ties to even
    p->x = (unsigned short)(u >> 16);
}

template <typename T>
__device__ void bias_act(const T* x, const T* b, T* y, unsigned int n, unsigned int c,
                         unsigned int hw, unsigned int relu) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = tof(x[i]) + tof(b[(i / hw) % c]);
    put(&y[i], (relu && v < 0.0f) ? 0.0f : v);
}

// One warp per (batch, channel) row of hw contiguous values.
template <typename T>
__device__ void channel_mean_bias(const T* x, const T* b, T* m, unsigned int rows,
                                  unsigned int c, unsigned int hw) {
    unsigned int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const T* p = x + (size_t)row * hw;
    float s = 0.0f;
    for (unsigned int j = lane; j < hw; j += 32) s += tof(p[j]);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) put(&m[row], s / (float)hw + tof(b[row % c]));
}

template <typename T>
__device__ void se_tail(const T* x, const T* b, const T* g, const T* r, T* y, unsigned int n,
                        unsigned int c, unsigned int hw) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int row = i / hw;
    float v = (tof(x[i]) + tof(b[row % c])) * tof(g[row]) + tof(r[i]);
    put(&y[i], v < 0.0f ? 0.0f : v);
}

#define KF_KERNELS(T, SUFFIX)                                                              \
extern "C" __global__ void kf_bias_act##SUFFIX(const T* x, const T* b, T* y,               \
        unsigned int n, unsigned int c, unsigned int hw, unsigned int relu) {               \
    bias_act<T>(x, b, y, n, c, hw, relu);                                                   \
}                                                                                           \
extern "C" __global__ void kf_channel_mean_bias##SUFFIX(const T* x, const T* b, T* m,      \
        unsigned int rows, unsigned int c, unsigned int hw) {                               \
    channel_mean_bias<T>(x, b, m, rows, c, hw);                                             \
}                                                                                           \
extern "C" __global__ void kf_se_tail##SUFFIX(const T* x, const T* b, const T* g,          \
        const T* r, T* y, unsigned int n, unsigned int c, unsigned int hw) {                \
    se_tail<T>(x, b, g, r, y, n, c, hw);                                                    \
}

KF_KERNELS(float, )
KF_KERNELS(h16, _f16)
KF_KERNELS(b16, _bf16)
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

    pub(super) fn view<'a, T: Elem>(s: &'a CudaStorage, l: &Layout) -> Result<CudaView<'a, T>> {
        let start = l.start_offset();
        Ok(s.as_cuda_slice::<T>()?
            .slice(start..start + l.shape().elem_count()))
    }

    pub(super) fn wrap<T: Elem>(out: CudaSlice<T>, like: &CudaStorage) -> CudaStorage {
        CudaStorage::wrap_cuda_slice(out, like.device.clone())
    }

    fn launch<T: Elem>(
        name: &str,
        dev: &CudaDevice,
        out_len: usize,
        cfg: LaunchConfig,
        inputs: &[CudaView<'_, T>],
        params: &[u32],
    ) -> Result<CudaSlice<T>> {
        let func = dev.get_or_load_custom_func(name, MODULE, ptx()?)?;
        // SAFETY: every element is written by the kernel.
        let out = unsafe { dev.alloc::<T>(out_len)? };
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
    pub(super) fn launch_elementwise<T: Elem>(
        name: &str,
        dev: &CudaDevice,
        n: usize,
        inputs: &[CudaView<'_, T>],
        params: &[u32],
    ) -> Result<CudaSlice<T>> {
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
    pub(super) fn launch_rows<T: Elem>(
        name: &str,
        dev: &CudaDevice,
        rows: usize,
        inputs: &[CudaView<'_, T>],
        params: &[u32],
    ) -> Result<CudaSlice<T>> {
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
