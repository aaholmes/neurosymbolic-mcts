//! A stride-1 cuDNN convolution that keeps its setup between calls.
//!
//! candle's cuDNN convolution creates its descriptors and allocates and zeroes a
//! workspace on every call, which costs more than the convolution itself at
//! self-play batch sizes. In f16 and bf16 it also accumulates in half precision
//! without tensor-core math, so cuDNN falls back to slow kernels. Here the
//! descriptors are built once per (stream, precision, shape), the workspace is
//! kept per stream, and half-precision inputs use tensor cores with f32
//! accumulation.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::cudnn::{
    sys, ConvDescriptor, ConvForward, Cudnn, CudnnDataType, FilterDescriptor, TensorDescriptor,
};
use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits,
};
use candle_core::cuda_backend::{CudaDType, WrapErr};
use candle_core::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor, WithDType,
};
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// `conv2d(x, w)` with stride 1 and `padding` on each side, for `x: [B, Cin, H, W]`
/// and `w: [Cout, Cin, K, K]` on a CUDA device, in f32, f16 or bf16.
pub fn conv2d(x: &Tensor, w: &Tensor, padding: usize) -> Result<Tensor> {
    let (_, cin, _, _) = x.dims4()?;
    let (_, cin_w, kh, kw) = w.dims4()?;
    if cin != cin_w || kh != kw {
        candle_core::bail!(
            "conv2d: input {:?} does not fit filter {:?}",
            x.dims(),
            w.dims()
        );
    }
    x.contiguous()?
        .apply_op2_no_bwd(&w.contiguous()?, &Conv { padding })
}

struct Conv {
    padding: usize,
}

impl CustomOp2 for Conv {
    fn name(&self) -> &'static str {
        "kf-cudnn-conv2d"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("kf-cudnn-conv2d runs on CUDA only")
    }

    fn cuda_fwd(
        &self,
        xs: &CudaStorage,
        xl: &Layout,
        ws: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match xs.dtype() {
            DType::F32 => self.run::<f32>(xs, xl, ws, wl),
            DType::F16 => self.run::<half::f16>(xs, xl, ws, wl),
            DType::BF16 => self.run::<half::bf16>(xs, xl, ws, wl),
            d => candle_core::bail!("kf-cudnn-conv2d does not support {d:?}"),
        }
    }
}

/// Descriptors and algorithm for one (stream, precision, shape).
struct Plan<T: CudnnDataType> {
    conv: ConvDescriptor<f32>,
    x: TensorDescriptor<T>,
    w: FilterDescriptor<T>,
    y: TensorDescriptor<T>,
    algo: sys::cudnnConvolutionFwdAlgo_t,
    workspace: usize,
}

type Key = (usize, DType, [usize; 6]);

thread_local! {
    static HANDLES: RefCell<HashMap<usize, Arc<Cudnn>>> = RefCell::new(HashMap::new());
    static PLANS: RefCell<HashMap<Key, Box<dyn Any>>> = RefCell::new(HashMap::new());
    static WORKSPACES: RefCell<HashMap<usize, CudaSlice<u8>>> = RefCell::new(HashMap::new());
}

impl Conv {
    fn run<T>(
        &self,
        xs: &CudaStorage,
        xl: &Layout,
        ws: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: CudnnDataType + CudaDType + DeviceRepr + ValidAsZeroBits + WithDType,
    {
        let (b, cin, h, w) = xl.shape().dims4()?;
        let (cout, _, k, _) = wl.shape().dims4()?;
        let p = self.padding;
        let (ho, wo) = (h + 2 * p + 1 - k, w + 2 * p + 1 - k);
        let stream: Arc<CudaStream> = xs.device.cuda_stream();
        let sid = Arc::as_ptr(&stream) as usize;
        let key: Key = (sid, T::DTYPE, [b, cin, h, w, cout, k]);

        let cudnn = HANDLES.with(|m| -> Result<Arc<Cudnn>> {
            if let Some(c) = m.borrow().get(&sid) {
                return Ok(c.clone());
            }
            let c = Cudnn::new(stream.clone())?;
            m.borrow_mut().insert(sid, c.clone());
            Ok(c)
        })?;

        let x = xs.as_cuda_slice::<T>()?;
        let x = x.slice(xl.start_offset()..xl.start_offset() + xl.shape().elem_count());
        let f = ws.as_cuda_slice::<T>()?;
        let f = f.slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
        // SAFETY: cuDNN writes every output element (beta = 0).
        let mut out = unsafe { stream.alloc::<T>(b * cout * ho * wo) }.w()?;

        PLANS.with(|plans| -> Result<()> {
            let mut plans = plans.borrow_mut();
            let plan =
                match plans.entry(key) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => e.insert(Box::new(
                        make_plan::<T>(&cudnn, [b, cin, h, w], [cout, k], p, [ho, wo])?,
                    )),
                };
            let plan = plan
                .downcast_ref::<Plan<T>>()
                .expect("plans are keyed by dtype");
            WORKSPACES.with(|wsm| -> Result<()> {
                let mut wsm = wsm.borrow_mut();
                let too_small = wsm.get(&sid).is_none_or(|w| w.len() < plan.workspace);
                if too_small {
                    // SAFETY: cuDNN uses the workspace as scratch space only.
                    let buf = unsafe { stream.alloc::<u8>(plan.workspace.max(1)) }.w()?;
                    wsm.insert(sid, buf);
                }
                let workspace = wsm.get_mut(&sid).expect("just inserted");
                let fwd = ConvForward {
                    conv: &plan.conv,
                    x: &plan.x,
                    w: &plan.w,
                    y: &plan.y,
                };
                let one = T::from_f64(1.0);
                let zero = T::from_f64(0.0);
                // SAFETY: the descriptors were built for exactly these shapes and dtype.
                unsafe { fwd.launch(plan.algo, Some(workspace), (one, zero), &x, &f, &mut out) }?;
                Ok(())
            })
        })?;

        Ok((
            CudaStorage::wrap_cuda_slice(out, xs.device.clone()),
            Shape::from((b, cout, ho, wo)),
        ))
    }
}

fn make_plan<T: CudnnDataType>(
    cudnn: &Arc<Cudnn>,
    [b, cin, h, w]: [usize; 4],
    [cout, k]: [usize; 2],
    pad: usize,
    [ho, wo]: [usize; 2],
) -> Result<Plan<T>> {
    let i = |v: usize| v as i32;
    let nchw = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    let mut conv = cudnn.create_conv2d::<f32>(
        [i(pad), i(pad)],
        [1, 1],
        [1, 1],
        sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    // f32 stays exact (no TF32 tensor cores), so it remains the reference;
    // half precision needs tensor-op math explicitly.
    let math = if T::DATA_TYPE == f32::DATA_TYPE {
        sys::cudnnMathType_t::CUDNN_FMA_MATH
    } else {
        sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH
    };
    conv.set_math_type(math)?;
    let x = cudnn.create_4d_tensor::<T>(nchw, [i(b), i(cin), i(h), i(w)])?;
    let wd = cudnn.create_4d_filter::<T>(nchw, [i(cout), i(cin), i(k), i(k)])?;
    let y = cudnn.create_4d_tensor::<T>(nchw, [i(b), i(cout), i(ho), i(wo)])?;
    let algo = sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    let workspace = ConvForward {
        conv: &conv,
        x: &x,
        w: &wd,
        y: &y,
    }
    .get_workspace_size(algo)?;
    Ok(Plan {
        conv,
        x,
        w: wd,
        y,
        algo,
        workspace,
    })
}
