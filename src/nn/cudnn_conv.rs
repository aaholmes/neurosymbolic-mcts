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
    result, sys, ConvDescriptor, ConvForward, Cudnn, CudnnDataType, FilterDescriptor,
    TensorDescriptor,
};
use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
};
use candle_core::cuda_backend::{CudaDType, WrapErr};
use candle_core::{
    CpuStorage, CudaStorage, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor, WithDType,
};
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// `conv2d(x, w)` with stride 1 and `padding` on each side, for `x: [B, Cin, H, W]`
/// and `w: [Cout, Cin, K, K]` on a CUDA device, in f32, f16 or bf16.
pub fn conv2d(x: &Tensor, w: &Tensor, padding: usize) -> Result<Tensor> {
    conv_with(x, w, padding, false)
}

fn conv_with(x: &Tensor, w: &Tensor, padding: usize, nhwc: bool) -> Result<Tensor> {
    let Geometry {
        cin, cin_w, k, kw, ..
    } = Geometry::of(x, w, nhwc)?;
    if cin != cin_w || k != kw {
        candle_core::bail!(
            "conv2d: input {:?} does not fit filter {:?}",
            x.dims(),
            w.dims()
        );
    }
    x.contiguous()?
        .apply_op2_no_bwd(&w.contiguous()?, &Conv { padding, nhwc })
}

/// Sizes read from the input and filter shapes in either layout.
struct Geometry {
    b: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    cin_w: usize,
    k: usize,
    kw: usize,
}

impl Geometry {
    fn of(x: &Tensor, w: &Tensor, nhwc: bool) -> Result<Self> {
        Self::of_dims(x.dims(), w.dims(), nhwc)
    }

    fn of_dims(x: &[usize], w: &[usize], nhwc: bool) -> Result<Self> {
        if x.len() != 4 || w.len() != 4 {
            candle_core::bail!("conv2d: input {x:?} and filter {w:?} must be 4-D");
        }
        Ok(if nhwc {
            Self {
                b: x[0],
                h: x[1],
                w: x[2],
                cin: x[3],
                cout: w[0],
                k: w[1],
                kw: w[2],
                cin_w: w[3],
            }
        } else {
            Self {
                b: x[0],
                cin: x[1],
                h: x[2],
                w: x[3],
                cout: w[0],
                cin_w: w[1],
                k: w[2],
                kw: w[3],
            }
        })
    }

    fn out_shape(&self, pad: usize, nhwc: bool) -> Shape {
        let (ho, wo) = (self.h + 2 * pad + 1 - self.k, self.w + 2 * pad + 1 - self.k);
        if nhwc {
            Shape::from((self.b, ho, wo, self.cout))
        } else {
            Shape::from((self.b, self.cout, ho, wo))
        }
    }
}

fn format(nhwc: bool) -> sys::cudnnTensorFormat_t {
    if nhwc {
        sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC
    } else {
        sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW
    }
}

/// `relu?(conv2d(x, w) + bias)` in one cuDNN call, for `bias: [Cout]` in the dtype
/// of `x`; shapes and padding as for [`conv2d`]. Saves the separate bias kernel
/// and one pass over the output.
pub fn conv2d_bias_act(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    padding: usize,
    relu: bool,
) -> Result<Tensor> {
    conv_bias_act_with(x, w, bias, padding, relu, false)
}

fn conv_bias_act_with(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    padding: usize,
    relu: bool,
    nhwc: bool,
) -> Result<Tensor> {
    let Geometry {
        cin,
        cin_w,
        k,
        kw,
        cout,
        ..
    } = Geometry::of(x, w, nhwc)?;
    if cin != cin_w || k != kw || bias.dims() != [cout] {
        candle_core::bail!(
            "conv2d_bias_act: input {:?}, filter {:?} and bias {:?} do not fit",
            x.dims(),
            w.dims(),
            bias.dims()
        );
    }
    x.contiguous()?.apply_op3_no_bwd(
        &w.contiguous()?,
        &bias.contiguous()?,
        &FusedConv {
            padding,
            relu,
            nhwc,
        },
    )
}

/// As [`conv2d`], for channels-last `x: [B, H, W, Cin]` and `w: [Cout, K, K, Cin]`,
/// giving `[B, H', W', Cout]`.
pub fn conv2d_nhwc(x: &Tensor, w: &Tensor, padding: usize) -> Result<Tensor> {
    conv_with(x, w, padding, true)
}

/// As [`conv2d_bias_act`], in the channels-last layout of [`conv2d_nhwc`].
pub fn conv2d_bias_act_nhwc(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    padding: usize,
    relu: bool,
) -> Result<Tensor> {
    conv_bias_act_with(x, w, bias, padding, relu, true)
}

struct FusedConv {
    padding: usize,
    relu: bool,
    nhwc: bool,
}

impl CustomOp3 for FusedConv {
    fn name(&self) -> &'static str {
        "kf-cudnn-conv2d-bias-act"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("kf-cudnn-conv2d-bias-act runs on CUDA only")
    }

    fn cuda_fwd(
        &self,
        xs: &CudaStorage,
        xl: &Layout,
        ws: &CudaStorage,
        wl: &Layout,
        bs: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match xs.dtype() {
            DType::F16 => self.run::<half::f16>(xs, xl, ws, wl, bs, bl),
            DType::BF16 => self.run::<half::bf16>(xs, xl, ws, wl, bs, bl),
            d => candle_core::bail!("kf-cudnn-conv2d-bias-act does not support {d:?}"),
        }
    }
}

/// Raw cuDNN descriptors for one fused convolution, destroyed on drop. cudarc's safe
/// wrapper takes the unused residual input `z` and the output as separate borrows,
/// but with alpha2 = 0 cuDNN expects `z` to alias the output.
struct FusedPlan {
    x: sys::cudnnTensorDescriptor_t,
    w: sys::cudnnFilterDescriptor_t,
    y: sys::cudnnTensorDescriptor_t,
    bias: sys::cudnnTensorDescriptor_t,
    conv: sys::cudnnConvolutionDescriptor_t,
    act: sys::cudnnActivationDescriptor_t,
    workspace: usize,
}

impl Drop for FusedPlan {
    fn drop(&mut self) {
        // SAFETY: each descriptor was created by this plan and is destroyed once.
        unsafe {
            let _ = result::destroy_tensor_descriptor(self.x);
            let _ = result::destroy_filter_descriptor(self.w);
            let _ = result::destroy_tensor_descriptor(self.y);
            let _ = result::destroy_tensor_descriptor(self.bias);
            let _ = result::destroy_convolution_descriptor(self.conv);
            let _ = sys::cudnnDestroyActivationDescriptor(self.act);
        }
    }
}

/// A raw cuDNN handle bound to one stream.
struct RawHandle(sys::cudnnHandle_t);

impl Drop for RawHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `raw_handle` and is destroyed once.
        unsafe {
            let _ = result::destroy_handle(self.0);
        }
    }
}

type FusedKey = (usize, DType, [usize; 9]);

thread_local! {
    static RAW_HANDLES: RefCell<HashMap<usize, RawHandle>> = RefCell::new(HashMap::new());
    static FUSED_PLANS: RefCell<HashMap<FusedKey, FusedPlan>> = RefCell::new(HashMap::new());
}

fn raw_handle(sid: usize, stream: &CudaStream) -> Result<sys::cudnnHandle_t> {
    RAW_HANDLES.with(|m| -> Result<sys::cudnnHandle_t> {
        if let Some(h) = m.borrow().get(&sid) {
            return Ok(h.0);
        }
        let h = result::create_handle()?;
        // SAFETY: `h` is a fresh handle and the stream outlives this thread's use of it.
        unsafe { result::set_stream(h, stream.cu_stream() as _) }?;
        m.borrow_mut().insert(sid, RawHandle(h));
        Ok(h)
    })
}

fn make_fused_plan(
    handle: sys::cudnnHandle_t,
    dtype: sys::cudnnDataType_t,
    [b, cin, h, w, cout, k, pad, relu, nhwc]: [usize; 9],
) -> Result<FusedPlan> {
    let i = |v: usize| v as i32;
    let fmt = format(nhwc == 1);
    let nchw = sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    let (ho, wo) = (h + 2 * pad + 1 - k, w + 2 * pad + 1 - k);
    // SAFETY: every descriptor is freshly created before it is set, and owned by the plan.
    unsafe {
        let mut plan = FusedPlan {
            x: result::create_tensor_descriptor()?,
            w: result::create_filter_descriptor()?,
            y: result::create_tensor_descriptor()?,
            bias: result::create_tensor_descriptor()?,
            conv: result::create_convolution_descriptor()?,
            act: result::create_activation_descriptor()?,
            workspace: 0,
        };
        result::set_tensor4d_descriptor(plan.x, fmt, dtype, [i(b), i(cin), i(h), i(w)])?;
        result::set_filter4d_descriptor(plan.w, dtype, fmt, [i(cout), i(cin), i(k), i(k)])?;
        result::set_tensor4d_descriptor(plan.y, fmt, dtype, [i(b), i(cout), i(ho), i(wo)])?;
        result::set_tensor4d_descriptor(plan.bias, nchw, dtype, [1, i(cout), 1, 1])?;
        result::set_convolution2d_descriptor(
            plan.conv,
            i(pad),
            i(pad),
            1,
            1,
            1,
            1,
            sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
        )?;
        result::set_convolution_math_type(plan.conv, sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)?;
        let mode = if relu == 1 {
            sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU
        } else {
            sys::cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY
        };
        result::set_activation_descriptor(
            plan.act,
            mode,
            sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            0.0,
        )?;
        let workspace = result::get_convolution_forward_workspace_size(
            handle, plan.x, plan.w, plan.conv, plan.y, FUSED_ALGO,
        )?;
        plan.workspace = workspace;
        Ok(plan)
    }
}

/// cuDNN allows the identity activation only with this algorithm.
const FUSED_ALGO: sys::cudnnConvolutionFwdAlgo_t =
    sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

impl FusedConv {
    #[allow(clippy::too_many_arguments)]
    fn run<T>(
        &self,
        xs: &CudaStorage,
        xl: &Layout,
        ws: &CudaStorage,
        wl: &Layout,
        bs: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: CudnnDataType + CudaDType + DeviceRepr + ValidAsZeroBits + WithDType,
    {
        let g = Geometry::of_dims(xl.shape().dims(), wl.shape().dims(), self.nhwc)?;
        let Geometry {
            b,
            cin,
            h,
            w,
            cout,
            k,
            ..
        } = g;
        let p = self.padding;
        let out_shape = g.out_shape(p, self.nhwc);
        let stream: Arc<CudaStream> = xs.device.cuda_stream();
        let sid = Arc::as_ptr(&stream) as usize;
        let handle = raw_handle(sid, &stream)?;
        let dims = [
            b,
            cin,
            h,
            w,
            cout,
            k,
            p,
            self.relu as usize,
            self.nhwc as usize,
        ];
        let key: FusedKey = (sid, T::DTYPE, dims);

        let x = xs.as_cuda_slice::<T>()?;
        let x = x.slice(xl.start_offset()..xl.start_offset() + xl.shape().elem_count());
        let f = ws.as_cuda_slice::<T>()?;
        let f = f.slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
        let bias = bs.as_cuda_slice::<T>()?;
        let bias = bias.slice(bl.start_offset()..bl.start_offset() + bl.shape().elem_count());
        // SAFETY: cuDNN writes every output element (alpha2 = 0, so `z` is not used).
        let mut out = unsafe { stream.alloc::<T>(out_shape.elem_count()) }.w()?;

        FUSED_PLANS.with(|plans| -> Result<()> {
            let mut plans = plans.borrow_mut();
            let plan = match plans.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(make_fused_plan(handle, T::DATA_TYPE, dims)?)
                }
            };
            WORKSPACES.with(|wsm| -> Result<()> {
                let mut wsm = wsm.borrow_mut();
                if wsm.get(&sid).is_none_or(|w| w.len() < plan.workspace) {
                    // SAFETY: cuDNN uses the workspace as scratch space only.
                    let buf = unsafe { stream.alloc::<u8>(plan.workspace.max(1)) }.w()?;
                    wsm.insert(sid, buf);
                }
                let workspace = wsm.get_mut(&sid).expect("just inserted");
                let (one, zero) = (1f32, 0f32);
                let (xp, _gx) = x.device_ptr(&stream);
                let (fp, _gf) = f.device_ptr(&stream);
                let (bp, _gb) = bias.device_ptr(&stream);
                let (wp, _gw) = workspace.device_ptr_mut(&stream);
                let (yp, _gy) = out.device_ptr_mut(&stream);
                // SAFETY: the descriptors match these buffers' shapes and dtype; half-precision
                // data takes f32 scaling factors; `z` aliases `y` as cuDNN allows.
                unsafe {
                    result::convolution_bias_activation_forward(
                        handle,
                        (&one) as *const f32 as *const _,
                        plan.x,
                        xp as *const _,
                        plan.w,
                        fp as *const _,
                        plan.conv,
                        FUSED_ALGO,
                        wp as *mut _,
                        plan.workspace,
                        (&zero) as *const f32 as *const _,
                        plan.y,
                        yp as *const _,
                        plan.bias,
                        bp as *const _,
                        plan.act,
                        plan.y,
                        yp as *mut _,
                    )
                }?;
                Ok(())
            })
        })?;

        Ok((
            CudaStorage::wrap_cuda_slice(out, xs.device.clone()),
            out_shape,
        ))
    }
}

struct Conv {
    padding: usize,
    nhwc: bool,
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

type Key = (usize, DType, [usize; 7]);

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
        let g = Geometry::of_dims(xl.shape().dims(), wl.shape().dims(), self.nhwc)?;
        let Geometry {
            b,
            cin,
            h,
            w,
            cout,
            k,
            ..
        } = g;
        let p = self.padding;
        let out_shape = g.out_shape(p, self.nhwc);
        let (ho, wo) = (h + 2 * p + 1 - k, w + 2 * p + 1 - k);
        let stream: Arc<CudaStream> = xs.device.cuda_stream();
        let sid = Arc::as_ptr(&stream) as usize;
        let key: Key = (sid, T::DTYPE, [b, cin, h, w, cout, k, self.nhwc as usize]);

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
                        make_plan::<T>(&cudnn, [b, cin, h, w], [cout, k], p, [ho, wo], self.nhwc)?,
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
            out_shape,
        ))
    }
}

fn make_plan<T: CudnnDataType>(
    cudnn: &Arc<Cudnn>,
    [b, cin, h, w]: [usize; 4],
    [cout, k]: [usize; 2],
    pad: usize,
    [ho, wo]: [usize; 2],
    nhwc: bool,
) -> Result<Plan<T>> {
    let i = |v: usize| v as i32;
    let nchw = format(nhwc);
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
