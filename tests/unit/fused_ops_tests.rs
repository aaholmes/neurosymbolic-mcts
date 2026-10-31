//! Fused elementwise kernels (`kingfisher::nn::fused`) must match the candle
//! ops they replace, on the CPU and (with CUDA) on the GPU.

#![cfg(feature = "neural")]

use candle_core::{Device, Tensor};
use kingfisher::nn::fused::{bias_act, channel_mean_bias, se_tail};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims());
    let a = a.to_device(&Device::Cpu).unwrap();
    let b = b.to_device(&Device::Cpu).unwrap();
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// B=3, C=20, 8x8: 60 channel rows, which does not fill whole thread blocks.
fn inputs(dev: &Device) -> (Tensor, Tensor, Tensor, Tensor) {
    let x = Tensor::randn(0f32, 1., (3, 20, 8, 8), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 1., 20, &Device::Cpu).unwrap();
    let gate = Tensor::rand(0f32, 1., (3, 20), &Device::Cpu).unwrap();
    let res = Tensor::randn(0f32, 1., (3, 20, 8, 8), &Device::Cpu).unwrap();
    (
        x.to_device(dev).unwrap(),
        bias.to_device(dev).unwrap(),
        gate.to_device(dev).unwrap(),
        res.to_device(dev).unwrap(),
    )
}

fn unfused_bias_act(x: &Tensor, bias: &Tensor, relu: bool) -> Tensor {
    let y = x
        .broadcast_add(&bias.reshape((1, (), 1, 1)).unwrap())
        .unwrap();
    if relu {
        y.relu().unwrap()
    } else {
        y
    }
}

fn unfused_mean(x: &Tensor, bias: &Tensor) -> Tensor {
    x.mean((2, 3))
        .unwrap()
        .broadcast_add(&bias.unsqueeze(0).unwrap())
        .unwrap()
}

fn unfused_tail(x: &Tensor, bias: &Tensor, gate: &Tensor, res: &Tensor) -> Tensor {
    let (b, c) = gate.dims2().unwrap();
    let y = unfused_bias_act(x, bias, false);
    (y.broadcast_mul(&gate.reshape((b, c, 1, 1)).unwrap())
        .unwrap()
        + res)
        .unwrap()
        .relu()
        .unwrap()
}

fn check_all(dev: &Device, tol: f32) {
    let (x, bias, gate, res) = inputs(dev);
    for relu in [false, true] {
        let d = max_abs_diff(
            &bias_act(&x, &bias, relu).unwrap(),
            &unfused_bias_act(&x, &bias, relu),
        );
        assert!(d <= tol, "bias_act relu={relu}: {d}");
    }
    let d = max_abs_diff(
        &channel_mean_bias(&x, &bias).unwrap(),
        &unfused_mean(&x, &bias),
    );
    assert!(d <= tol, "channel_mean_bias: {d}");
    let d = max_abs_diff(
        &se_tail(&x, &bias, &gate, &res).unwrap(),
        &unfused_tail(&x, &bias, &gate, &res),
    );
    assert!(d <= tol, "se_tail: {d}");
}

#[test]
fn fused_ops_match_unfused_on_cpu() {
    check_all(&Device::Cpu, 1e-6);
}

#[test]
fn fused_ops_accept_non_contiguous_input() {
    let (x, bias, _, _) = inputs(&Device::Cpu);
    // Same values, transposed in memory and back
    let xt = x.transpose(2, 3).unwrap().contiguous().unwrap();
    let xt = xt.transpose(2, 3).unwrap();
    assert!(!xt.is_contiguous());
    let d = max_abs_diff(
        &bias_act(&xt, &bias, true).unwrap(),
        &bias_act(&x, &bias, true).unwrap(),
    );
    assert_eq!(d, 0.0);
}

#[test]
fn fused_ops_reject_mismatched_bias() {
    let (x, _, _, _) = inputs(&Device::Cpu);
    let wrong = Tensor::zeros(7, candle_core::DType::F32, &Device::Cpu).unwrap();
    assert!(bias_act(&x, &wrong, true).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn fused_ops_match_unfused_on_gpu() {
    let dev = Device::new_cuda(0).expect("CUDA device");
    // The GPU may contract multiply-add into one rounding, so allow a few ulps.
    check_all(&dev, 1e-5);
}

#[cfg(feature = "cuda")]
#[test]
fn tiny_net_on_gpu_matches_pytorch_golden_outputs() {
    use kingfisher::nn::oracle_net::OracleNet;
    use kingfisher::nn::weights::load_state_dict;
    use kingfisher::nn::PolicyValueNet;
    use std::path::Path;

    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let dev = Device::new_cuda(0).expect("CUDA device");
    let sd = load_state_dict(&fixtures.join("oraclenet_tiny.safetensors"), &dev).unwrap();
    let net = OracleNet::from_state_dict(sd, &dev).unwrap();
    let io = candle_core::safetensors::load(fixtures.join("oraclenet_tiny_io.safetensors"), &dev)
        .unwrap();
    let out = net.forward(&io["input"], &io["scalars"]).unwrap();
    assert!(max_abs_diff(&out.log_policy, &io["log_policy"]) < 1e-4);
    assert!(max_abs_diff(&out.v_logit, &io["v_logit"]) < 1e-4);
}

/// Largest |a - b| relative to the largest |b| (at least 1).
#[cfg(feature = "cuda")]
fn rel_diff(a: &Tensor, b: &Tensor) -> f32 {
    let b32 = b.to_dtype(candle_core::DType::F32).unwrap();
    let scale = b32
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .max(1.0);
    max_abs_diff(&a.to_dtype(candle_core::DType::F32).unwrap(), &b32) / scale
}

/// The half-precision kernels compute in f32 and round once on output, so they
/// match the f32 kernels to within the rounding of their inputs and outputs.
#[cfg(feature = "cuda")]
#[test]
fn fused_ops_in_half_precision_match_f32_on_gpu() {
    use candle_core::DType;
    let dev = Device::new_cuda(0).expect("CUDA device");
    let (x, bias, gate, res) = inputs(&dev);
    for (dtype, tol) in [(DType::F16, 2e-3), (DType::BF16, 2e-2)] {
        let h = |t: &Tensor| t.to_dtype(dtype).unwrap();
        for relu in [false, true] {
            let got = bias_act(&h(&x), &h(&bias), relu).unwrap();
            assert_eq!(got.dtype(), dtype);
            let d = rel_diff(&got, &bias_act(&x, &bias, relu).unwrap());
            assert!(d <= tol, "{dtype:?} bias_act relu={relu}: {d}");
        }
        let got = channel_mean_bias(&h(&x), &h(&bias)).unwrap();
        assert_eq!(got.dtype(), dtype);
        let d = rel_diff(&got, &channel_mean_bias(&x, &bias).unwrap());
        assert!(d <= tol, "{dtype:?} channel_mean_bias: {d}");
        let got = se_tail(&h(&x), &h(&bias), &h(&gate), &h(&res)).unwrap();
        assert_eq!(got.dtype(), dtype);
        let d = rel_diff(&got, &se_tail(&x, &bias, &gate, &res).unwrap());
        assert!(d <= tol, "{dtype:?} se_tail: {d}");
    }
}

#[test]
fn fused_ops_reject_half_precision_on_cpu() {
    let (x, bias, _, _) = inputs(&Device::Cpu);
    let h = |t: &Tensor| t.to_dtype(candle_core::DType::F16).unwrap();
    let err = bias_act(&h(&x), &h(&bias), true).unwrap_err().to_string();
    assert!(err.contains("CUDA"), "unexpected error: {err}");
}
