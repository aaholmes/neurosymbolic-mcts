//! The cached cuDNN convolution (`kingfisher::nn::cudnn_conv`) must match candle's
//! conv2d, for every precision, kernel size and batch size the network uses.

#![cfg(feature = "cudnn")]

use candle_core::{DType, Device, Tensor};
use kingfisher::nn::cudnn_conv::conv2d;

/// Largest |a - b| relative to the largest |b| (at least 1).
fn rel_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims());
    let a = a
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    let b = b
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    let max = |t: Tensor| {
        t.abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    max((&a - &b).unwrap()) / max(b).max(1.0)
}

/// (in channels, out channels, kernel): the start conv, a trunk conv, the policy
/// head and the value conv of a 6x128 net.
const SHAPES: [(usize, usize, usize); 4] = [(17, 128, 3), (128, 128, 3), (128, 73, 1), (128, 1, 1)];

#[test]
fn cached_conv_matches_candle_conv_in_every_precision() {
    let dev = Device::new_cuda(0).expect("CUDA device");
    for (cin, cout, k) in SHAPES {
        let w = Tensor::randn(0f32, 0.1, (cout, cin, k, k), &dev).unwrap();
        // Batch sizes alternate, so cached descriptors must be keyed by batch.
        for b in [1, 7, 64, 130, 7] {
            let x = Tensor::randn(0f32, 1., (b, cin, 8, 8), &dev).unwrap();
            let reference = x.conv2d(&w, k / 2, 1, 1, 1).unwrap();
            for (dtype, tol) in [(DType::F32, 2e-3), (DType::F16, 5e-3), (DType::BF16, 3e-2)] {
                let got = conv2d(
                    &x.to_dtype(dtype).unwrap(),
                    &w.to_dtype(dtype).unwrap(),
                    k / 2,
                )
                .unwrap();
                assert_eq!(got.dtype(), dtype);
                let d = rel_diff(&got, &reference);
                assert!(
                    d <= tol,
                    "{dtype:?} cin {cin} cout {cout} k {k} batch {b}: {d}"
                );
            }
        }
    }
}

#[test]
fn cached_conv_rejects_mismatched_channels() {
    let dev = Device::new_cuda(0).expect("CUDA device");
    let x = Tensor::zeros((2, 16, 8, 8), DType::F32, &dev).unwrap();
    let w = Tensor::zeros((8, 17, 3, 3), DType::F32, &dev).unwrap();
    assert!(conv2d(&x, &w, 1).is_err());
}
