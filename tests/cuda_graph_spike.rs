//! Spike: does replaying a captured CUDA graph of one SeresNet forward pass
//! match the normal forward pass, and how much per-call time does it save?
//!
//! BENCH_MODEL=weights/gen_34.pt \
//!   cargo test --release --features cudnn --test cuda_graph_spike -- --ignored --nocapture

use candle_core::cuda::cudarc::driver::sys;
use candle_core::{DType, Device, Tensor};
use kingfisher::nn::seres_net::SeresNet;
use kingfisher::nn::weights::load_state_dict;
use kingfisher::nn::PolicyValueNet;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;
use std::time::Instant;

const POLICY: usize = 4672;

fn host_inputs(bs: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<f32>) {
    let planes = (0..bs * 17 * 64)
        .map(|_| if rng.gen_bool(0.15) { 1.0 } else { 0.0 })
        .collect();
    let scalars = (0..bs)
        .flat_map(|_| [rng.gen_range(-5.0f32..5.0), 1.0])
        .collect();
    (planes, scalars)
}

fn median_us(mut f: impl FnMut()) -> (f64, f64, f64) {
    for _ in 0..50 {
        f();
    }
    let mut t: Vec<f64> = (0..500)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (t[250], t[125], t[375])
}

#[test]
#[ignore]
fn cuda_graph_spike() -> candle_core::Result<()> {
    let path = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "weights/gen_34.pt".to_string());
    let dev = Device::new_cuda_with_stream(0)?;
    let Device::Cuda(cuda) = &dev else {
        panic!("needs CUDA")
    };
    // Event tracking makes captured ops wait on uncaptured events; this device uses one stream.
    unsafe { cuda.disable_event_tracking() };
    let net = SeresNet::from_state_dict(load_state_dict(Path::new(&path), &dev).unwrap(), &dev)?;
    let stream = cuda.cuda_stream();
    let mut rng = StdRng::seed_from_u64(3);

    for &bs in &[1usize, 16, 32, 128] {
        let (hp, hs) = host_inputs(bs, &mut rng);

        // Normal path: upload, forward, read back.
        let eager = |hp: &[f32], hs: &[f32]| -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
            let x = Tensor::from_slice(hp, (bs, 17, 8, 8), &dev)?;
            let s = Tensor::from_slice(hs, (bs, 2), &dev)?;
            let out = net.forward(&x, &s)?;
            Ok((
                out.log_policy.flatten_all()?.to_vec1()?,
                out.v_logit.flatten_all()?.to_vec1()?,
            ))
        };

        // Graph path: fixed input and output buffers, captured once.
        let x_in = Tensor::zeros((bs, 17, 8, 8), DType::F32, &dev)?;
        let s_in = Tensor::zeros((bs, 2), DType::F32, &dev)?;
        let p_out = Tensor::zeros((bs, POLICY), DType::F32, &dev)?;
        let v_out = Tensor::zeros((bs, 1), DType::F32, &dev)?;
        let graph = {
            let _cache = cuda.enable_cuda_graph_htod_cache();
            // Warm up so cuDNN/cuBLAS handles exist and shape metadata is cached on the device.
            for _ in 0..3 {
                let o = net.forward(&x_in, &s_in)?;
                p_out.slice_set(&o.log_policy, 0, 0)?;
                v_out.slice_set(&o.v_logit, 0, 0)?;
            }
            stream.synchronize().unwrap();
            let mode = match std::env::var("SPIKE_CAPTURE").as_deref() {
                Ok("relaxed") => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                Ok("global") => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                _ => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            };
            eprintln!("batch {bs}: warm-up done, capturing in {mode:?}");
            stream.begin_capture(mode).unwrap();
            let captured = (|| -> candle_core::Result<()> {
                let o = net.forward(&x_in, &s_in)?;
                p_out.slice_set(&o.log_policy, 0, 0)?;
                v_out.slice_set(&o.v_logit, 0, 0)?;
                Ok(())
            })();
            let g = stream.end_capture(sys::CUgraphInstantiate_flags(0));
            captured?;
            let g = g.unwrap().expect("empty graph");
            g.upload().unwrap();
            g
        };
        let replay = |hp: &[f32], hs: &[f32]| -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
            x_in.slice_set(&Tensor::from_slice(hp, (bs, 17, 8, 8), &dev)?, 0, 0)?;
            s_in.slice_set(&Tensor::from_slice(hs, (bs, 2), &dev)?, 0, 0)?;
            graph.launch().unwrap();
            Ok((
                p_out.flatten_all()?.to_vec1()?,
                v_out.flatten_all()?.to_vec1()?,
            ))
        };

        // Outputs must match on fresh inputs, not just the captured ones.
        let mut max_diff = 0f32;
        for _ in 0..3 {
            let (hp, hs) = host_inputs(bs, &mut rng);
            let (pe, ve) = eager(&hp, &hs)?;
            let (pg, vg) = replay(&hp, &hs)?;
            for (a, b) in pe.iter().chain(&ve).zip(pg.iter().chain(&vg)) {
                max_diff = max_diff.max((a - b).abs());
            }
        }

        let te = median_us(|| {
            eager(&hp, &hs).unwrap();
        });
        let tg = median_us(|| {
            replay(&hp, &hs).unwrap();
        });
        println!(
            "batch {bs:>3}: eager {:.1} us [{:.1}, {:.1}]  graph {:.1} us [{:.1}, {:.1}]  \
             saving {:.0}%  max |diff| {max_diff:.2e}",
            te.0,
            te.1,
            te.2,
            tg.0,
            tg.1,
            tg.2,
            100.0 * (1.0 - tg.0 / te.0)
        );
    }
    Ok(())
}

/// Which single candle op first breaks stream capture.
#[test]
#[ignore]
fn capture_single_ops() -> candle_core::Result<()> {
    let dev = Device::new_cuda_with_stream(0)?;
    let Device::Cuda(cuda) = &dev else {
        panic!("needs CUDA")
    };
    // Event tracking makes captured ops wait on uncaptured events; this device uses one stream.
    unsafe { cuda.disable_event_tracking() };
    let stream = cuda.cuda_stream();
    let x = Tensor::randn(0f32, 1., (16, 128, 8, 8), &dev)?;
    let w = Tensor::randn(0f32, 0.1, (128, 128, 3, 3), &dev)?;
    let bias = Tensor::randn(0f32, 1., (1, 128, 1, 1), &dev)?;
    let out = Tensor::zeros((16, 128, 8, 8), DType::F32, &dev)?;
    let ops: Vec<(&str, Box<dyn Fn() -> candle_core::Result<Tensor>>)> = vec![
        ("relu", Box::new(|| x.relu())),
        ("broadcast_add", Box::new(|| x.broadcast_add(&bias))),
        ("conv2d", Box::new(|| x.conv2d(&w, 1, 1, 1, 1))),
        (
            "conv2d_implicit",
            Box::new(|| {
                x.conv2d_with_algo(
                    &w,
                    1,
                    1,
                    1,
                    1,
                    Some(candle_core::conv::CudnnFwdAlgo::ImplicitGemm),
                )
            }),
        ),
    ];
    let _cache = cuda.enable_cuda_graph_htod_cache();
    for (name, op) in &ops {
        for _ in 0..3 {
            out.slice_set(&op()?, 0, 0)?;
        }
        stream.synchronize().unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        let r = op().and_then(|y| out.slice_set(&y, 0, 0));
        let g = stream.end_capture(sys::CUgraphInstantiate_flags(0));
        println!(
            "{name}: op {:?}, capture {:?}",
            r.as_ref().err(),
            g.as_ref().err()
        );
        if let Ok(Some(g)) = g {
            println!("{name}: launch {:?}", g.launch());
            stream.synchronize().unwrap();
        }
    }
    Ok(())
}
