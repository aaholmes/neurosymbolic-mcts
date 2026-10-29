//! Compare the candle backend with the LibTorch (tch) backend on a trained model.
//! Temporary: removed together with the `neural-tch` feature.
//!
//! PARITY_MODEL=weights/gen_34.pt \
//!   cargo test --release --features cuda,neural-tch --test tch_parity_tests -- --ignored --nocapture

use candle_core::Device;
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::neural_net::tch_impl::NeuralNetPolicy as TchPolicy;
use kingfisher::neural_net::NeuralNetPolicy as CandlePolicy;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn model_path() -> String {
    std::env::var("PARITY_MODEL").unwrap_or_else(|_| "weights/gen_34.pt".to_string())
}

fn legal_moves(board: &Board, mg: &MoveGen) -> Vec<kingfisher::move_types::Move> {
    let (c, q) = mg.gen_pseudo_legal_moves(board);
    c.into_iter()
        .chain(q)
        .filter(|&m| board.apply_move_to_board(m).is_legal(mg))
        .collect()
}

/// `n` positions from seeded random games (2-60 plies), with random q-search inputs.
fn sample_positions(n: usize, seed: u64) -> (Vec<Board>, Vec<bool>, Vec<f32>) {
    let mg = MoveGen::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut boards = Vec::with_capacity(n);
    while boards.len() < n {
        let mut b = Board::new();
        let plies = rng.gen_range(2..60);
        for _ in 0..plies {
            let moves = legal_moves(&b, &mg);
            if moves.is_empty() {
                break;
            }
            b = b.apply_move_to_board(moves[rng.gen_range(0..moves.len())]);
        }
        if !legal_moves(&b, &mg).is_empty() {
            boards.push(b);
        }
    }
    let flags = (0..n).map(|_| rng.gen_bool(0.8)).collect();
    let qs = (0..n).map(|_| rng.gen_range(-5.0f32..5.0)).collect();
    (boards, flags, qs)
}

fn percentile(sorted: &[f32], p: f64) -> f32 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

type Preds = Vec<Option<(Vec<f32>, f32, f32)>>;

/// Max |Δ policy prob| and |Δ v_logit| per position, sorted, plus top-move agreement.
fn compare(name: &str, a: &Preds, b: &Preds, boards: &[Board]) -> (Vec<f32>, Vec<f32>) {
    let mg = MoveGen::new();
    let policy = CandlePolicy::with_device(Device::Cpu);
    let (mut dp, mut dv, mut agree) = (Vec::new(), Vec::new(), 0);
    for i in 0..boards.len() {
        let (pa, va, ka) = a[i].clone().unwrap();
        let (pb, vb, kb) = b[i].clone().unwrap();
        assert!((ka - kb).abs() < 1e-6, "k {ka} vs {kb}");
        dp.push(
            pa.iter()
                .zip(&pb)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max),
        );
        dv.push((va - vb).abs());
        let moves = legal_moves(&boards[i], &mg);
        let best = |p: &[f32]| {
            policy
                .policy_to_move_priors(p, &moves, &boards[i])
                .into_iter()
                .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0
        };
        agree += (best(&pa) == best(&pb)) as usize;
    }
    dp.sort_by(|x, y| x.partial_cmp(y).unwrap());
    dv.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = boards.len();
    println!("{name}:");
    println!(
        "  |Δ policy prob| (max over 4672 per position): p50 {:.2e}  p99 {:.2e}  max {:.2e}",
        percentile(&dp, 0.5),
        percentile(&dp, 0.99),
        dp[n - 1]
    );
    println!(
        "  |Δ v_logit|: p50 {:.2e}  p99 {:.2e}  max {:.2e}",
        percentile(&dv, 0.5),
        percentile(&dv, 0.99),
        dv[n - 1]
    );
    println!("  top legal move agrees on {agree}/{n}");
    (dp, dv)
}

/// candle on CPU is plain fp32 and serves as the reference; both GPU backends
/// are compared with it, and with each other.
#[test]
#[ignore]
fn real_model_parity() {
    let path = model_path();
    let (boards, flags, qs) = sample_positions(512, 7);
    let mut run = |nn: &mut dyn FnMut(&[Board], &[bool], &[f32]) -> Preds| nn(&boards, &flags, &qs);

    let mut tch = TchPolicy::new();
    tch.load(&path).unwrap();
    let t = run(&mut |b, f, q| tch.predict_batch(b, f, q));
    let mut gpu = CandlePolicy::new();
    gpu.load(&path).unwrap();
    let g = run(&mut |b, f, q| gpu.predict_batch(b, f, q));
    let mut cpu = CandlePolicy::with_device(Device::Cpu);
    cpu.load(&path).unwrap();
    let c = run(&mut |b, f, q| cpu.predict_batch(b, f, q));

    println!("model {path}, n = {} positions", boards.len());
    let (dp, dv) = compare("candle GPU vs candle CPU (fp32 reference)", &g, &c, &boards);
    compare("tch GPU vs candle CPU (fp32 reference)", &t, &c, &boards);
    compare("tch GPU vs candle GPU", &t, &g, &boards);
    // cuDNN runs fp32 convolutions in TF32 on recent GPUs (as LibTorch does), which
    // moves outputs by ~1e-3; candle's own CUDA kernels stay within fp32 rounding.
    let tol = if cfg!(feature = "cudnn") { 1e-2 } else { 1e-4 };
    let n = boards.len();
    assert!(dp[n - 1] < tol, "candle GPU policy max diff {}", dp[n - 1]);
    assert!(dv[n - 1] < tol, "candle GPU v_logit max diff {}", dv[n - 1]);
}

fn time_batches(
    name: &str,
    mut run: impl FnMut(&[Board], &[bool], &[f32]),
    data: &(Vec<Board>, Vec<bool>, Vec<f32>),
) {
    for &bs in &[1usize, 8, 16, 32, 64, 128, 256] {
        let (b, f, q) = (&data.0[..bs], &data.1[..bs], &data.2[..bs]);
        for _ in 0..50 {
            run(b, f, q);
        }
        let mut t: Vec<f64> = (0..500)
            .map(|_| {
                let s = Instant::now();
                run(b, f, q);
                s.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        t.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!(
            "{name:>6} batch {bs:>3}: median {:>8.1} us  IQR [{:.1}, {:.1}]  ({:.2} us/position)",
            t[250],
            t[125],
            t[375],
            t[250] / bs as f64
        );
    }
}

/// End-to-end `predict_batch` latency (plane building, copies, forward, readback).
#[test]
#[ignore]
fn bench_predict_batch() {
    let path = model_path();
    let data = sample_positions(256, 11);
    let mut tch = TchPolicy::new();
    tch.load(&path).unwrap();
    time_batches(
        "tch",
        |b, f, q| {
            tch.predict_batch(b, f, q);
        },
        &data,
    );
    drop(tch);
    let mut candle = CandlePolicy::new();
    candle.load(&path).unwrap();
    time_batches(
        "candle",
        |b, f, q| {
            candle.predict_batch(b, f, q);
        },
        &data,
    );
    candle.print_inference_timing();
}
