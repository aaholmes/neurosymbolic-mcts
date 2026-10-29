//! Latency of `NeuralNetPolicy::predict_batch` (plane building, copies, forward
//! pass, readback) for a trained model, by batch size.
//!
//! BENCH_MODEL=weights/gen_34.pt \
//!   cargo test --release --features cudnn --test nn_bench -- --ignored --nocapture

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::neural_net::NeuralNetPolicy;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

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

#[test]
#[ignore]
fn bench_predict_batch() {
    let path = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "weights/gen_34.pt".to_string());
    let data = sample_positions(256, 11);
    let mut nn = NeuralNetPolicy::new();
    nn.load(&path).unwrap();
    time_batches(
        "candle",
        |b, f, q| {
            nn.predict_batch(b, f, q);
        },
        &data,
    );
}
