//! Parity tests for the candle OracleNet port, using golden outputs written by
//! `python/export_parity_fixture.py` (a 2-block, hidden-32 OracleNet with all
//! parameters randomized).

#![cfg(feature = "neural")]

use crate::common::{board_from_fen, positions};
use candle_core::{Device, Tensor};
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::nn::oracle_net::OracleNet;
use kingfisher::nn::weights::{load_state_dict, resolve_weights, WeightsFile};
use kingfisher::nn::PolicyValueNet;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn io() -> HashMap<String, Tensor> {
    candle_core::safetensors::load(fixture("oraclenet_tiny_io.safetensors"), &Device::Cpu).unwrap()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims());
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

fn tiny_net(file: &str) -> OracleNet {
    let sd = load_state_dict(&fixture(file), &Device::Cpu).unwrap();
    OracleNet::from_state_dict(sd, &Device::Cpu).unwrap()
}

#[test]
fn forward_matches_pytorch_golden_outputs() {
    let io = io();
    let net = tiny_net("oraclenet_tiny.safetensors");
    let out = net.forward(&io["input"], &io["scalars"]).unwrap();

    let dp = max_abs_diff(&out.log_policy, &io["log_policy"]);
    let dv = max_abs_diff(&out.v_logit, &io["v_logit"]);
    let k_ref: Vec<f32> = io["k"].flatten_all().unwrap().to_vec1().unwrap();
    assert!(dp < 1e-4, "log_policy max |diff| = {dp}");
    assert!(dv < 1e-4, "v_logit max |diff| = {dv}");
    assert!(
        (out.k - k_ref[0]).abs() < 1e-6,
        "k {} vs {}",
        out.k,
        k_ref[0]
    );
}

#[test]
fn config_is_inferred_from_weight_shapes() {
    let cfg = tiny_net("oraclenet_tiny.safetensors").config();
    assert_eq!(cfg.hidden_dim, 32);
    assert_eq!(cfg.num_blocks, 2);
}

#[test]
fn pth_and_safetensors_give_identical_outputs() {
    let io = io();
    let a = tiny_net("oraclenet_tiny.safetensors")
        .forward(&io["input"], &io["scalars"])
        .unwrap();
    let b = tiny_net("oraclenet_tiny.pth")
        .forward(&io["input"], &io["scalars"])
        .unwrap();
    assert_eq!(max_abs_diff(&a.log_policy, &b.log_policy), 0.0);
    assert_eq!(max_abs_diff(&a.v_logit, &b.v_logit), 0.0);
    assert_eq!(a.k, b.k);
}

#[test]
fn resolve_prefers_sibling_safetensors_then_pth() {
    let dir = tempfile::tempdir().unwrap();
    let pt = dir.path().join("gen_3.pt");
    std::fs::write(&pt, b"torchscript").unwrap();

    assert!(resolve_weights(&pt).is_err(), "no state dict next to .pt");

    std::fs::copy(fixture("oraclenet_tiny.pth"), dir.path().join("gen_3.pth")).unwrap();
    assert_eq!(
        resolve_weights(&pt).unwrap(),
        WeightsFile::Pth(dir.path().join("gen_3.pth"))
    );

    std::fs::copy(
        fixture("oraclenet_tiny.safetensors"),
        dir.path().join("gen_3.safetensors"),
    )
    .unwrap();
    assert_eq!(
        resolve_weights(&pt).unwrap(),
        WeightsFile::Safetensors(dir.path().join("gen_3.safetensors"))
    );
}

#[test]
fn resolve_errors_on_missing_file() {
    assert!(resolve_weights(Path::new("/nonexistent/model.safetensors")).is_err());
}

#[test]
fn missing_tensor_is_an_error_not_a_panic() {
    let mut sd = load_state_dict(&fixture("oraclenet_tiny.safetensors"), &Device::Cpu).unwrap();
    sd.remove("p_head.weight");
    assert!(OracleNet::from_state_dict(sd, &Device::Cpu).is_err());
}

#[test]
fn policy_load_reports_missing_file() {
    let mut nn = NeuralNetPolicy::new();
    assert!(nn.load("/nonexistent/model.pt").is_err());
    assert!(!nn.is_available());
}

/// A batch can be launched, left running while another is launched, and read
/// back later; the results equal `predict_batch`.
#[test]
fn launched_batches_finish_with_predict_batch_results() {
    let mut nn = NeuralNetPolicy::new();
    nn.load(fixture("oraclenet_tiny.safetensors").to_str().unwrap())
        .unwrap();
    let a: Vec<_> = [positions::STARTING, positions::CASTLING_BOTH]
        .iter()
        .map(|f| board_from_fen(f))
        .collect();
    let b = vec![board_from_fen(positions::EN_PASSANT)];

    let expected_a = nn.predict_batch(&a, &[true, false], &[0.5, -1.0]);
    let expected_b = nn.predict_batch(&b, &[true], &[2.0]);

    let first = nn.launch_batch(&a, &[true, false], &[0.5, -1.0]);
    let second = nn.launch_batch(&b, &[true], &[2.0]);
    let got_a = nn.finish_batch(first);
    let got_b = nn.finish_batch(second);
    for (got, exp) in got_a
        .iter()
        .chain(&got_b)
        .zip(expected_a.iter().chain(&expected_b))
    {
        let (p, v, k) = got.clone().unwrap();
        let (pe, ve, ke) = exp.clone().unwrap();
        assert_eq!(p, pe);
        assert_eq!((v, k), (ve, ke));
    }
}

#[test]
fn launching_without_a_model_finishes_with_no_predictions() {
    let mut nn = NeuralNetPolicy::new();
    let pending = nn.launch_batch(&[board_from_fen(positions::STARTING)], &[true], &[0.0]);
    assert!(nn.is_finished(&pending));
    assert_eq!(nn.finish_batch(pending), vec![None]);
}

#[test]
fn predict_batch_matches_single_predictions() {
    let mut nn = NeuralNetPolicy::new();
    nn.load(fixture("oraclenet_tiny.safetensors").to_str().unwrap())
        .unwrap();
    assert!(nn.is_available());

    let fens = [
        positions::STARTING,
        positions::CASTLING_BOTH,
        positions::EN_PASSANT,
        "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
    ];
    let boards: Vec<_> = fens.iter().map(|f| board_from_fen(f)).collect();
    let flags = [true, false, true, false];
    let qs = [0.0, 1.5, -2.0, 3.0];

    let batch = nn.predict_batch(&boards, &flags, &qs);
    assert_eq!(batch.len(), boards.len());
    for (i, b) in boards.iter().enumerate() {
        let (p1, v1, k1) = nn.predict(b, flags[i], qs[i]).unwrap();
        let (pb, vb, kb) = batch[i].clone().unwrap();
        assert_eq!(p1.len(), 4672);
        let sum: f32 = p1.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy sums to {sum}");
        let dp = p1
            .iter()
            .zip(&pb)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(dp < 1e-5, "board {i}: policy diff {dp}");
        assert!((v1 - vb).abs() < 1e-5, "board {i}: v_logit {v1} vs {vb}");
        assert_eq!(k1, kb);
    }
}

#[test]
fn board_to_tensor_has_17_planes() {
    let nn = NeuralNetPolicy::new();
    let t = nn.board_to_tensor(&board_from_fen(positions::STARTING));
    assert_eq!(t.dims(), &[17, 8, 8]);
}

#[cfg(unix)]
#[test]
fn resolve_follows_symlinked_model_path() {
    // weights/latest.pt is a symlink to gen_N.pt, with no latest.pth beside it
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("gen_3.pt"), b"torchscript").unwrap();
    std::fs::copy(fixture("oraclenet_tiny.pth"), dir.path().join("gen_3.pth")).unwrap();
    let latest = dir.path().join("latest.pt");
    std::os::unix::fs::symlink(dir.path().join("gen_3.pt"), &latest).unwrap();
    assert_eq!(
        resolve_weights(&latest).unwrap(),
        WeightsFile::Pth(dir.path().join("gen_3.pth"))
    );
}

#[cfg(unix)]
#[test]
fn resolve_pt_name_without_file_finds_symlinked_pth() {
    // The orchestrator no longer writes TorchScript: weights/latest.pt is only a
    // name, and weights/latest.pth is a symlink to gen_N.pth.
    let dir = tempfile::tempdir().unwrap();
    let gen = dir.path().join("gen_5.pth");
    std::fs::copy(fixture("oraclenet_tiny.pth"), &gen).unwrap();
    let latest_pth = dir.path().join("latest.pth");
    std::os::unix::fs::symlink(&gen, &latest_pth).unwrap();
    assert_eq!(
        resolve_weights(&dir.path().join("latest.pt")).unwrap(),
        WeightsFile::Pth(latest_pth.clone())
    );
    let sd = load_state_dict(&dir.path().join("latest.pt"), &Device::Cpu).unwrap();
    assert!(sd.contains_key("k_logit"));
}

#[test]
fn predict_batch_records_batch_sizes() {
    let mut nn = NeuralNetPolicy::new();
    nn.load(fixture("oraclenet_tiny.safetensors").to_str().unwrap())
        .unwrap();
    let boards: Vec<_> = (0..4)
        .map(|_| board_from_fen(positions::STARTING))
        .collect();
    nn.predict_batch(&boards, &[true; 4], &[0.0; 4]);
    nn.predict_batch(&boards[..2], &[true; 2], &[0.0; 2]);
    assert_eq!(nn.batch_sizes.count, 2);
    assert!((nn.batch_sizes.mean_val() - 3.0).abs() < 1e-12);
}

/// Each inference worker owns its stream and never shares a tensor with another
/// stream, so candle's cross-stream event tracking is pure overhead (hundreds of
/// CUDA event calls per batch) and must be off.
#[cfg(feature = "cuda")]
#[test]
fn own_stream_policy_turns_off_event_tracking() {
    let nn = NeuralNetPolicy::new_on_own_stream();
    let Device::Cuda(dev) = nn.device() else {
        panic!("expected a CUDA device")
    };
    assert!(!dev.is_event_tracking());
}

fn legal_moves(board: &kingfisher::board::Board) -> Vec<kingfisher::move_types::Move> {
    let mg = kingfisher::move_generation::MoveGen::new();
    let (captures, quiet) = mg.gen_pseudo_legal_moves(board);
    captures
        .into_iter()
        .chain(quiet)
        .filter(|&m| board.apply_move_to_board(m).is_legal(&mg))
        .collect()
}

/// Priors computed for the legal moves only (gathered on the device, softmax on
/// the CPU) equal the full policy renormalized over those moves, and a request
/// without moves in the same batch still gets the full 4672-slot policy.
#[test]
fn priors_for_given_moves_match_renormalized_full_policy() {
    let mut nn = NeuralNetPolicy::new();
    nn.load(fixture("oraclenet_tiny.safetensors").to_str().unwrap())
        .unwrap();
    let fens = [
        positions::STARTING,
        positions::CASTLING_BOTH,
        positions::EN_PASSANT,
        positions::PROMOTION,
        "4k3/8/8/8/8/8/1p6/4K2R b K - 0 1", // Black to move, promotions on b1
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",
    ];
    let boards: Vec<_> = fens.iter().map(|f| board_from_fen(f)).collect();
    let flags: Vec<bool> = (0..boards.len()).map(|i| i % 2 == 0).collect();
    let qs: Vec<f32> = (0..boards.len()).map(|i| i as f32 - 2.0).collect();
    let full = nn.predict_batch(&boards, &flags, &qs);

    let mut moves: Vec<Option<Vec<_>>> = boards.iter().map(|b| Some(legal_moves(b))).collect();
    moves[2] = None;
    let pending = nn.launch_batch_with_moves(&boards, &flags, &qs, &moves);
    let got = nn.finish_batch(pending);

    for (i, b) in boards.iter().enumerate() {
        let (pf, vf, kf) = full[i].clone().unwrap();
        let (pg, vg, kg) = got[i].clone().unwrap();
        assert!((vf - vg).abs() < 1e-5 && kf == kg, "board {i}: value");
        match &moves[i] {
            Some(mv) => {
                assert_eq!(pg.len(), mv.len(), "board {i}");
                let expected = kingfisher::tensor::policy_to_move_priors(&pf, mv, b);
                for ((m, pe), p) in expected.iter().zip(&pg) {
                    assert!((pe - p).abs() < 1e-5, "board {i} move {m:?}: {pe} vs {p}");
                }
            }
            None => {
                assert_eq!(pg.len(), 4672);
                let d = pf
                    .iter()
                    .zip(&pg)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(d < 1e-5, "board {i}: full policy diff {d}");
            }
        }
    }
}
