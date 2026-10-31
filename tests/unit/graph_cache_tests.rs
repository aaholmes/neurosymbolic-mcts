//! CUDA graph replay of the half-precision forward pass
//! (`kingfisher::nn::graph_cache`): batches are padded to a few fixed sizes, and
//! replaying a recorded graph must give what the network computes directly.

#![cfg(feature = "neural")]

use kingfisher::nn::graph_cache::bucket;
use std::collections::BTreeSet;

#[test]
fn batches_are_padded_to_eight_then_to_multiples_of_sixteen() {
    let cap = 128;
    let sizes: BTreeSet<usize> = (1..=cap).map(|b| bucket(b, cap)).collect();
    assert_eq!(
        sizes.into_iter().collect::<Vec<_>>(),
        [8, 16, 32, 48, 64, 80, 96, 112, 128]
    );
    for b in 1..=cap {
        let k = bucket(b, cap);
        assert!(b <= k && k <= cap, "batch {b} -> {k}");
    }
    // A cap that is not a multiple of 16 is its own largest size.
    assert_eq!(bucket(99, 100), 100);
    assert_eq!(bucket(100, 100), 100);
    assert_eq!(bucket(3, 4), 4);
}

#[cfg(feature = "cudnn")]
mod gpu {
    use crate::common::board_from_fen;
    use candle_core::{Device, Tensor};
    use kingfisher::board::Board;
    use kingfisher::move_generation::MoveGen;
    use kingfisher::neural_net::NeuralNetPolicy;
    use kingfisher::nn::graph_cache::GraphCache;
    use kingfisher::nn::seres_net::SeresNet;
    use kingfisher::nn::weights::load_state_dict;
    use kingfisher::nn::Precision;
    use kingfisher::tensor::NHWC_CHANNELS;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/seresnet_tiny.safetensors")
    }

    fn own_stream() -> Device {
        let dev = Device::new_cuda_with_stream(0).expect("CUDA device");
        if let Device::Cuda(d) = &dev {
            // SAFETY: every tensor of these tests stays on this stream.
            unsafe { d.disable_event_tracking() };
        }
        dev
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
            .to_device(&Device::Cpu)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Random 0/1 planes `[b, 8, 8, NHWC_CHANNELS]` and scalars `[b, 2]`.
    fn inputs(b: usize, seed: u64, dev: &Device) -> (Tensor, Tensor) {
        let mut rng = StdRng::seed_from_u64(seed);
        let planes: Vec<f32> = (0..b * 64 * NHWC_CHANNELS)
            .map(|i| (i % NHWC_CHANNELS < 17 && rng.gen_bool(0.1)) as u8 as f32)
            .collect();
        let scalars: Vec<f32> = (0..b * 2).map(|_| rng.gen_range(-3.0..3.0)).collect();
        (
            Tensor::from_vec(planes, (b, 8, 8, NHWC_CHANNELS), dev).unwrap(),
            Tensor::from_vec(scalars, (b, 2), dev).unwrap(),
        )
    }

    /// Replaying the graph for a batch's size gives the logits and value the
    /// network computes directly: exactly when the batch fills its size (same
    /// kernels on the same shapes), and to within half-precision rounding when it is
    /// padded, since cuDNN may then tile the batch differently.
    #[test]
    fn graph_replay_matches_the_network_for_every_batch_size() {
        let dev = own_stream();
        let sd = load_state_dict(&fixture(), &dev).unwrap();
        let net = SeresNet::from_state_dict_with_precision(sd, &dev, Precision::F16).unwrap();
        assert!(net.channels_last());
        let mut graphs = GraphCache::new(64);
        for (i, b) in [16, 5, 48, 9, 37, 64, 1, 16].into_iter().enumerate() {
            let (planes, scalars) = inputs(b, i as u64, &dev);
            let (gl, gv) = graphs.forward(&net, &planes, &scalars).unwrap();
            let (el, ev) = net.forward_logits(&planes, &scalars).unwrap();
            assert_eq!(gl.dims(), [b, net.policy_width()]);
            assert_eq!(gv.dims(), [b, 1]);
            let (dl, dv) = (max_abs_diff(&gl, &el), max_abs_diff(&gv, &ev));
            if b == kingfisher::nn::graph_cache::bucket(b, 64) {
                assert_eq!((dl, dv), (0.0, 0.0), "batch {b}");
            } else {
                assert!(dl < 2e-2 && dv < 2e-2, "batch {b}: logits {dl}, value {dv}");
            }
        }
    }

    /// The value returned for one batch is not overwritten by the next replay of
    /// the same graph, so a launched batch can be read back after the next one starts.
    #[test]
    fn a_batch_keeps_its_outputs_after_the_next_replay() {
        let dev = own_stream();
        let sd = load_state_dict(&fixture(), &dev).unwrap();
        let net = SeresNet::from_state_dict_with_precision(sd, &dev, Precision::F16).unwrap();
        let mut graphs = GraphCache::new(16);
        let (p1, s1) = inputs(16, 1, &dev);
        let (p2, s2) = inputs(16, 2, &dev);
        let (_, v1) = graphs.forward(&net, &p1, &s1).unwrap();
        let v1_before = v1.to_device(&Device::Cpu).unwrap();
        let _ = graphs.forward(&net, &p2, &s2).unwrap();
        assert_eq!(
            max_abs_diff(&v1.to_device(&Device::Cpu).unwrap(), &v1_before),
            0.0
        );
    }

    /// 200 positions from seeded random games.
    fn positions() -> Vec<Board> {
        let mg = MoveGen::new();
        let mut rng = StdRng::seed_from_u64(7);
        let mut out = Vec::new();
        while out.len() < 200 {
            let mut b = board_from_fen(crate::common::positions::STARTING);
            for _ in 0..rng.gen_range(0..60) {
                let moves = legal(&b, &mg);
                if moves.is_empty() {
                    break;
                }
                b = b.apply_move_to_board(moves[rng.gen_range(0..moves.len())]);
            }
            if !legal(&b, &mg).is_empty() {
                out.push(b);
            }
        }
        out
    }

    fn legal(b: &Board, mg: &MoveGen) -> Vec<kingfisher::move_types::Move> {
        let (c, q) = mg.gen_pseudo_legal_moves(b);
        c.into_iter()
            .chain(q)
            .filter(|&m| b.apply_move_to_board(m).is_legal(mg))
            .collect()
    }

    type Predictions = Vec<Option<(Vec<f32>, f32, f32)>>;

    /// Legal-move priors for `boards` in batches of `sizes`, with or without graphs.
    fn priors(boards: &[Board], sizes: &[usize], graphs: bool) -> Predictions {
        let mut nn = NeuralNetPolicy::new_on_own_stream();
        nn.load_with_precision(fixture().to_str().unwrap(), Precision::F16)
            .unwrap();
        if graphs {
            nn.enable_cuda_graphs(64);
        }
        let mg = MoveGen::new();
        let mut out = Vec::new();
        let mut start = 0;
        for &n in sizes {
            let batch = &boards[start..start + n];
            let flags: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
            let qs: Vec<f32> = (0..n).map(|i| (i % 5) as f32 - 2.0).collect();
            let moves: Vec<_> = batch.iter().map(|b| Some(legal(b, &mg))).collect();
            let pending = nn.launch_batch_with_moves(batch, &flags, &qs, &moves);
            out.extend(nn.finish_batch(pending));
            start += n;
        }
        out
    }

    /// With graphs on, the policy gives the priors and values it gives without
    /// them, on two threads at once, each with its own stream and graphs.
    #[test]
    fn policy_with_cuda_graphs_matches_policy_without() {
        let boards = positions();
        let sizes = [1, 9, 37, 64, 50, 39];
        let reference = priors(&boards, &sizes, false);
        let runs: Vec<Predictions> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2)
                .map(|_| s.spawn(|| priors(&boards, &sizes, true)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for got in runs {
            let (mut dp, mut dv, mut same_top) = (0f32, 0f32, 0);
            for (r, g) in reference.iter().zip(&got) {
                let ((pr, vr, _), (pg, vg, _)) = (r.clone().unwrap(), g.clone().unwrap());
                for (a, b) in pr.iter().zip(&pg) {
                    dp = dp.max((a - b).abs());
                }
                dv = dv.max((vr - vg).abs());
                let top = |p: &[f32]| {
                    p.iter()
                        .enumerate()
                        .max_by(|x, y| x.1.total_cmp(y.1))
                        .map(|x| x.0)
                };
                same_top += (top(&pr) == top(&pg)) as usize;
            }
            assert!(dp < 2e-3, "prior diff {dp}");
            assert!(dv < 2e-2, "value diff {dv}");
            assert!(same_top >= 198, "top move agrees in {same_top}/200");
        }
    }
}
