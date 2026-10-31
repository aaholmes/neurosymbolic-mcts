//! Tests for InferenceServer mock constructors and predict_async

use kingfisher::board::Board;
use kingfisher::mcts::InferenceServer;

#[test]
fn test_mock_server_returns_result() {
    let server = InferenceServer::new_mock();
    let board = Board::new();
    let receiver = server.predict_async(board, true, 0.0);
    let result = receiver.recv().expect("Should receive a response");
    assert!(result.is_some(), "Mock server should return Some");
    let (policy, v_logit, k) = result.unwrap();
    assert_eq!(policy.len(), 4672, "Policy should have 4672 elements");
    // v_logit is unbounded (raw logit from NN), but mock returns in [-2, 2]
    assert!(
        v_logit >= -2.0 && v_logit <= 2.0,
        "v_logit {v_logit} should be in [-2, 2] for mock"
    );
    assert!(k >= 0.0 && k <= 1.0, "k={k} should be in [0, 1]");
}

#[test]
fn test_mock_biased_returns_correct_v_logit() {
    let server = InferenceServer::new_mock_biased(0.5);
    let board = Board::new();
    let receiver = server.predict_async(board, true, 0.0);
    let result = receiver.recv().expect("Should receive a response");
    let (_, v_logit, k) = result.unwrap();
    assert!(
        (v_logit - 0.5).abs() < 1e-6,
        "Biased mock should return v_logit=0.5, got {v_logit}"
    );
    assert!(
        (k - 0.5).abs() < 1e-6,
        "Biased mock should return k=0.5, got {k}"
    );
}

#[test]
fn test_mock_server_handles_multiple_requests() {
    let server = InferenceServer::new_mock();
    let board = Board::new();

    for i in 0..5 {
        let receiver = server.predict_async(board.clone(), true, 0.0);
        let result = receiver
            .recv()
            .expect(&format!("Request {i} should get response"));
        assert!(result.is_some(), "Request {i} should return Some");
    }
}

#[test]
fn test_mock_server_policy_length() {
    let server = InferenceServer::new_mock_biased(0.0);
    let board = Board::new();
    let receiver = server.predict_async(board, true, 0.0);
    let (policy, _, _) = receiver.recv().unwrap().unwrap();
    assert_eq!(
        policy.len(),
        4672,
        "Policy vector should have exactly 4672 elements"
    );
}

#[test]
fn test_mock_server_reports_no_worker_stats() {
    let server = InferenceServer::new_mock();
    assert!(server.worker_stats().is_empty());
}

#[cfg(feature = "neural")]
mod with_model {
    use kingfisher::board::Board;
    use kingfisher::mcts::InferenceServer;
    use kingfisher::neural_net::NeuralNetPolicy;

    pub fn tiny_policy() -> NeuralNetPolicy {
        let mut nn = NeuralNetPolicy::with_device(candle_core::Device::Cpu);
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/oraclenet_tiny.safetensors"
        );
        nn.load(path).unwrap();
        nn
    }

    #[test]
    fn server_records_collect_and_processing_time_per_batch() {
        let server = InferenceServer::new(tiny_policy(), 16);
        for _ in 0..3 {
            // One request at a time, so each forms its own batch
            let r = server
                .predict_async(Board::new(), true, 0.0)
                .recv()
                .unwrap();
            assert!(r.is_some());
        }
        let stats = server.worker_stats();
        assert_eq!(stats.len(), 1);
        let s = &stats[0];
        assert_eq!(s.batches, 3);
        assert_eq!(s.positions, 3);
        assert_eq!(s.collect.count, 3);
        assert_eq!(s.process.count, 3);
        assert!(s.process.total > std::time::Duration::ZERO);
    }

    fn sample_boards() -> Vec<Board> {
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
            "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        ]
        .iter()
        .map(|f| Board::new_from_fen(f))
        .collect()
    }

    #[test]
    fn two_workers_answer_each_request_with_its_own_result() {
        let boards = sample_boards();
        let mut reference = tiny_policy();
        let expected: Vec<_> = (0..200)
            .map(|i| {
                let b = &boards[i % boards.len()];
                reference.predict(b, i % 2 == 0, i as f32 * 0.01).unwrap()
            })
            .collect();

        let server = std::sync::Arc::new(InferenceServer::with_workers(
            vec![tiny_policy(), tiny_policy()],
            16,
        ));
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let server = server.clone();
                let boards = boards.clone();
                std::thread::spawn(move || {
                    (t..200)
                        .step_by(8)
                        .map(|i| {
                            let b = boards[i % boards.len()].clone();
                            let r = server
                                .predict_async(b, i % 2 == 0, i as f32 * 0.01)
                                .recv()
                                .unwrap()
                                .unwrap();
                            (i, r)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, (p, v, k)) in h.join().unwrap() {
                let (pe, ve, ke) = &expected[i];
                let dp = p
                    .iter()
                    .zip(pe)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(
                    dp < 1e-5 && (v - ve).abs() < 1e-5 && k == *ke,
                    "request {i}"
                );
            }
        }

        let stats = std::sync::Arc::try_unwrap(server).unwrap().shutdown();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats.iter().map(|s| s.positions).sum::<u64>(), 200);
    }

    #[test]
    fn pipelined_worker_answers_each_request_with_its_own_result() {
        let boards = sample_boards();
        let mut reference = tiny_policy();
        let expected: Vec<_> = (0..200)
            .map(|i| {
                let b = &boards[i % boards.len()];
                reference.predict(b, i % 2 == 0, i as f32 * 0.01).unwrap()
            })
            .collect();
        let server = std::sync::Arc::new(InferenceServer::new(tiny_policy(), 16));
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let server = server.clone();
                let boards = boards.clone();
                std::thread::spawn(move || {
                    (t..200)
                        .step_by(8)
                        .map(|i| {
                            let b = boards[i % boards.len()].clone();
                            let r = server
                                .predict_async(b, i % 2 == 0, i as f32 * 0.01)
                                .recv()
                                .unwrap()
                                .unwrap();
                            (i, r)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, (p, v, k)) in h.join().unwrap() {
                let (pe, ve, ke) = &expected[i];
                let dp = p
                    .iter()
                    .zip(pe)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(
                    dp < 1e-5 && (v - ve).abs() < 1e-5 && k == *ke,
                    "request {i}"
                );
            }
        }
        let stats = std::sync::Arc::try_unwrap(server).unwrap().shutdown();
        assert_eq!(stats[0].positions, 200);
    }

    #[test]
    fn requests_queued_before_shutdown_are_all_answered() {
        let server = InferenceServer::new(tiny_policy(), 4);
        let receivers: Vec<_> = (0..50)
            .map(|_| server.predict_async(Board::new(), true, 0.0))
            .collect();
        let stats = server.shutdown();
        for r in receivers {
            assert!(r.recv().unwrap().is_some());
        }
        assert_eq!(stats[0].positions, 50);
    }

    #[test]
    fn worker_records_gpu_idle_time_between_batches() {
        let server = InferenceServer::new(tiny_policy(), 16);
        for _ in 0..3 {
            server
                .predict_async(Board::new(), true, 0.0)
                .recv()
                .unwrap();
        }
        let s = server.shutdown().remove(0);
        assert_eq!(s.batches, 3);
        // Measured from each batch's readback to the next launch
        assert_eq!(s.gpu_idle.count, 2);
    }

    #[test]
    fn zero_idle_window_answers_a_lone_request_without_waiting() {
        use std::time::{Duration, Instant};
        let slow = InferenceServer::with_workers_and_window(
            vec![tiny_policy()],
            16,
            Duration::from_millis(300),
        );
        let fast =
            InferenceServer::with_workers_and_window(vec![tiny_policy()], 16, Duration::ZERO);
        let time = |s: &InferenceServer| {
            let t = Instant::now();
            s.predict_async(Board::new(), true, 0.0).recv().unwrap();
            t.elapsed()
        };
        assert!(time(&slow) >= Duration::from_millis(300));
        assert!(time(&fast) < Duration::from_millis(150));
    }

    #[test]
    fn shutdown_joins_every_worker() {
        let server =
            InferenceServer::with_workers(vec![tiny_policy(), tiny_policy(), tiny_policy()], 4);
        let r = server
            .predict_async(Board::new(), true, 0.0)
            .recv()
            .unwrap();
        assert!(r.is_some());
        let stats = server.shutdown();
        assert_eq!(stats.len(), 3);
        assert_eq!(stats.iter().map(|s| s.positions).sum::<u64>(), 1);
    }

    #[test]
    fn load_starts_the_requested_number_of_workers() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/oraclenet_tiny.safetensors"
        );
        let server = InferenceServer::load(path, 2, 8).unwrap();
        assert!(server
            .predict_async(Board::new(), true, 0.0)
            .recv()
            .unwrap()
            .is_some());
        assert_eq!(server.shutdown().len(), 2);
    }

    #[test]
    fn load_reports_a_missing_model() {
        assert!(InferenceServer::load("/nonexistent/model.pt", 2, 8).is_err());
    }
}
