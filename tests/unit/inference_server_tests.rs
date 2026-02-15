//! Tests for InferenceServer mock constructors and predict_async

use kingfisher::board::Board;
use kingfisher::mcts::InferenceServer;

#[test]
fn test_mock_server_returns_result() {
    let server = InferenceServer::new_mock();
    let board = Board::new();
    let receiver = server.predict_async(board, true);
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
    let receiver = server.predict_async(board, true);
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
        let receiver = server.predict_async(board.clone(), true);
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
    let receiver = server.predict_async(board, true);
    let (policy, _, _) = receiver.recv().unwrap().unwrap();
    assert_eq!(
        policy.len(),
        4672,
        "Policy vector should have exactly 4672 elements"
    );
}
