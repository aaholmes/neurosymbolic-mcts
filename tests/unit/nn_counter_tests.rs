//! Tests for mcts::nn_counter module

use kingfisher::board::Board;
use kingfisher::mcts::nn_counter::{CountingNeuralNetPolicy, EfficiencyComparison};

// === CountingNeuralNetPolicy Tests ===

#[test]
fn test_new_counter_starts_at_zero() {
    let counter = CountingNeuralNetPolicy::new();
    assert_eq!(counter.get_call_count(), 0);
    assert_eq!(counter.get_total_positions(), 0);
}

#[test]
fn test_mock_evaluate_increments_both_counters() {
    let counter = CountingNeuralNetPolicy::new();
    let board = Board::new();

    counter.mock_evaluate(&board).unwrap();
    assert_eq!(counter.get_call_count(), 1);
    assert_eq!(counter.get_total_positions(), 1);
}

#[test]
fn test_track_without_nn_only_increments_positions() {
    let counter = CountingNeuralNetPolicy::new();

    counter.track_position_without_nn();
    assert_eq!(counter.get_call_count(), 0);
    assert_eq!(counter.get_total_positions(), 1);
}

#[test]
fn test_reset_counters() {
    let counter = CountingNeuralNetPolicy::new();
    let board = Board::new();

    counter.mock_evaluate(&board).unwrap();
    counter.mock_evaluate(&board).unwrap();
    counter.track_position_without_nn();

    assert_eq!(counter.get_call_count(), 2);
    assert_eq!(counter.get_total_positions(), 3);

    counter.reset_counters();
    assert_eq!(counter.get_call_count(), 0);
    assert_eq!(counter.get_total_positions(), 0);
}

#[test]
fn test_efficiency_ratio_zero_positions() {
    let counter = CountingNeuralNetPolicy::new();
    assert_eq!(counter.get_efficiency_ratio(), 0.0);
}

#[test]
fn test_efficiency_ratio_all_nn() {
    let counter = CountingNeuralNetPolicy::new();
    let board = Board::new();

    counter.mock_evaluate(&board).unwrap();
    counter.mock_evaluate(&board).unwrap();
    // 2 calls / 2 positions = 1.0
    assert_eq!(counter.get_efficiency_ratio(), 1.0);
}

#[test]
fn test_efficiency_ratio_mixed() {
    let counter = CountingNeuralNetPolicy::new();
    let board = Board::new();

    counter.mock_evaluate(&board).unwrap(); // 1 call, 1 position
    counter.track_position_without_nn(); // 0 calls, 1 position
    counter.track_position_without_nn(); // 0 calls, 1 position
    counter.mock_evaluate(&board).unwrap(); // 1 call, 1 position

    // 2 calls / 4 positions = 0.5
    assert_eq!(counter.get_efficiency_ratio(), 0.5);
}

#[test]
fn test_mock_evaluate_returns_valid_policy() {
    let counter = CountingNeuralNetPolicy::new();
    let board = Board::new();

    let (policy, value) = counter.mock_evaluate(&board).unwrap();
    assert_eq!(policy.len(), 64);
    assert!(policy.iter().all(|&p| p >= 0.0));
    assert!((value - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_multiple_resets() {
    let counter = CountingNeuralNetPolicy::new();
    let board = Board::new();

    counter.mock_evaluate(&board).unwrap();
    counter.reset_counters();
    assert_eq!(counter.get_call_count(), 0);

    counter.mock_evaluate(&board).unwrap();
    counter.mock_evaluate(&board).unwrap();
    assert_eq!(counter.get_call_count(), 2);

    counter.reset_counters();
    assert_eq!(counter.get_call_count(), 0);
    assert_eq!(counter.get_total_positions(), 0);
}

// === EfficiencyComparison Tests ===

#[test]
fn test_efficiency_comparison_new() {
    let cmp = EfficiencyComparison::new();
    assert_eq!(cmp.tactical_mcts_calls, 0);
    assert_eq!(cmp.tactical_mcts_positions, 0);
    assert_eq!(cmp.classical_mcts_calls, 0);
    assert_eq!(cmp.classical_mcts_positions, 0);
}

#[test]
fn test_tactical_efficiency_zero_positions() {
    let cmp = EfficiencyComparison::new();
    assert_eq!(cmp.tactical_efficiency(), 0.0);
}

#[test]
fn test_classical_efficiency_zero_positions() {
    let cmp = EfficiencyComparison::new();
    assert_eq!(cmp.classical_efficiency(), 0.0);
}

#[test]
fn test_improvement_percentage_zero_classical() {
    let cmp = EfficiencyComparison::new();
    assert_eq!(cmp.improvement_percentage(), 0.0);
}

#[test]
fn test_improvement_percentage_50_percent() {
    let mut cmp = EfficiencyComparison::new();
    cmp.tactical_mcts_calls = 50;
    cmp.tactical_mcts_positions = 100;
    cmp.classical_mcts_calls = 100;
    cmp.classical_mcts_positions = 100;

    // tactical = 0.5, classical = 1.0
    // improvement = (1.0 - 0.5) / 1.0 * 100 = 50%
    assert!((cmp.improvement_percentage() - 50.0).abs() < 0.01);
}

#[test]
fn test_improvement_percentage_no_improvement() {
    let mut cmp = EfficiencyComparison::new();
    cmp.tactical_mcts_calls = 100;
    cmp.tactical_mcts_positions = 100;
    cmp.classical_mcts_calls = 100;
    cmp.classical_mcts_positions = 100;

    assert!((cmp.improvement_percentage()).abs() < 0.01);
}

#[test]
fn test_improvement_percentage_complete_savings() {
    let mut cmp = EfficiencyComparison::new();
    cmp.tactical_mcts_calls = 0;
    cmp.tactical_mcts_positions = 100;
    cmp.classical_mcts_calls = 100;
    cmp.classical_mcts_positions = 100;

    assert!((cmp.improvement_percentage() - 100.0).abs() < 0.01);
}
