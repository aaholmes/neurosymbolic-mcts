//! Tests for mcts::neural_mcts module

use kingfisher::board::Board;
use kingfisher::mcts::neural_mcts::neural_mcts_search;
use kingfisher::move_generation::MoveGen;
use std::time::Duration;

#[test]
fn test_neural_mcts_search_no_nn_returns_move() {
    let move_gen = MoveGen::new();
    let board = Board::new();
    let mut nn = None;

    let result = neural_mcts_search(
        board,
        &move_gen,
        &mut nn,
        3,
        Some(50),
        Some(Duration::from_secs(5)),
    );
    assert!(result.is_some(), "Should return a move from starting position");
}

#[test]
fn test_neural_mcts_search_mate_in_1() {
    let move_gen = MoveGen::new();
    // Back rank mate: Re8#
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");
    let mut nn = None;

    let result = neural_mcts_search(
        board,
        &move_gen,
        &mut nn,
        3,
        Some(100),
        None,
    );
    assert!(result.is_some());
    let mv = result.unwrap();
    assert_eq!(mv.to_uci(), "e1e8", "Should find Re8# mate");
}

#[test]
fn test_neural_mcts_search_defaults() {
    let move_gen = MoveGen::new();
    let board = Board::new();
    let mut nn = None;

    // Use None for both iterations and time_limit to test defaults
    let result = neural_mcts_search(
        board,
        &move_gen,
        &mut nn,
        3,
        None,
        None,
    );
    assert!(result.is_some(), "Should return a move with default settings");
}

#[test]
fn test_neural_mcts_search_few_iterations() {
    let move_gen = MoveGen::new();
    let board = Board::new();
    let mut nn = None;

    let result = neural_mcts_search(
        board,
        &move_gen,
        &mut nn,
        3,
        Some(10),
        Some(Duration::from_secs(10)),
    );
    assert!(result.is_some(), "Should return a move even with few iterations");
}
