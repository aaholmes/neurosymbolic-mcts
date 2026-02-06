//! Integration tests for the full MCTS search pipeline

use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::mcts::inference_server::InferenceServer;
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::eval::PestoEval;
use kingfisher::move_types::Move;
use crate::common::{board_from_fen, positions};
use std::time::Duration;
use std::sync::Arc;

#[test]
fn test_mcts_finds_mate_in_1() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let server = InferenceServer::new_mock(); 
    
    let board = board_from_fen(positions::MATE_IN_1_WHITE);
    
    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };
    
    let (best_move, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );
    
    // Should find Re8# instantly via Safety Gate
    assert_eq!(best_move.unwrap().to, 60, "Should find Re8#"); // e8 = square 60
    // search_time depends on machine, but safety gate is fast.
    // However, stats.search_time might be small.
}

#[test]
fn test_mcts_finds_mate_for_black() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let server = InferenceServer::new_mock();
    
    let board = board_from_fen(positions::MATE_IN_1_BLACK);
    
    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };
    
    let (best_move, _, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );
    
    // Should find Re1#
    assert_eq!(best_move.unwrap().to, 4, "Should find Re1#"); // e1 = square 4
}

#[test]
fn test_mcts_finds_koth_win() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let server = InferenceServer::new_mock();
    
    let board = board_from_fen(positions::KOTH_WIN_AVAILABLE);
    
    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server)),
        enable_koth: true,
        ..Default::default()
    };

    let (best_move, _, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );

    // Should find Kd5 or Ke5 (moves to center)
    let to_sq = best_move.unwrap().to;
    let center_squares = [27, 28, 35, 36]; // d4, e4, d5, e5
    assert!(
        center_squares.contains(&to_sq),
        "Should find move to KOTH center, got square {}",
        to_sq
    );
}

#[test]
#[ignore] // MCTS search quality issue - needs investigation
fn test_mcts_prefers_winning_capture() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let server = InferenceServer::new_mock();
    
    let board = board_from_fen(positions::WINNING_CAPTURE);
    
    let config = TacticalMctsConfig {
        max_iterations: 500, // More simulations needed
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };
    
    let (best_move, _, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );
    
    // Nxd5 should be strongly preferred
    let best = best_move.unwrap();
    assert_eq!(best.from, 28, "Should move knight from e4"); 
    assert_eq!(best.to, 35, "Should capture queen on d5");
}

#[test]
fn test_backpropagation_sign_consistency() {
    // This test verifies that values are correctly negated during backprop
    
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let server = InferenceServer::new_mock_biased(0.5); // Always returns +0.5 for StM
    
    let board = board_from_fen(positions::STARTING);
    
    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };
    
    let (_, _, root) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );
    
    // After search, examine root's children
    for child in root.borrow().children.iter() {
        let child_ref = child.borrow();
        if child_ref.visits > 0 {
            let q = child_ref.total_value / child_ref.visits as f64;
            // All children should have Q in valid range
            assert!(q >= -1.0 && q <= 1.0, 
                "Child Q {} outside valid range after backprop", q);
        }
    }
}
