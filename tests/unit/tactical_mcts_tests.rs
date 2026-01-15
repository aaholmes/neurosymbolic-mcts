//! Unit tests for tactical MCTS search

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig, TacticalMctsStats};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use std::time::Duration;

/// Helper to create a default config with limited iterations for testing
fn test_config(iterations: u32) -> TacticalMctsConfig {
    TacticalMctsConfig {
        max_iterations: iterations,
        time_limit: Duration::from_secs(30),
        ..Default::default()
    }
}

#[test]
fn test_tactical_mcts_returns_legal_move() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let board = Board::new();

    let config = test_config(50);
    let (best_move, stats, _root) = tactical_mcts_search(
        board.clone(),
        &move_gen,
        &pesto,
        &mut None,
        config,
    );

    assert!(best_move.is_some(), "Should return a move from starting position");

    let mv = best_move.unwrap();
    // Verify the move is legal
    let next = board.apply_move_to_board(mv);
    assert!(next.is_legal(&move_gen), "Returned move should be legal");

    // Verify stats are populated
    assert!(stats.iterations > 0, "Should have done some iterations");
}

#[test]
fn test_tactical_mcts_finds_mate_in_1() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();

    // Position: White to move, Re8 is mate
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        enable_tier1_gate: true,
        mate_search_depth: 2,
        ..Default::default()
    };

    let (best_move, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );

    assert!(best_move.is_some(), "Should find a move");
    let mv = best_move.unwrap();

    // Re8 is e1->e8, which is square 4 to square 60
    assert_eq!(mv.to, 60, "Should find Re8# (to e8 = square 60)");

    // Should have found at least one mate
    assert!(stats.mates_found > 0 || stats.tier1_solutions > 0,
        "Should detect mate via tier1 gate");
}

#[test]
fn test_tactical_mcts_config_defaults() {
    let config = TacticalMctsConfig::default();

    assert!(config.enable_tier1_gate, "Tier 1 should be enabled by default");
    assert!(config.enable_tier2_graft, "Tier 2 should be enabled by default");
    assert!(config.mate_search_depth > 0, "Mate search depth should be positive");
    assert!(config.exploration_constant > 0.0, "Exploration constant should be positive");
}

#[test]
fn test_tactical_mcts_tier1_disabled() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let board = Board::new();

    let config = TacticalMctsConfig {
        max_iterations: 20,
        time_limit: Duration::from_secs(10),
        enable_tier1_gate: false,
        enable_tier2_graft: false,
        ..Default::default()
    };

    let (best_move, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );

    assert!(best_move.is_some(), "Should still return a move with tiers disabled");
    // With tier1 disabled, no tier1 solutions should be found
    assert_eq!(stats.tier1_solutions, 0, "No tier1 solutions with tier1 disabled");
}

#[test]
fn test_tactical_mcts_stats_populated() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let board = Board::new();

    let config = test_config(100);
    let (_, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );

    // Basic stats should be populated
    assert!(stats.iterations > 0, "iterations should be tracked");
    assert!(stats.search_time.as_nanos() > 0, "search_time should be tracked");
}

#[test]
fn test_tactical_mcts_terminal_position() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();

    // Stalemate position - black to move but no legal moves
    let board = Board::new_from_fen("k7/1R6/K7/8/8/8/8/8 b - - 0 1");

    let config = test_config(10);
    let (best_move, _, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );

    // In a stalemate position, there might be no move to return
    // The function should handle this gracefully
    // (best_move could be None for terminal positions)
}

#[test]
fn test_tactical_mcts_checkmate_position() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();

    // Position where white is already checkmated
    let board = Board::new_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");

    let config = test_config(10);
    let (best_move, _, root) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );

    // Root should be marked as terminal
    let root_ref = root.borrow();
    assert!(root_ref.is_terminal, "Checkmate position should be terminal");
    assert_eq!(root_ref.terminal_or_mate_value, Some(-1.0),
        "Checkmated side should have value -1.0");
}

#[test]
fn test_tactical_mcts_time_limit_respected() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let board = Board::new();

    let time_limit = Duration::from_millis(500); // Longer time limit for stability
    let config = TacticalMctsConfig {
        max_iterations: 1_000_000, // High iteration count
        time_limit,
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let (best_move, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );
    let elapsed = start.elapsed();

    assert!(best_move.is_some(), "Should return a move");
    // Allow generous margin for overhead (mate search etc. can add time)
    assert!(elapsed < time_limit + Duration::from_millis(500),
        "Should respect time limit (elapsed: {:?})", elapsed);
}

#[test]
fn test_tactical_mcts_iteration_limit_respected() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let board = Board::new();

    let max_iters = 50;
    let config = TacticalMctsConfig {
        max_iterations: max_iters,
        time_limit: Duration::from_secs(60),
        ..Default::default()
    };

    let (_, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );

    assert!(stats.iterations <= max_iters,
        "Should not exceed iteration limit: {} > {}", stats.iterations, max_iters);
}
