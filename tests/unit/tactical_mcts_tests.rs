//! Unit tests for tactical MCTS search

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
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
    let board = Board::new();

    let config = test_config(50);
    let (best_move, stats, _root) = tactical_mcts_search(
        board.clone(),
        &move_gen,
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
    let board = Board::new();

    let config = test_config(100);
    let (_, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
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

    // Stalemate position - black to move but no legal moves
    let board = Board::new_from_fen("k7/1R6/K7/8/8/8/8/8 b - - 0 1");

    let config = test_config(10);
    let (best_move, _, _) = tactical_mcts_search(
        board,
        &move_gen,
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

    // Position where white is already checkmated
    let board = Board::new_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");

    let config = test_config(10);
    let (best_move, _, root) = tactical_mcts_search(
        board,
        &move_gen,
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
#[cfg(feature = "slow-tests")]
fn test_tactical_mcts_time_limit_respected() {
    let move_gen = MoveGen::new();
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
        &mut None,
        config,
    );

    assert!(stats.iterations <= max_iters,
        "Should not exceed iteration limit: {} > {}", stats.iterations, max_iters);
}

#[test]
fn test_q_value_sign_convention() {
    // After 1.f4, Black's e7e5 loses a pawn to fxe5.
    // The child node's total_value/visits (Q) should be negative = bad for Black.
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(30),
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(
        board,
        &move_gen,
        &mut None,
        config,
    );

    // Find the e7e5 child (e7=52, e5=36)
    let root_ref = root.borrow();
    let mut found_e7e5 = false;
    for child in &root_ref.children {
        let c = child.borrow();
        if let Some(mv) = c.action {
            if mv.from == 52 && mv.to == 36 && c.visits > 0 {
                found_e7e5 = true;
                let q = c.total_value / c.visits as f64;
                assert!(q < 0.0,
                    "e7e5 Q should be negative (bad for Black who loses a pawn), got {:.4}", q);
            }
        }
    }
    assert!(found_e7e5, "e7e5 should have been visited with 50 iterations");
}

#[test]
#[cfg(feature = "slow-tests")]
fn test_koth_gate_value_not_diluted_by_expansion() {
    // Position from game log: White to move, White king on e3, Black king on e7.
    // KOTH center squares (d4,e4,d5,e5): d4 blocked by Qf6, e4 blocked by pawn f5.
    // After a non-useful White move (e.g. a2a4), Black can force KOTH center in 2 moves.
    // The child node for such moves should have Q ≈ -1.0 (lost for White),
    // and the KOTH gate value should NOT be diluted by further tree expansion.
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("r1b4r/ppp1k2p/n3pq1n/5pp1/8/1PP1K3/P3PPPP/RNQ2BNR w - - 0 11");

    let config = TacticalMctsConfig {
        max_iterations: 1000,
        time_limit: Duration::from_secs(60),
        enable_koth: true,
        ..Default::default()
    };

    let (_best_move, _stats, root) = tactical_mcts_search(
        board,
        &move_gen,
        &mut None,
        config,
    );

    // Check ALL children that have terminal_or_mate_value set:
    // their Q-value must remain consistent with the cached value
    // (not diluted by child evaluations).
    let root_ref = root.borrow();
    for child in &root_ref.children {
        let c = child.borrow();
        if let (Some(cached_val), Some(mv)) = (c.terminal_or_mate_value, c.action) {
            if c.visits > 0 {
                let q = c.total_value / c.visits as f64;
                // Q should be consistent with the gate value.
                // terminal_or_mate_value is from STM perspective at this node.
                // total_value is from parent's perspective (negated).
                let expected_q = -cached_val;
                assert!((q - expected_q).abs() < 0.01,
                    "{}: Q={:.4} but terminal_or_mate_value={:.1}, expected Q={:.1} (visits={})",
                    mv.to_uci(), q, cached_val, expected_q, c.visits);
            }
        }
    }

    // Additionally verify gate-resolved nodes have no children (weren't expanded)
    for child in &root_ref.children {
        let c = child.borrow();
        if c.terminal_or_mate_value.is_some() {
            assert!(c.children.is_empty(),
                "Node with terminal_or_mate_value should not be expanded (has {} children)",
                c.children.len());
        }
    }
}

#[test]
fn test_gate_resolved_nodes_not_expanded() {
    // Simple KOTH position: White king on e3 can reach d4 in 1 move, but
    // Black queen on f6 covers d4 and Black king on e7 is 2 away from center.
    // After any non-king White move, KOTH gate should fire for Black.
    // Gate-resolved nodes must not be expanded (no children).
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("r1b4r/ppp1k2p/n3pq1n/5pp1/8/1PP1K3/P3PPPP/RNQ2BNR w - - 0 11");

    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(30),
        enable_koth: true,
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(board, &move_gen, &mut None, config);

    let root_ref = root.borrow();
    let mut gate_resolved_count = 0;
    for child in &root_ref.children {
        let c = child.borrow();
        if c.terminal_or_mate_value.is_some() {
            gate_resolved_count += 1;
            assert!(c.children.is_empty(),
                "Gate-resolved node {} should have no children, has {}",
                c.action.map(|m| m.to_uci()).unwrap_or_default(), c.children.len());
        }
    }
    assert!(gate_resolved_count > 0, "Some children should have gate-resolved values in this KOTH position");
}

#[test]
fn test_proven_loss_children_get_minimal_visits() {
    // Position: White to move, every move except c3c4 loses to Black's KOTH-in-1.
    // After the fix, proven-loss children (terminal_or_mate_value > 0 from child's STM)
    // should get only 1 visit (the initial root-exploration visit), not 40+ from UCB.
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("r1b4r/ppp1k2p/n3pq1n/5pp1/8/1PP1K3/P3PPPP/RNQ2BNR w - - 0 11");

    let config = TacticalMctsConfig {
        max_iterations: 200,
        time_limit: Duration::from_secs(30),
        enable_koth: true,
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(board, &move_gen, &mut None, config);

    let root_ref = root.borrow();
    let mut proven_loss_count = 0;
    let mut non_loss_count = 0;

    for child in &root_ref.children {
        let c = child.borrow();
        let mv_str = c.action.map(|m| m.to_uci()).unwrap_or_default();

        if c.terminal_or_mate_value.map_or(false, |v| v > 0.0) {
            // Proven loss for the parent: child's STM wins
            proven_loss_count += 1;
            assert_eq!(c.visits, 1,
                "Proven-loss child {} should have exactly 1 visit, got {}",
                mv_str, c.visits);
        } else if c.visits > 0 {
            non_loss_count += 1;
        }
    }

    assert!(proven_loss_count > 0, "Expected some proven-loss children in this KOTH position");
    assert!(non_loss_count > 0, "Expected some non-loss children with visits");
}
