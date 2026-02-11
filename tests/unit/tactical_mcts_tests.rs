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
        ..Default::default()
    };

    let (best_move, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
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

    let (_, _, root) = tactical_mcts_search(board, &move_gen, config);

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

    let (_, _, root) = tactical_mcts_search(board, &move_gen, config);

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

#[test]
fn test_koth_opponent_on_center_is_terminal() {
    // Position where White king is already on e4 (center), Black to move.
    // Black should see this as a terminal loss (opponent already won KOTH).
    let move_gen = MoveGen::new();
    // White king on e4 (center square), Black king on a7
    let board = Board::new_from_fen("8/k7/8/8/4K3/8/8/8 b - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        enable_koth: true,
        ..Default::default()
    };

    let (_best_move, _stats, root) = tactical_mcts_search(
        board,
        &move_gen,
        config,
    );

    // Root should be terminal: opponent (White) already occupies center
    let root_ref = root.borrow();
    assert!(root_ref.terminal_or_mate_value.is_some(),
        "Position with opponent on center should have terminal value");
    assert_eq!(root_ref.terminal_or_mate_value, Some(-1.0),
        "STM (Black) should lose: opponent king on center");
}

#[test]
fn test_koth_win_in_1_selected_and_search_aborts() {
    // Position: White king on d3, can step to d4 or e4 (KOTH center) for instant win.
    // d3c3 captures a rook but is slower. Search should pick d3d4 or d3e4 and abort early.
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("3N4/k6p/p6P/2P1R3/2n3P1/P1rK4/8/5B2 w - - 4 47");

    let config = TacticalMctsConfig {
        max_iterations: 1000,
        time_limit: Duration::from_secs(30),
        enable_koth: true,
        ..Default::default()
    };

    let (best_move, stats, _root) = tactical_mcts_search(
        board,
        &move_gen,
        config,
    );

    assert!(best_move.is_some(), "Should find a move");
    let mv = best_move.unwrap();

    // d3=19, d4=27, e4=28 in LERF
    let is_koth_win_move = (mv.from == 19 && mv.to == 27) || (mv.from == 19 && mv.to == 28);
    assert!(is_koth_win_move,
        "Should select KOTH win-in-1 (d3d4 or d3e4), got {}",
        mv.to_uci());

    // Early termination: should use far fewer than 1000 iterations
    assert!(stats.iterations < 100,
        "Search should abort early with win-in-1, used {} iterations", stats.iterations);
}

// === evaluate_leaf_node: classical fallback and material balance ===

#[test]
fn test_classical_fallback_material_advantage() {
    // White is up a queen — classical fallback should give positive value
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        enable_tier3_neural: false, // Force classical fallback
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(board, &move_gen, config);

    // Root's Q should be positive (White is winning with queen advantage)
    let root_ref = root.borrow();
    if root_ref.visits > 0 {
        let root_q = -(root_ref.total_value / root_ref.visits as f64);
        assert!(root_q > 0.0,
            "White up a queen should have positive Q, got {:.4}", root_q);
    }
}

#[test]
fn test_classical_fallback_material_disadvantage() {
    // White is down a queen — should give negative value
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("3qk3/8/8/8/8/8/8/4K3 w - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        enable_tier3_neural: false,
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(board, &move_gen, config);

    let root_ref = root.borrow();
    if root_ref.visits > 0 {
        let root_q = -(root_ref.total_value / root_ref.visits as f64);
        assert!(root_q < 0.0,
            "White down a queen should have negative Q, got {:.4}", root_q);
    }
}

#[test]
fn test_classical_fallback_equal_material() {
    // Equal material — value should be near 0
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        enable_tier3_neural: false,
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(board, &move_gen, config);

    let root_ref = root.borrow();
    if root_ref.visits > 0 {
        let root_q = -(root_ref.total_value / root_ref.visits as f64);
        assert!(root_q.abs() < 0.5,
            "Equal material should have Q near 0, got {:.4}", root_q);
    }
}

#[test]
fn test_leaf_values_in_tanh_range() {
    // All leaf node values should be in [-1, 1]
    let move_gen = MoveGen::new();
    let board = Board::new();

    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(board, &move_gen, config);

    fn check_tree_values(node: &std::rc::Rc<std::cell::RefCell<kingfisher::mcts::node::MctsNode>>) {
        let n = node.borrow();
        if let Some(v) = n.nn_value {
            assert!(v >= -1.0 && v <= 1.0,
                "nn_value {} outside tanh domain [-1, 1]", v);
        }
        if let Some(v) = n.terminal_or_mate_value {
            assert!(v >= -1.0 && v <= 1.0,
                "terminal_or_mate_value {} outside [-1, 1]", v);
        }
        for child in &n.children {
            check_tree_values(child);
        }
    }
    check_tree_values(&root);
}


// === select_best_move_from_root edge cases ===

#[test]
fn test_select_best_move_prefers_koth_win_over_captures() {
    // KOTH position where a capture exists but KOTH win is available
    // White king can step to center OR capture a piece
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("3N4/k6p/p6P/2P1R3/2n3P1/P1rK4/8/5B2 w - - 4 47");

    let config = TacticalMctsConfig {
        max_iterations: 200,
        time_limit: Duration::from_secs(10),
        enable_koth: true,
        ..Default::default()
    };

    let (best_move, stats, _root) = tactical_mcts_search(board, &move_gen, config);

    assert!(best_move.is_some());
    let mv = best_move.unwrap();
    // Must be d3d4 (19->27) or d3e4 (19->28), NOT d3c3 (19->18, captures rook)
    let is_koth = (mv.from == 19 && mv.to == 27) || (mv.from == 19 && mv.to == 28);
    assert!(is_koth, "Should prefer KOTH win over rook capture, got {}", mv.to_uci());

    // Pre-search KOTH gate should have fired (tier1_solutions > 0)
    assert!(stats.tier1_solutions > 0, "Pre-search KOTH gate should fire for win-in-1");
}

// === Training search ===

#[test]
fn test_training_search_returns_policy() {
    use kingfisher::mcts::tactical_mcts::tactical_mcts_search_for_training;

    let move_gen = MoveGen::new();
    let board = Board::new();

    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        ..Default::default()
    };

    let result = tactical_mcts_search_for_training(board, &move_gen, config);

    assert!(result.best_move.is_some(), "Should return a best move");
    assert!(!result.root_policy.is_empty(), "Should return non-empty policy");

    // Policy visit counts should sum to something reasonable
    let total_visits: u32 = result.root_policy.iter().map(|(_, v)| v).sum();
    assert!(total_visits > 0, "Total policy visits should be positive");

    // Root value should be in tanh domain
    assert!(result.root_value_prediction >= -1.0 && result.root_value_prediction <= 1.0,
        "Root value {} outside [-1, 1]", result.root_value_prediction);
}

#[test]
fn test_training_search_koth_early_termination() {
    use kingfisher::mcts::tactical_mcts::tactical_mcts_search_for_training;

    let move_gen = MoveGen::new();
    // KOTH win-in-1 position
    let board = Board::new_from_fen("3N4/k6p/p6P/2P1R3/2n3P1/P1rK4/8/5B2 w - - 4 47");

    let config = TacticalMctsConfig {
        max_iterations: 1000,
        time_limit: Duration::from_secs(30),
        enable_koth: true,
        ..Default::default()
    };

    let result = tactical_mcts_search_for_training(board, &move_gen, config);

    assert!(result.best_move.is_some());
    assert!(result.stats.iterations < 100,
        "Training search should also abort early on KOTH win, used {} iters",
        result.stats.iterations);
}

#[test]
fn test_training_search_with_reuse_basic() {
    use kingfisher::mcts::tactical_mcts::{tactical_mcts_search_for_training_with_reuse, reuse_subtree};
    use kingfisher::transposition::TranspositionTable;

    let move_gen = MoveGen::new();
    let board = Board::new();
    let mut tt = TranspositionTable::new();

    let config = TacticalMctsConfig {
        max_iterations: 30,
        time_limit: Duration::from_secs(10),
        ..Default::default()
    };

    // First search
    let result1 = tactical_mcts_search_for_training_with_reuse(
        board.clone(), &move_gen, config.clone(), None, &mut tt,
    );

    assert!(result1.best_move.is_some());
    let best = result1.best_move.unwrap();

    // Reuse subtree for the best move
    let reused = reuse_subtree(result1.root_node, best);
    assert!(reused.is_some(), "Should find subtree for the played move");

    // Second search with reused tree
    let next_board = board.apply_move_to_board(best);
    let result2 = tactical_mcts_search_for_training_with_reuse(
        next_board, &move_gen, config, reused, &mut tt,
    );

    assert!(result2.best_move.is_some(), "Second search should return a move");
}

// === Dirichlet noise ===

#[test]
fn test_dirichlet_noise_modifies_priors() {
    use kingfisher::mcts::node::MctsNode;
    use kingfisher::mcts::tactical_mcts::apply_dirichlet_noise;

    let move_gen = MoveGen::new();
    let board = Board::new();
    let root = MctsNode::new_root(board, &move_gen);

    // Expand the root so it has children
    {
        let mut root_ref = root.borrow_mut();
        let board = root_ref.state.clone();
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
        for mv in captures.iter().chain(moves.iter()) {
            let next = board.apply_move_to_board(*mv);
            if next.is_legal(&move_gen) {
                let child = MctsNode::new_child(std::rc::Rc::downgrade(&root), *mv, next, &move_gen);
                root_ref.children.push(child);
            }
        }
        // Set uniform priors
        let n = root_ref.children.len() as f64;
        let child_moves: Vec<Move> = root_ref.children.iter()
            .filter_map(|c| c.borrow().action)
            .collect();
        for mv in child_moves {
            root_ref.move_priorities.insert(mv, 1.0 / n);
        }
        root_ref.policy_evaluated = true;
    }

    // Record priors before noise
    let priors_before: Vec<f64> = {
        let root_ref = root.borrow();
        root_ref.children.iter()
            .filter_map(|c| {
                let mv = c.borrow().action?;
                root_ref.move_priorities.get(&mv).copied()
            })
            .collect()
    };

    // Apply Dirichlet noise
    apply_dirichlet_noise(&root, 0.3, 0.25);

    // Record priors after noise
    let priors_after: Vec<f64> = {
        let root_ref = root.borrow();
        root_ref.children.iter()
            .filter_map(|c| {
                let mv = c.borrow().action?;
                root_ref.move_priorities.get(&mv).copied()
            })
            .collect()
    };

    // At least some priors should have changed
    let changed = priors_before.iter().zip(priors_after.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-6)
        .count();
    assert!(changed > 0, "Dirichlet noise should modify at least some priors");

    // All priors should still be positive
    for p in &priors_after {
        assert!(*p > 0.0, "All priors should remain positive after noise, got {}", p);
    }
}

// === Stats tracking ===

#[test]
fn test_tier1_stats_with_mate() {
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        enable_tier1_gate: true,
        mate_search_depth: 2,
        ..Default::default()
    };

    let (_, stats, _) = tactical_mcts_search(board, &move_gen, config);

    // Should have found mate via tier1 (pre-search gate or in-tree)
    assert!(stats.tier1_solutions > 0 || stats.mates_found > 0,
        "Should detect mate via tier1");
    // Pre-search gate finds mate before any tree is built, so nn_saved
    // may be 0. The key metric is that tier1 found the solution.
    assert!(stats.tier1_solutions > 0 || stats.nn_saved_by_tier1 > 0,
        "Should have tier1 solution or saved NN evals");
}

#[test]
fn test_all_tiers_disabled() {
    let move_gen = MoveGen::new();
    let board = Board::new();

    let config = TacticalMctsConfig {
        max_iterations: 30,
        time_limit: Duration::from_secs(10),
        enable_tier1_gate: false,
        enable_tier3_neural: false,
        ..Default::default()
    };

    let (best_move, stats, _) = tactical_mcts_search(board, &move_gen, config);

    assert!(best_move.is_some(), "Should still return a move with all tiers disabled");
    assert_eq!(stats.tier1_solutions, 0);
    assert_eq!(stats.nn_evaluations, 0);
}

// === InferenceServer integration tests ===

#[test]
fn test_mock_inference_server_used_in_search() {
    use kingfisher::mcts::InferenceServer;
    use std::sync::Arc;

    let move_gen = MoveGen::new();
    let board = Board::new();

    let server = InferenceServer::new_mock_uniform();
    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server)),
        enable_tier3_neural: true,
        use_neural_policy: true,
        ..Default::default()
    };

    let (best_move, stats, _) = tactical_mcts_search(board, &move_gen, config);

    assert!(best_move.is_some(), "Should return a move with mock inference server");
    assert!(stats.nn_evaluations > 0,
        "Should have NN evaluations when inference server is provided, got {}", stats.nn_evaluations);
}

#[test]
fn test_mock_uniform_matches_classical() {
    use kingfisher::mcts::InferenceServer;
    use std::sync::Arc;

    let move_gen = MoveGen::new();
    // Use a position with clear material advantage so Q-values are deterministic
    let board = Board::new_from_fen("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1");

    // Classical fallback (no server)
    let config_classical = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        enable_tier3_neural: false,
        ..Default::default()
    };

    let (move_classical, _, root_classical) = tactical_mcts_search(
        board.clone(), &move_gen, config_classical,
    );

    // Mock uniform server (v_logit=0, k=0.5 — identical to classical fallback)
    let server = InferenceServer::new_mock_uniform();
    let config_nn = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server)),
        enable_tier3_neural: true,
        use_neural_policy: true,
        ..Default::default()
    };

    let (move_nn, stats_nn, root_nn) = tactical_mcts_search(
        board, &move_gen, config_nn,
    );

    // Both should produce the same best move (same evaluation)
    assert_eq!(move_classical, move_nn,
        "Uniform mock (v_logit=0) should match classical fallback best move");

    // Q-values should be similar
    let q_classical = {
        let r = root_classical.borrow();
        if r.visits > 0 { -(r.total_value / r.visits as f64) } else { 0.0 }
    };
    let q_nn = {
        let r = root_nn.borrow();
        if r.visits > 0 { -(r.total_value / r.visits as f64) } else { 0.0 }
    };

    assert!((q_classical - q_nn).abs() < 0.15,
        "Q-values should be similar: classical={:.4}, nn={:.4}", q_classical, q_nn);

    // NN path should have nn_evaluations > 0
    assert!(stats_nn.nn_evaluations > 0, "NN path should track evaluations");
}

#[test]
fn test_biased_mock_changes_evaluation() {
    use kingfisher::mcts::InferenceServer;
    use std::sync::Arc;

    let move_gen = MoveGen::new();
    // Equal material position where bias should shift evaluation
    let board = Board::new_from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1");

    // Strongly positive bias (thinks position is great for STM)
    let server_pos = InferenceServer::new_mock_biased(2.0);
    let config_pos = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server_pos)),
        enable_tier3_neural: true,
        use_neural_policy: true,
        ..Default::default()
    };

    let (_, _, root_pos) = tactical_mcts_search(
        board.clone(), &move_gen, config_pos,
    );

    // Strongly negative bias (thinks position is terrible for STM)
    let server_neg = InferenceServer::new_mock_biased(-2.0);
    let config_neg = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        inference_server: Some(Arc::new(server_neg)),
        enable_tier3_neural: true,
        use_neural_policy: true,
        ..Default::default()
    };

    let (_, _, root_neg) = tactical_mcts_search(
        board, &move_gen, config_neg,
    );

    let q_pos = {
        let r = root_pos.borrow();
        if r.visits > 0 { -(r.total_value / r.visits as f64) } else { 0.0 }
    };
    let q_neg = {
        let r = root_neg.borrow();
        if r.visits > 0 { -(r.total_value / r.visits as f64) } else { 0.0 }
    };

    // Different biases should produce different Q-values, proving the NN is queried
    assert!((q_pos - q_neg).abs() > 0.05,
        "Different biases should produce different Q-values: pos={:.4}, neg={:.4}", q_pos, q_neg);
}

// === Subtree reuse edge cases ===

#[test]
fn test_reused_root_with_stale_terminal_value_still_finds_move() {
    // Regression test: when a node with terminal_or_mate_value (from mate search
    // or KOTH center_in_3) is reused as root, select_leaf_node would immediately
    // return the root as a leaf (because terminal_or_mate_value.is_some()), so no
    // child ever got visited and best_move was None — ending the game as "move limit".
    //
    // The fix: reuse_subtree clears terminal_or_mate_value on non-terminal nodes.
    use kingfisher::mcts::tactical_mcts::{tactical_mcts_search_for_training_with_reuse, reuse_subtree};
    use kingfisher::mcts::node::MctsNode;
    use kingfisher::transposition::TranspositionTable;

    let move_gen = MoveGen::new();
    let board = Board::new();
    let mut tt = TranspositionTable::new();

    let config = TacticalMctsConfig {
        max_iterations: 30,
        time_limit: Duration::from_secs(10),
        ..Default::default()
    };

    // Do an initial search to build a tree
    let result1 = tactical_mcts_search_for_training_with_reuse(
        board.clone(), &move_gen, config.clone(), None, &mut tt,
    );
    assert!(result1.best_move.is_some());
    let move1 = result1.best_move.unwrap();

    // Get the reused subtree (the child for the played move)
    let reused = reuse_subtree(result1.root_node, move1);
    assert!(reused.is_some());
    let reused_root = reused.unwrap();

    // Simulate the bug: artificially set terminal_or_mate_value and clear children
    // on the reused root, mimicking what happens when a leaf node deep in a previous
    // search gets flagged by mate_search or koth_center_in_3 and is later reused as root.
    {
        let mut node = reused_root.borrow_mut();
        assert!(!node.is_terminal, "Reused root should not be truly terminal");
        node.terminal_or_mate_value = Some(1.0); // Stale mate search value
        node.children.clear(); // Was a leaf in previous search (never expanded)
    }

    // The fix should clear terminal_or_mate_value since is_terminal is false
    let next_board = board.apply_move_to_board(move1);
    let result2 = tactical_mcts_search_for_training_with_reuse(
        next_board, &move_gen, config, Some(reused_root), &mut tt,
    );

    // Before the fix, best_move would be None here
    assert!(result2.best_move.is_some(),
        "Reused root with stale terminal_or_mate_value must still find a move");
    assert!(!result2.root_policy.is_empty(),
        "Should have non-empty policy");
}

// === Material value toggle ===

#[test]
fn test_enable_material_value_toggle() {
    // White is up a queen — with material enabled, classical fallback should give
    // strongly positive Q. With material disabled, classical fallback = 0.0 for every
    // leaf, so Q should be near 0.
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1");

    // With material enabled (default)
    let config_material = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        enable_tier3_neural: false,
        enable_material_value: true,
        ..Default::default()
    };

    let (_, _, root_material) = tactical_mcts_search(board.clone(), &move_gen, config_material);
    let q_material = {
        let r = root_material.borrow();
        if r.visits > 0 { -(r.total_value / r.visits as f64) } else { 0.0 }
    };

    // With material disabled (pure AlphaZero mode)
    let config_no_material = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        enable_tier3_neural: false,
        enable_material_value: false,
        ..Default::default()
    };

    let (_, _, root_no_material) = tactical_mcts_search(board, &move_gen, config_no_material);
    let q_no_material = {
        let r = root_no_material.borrow();
        if r.visits > 0 { -(r.total_value / r.visits as f64) } else { 0.0 }
    };

    // Material-enabled should detect the queen advantage
    assert!(q_material > 0.3,
        "With material enabled, white up a queen should have Q > 0.3, got {:.4}", q_material);

    // Material-disabled classical fallback gives 0.0 for every leaf
    assert!(q_no_material.abs() < 0.01,
        "With material disabled and no NN, Q should be ~0.0, got {:.4}", q_no_material);

    // The two should differ significantly
    assert!((q_material - q_no_material).abs() > 0.2,
        "Material toggle should produce different Q: material={:.4}, no_material={:.4}",
        q_material, q_no_material);
}

// === Edge cases ===

#[test]
fn test_single_legal_move() {
    // Position with only one legal move
    let move_gen = MoveGen::new();
    // Black king in corner, only move is Ka8->Ka7 (or similar)
    let board = Board::new_from_fen("k7/8/1K6/8/8/8/8/8 b - - 0 1");

    let config = test_config(50);
    let (best_move, _, _) = tactical_mcts_search(board, &move_gen, config);

    // Should still return the single legal move
    assert!(best_move.is_some(), "Should return the only legal move");
}
