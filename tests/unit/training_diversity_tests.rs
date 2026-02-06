/// Tests for training diversity features:
/// - Dirichlet noise application
/// - KOTH gating config
/// - Self-play game diversity

use kingfisher::board::Board;

use kingfisher::eval::PestoEval;
use kingfisher::mcts::{
    apply_dirichlet_noise,
    tactical_mcts_search_for_training_with_reuse,
    MctsNode, TacticalMctsConfig,
};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::transposition::TranspositionTable;
use std::time::Duration;

/// apply_dirichlet_noise modifies root priors and they still sum to ~1.0.
#[test]
fn test_dirichlet_noise_modifies_priors() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let root = MctsNode::new_root(board, &move_gen);

    // Expand and set uniform priors
    {
        let mut node = root.borrow_mut();
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(&node.state);
        for mv in captures.iter().chain(moves.iter()) {
            let new_board = node.state.apply_move_to_board(*mv);
            if new_board.is_legal(&move_gen) {
                let child = MctsNode::new_child(
                    std::rc::Rc::downgrade(&root),
                    *mv,
                    new_board,
                    &move_gen,
                );
                node.children.push(child);
            }
        }
        let n = node.children.len() as f64;
        let moves: Vec<Move> = node.children.iter()
            .filter_map(|c| c.borrow().action)
            .collect();
        for mv in moves {
            node.move_priorities.insert(mv, 1.0 / n);
        }
        node.policy_evaluated = true;
    }

    // Record original priors
    let original_priors: std::collections::HashMap<Move, f64> =
        root.borrow().move_priorities.clone();

    // Apply noise
    apply_dirichlet_noise(&root, 0.3, 0.25);

    let noisy_priors = &root.borrow().move_priorities;

    // Priors should sum to ~1.0
    let sum: f64 = noisy_priors.values().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Noisy priors should sum to ~1.0, got {}",
        sum
    );

    // At least some priors should have changed
    let mut changed = 0;
    for (mv, original) in &original_priors {
        let noisy = noisy_priors.get(mv).unwrap();
        if (*noisy - *original).abs() > 1e-6 {
            changed += 1;
        }
    }
    assert!(changed > 0, "Dirichlet noise should modify at least some priors");
}

/// apply_dirichlet_noise with epsilon=0 leaves priors unchanged.
#[test]
fn test_dirichlet_noise_epsilon_zero_no_change() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let root = MctsNode::new_root(board, &move_gen);

    {
        let mut node = root.borrow_mut();
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(&node.state);
        for mv in captures.iter().chain(moves.iter()) {
            let new_board = node.state.apply_move_to_board(*mv);
            if new_board.is_legal(&move_gen) {
                let child = MctsNode::new_child(
                    std::rc::Rc::downgrade(&root),
                    *mv,
                    new_board,
                    &move_gen,
                );
                node.children.push(child);
            }
        }
        let n = node.children.len() as f64;
        let moves: Vec<Move> = node.children.iter()
            .filter_map(|c| c.borrow().action)
            .collect();
        for mv in moves {
            node.move_priorities.insert(mv, 1.0 / n);
        }
        node.policy_evaluated = true;
    }

    let original: std::collections::HashMap<Move, f64> =
        root.borrow().move_priorities.clone();

    apply_dirichlet_noise(&root, 0.3, 0.0);

    let after = &root.borrow().move_priorities;
    for (mv, orig) in &original {
        let new_val = after.get(mv).unwrap();
        assert!(
            (*new_val - *orig).abs() < 1e-10,
            "epsilon=0 should leave priors unchanged"
        );
    }
}

/// KOTH gating: enable_koth=false should skip KOTH checks (no early return from KOTH).
#[test]
fn test_koth_gating_disabled_by_default() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let mut nn = None;
    let mut tt = TranspositionTable::new();

    let config = TacticalMctsConfig {
        max_iterations: 10,
        time_limit: Duration::from_secs(10),
        enable_koth: false, // default
        ..Default::default()
    };

    // This should complete without KOTH interfering
    let result = tactical_mcts_search_for_training_with_reuse(
        board,
        &move_gen,
        &pesto,
        &mut nn,
        config,
        None,
        &mut tt,
    );

    assert!(result.best_move.is_some(), "Should produce a move");
    assert!(result.stats.iterations > 0, "Should complete iterations");
}

/// Self-play with Dirichlet noise produces different search results across runs.
#[test]
fn test_dirichlet_noise_produces_different_searches() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();

    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        ..Default::default()
    };

    // Run multiple searches and collect root policies
    let mut policies = Vec::new();
    for _ in 0..5 {
        let mut nn = None;
        let mut tt = TranspositionTable::new();
        let result = tactical_mcts_search_for_training_with_reuse(
            board.clone(),
            &move_gen,
            &pesto,
            &mut nn,
            config.clone(),
            None,
            &mut tt,
        );
        policies.push(result.root_policy);
    }

    // At least some policies should differ (visit counts should vary due to noise)
    let mut any_different = false;
    for i in 1..policies.len() {
        let p0: Vec<u32> = policies[0].iter().map(|(_, v)| *v).collect();
        let pi: Vec<u32> = policies[i].iter().map(|(_, v)| *v).collect();
        if p0 != pi {
            any_different = true;
            break;
        }
    }
    assert!(
        any_different,
        "Dirichlet noise should cause different search results across runs"
    );
}
