/// Tests for MCTS selection optimization: children created during expansion
/// are already legality-checked, so selection should NOT re-validate them.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::node::MctsNode;
use kingfisher::mcts::selection::{select_child_with_tactical_priority, calculate_ucb_value};
use kingfisher::mcts::tactical_mcts::{TacticalMctsStats, TacticalMctsConfig};
use std::rc::Rc;

use crate::common::board_from_fen;

/// After expansion, selection should return a child with a valid action
/// that exists in the children list — no external gen_pseudo_legal_moves needed.
#[test]
fn test_selection_returns_valid_child() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let root = MctsNode::new_root(board, &move_gen);
    let config = TacticalMctsConfig::default();
    let mut stats = TacticalMctsStats::default();
    let mut nn_policy = None;

    // First selection should expand and return a child
    let child = select_child_with_tactical_priority(
        root.clone(), &config, &move_gen, &mut nn_policy, &mut stats, None, 0,
    );

    assert!(child.is_some(), "Selection should return a child from starting position");
    let child = child.unwrap();
    let child_action = child.borrow().action;
    assert!(child_action.is_some(), "Child should have an action");

    // Verify the returned child is actually in the root's children list
    let root_ref = root.borrow();
    let found = root_ref.children.iter().any(|c| {
        c.borrow().action == child_action
    });
    assert!(found, "Selected child should exist in root's children list");
}

/// All children created during expansion have legal moves.
/// Selection should trust this and not re-validate.
#[test]
fn test_selection_consistent_with_expansion() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let root = MctsNode::new_root(board, &move_gen);
    let config = TacticalMctsConfig::default();
    let mut stats = TacticalMctsStats::default();
    let mut nn_policy = None;

    // Trigger expansion via selection
    let _ = select_child_with_tactical_priority(
        root.clone(), &config, &move_gen, &mut nn_policy, &mut stats, None, 0,
    );

    // Verify all children have legal actions
    let root_ref = root.borrow();
    assert_eq!(root_ref.children.len(), 20, "Starting position should have 20 legal moves");

    for child in &root_ref.children {
        let child_ref = child.borrow();
        let action = child_ref.action.expect("Child must have an action");
        // The child's state should be the result of a legal move
        assert!(child_ref.state.is_legal(&move_gen),
            "Child state for move {} should be legal", action.to_uci());
    }
}

/// Selection should return None for terminal positions (no children).
#[test]
fn test_selection_handles_no_children() {
    // Stalemate position — black has no legal moves
    let board = board_from_fen("k7/1R6/K7/8/8/8/8/8 b - - 0 1");
    let move_gen = MoveGen::new();
    let root = MctsNode::new_root(board, &move_gen);
    let config = TacticalMctsConfig::default();
    let mut stats = TacticalMctsStats::default();
    let mut nn_policy = None;

    let child = select_child_with_tactical_priority(
        root.clone(), &config, &move_gen, &mut nn_policy, &mut stats, None, 0,
    );

    assert!(child.is_none(), "Terminal position should return None");
}

/// UCB selection should pick the child with highest UCB value.
#[test]
fn test_ucb_selection_picks_highest_value() {
    let board = Board::new();
    let move_gen = MoveGen::new();

    // Test UCB calculation directly
    let parent = MctsNode::new_root(board.clone(), &move_gen);
    let e2e4 = kingfisher::move_types::Move::new(12, 28, None);
    let child_board = board.apply_move_to_board(e2e4);
    let child = MctsNode::new_child(Rc::downgrade(&parent), e2e4, child_board, &move_gen);

    // Unvisited node with high prior should have high UCB
    let ucb_high_prior = calculate_ucb_value(&child.borrow(), 100, 0.5, 1.414);

    // Low prior
    let ucb_low_prior = calculate_ucb_value(&child.borrow(), 100, 0.01, 1.414);

    assert!(ucb_high_prior > ucb_low_prior,
        "Higher prior should produce higher UCB: {} vs {}", ucb_high_prior, ucb_low_prior);
}

/// select_best_explored_child should return the highest-PUCT child directly
/// without re-checking move legality (children are already legal).
#[test]
fn test_best_explored_child_no_revalidation() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let root = MctsNode::new_root(board, &move_gen);
    let config = TacticalMctsConfig::default();
    let mut stats = TacticalMctsStats::default();
    let mut nn_policy = None;

    // Expand the root
    let _ = select_child_with_tactical_priority(
        root.clone(), &config, &move_gen, &mut nn_policy, &mut stats, None, 0,
    );

    // Give one child more visits so it has a defined Q-value
    {
        let root_ref = root.borrow();
        let first_child = &root_ref.children[0];
        first_child.borrow_mut().visits = 10;
        first_child.borrow_mut().total_value = 5.0;
    }

    // select_best_explored_child should work without issues
    let root_ref = root.borrow();
    let best = root_ref.select_best_explored_child(&move_gen, 1.414);
    assert!(best.borrow().action.is_some(), "Best child should have an action");
}

/// Selection in a tactical position should still work correctly.
#[test]
fn test_selection_tactical_position() {
    // Position with captures available
    let board = board_from_fen("8/8/8/3q4/4N3/8/8/K6k w - - 0 1");
    let move_gen = MoveGen::new();
    let root = MctsNode::new_root(board, &move_gen);
    let config = TacticalMctsConfig::default();
    let mut stats = TacticalMctsStats::default();
    let mut nn_policy = None;

    let child = select_child_with_tactical_priority(
        root.clone(), &config, &move_gen, &mut nn_policy, &mut stats, None, 0,
    );

    assert!(child.is_some(), "Should select a child in tactical position");
}

/// Multiple selections should all return valid children without errors.
#[test]
fn test_repeated_selection_no_errors() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let root = MctsNode::new_root(board, &move_gen);
    let config = TacticalMctsConfig::default();
    let mut stats = TacticalMctsStats::default();
    let mut nn_policy = None;

    // Run selection many times (simulating MCTS iterations)
    for _ in 0..50 {
        let child = select_child_with_tactical_priority(
            root.clone(), &config, &move_gen, &mut nn_policy, &mut stats, None, 0,
        );
        assert!(child.is_some(), "Selection should always return a child from starting position");
    }
}
