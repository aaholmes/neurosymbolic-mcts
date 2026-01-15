//! Tests for MCTS node logic and value handling

use kingfisher::mcts::node::{MctsNode, MoveCategory, NodeOrigin};
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use std::rc::Rc;
use crate::common::{board_from_fen, assert_in_tanh_domain, positions};

#[test]
fn test_terminal_value_checkmate() {
    let move_gen = MoveGen::new();

    // Position where side-to-move is checkmated
    let mated_board = board_from_fen("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1");
    let node = MctsNode::new_root(mated_board, &move_gen);
    let node_ref = node.borrow();

    // StM (Black) is mated, so value should be -1.0 (loss for StM)
    assert_eq!(
        node_ref.terminal_or_mate_value,
        Some(-1.0),
        "Checkmate should give -1.0 (loss for StM)"
    );
}

#[test]
fn test_terminal_value_stalemate() {
    let move_gen = MoveGen::new();
    let stalemate_board = board_from_fen(positions::STALEMATE);
    let node = MctsNode::new_root(stalemate_board, &move_gen);
    let node_ref = node.borrow();

    assert_eq!(
        node_ref.terminal_or_mate_value,
        Some(0.0),
        "Stalemate should give 0.0 (draw)"
    );
}

#[test]
fn test_values_in_tanh_domain() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let root = MctsNode::new_root(board, &move_gen);

    // Manually set some values to verify the checking function works on a mock structure
    root.borrow_mut().nn_value = Some(0.5);
    root.borrow_mut().visits = 10;
    root.borrow_mut().total_value = 5.0; // avg 0.5

    fn check_node_values(node: &MctsNode) {
        if let Some(v) = node.terminal_or_mate_value {
            assert_in_tanh_domain(v, "terminal_or_mate_value");
        }
        if let Some(v) = node.nn_value {
            assert_in_tanh_domain(v, "nn_value");
        }
        if node.visits > 0 {
            let avg_value = node.total_value / node.visits as f64;
            assert_in_tanh_domain(avg_value, "average Q value");
        }
        for (_, &v) in &node.tactical_values {
            assert_in_tanh_domain(v, "tactical_values entry");
        }
    }

    check_node_values(&root.borrow());
}

#[test]
fn test_new_root_starting_position() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let root = MctsNode::new_root(board, &move_gen);
    let root_ref = root.borrow();

    // Starting position is not terminal
    assert!(!root_ref.is_terminal, "Starting position should not be terminal");
    assert!(root_ref.action.is_none(), "Root should have no action");
    assert!(root_ref.parent.is_none(), "Root should have no parent");
    assert_eq!(root_ref.visits, 0, "New node should have 0 visits");
    assert_eq!(root_ref.total_value, 0.0, "New node should have 0 total value");
    assert!(root_ref.children.is_empty(), "New node should have no children");
}

#[test]
fn test_new_child_creation() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let root = MctsNode::new_root(board.clone(), &move_gen);

    // Create a child with e2e4
    let e2e4 = Move::from_uci("e2e4").unwrap();
    let new_state = board.apply_move_to_board(e2e4);
    let child = MctsNode::new_child(
        Rc::downgrade(&root),
        e2e4,
        new_state,
        &move_gen,
    );

    let child_ref = child.borrow();
    assert_eq!(child_ref.action, Some(e2e4), "Child should have the action");
    assert!(child_ref.parent.is_some(), "Child should have a parent");
    assert!(!child_ref.is_terminal, "e2e4 position should not be terminal");
}

#[test]
fn test_is_game_terminal() {
    let move_gen = MoveGen::new();

    // Non-terminal
    let normal_board = board_from_fen(positions::STARTING);
    let normal_node = MctsNode::new_root(normal_board, &move_gen);
    assert!(!normal_node.borrow().is_game_terminal());

    // Terminal (checkmate)
    let checkmate_board = board_from_fen("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1");
    let checkmate_node = MctsNode::new_root(checkmate_board, &move_gen);
    assert!(checkmate_node.borrow().is_game_terminal());

    // Terminal (stalemate)
    let stalemate_board = board_from_fen(positions::STALEMATE);
    let stalemate_node = MctsNode::new_root(stalemate_board, &move_gen);
    assert!(stalemate_node.borrow().is_game_terminal());
}

#[test]
fn test_is_evaluated_or_terminal() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let node = MctsNode::new_root(board, &move_gen);

    // Initially not evaluated
    assert!(!node.borrow().is_evaluated_or_terminal());

    // Set nn_value
    node.borrow_mut().nn_value = Some(0.1);
    assert!(node.borrow().is_evaluated_or_terminal());
}

#[test]
fn test_uct_value_unvisited() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let node = MctsNode::new_root(board, &move_gen);

    let uct = node.borrow().uct_value(100, 1.4);

    // Unvisited nodes should return infinity
    assert!(uct.is_infinite() && uct > 0.0, "Unvisited node should have infinite UCT");
}

#[test]
fn test_uct_value_visited() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let node = MctsNode::new_root(board, &move_gen);

    // Simulate some visits
    {
        let mut node_mut = node.borrow_mut();
        node_mut.visits = 10;
        node_mut.total_value = 5.0; // Q = 0.5
    }

    let uct = node.borrow().uct_value(100, 1.4);

    // UCT should be Q + exploration term
    // Q = 0.5, exploration = 1.4 * sqrt(ln(100)/10) ≈ 1.4 * sqrt(0.46) ≈ 0.95
    assert!(uct > 0.5, "UCT should be greater than Q alone");
    assert!(uct < 2.0, "UCT should be reasonable");
}

#[test]
fn test_puct_value() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let node = MctsNode::new_root(board, &move_gen);

    {
        let mut node_mut = node.borrow_mut();
        node_mut.visits = 10;
        node_mut.total_value = 5.0;
    }

    let puct = node.borrow().puct_value(100, 20, 1.5);

    // PUCT = Q + c * P * sqrt(N) / (1 + n)
    assert!(puct > 0.0, "PUCT should be positive");
}

#[test]
fn test_categorize_and_store_moves() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let node = MctsNode::new_root(board, &move_gen);

    node.borrow_mut().categorize_and_store_moves(&move_gen);

    let node_ref = node.borrow();

    // Should have some legal moves
    assert!(node_ref.num_legal_moves.is_some());
    assert_eq!(node_ref.num_legal_moves.unwrap(), 20, "Starting position has 20 legal moves");

    // Should have moves categorized
    assert!(!node_ref.unexplored_moves_by_cat.is_empty());

    // Starting position has no captures or checks
    assert!(node_ref.unexplored_moves_by_cat.get(&MoveCategory::Quiet).is_some());
}

#[test]
fn test_move_category_ordering() {
    // MoveCategory should have Check < Capture < Quiet (priority order)
    assert!(MoveCategory::Check < MoveCategory::Capture);
    assert!(MoveCategory::Capture < MoveCategory::Quiet);
}

#[test]
fn test_node_origin_colors() {
    // Verify NodeOrigin colors are valid strings
    assert!(!NodeOrigin::Gate.to_color().is_empty());
    assert!(!NodeOrigin::Grafted.to_color().is_empty());
    assert!(!NodeOrigin::Neural.to_color().is_empty());
    assert!(!NodeOrigin::Unknown.to_color().is_empty());
}

#[test]
fn test_node_origin_labels() {
    assert_eq!(NodeOrigin::Gate.to_label(), "T1:Gate");
    assert_eq!(NodeOrigin::Grafted.to_label(), "T2:QS");
    assert_eq!(NodeOrigin::Neural.to_label(), "T3:NN");
    assert_eq!(NodeOrigin::Unknown.to_label(), "?");
}

#[test]
fn test_is_fully_explored() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::STARTING);
    let node = MctsNode::new_root(board, &move_gen);

    // Initially not fully explored
    assert!(!node.borrow().is_fully_explored());

    // After categorizing, should have moves
    node.borrow_mut().categorize_and_store_moves(&move_gen);
    assert!(!node.borrow().is_fully_explored(), "Still has unexplored moves");

    // Clear unexplored moves
    node.borrow_mut().unexplored_moves_by_cat.clear();
    assert!(node.borrow().is_fully_explored(), "Now fully explored");
}

#[test]
fn test_categorize_moves_basic() {
    let move_gen = MoveGen::new();
    // Simple position: just kings
    let board = board_from_fen("8/8/8/8/8/8/8/K6k w - - 0 1");
    let node = MctsNode::new_root(board, &move_gen);

    node.borrow_mut().categorize_and_store_moves(&move_gen);

    let node_ref = node.borrow();

    // Should have legal moves (king moves)
    assert!(node_ref.num_legal_moves.is_some());
    assert!(node_ref.num_legal_moves.unwrap() > 0, "King should have moves");

    // Should have at least one category populated
    assert!(!node_ref.unexplored_moves_by_cat.is_empty(),
        "Should have categorized moves");
}
