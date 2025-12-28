//! Tests for MCTS node logic and value handling

use kingfisher::mcts::node::MctsNode;
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
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
