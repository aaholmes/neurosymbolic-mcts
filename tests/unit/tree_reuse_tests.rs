/// Tests for MCTS tree reuse between moves.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::mcts::tactical_mcts::{TacticalMctsConfig, reuse_subtree};

fn run_search(iterations: u32) -> (Option<Move>, std::rc::Rc<std::cell::RefCell<kingfisher::mcts::node::MctsNode>>) {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let config = TacticalMctsConfig {
        max_iterations: iterations,
        ..Default::default()
    };
    let (best, _stats, root) = kingfisher::mcts::tactical_mcts_search(
        board, &move_gen, config,
    );
    (best, root)
}

fn get_first_child_move(root: &std::rc::Rc<std::cell::RefCell<kingfisher::mcts::node::MctsNode>>) -> Move {
    let root_ref = root.borrow();
    let child = root_ref.children[0].clone();
    drop(root_ref);
    let action = child.borrow().action.unwrap();
    action
}

/// Given a root with children, reuse_subtree should find the child matching the played move.
#[test]
fn test_reuse_subtree_finds_played_child() {
    let (_best, root) = run_search(50);
    let played_move = get_first_child_move(&root);

    let reused = reuse_subtree(root.clone(), played_move);
    assert!(reused.is_some(), "Should find subtree for played move");

    let reused = reused.unwrap();
    assert_eq!(reused.borrow().action, Some(played_move),
        "Reused subtree root should have the played move as its action");
}

/// Reused subtree's root should have parent set to None.
#[test]
fn test_reuse_subtree_clears_parent() {
    let (_best, root) = run_search(50);
    let played_move = get_first_child_move(&root);

    let reused = reuse_subtree(root.clone(), played_move).unwrap();
    assert!(reused.borrow().parent.is_none(),
        "Reused subtree root should have no parent");
}

/// Reused subtree should preserve visit count and value.
#[test]
fn test_reuse_subtree_preserves_visits() {
    let (_best, root) = run_search(100);

    // Find a child with some visits
    let (played_move, expected_visits, expected_value) = {
        let root_ref = root.borrow();
        let child = root_ref.children.iter()
            .find(|c| c.borrow().visits > 0)
            .expect("Should have at least one visited child")
            .clone();
        drop(root_ref);
        let cr = child.borrow();
        (cr.action.unwrap(), cr.visits, cr.total_value)
    };

    let reused = reuse_subtree(root.clone(), played_move).unwrap();
    let reused_ref = reused.borrow();
    assert_eq!(reused_ref.visits, expected_visits,
        "Visit count should be preserved");
    assert!((reused_ref.total_value - expected_value).abs() < f64::EPSILON,
        "Total value should be preserved");
}

/// Returns None if the played move is not found among children.
#[test]
fn test_reuse_subtree_returns_none_for_missing_move() {
    let (_best, root) = run_search(50);

    // Use a move that doesn't exist (e.g., a8a1 in starting position is impossible)
    let fake_move = Move::new(56, 0, None);
    let reused = reuse_subtree(root.clone(), fake_move);
    assert!(reused.is_none(), "Should return None for missing move");
}

/// Grandchild nodes should still be accessible after reuse.
#[test]
fn test_reuse_subtree_preserves_grandchildren() {
    let (_best, root) = run_search(200);

    // Find a child that has been expanded (has grandchildren)
    let played_move = {
        let root_ref = root.borrow();
        let child = root_ref.children.iter()
            .find(|c| !c.borrow().children.is_empty())
            .expect("With 200 iterations, at least one child should be expanded")
            .clone();
        drop(root_ref);
        let mv = child.borrow().action.unwrap();
        mv
    };

    let grandchild_count_before = {
        let root_ref = root.borrow();
        let child = root_ref.children.iter()
            .find(|c| c.borrow().action == Some(played_move))
            .unwrap()
            .clone();
        drop(root_ref);
        let count = child.borrow().children.len();
        count
    };

    let reused = reuse_subtree(root.clone(), played_move).unwrap();
    assert_eq!(reused.borrow().children.len(), grandchild_count_before,
        "Grandchildren should be preserved after reuse");
    assert!(!reused.borrow().children.is_empty(),
        "Reused node should still have children (grandchildren of original root)");
}
