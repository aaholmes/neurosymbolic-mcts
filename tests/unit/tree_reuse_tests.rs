/// Tests for MCTS tree reuse between moves.
use kingfisher::board::Board;
use kingfisher::mcts::tactical_mcts::{reuse_subtree, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;

fn run_search(
    iterations: u32,
) -> (
    Option<Move>,
    std::rc::Rc<std::cell::RefCell<kingfisher::mcts::node::MctsNode>>,
) {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let config = TacticalMctsConfig {
        max_iterations: iterations,
        ..Default::default()
    };
    let (best, _stats, root) = kingfisher::mcts::tactical_mcts_search(board, &move_gen, config);
    (best, root)
}

fn get_first_child_move(
    root: &std::rc::Rc<std::cell::RefCell<kingfisher::mcts::node::MctsNode>>,
) -> Move {
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
    assert_eq!(
        reused.borrow().action,
        Some(played_move),
        "Reused subtree root should have the played move as its action"
    );
}

/// Reused subtree's root should have parent set to None.
#[test]
fn test_reuse_subtree_clears_parent() {
    let (_best, root) = run_search(50);
    let played_move = get_first_child_move(&root);

    let reused = reuse_subtree(root.clone(), played_move).unwrap();
    assert!(
        reused.borrow().parent.is_none(),
        "Reused subtree root should have no parent"
    );
}

/// Reused subtree should preserve visit count and value.
#[test]
fn test_reuse_subtree_preserves_visits() {
    let (_best, root) = run_search(100);

    // Find a child with some visits
    let (played_move, expected_visits, expected_value) = {
        let root_ref = root.borrow();
        let child = root_ref
            .children
            .iter()
            .find(|c| c.borrow().visits > 0)
            .expect("Should have at least one visited child")
            .clone();
        drop(root_ref);
        let cr = child.borrow();
        (cr.action.unwrap(), cr.visits, cr.total_value)
    };

    let reused = reuse_subtree(root.clone(), played_move).unwrap();
    let reused_ref = reused.borrow();
    assert_eq!(
        reused_ref.visits, expected_visits,
        "Visit count should be preserved"
    );
    assert!(
        (reused_ref.total_value - expected_value).abs() < f64::EPSILON,
        "Total value should be preserved"
    );
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
        let child = root_ref
            .children
            .iter()
            .find(|c| !c.borrow().children.is_empty())
            .expect("With 200 iterations, at least one child should be expanded")
            .clone();
        drop(root_ref);
        let mv = child.borrow().action.unwrap();
        mv
    };

    let grandchild_count_before = {
        let root_ref = root.borrow();
        let child = root_ref
            .children
            .iter()
            .find(|c| c.borrow().action == Some(played_move))
            .unwrap()
            .clone();
        drop(root_ref);
        let count = child.borrow().children.len();
        count
    };

    let reused = reuse_subtree(root.clone(), played_move).unwrap();
    assert_eq!(
        reused.borrow().children.len(),
        grandchild_count_before,
        "Grandchildren should be preserved after reuse"
    );
    assert!(
        !reused.borrow().children.is_empty(),
        "Reused node should still have children (grandchildren of original root)"
    );
}

// === The simulation budget counts the visits a reused subtree already has ===

use kingfisher::mcts::tactical_mcts::{
    tactical_mcts_search_for_training_with_reuse, MctsTrainingResult,
};
use kingfisher::transposition::TranspositionTable;

fn training_config(iterations: u32, leaves_per_iteration: u32) -> TacticalMctsConfig {
    TacticalMctsConfig {
        max_iterations: iterations,
        time_limit: std::time::Duration::from_secs(30),
        // No early stop on forced results, so the budget alone ends the search.
        enable_tier1_gate: false,
        leaves_per_iteration,
        ..Default::default()
    }
}

fn training_search(
    board: Board,
    config: TacticalMctsConfig,
    reused: Option<std::rc::Rc<std::cell::RefCell<kingfisher::mcts::node::MctsNode>>>,
) -> MctsTrainingResult {
    let move_gen = MoveGen::new();
    let mut tt = TranspositionTable::new();
    tactical_mcts_search_for_training_with_reuse(board, &move_gen, config, reused, &mut tt)
}

/// The most visited child of a fresh 200-simulation search from the start, the
/// position after it, and its visit count.
fn reused_child(
    leaves_per_iteration: u32,
) -> (
    std::rc::Rc<std::cell::RefCell<kingfisher::mcts::node::MctsNode>>,
    Board,
    u32,
) {
    let board = Board::new();
    let first = training_search(
        board.clone(),
        training_config(200, leaves_per_iteration),
        None,
    );
    let best = first.best_move.expect("a move from the start position");
    let child = reuse_subtree(first.root_node, best).expect("the played child");
    let visits = child.borrow().visits;
    assert!(
        visits > 1 && visits < 200,
        "the played child should hold part of the search, has {visits}"
    );
    (child, board.apply_move_to_board(best), visits)
}

#[test]
fn a_fresh_search_runs_the_whole_budget() {
    let result = training_search(Board::new(), training_config(200, 1), None);
    assert_eq!(result.stats.iterations, 200);
}

#[test]
fn a_reused_subtree_is_topped_up_to_the_budget() {
    for leaves in [1, 8] {
        let (child, board, visits) = reused_child(leaves);
        let result = training_search(board, training_config(200, leaves), Some(child));
        assert_eq!(
            result.stats.iterations + visits,
            200,
            "{leaves} leaves per iteration: {visits} reused visits"
        );
        assert!(result.best_move.is_some());
    }
}

#[test]
fn a_reused_subtree_at_or_over_the_budget_runs_no_new_simulations() {
    let (child, board, visits) = reused_child(1);
    let result = training_search(board, training_config(visits, 1), Some(child));
    assert_eq!(result.stats.iterations, 0);
    assert!(
        result.best_move.is_some(),
        "the reused tree still chooses a move"
    );
    let policy_visits: u32 = result.root_policy.iter().map(|(_, v)| *v).sum();
    assert!(policy_visits > 0, "the policy comes from the reused visits");
}
