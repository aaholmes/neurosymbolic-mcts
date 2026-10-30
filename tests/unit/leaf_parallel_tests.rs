//! Several leaves per MCTS iteration (`leaves_per_iteration`), steered apart by
//! virtual loss, so one game can have several positions waiting for the network.

use kingfisher::board::Board;
use kingfisher::mcts::node::MctsNode;
use kingfisher::mcts::selection::select_child_with_tactical_priority;
use kingfisher::mcts::tactical_mcts::{
    tactical_mcts_search, TacticalMctsConfig, TacticalMctsStats,
};
use kingfisher::mcts::InferenceServer;
use kingfisher::move_generation::MoveGen;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

fn config(iterations: u32, leaves: u32) -> TacticalMctsConfig {
    TacticalMctsConfig {
        max_iterations: iterations,
        time_limit: Duration::from_secs(60),
        inference_server: Some(Arc::new(InferenceServer::new_mock_uniform())),
        enable_tier3_neural: true,
        use_neural_policy: true,
        randomize_move_order: false,
        leaves_per_iteration: leaves,
        ..Default::default()
    }
}

fn root_child_visits(root: &Rc<RefCell<MctsNode>>) -> Vec<u32> {
    root.borrow()
        .children
        .iter()
        .map(|c| c.borrow().visits)
        .collect()
}

fn max_virtual_loss(node: &Rc<RefCell<MctsNode>>) -> u32 {
    let n = node.borrow();
    n.children
        .iter()
        .map(max_virtual_loss)
        .fold(n.virtual_loss, u32::max)
}

const MIDDLEGAME: &str = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";

/// Root child visit counts with one leaf per iteration, recorded before
/// leaves_per_iteration existed (mock uniform network, no move shuffling).
const K1_ROOT_VISITS: &[u32] = &[
    1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 236, 13, 1, 1, 1, 1, 1, 1, 7, 2, 6, 1, 1, 1, 1, 5, 1,
    1, 1,
];

#[test]
fn one_leaf_per_iteration_matches_recorded_search() {
    let mg = MoveGen::new();
    let (_, _, root) = tactical_mcts_search(Board::new_from_fen(MIDDLEGAME), &mg, config(300, 1));
    assert_eq!(root_child_visits(&root), K1_ROOT_VISITS);
}

#[test]
fn four_leaves_run_the_same_number_of_simulations() {
    let mg = MoveGen::new();
    let board = Board::new_from_fen(MIDDLEGAME);
    let (_, s1, root1) = tactical_mcts_search(board.clone(), &mg, config(300, 1));
    let (_, s4, root4) = tactical_mcts_search(board, &mg, config(300, 4));
    assert_eq!(s4.iterations, s1.iterations);
    assert_eq!(root4.borrow().visits, root1.borrow().visits);
}

#[test]
fn no_virtual_loss_remains_after_search() {
    let mg = MoveGen::new();
    let (_, _, root) = tactical_mcts_search(Board::new_from_fen(MIDDLEGAME), &mg, config(300, 4));
    assert_eq!(max_virtual_loss(&root), 0);
}

#[test]
fn four_leaves_still_find_mate_in_one() {
    let mg = MoveGen::new();
    // Scholar's mate pattern: Qxf7#
    let board =
        Board::new_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4");
    let (best, _, _) = tactical_mcts_search(board, &mg, config(200, 4));
    let mv = best.expect("a move");
    assert_eq!((mv.from, mv.to), (39, 53), "expected Qh5xf7#, got {:?}", mv);
}

#[test]
fn virtual_loss_steers_selection_away_and_is_fully_removed() {
    let mg = MoveGen::new();
    let cfg = config(100, 1);
    let (_, _, root) = tactical_mcts_search(Board::new_from_fen(MIDDLEGAME), &mg, cfg.clone());
    let mut stats = TacticalMctsStats::default();
    let pick = |stats: &mut TacticalMctsStats| {
        select_child_with_tactical_priority(root.clone(), &cfg, &mg, stats, None, 0).unwrap()
    };

    let first = pick(&mut stats);
    for _ in 0..50 {
        MctsNode::apply_virtual_loss(&first);
    }
    let second = pick(&mut stats);
    assert!(
        !Rc::ptr_eq(&first, &second),
        "virtual loss should steer selection elsewhere"
    );

    for _ in 0..50 {
        MctsNode::remove_virtual_loss(&first);
    }
    assert_eq!(max_virtual_loss(&root), 0);
    assert!(
        Rc::ptr_eq(&first, &pick(&mut stats)),
        "removal should restore the original choice"
    );
}
