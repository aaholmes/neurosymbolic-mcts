//! Exact leaf values without the tiers: with tier 1 off (vanilla search), a leaf
//! whose side to move can mate in one, or whose position the Syzygy tables
//! cover, gets its exact value instead of a network evaluation. The root is
//! always searched, so a move is always returned.

use kingfisher::board::Board;
use kingfisher::egtb::Tablebase;
use kingfisher::mcts::tactical_mcts::{
    reuse_subtree, tactical_mcts_search, tactical_mcts_search_for_training_with_reuse,
    TacticalMctsConfig,
};
use kingfisher::move_generation::MoveGen;
use kingfisher::transposition::TranspositionTable;
use std::time::Duration;

/// Vanilla search: no tier 1, no material term.
fn vanilla(iterations: u32) -> TacticalMctsConfig {
    TacticalMctsConfig {
        max_iterations: iterations,
        time_limit: Duration::from_secs(30),
        enable_tier1_gate: false,
        mate_search_depth: 0,
        enable_material_value: false,
        ..Default::default()
    }
}

fn tables() -> Option<Tablebase> {
    let dir = std::env::var("KF_SYZYGY_PATH").unwrap_or_else(|_| "/data/syzygy".into());
    if !std::path::Path::new(&dir).join("KRvK.rtbw").exists() {
        eprintln!("skipped: no Syzygy files in {dir}");
        return None;
    }
    Some(Tablebase::open(&dir).unwrap())
}

#[test]
fn exact_leaf_values_are_off_by_default() {
    let c = TacticalMctsConfig::default();
    assert!(!c.leaf_mate_in_1);
    assert!(c.tablebase.is_none());
}

#[test]
fn with_mate_in_one_on_a_root_with_a_mate_in_one_is_searched_and_mates() {
    let mg = MoveGen::new();
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");
    let config = TacticalMctsConfig {
        leaf_mate_in_1: true,
        ..vanilla(100)
    };
    let (best, _, root) = tactical_mcts_search(board, &mg, config);
    assert_eq!(best.map(|m| m.to()), Some(60), "Re8#");
    assert!(root.borrow().terminal_or_mate_value.is_none());
}

/// Black threatens Ra1. After Kh1 that is mate; after h3 the king escapes to h2.
/// Kh1 is a proven loss for White.
#[test]
fn with_mate_in_one_on_moves_that_allow_a_mate_in_one_are_proven_losses() {
    let mg = MoveGen::new();
    let fen = "r5k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1";
    let run = |on: bool| {
        let config = TacticalMctsConfig {
            leaf_mate_in_1: on,
            ..vanilla(400)
        };
        tactical_mcts_search(Board::new_from_fen(fen), &mg, config)
    };
    let (_, stats, root) = run(true);
    let lost: Vec<String> = root
        .borrow()
        .children
        .iter()
        .filter(|c| c.borrow().terminal_or_mate_value == Some(1.0))
        .map(|c| c.borrow().action.unwrap().to_uci())
        .collect();
    assert!(lost.contains(&"g1h1".to_string()), "lost: {lost:?}");
    assert!(!lost.contains(&"h2h3".to_string()), "lost: {lost:?}");
    assert!(stats.exact_leaf_values > 0);
    for c in root.borrow().children.iter() {
        let c = c.borrow();
        if c.terminal_or_mate_value == Some(1.0) {
            assert_eq!(c.visits, 1, "{:?}", c.action);
        }
    }

    let (_, stats, root) = run(false);
    assert_eq!(stats.exact_leaf_values, 0);
    assert!(root
        .borrow()
        .children
        .iter()
        .all(|c| c.borrow().terminal_or_mate_value.is_none()));
}

#[test]
fn with_tables_every_leaf_below_a_covered_root_is_exact() {
    let Some(tb) = tables() else { return };
    let mg = MoveGen::new();
    let config = TacticalMctsConfig {
        tablebase: Some(tb),
        ..vanilla(60)
    };
    let board = Board::new_from_fen("8/8/8/8/8/2k5/8/R3K3 w - - 0 1");
    let (best, stats, root) = tactical_mcts_search(board, &mg, config);
    assert!(best.is_some());
    assert_eq!(stats.nn_evaluations, 0);
    assert!(stats.exact_leaf_values > 0);
    let root = root.borrow();
    assert!(
        root.terminal_or_mate_value.is_none(),
        "the root is searched"
    );
    for c in root.children.iter().filter(|c| c.borrow().visits > 0) {
        let v = c.borrow().terminal_or_mate_value;
        assert!(v.is_some(), "{:?} has no exact value", c.borrow().action);
    }
    // Some rook checks let the black king take the rook; the move chosen keeps
    // the win.
    let best = best.unwrap();
    let after = Board::new_from_fen("8/8/8/8/8/2k5/8/R3K3 w - - 0 1").apply_move_to_board(best);
    let child_value = Tablebase::open("/data/syzygy")
        .unwrap()
        .probe_value(&after)
        .unwrap();
    assert_eq!(child_value, -1.0, "{} should keep the win", best.to_uci());
}

#[test]
fn the_tables_are_not_used_in_king_of_the_hill() {
    let Some(tb) = tables() else { return };
    let mg = MoveGen::new();
    let config = TacticalMctsConfig {
        tablebase: Some(tb),
        enable_koth: true,
        ..vanilla(40)
    };
    let board = Board::new_from_fen("8/8/8/8/8/2k5/8/R3K3 w - - 0 1");
    let (_, stats, root) = tactical_mcts_search(board, &mg, config);
    assert_eq!(stats.exact_leaf_values, 0);
    assert!(root
        .borrow()
        .children
        .iter()
        .all(|c| c.borrow().terminal_or_mate_value.is_none()));
}

#[test]
fn a_reused_root_the_tables_solved_is_searched_again() {
    let Some(tb) = tables() else { return };
    let mg = MoveGen::new();
    let mut tt = TranspositionTable::new();
    let config = TacticalMctsConfig {
        tablebase: Some(tb),
        leaf_mate_in_1: true,
        ..vanilla(60)
    };
    let board = Board::new_from_fen("8/8/8/8/8/2k5/8/R3K3 w - - 0 1");
    let first = tactical_mcts_search_for_training_with_reuse(
        board.clone(),
        &mg,
        config.clone(),
        None,
        &mut tt,
    );
    let played = first.best_move.unwrap();
    let reused = reuse_subtree(first.root_node, played).expect("child was visited");
    assert!(reused.borrow().terminal_or_mate_value.is_some());
    let second = tactical_mcts_search_for_training_with_reuse(
        board.apply_move_to_board(played),
        &mg,
        config,
        Some(reused),
        &mut tt,
    );
    assert!(second.best_move.is_some());
}
