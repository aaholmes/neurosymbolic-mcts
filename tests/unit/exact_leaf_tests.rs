//! Exact leaf values without the tiers: with tier 1 off (vanilla search), a leaf
//! whose side to move can mate in one, or whose position the Syzygy tables
//! cover, gets its exact value instead of a network evaluation. The root is
//! always searched, so a move is always returned.

use kingfisher::board::Board;
use kingfisher::egtb::{MateSearch, Tablebase};
use kingfisher::mcts::tactical_mcts::{
    reuse_subtree, tactical_mcts_search, tactical_mcts_search_for_training_with_reuse,
    TacticalMctsConfig,
};
use kingfisher::move_generation::MoveGen;
use kingfisher::search::mate_search::mate_search;
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
    assert_eq!(c.leaf_mate, MateSearch::OFF);
    assert!(c.tablebase.is_none());
}

#[test]
fn with_mate_in_one_on_a_root_with_a_mate_in_one_is_searched_and_mates() {
    let mg = MoveGen::new();
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");
    let config = TacticalMctsConfig {
        leaf_mate: MateSearch::exhaustive(1),
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
            leaf_mate: if on {
                MateSearch::exhaustive(1)
            } else {
                MateSearch::OFF
            },
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
        leaf_mate: MateSearch::exhaustive(1),
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

/// 1. Kb6 Kb8 2. Rh8# is a mate in two for White that starts with a quiet
/// move. With a mate-in-two search at leaves, positions where White has such a
/// mate get the exact value +1, and the search keeps a forced win.
#[test]
fn a_mate_in_two_search_at_leaves_gives_a_quiet_mating_line_its_exact_value() {
    let mg = MoveGen::new();
    let fen = "k7/8/2K5/8/8/8/8/7R w - - 0 1";
    let config = TacticalMctsConfig {
        leaf_mate: MateSearch::exhaustive(2),
        ..vanilla(200)
    };
    let (best, stats, root) = tactical_mcts_search(Board::new_from_fen(fen), &mg, config);
    assert!(stats.exact_leaf_values > 0);
    let kb6 = root
        .borrow()
        .children
        .iter()
        .find(|c| c.borrow().action.unwrap().to_uci() == "c6b6")
        .cloned()
        .unwrap();
    assert!(kb6.borrow().visits > 0);
    // The move chosen keeps a forced win: every reply leaves White a mate in
    // two. (A mate in three, such as Ra1+, counts.)
    let after = Board::new_from_fen(fen).apply_move_to_board(best.unwrap());
    let (caps, quiets) = mg.gen_pseudo_legal_moves(&after);
    let replies: Vec<_> = caps
        .iter()
        .chain(quiets.iter())
        .filter(|&&m| after.is_legal_after_move(m, &mg))
        .collect();
    assert!(
        !replies.is_empty(),
        "{} is not stalemate",
        best.unwrap().to_uci()
    );
    for &&r in &replies {
        let (score, _, _) = mate_search(&after.apply_move_to_board(r), &mg, 2, false, 2);
        assert!(
            score >= 1_000_000,
            "{} then {} is not mate in two",
            best.unwrap().to_uci(),
            r.to_uci()
        );
    }
}
