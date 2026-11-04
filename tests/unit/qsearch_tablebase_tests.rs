//! Syzygy tables inside the quiescence search: a line of captures that reaches
//! a position the tables cover gets its exact result there. These need the
//! 5-piece WDL files, in KF_SYZYGY_PATH or /data/syzygy; without them each test
//! is skipped.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::egtb::Tablebase;
use kingfisher::eval::PestoEval;
use kingfisher::mcts::tactical_mcts::{qsearch_inputs, QSearchVariant};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::search::quiescence::{
    forced_ext_pesto_balance_tables, forced_ext_pesto_balance_with, ExtConfig, TABLEBASE_WIN_CP,
};

use crate::common::board_from_fen;

fn tables() -> Option<Tablebase> {
    let dir = std::env::var("KF_SYZYGY_PATH").unwrap_or_else(|_| "/data/syzygy".into());
    if !std::path::Path::new(&dir).join("KRvK.rtbw").exists() {
        eprintln!("skipped: no Syzygy files in {dir}");
        return None;
    }
    let tb = Tablebase::open(&dir).expect("tablebase opens");
    // The tests below assume the 5-piece limit: 6-piece positions must not be covered.
    if tb.max_pieces() != 5 {
        eprintln!("skipped: these positions assume 5-piece tables");
        return None;
    }
    Some(tb)
}

fn score(fen: &str, tb: Option<&Tablebase>) -> f32 {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let mut stack = BoardStack::with_board(board_from_fen(fen));
    forced_ext_pesto_balance_tables(&mut stack, &mg, &pesto, &ExtConfig::default(), tb).0
}

/// White can take a rook with the bishop, leaving king, dark-squared bishop and
/// a-pawn versus king and pawn, with Black's king in the a8 corner: a draw.
const BISHOP_TAKES_ROOK: &str = "k7/7p/8/P1r5/8/8/5B2/4K3 w - - 0 1";
const AFTER_CAPTURE_MOVE: (usize, usize) = (13, 34); // f2 takes c5

#[test]
fn the_capture_leads_to_a_position_the_tables_call_a_draw() {
    let Some(tb) = tables() else { return };
    let board = Board::new_from_fen(BISHOP_TAKES_ROOK);
    assert_eq!(board.get_all_occupancy().count_ones(), 6);
    assert_eq!(
        tb.probe_value(&board),
        None,
        "the root itself is not covered"
    );
    let after =
        board.apply_move_to_board(Move::new(AFTER_CAPTURE_MOVE.0, AFTER_CAPTURE_MOVE.1, None));
    assert_eq!(tb.probe_value(&after), Some(0.0));
}

#[test]
fn without_tables_the_capture_looks_winning() {
    assert!(score(BISHOP_TAKES_ROOK, None) > 2.0);
}

#[test]
fn with_tables_the_capture_is_scored_as_the_draw_it_is() {
    let Some(tb) = tables() else { return };
    // Standing pat is worse than a draw (a rook versus bishop and pawn), so the
    // best line is the capture, worth exactly a draw.
    let s = score(BISHOP_TAKES_ROOK, Some(&tb));
    assert!(s.abs() < 1e-6, "{s}");
}

#[test]
fn a_position_the_tables_cover_gets_their_result() {
    let Some(tb) = tables() else { return };
    let win = TABLEBASE_WIN_CP as f32 / 100.0;
    assert_eq!(score("8/8/8/4k3/8/8/8/R3K3 w - - 0 1", Some(&tb)), win); // KRvK, White to move
    assert_eq!(score("8/8/8/4k3/8/8/8/R3K3 b - - 0 1", Some(&tb)), -win);
}

#[test]
fn a_table_win_is_worth_almost_a_full_point() {
    assert!((0.326_f64 * TABLEBASE_WIN_CP as f64 / 100.0).tanh() > 0.99);
}

#[test]
fn positions_far_from_the_limit_are_unchanged() {
    let Some(tb) = tables() else { return };
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let leaves = include_str!("../fixtures/qsearch_leaves.fen");
    let mut n = 0;
    for fen in leaves.lines().filter(|l| !l.trim().is_empty()) {
        let board = board_from_fen(fen);
        if board.get_all_occupancy().count_ones() < 16 {
            continue;
        }
        let with = forced_ext_pesto_balance_tables(
            &mut BoardStack::with_board(board.clone()),
            &mg,
            &pesto,
            &ExtConfig::default(),
            Some(&tb),
        );
        let without = forced_ext_pesto_balance_with(
            &mut BoardStack::with_board(board),
            &mg,
            &pesto,
            &ExtConfig::default(),
        );
        assert_eq!(with, without, "{fen}");
        n += 1;
    }
    assert!(n > 50);
}

#[test]
fn the_search_inputs_use_the_tables() {
    let Some(tb) = tables() else { return };
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let board = Board::new_from_fen(BISHOP_TAKES_ROOK);
    let with = qsearch_inputs(
        &board,
        &mg,
        &pesto,
        QSearchVariant::Extended,
        true,
        Some(&tb),
    );
    let without = qsearch_inputs(&board, &mg, &pesto, QSearchVariant::Extended, true, None);
    assert!(with.score.abs() < 1e-6, "{}", with.score);
    assert!(without.score > 2.0);
}
