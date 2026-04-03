//! Unit tests for PeSTO-based quiescence search

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::quiescence::{forced_pesto_balance, pesto_qsearch};

/// Helper to create test components
fn setup() -> (MoveGen, PestoEval) {
    (MoveGen::new(), PestoEval::new())
}

// ======== pst_eval_cp Tests ========

#[test]
fn test_pst_only_eval_starting_position_near_zero() {
    let pesto = PestoEval::new();
    let board = Board::new();
    let score = pesto.pst_eval_cp(&board);
    // Symmetric starting position should be near 0
    assert!(
        score.abs() < 50,
        "Starting position PST eval should be near 0, got {score}"
    );
}

#[test]
fn test_pst_eval_cp_starting_position() {
    let pesto = PestoEval::new();
    let board = Board::new();
    let score = pesto.pst_eval_cp(&board);
    // Tapered result near 0 cp for starting position
    assert!(
        score.abs() < 50,
        "Starting position should evaluate near 0 cp, got {score}"
    );
}

#[test]
fn test_pst_eval_cp_queen_advantage() {
    let pesto = PestoEval::new();
    // White has an extra queen
    let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let base = pesto.pst_eval_cp(&board);

    // Remove black queen
    let board_up_q =
        Board::new_from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let up_q = pesto.pst_eval_cp(&board_up_q);

    // White up a queen should be ~900+ cp better
    let advantage = up_q - base;
    assert!(
        advantage > 800,
        "Queen advantage should be ~900+ cp, got {advantage}"
    );
}

#[test]
fn test_pst_eval_cp_stm_perspective() {
    let pesto = PestoEval::new();
    // Same material but different side to move
    let board_w = Board::new_from_fen("4k3/8/8/8/8/8/8/4K2Q w - - 0 1");
    let board_b = Board::new_from_fen("4k3/8/8/8/8/8/8/4K2Q b - - 0 1");

    let score_w = pesto.pst_eval_cp(&board_w);
    let score_b = pesto.pst_eval_cp(&board_b);

    // White has a queen, so from white's perspective it's positive,
    // from black's perspective it's negative
    assert!(score_w > 0, "White with queen should be positive: {score_w}");
    assert!(
        score_b < 0,
        "Black facing queen should be negative: {score_b}"
    );
    // They should be negations of each other (approximately, since game phase may differ slightly)
    assert!(
        (score_w + score_b).abs() < 10,
        "Scores should be approximately negated: w={score_w}, b={score_b}"
    );
}

// ======== forced_pesto_balance Tests ========

#[test]
fn test_forced_pesto_balance_starting_position() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    let (result, completed) = forced_pesto_balance(&mut stack, &move_gen, &pesto);

    // Starting position: no captures, should be near 0 pawn units
    assert!(
        result.abs() < 0.5,
        "Starting position PeSTO balance should be near 0, got {result}"
    );
    assert!(
        completed,
        "Starting position Q-search should complete (no captures)"
    );
}

#[test]
fn test_forced_pesto_balance_free_piece() {
    let (move_gen, pesto) = setup();
    // White can capture undefended black knight with pawn
    let board = Board::new_from_fen("4k3/8/8/8/4n3/3P4/8/4K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board.clone());

    let stand_pat = pesto.pst_eval_cp(&board) as f32 / 100.0;
    let (result, completed) = forced_pesto_balance(&mut stack, &move_gen, &pesto);

    // After Pxe4 capturing the knight, balance should improve
    assert!(
        result > stand_pat,
        "Should find winning capture: result={result}, stand_pat={stand_pat}"
    );
    assert!(completed, "Single capture resolves: should complete");
}

#[test]
fn test_forced_pesto_balance_returns_float_pawn_units() {
    let (move_gen, pesto) = setup();
    // White up a queen (queen vs nothing)
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/4K2Q w - - 0 1");
    let mut stack = BoardStack::with_board(board);

    let (result, _completed) = forced_pesto_balance(&mut stack, &move_gen, &pesto);

    // Should be in pawn units (~9-11), not centipawns (900+)
    assert!(
        result > 5.0 && result < 15.0,
        "Queen advantage should be ~9-11 pawn units, got {result}"
    );
}

// ======== pesto_qsearch Tests ========

#[test]
fn test_pesto_qsearch_quiet_position() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    let (score, completed) = pesto_qsearch(&mut stack, &move_gen, &pesto, -100_000, 100_000, 8);

    // Quiet starting position should return stand-pat
    assert!(
        score.abs() < 100,
        "Starting position should have reasonable eval: {score}"
    );
    assert!(completed, "No captures available, should complete");
}

#[test]
fn test_pesto_qsearch_captures_improve_score() {
    let (move_gen, pesto) = setup();
    // White pawn can capture undefended black knight
    let board = Board::new_from_fen("4k3/8/8/8/4n3/3P4/8/4K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board.clone());

    let stand_pat = pesto.pst_eval_cp(&board);
    let (score, _) = pesto_qsearch(&mut stack, &move_gen, &pesto, -100_000, 100_000, 8);

    assert!(
        score >= stand_pat,
        "Q-search score should be >= stand-pat: score={score}, stand_pat={stand_pat}"
    );
}

// ======== Classical Fallback Calibration Tests ========

#[test]
fn test_classical_fallback_texel_calibration() {
    // Verify tanh(0.326 * q) gives expected win probabilities
    let k = 0.326f64;

    // q=0: draw → tanh(0) = 0.0
    let v0 = (k * 0.0).tanh();
    assert!(
        v0.abs() < 0.001,
        "q=0 should give ~0.0 value, got {v0}"
    );

    // q=1 (1 pawn up): slight advantage
    let v1 = (k * 1.0).tanh();
    assert!(
        v1 > 0.2 && v1 < 0.5,
        "q=1 should give moderate advantage, got {v1}"
    );

    // q=3 (minor piece up): significant advantage
    let v3 = (k * 3.0).tanh();
    assert!(
        v3 > 0.6 && v3 < 0.9,
        "q=3 should give significant advantage, got {v3}"
    );

    // q=5 (rook up): large advantage
    let v5 = (k * 5.0).tanh();
    assert!(
        v5 > 0.85 && v5 < 0.99,
        "q=5 should give large advantage, got {v5}"
    );

    // q=9 (queen up): near-winning
    let v9 = (k * 9.0).tanh();
    assert!(
        v9 > 0.95,
        "q=9 should give near-winning value, got {v9}"
    );
}
