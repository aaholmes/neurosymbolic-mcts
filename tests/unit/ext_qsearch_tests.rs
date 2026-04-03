//! Unit tests for extended PeSTO quiescence search with tactical moves
//! (non-capture checks, pawn forks, knight forks, forked piece retreats)

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::quiescence::{
    ext_pesto_qsearch_counted, forced_ext_pesto_balance, forced_ext_pesto_balance_counted,
    forced_pesto_balance, pesto_qsearch_counted,
};

fn setup() -> (MoveGen, PestoEval) {
    (MoveGen::new(), PestoEval::new())
}

// ======== Test 1: Quiet position matches pure capture qsearch ========

#[test]
fn test_ext_quiet_position_matches_capture_qsearch() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();
    let (ext_score, ext_completed, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, 0,
    );

    let mut stack2 = BoardStack::new();
    let (orig_score, orig_completed, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    assert_eq!(
        ext_score, orig_score,
        "Quiet position: ext={ext_score} should match orig={orig_score}"
    );
    assert_eq!(ext_completed, orig_completed);
}

// ======== Test 2: Knight fork position ========
// White Nc4 can play Ne5 forking Rc6 and Rg6

#[test]
fn test_ext_knight_fork_improves_score() {
    let (move_gen, pesto) = setup();
    let board = Board::new_from_fen("7k/8/2r3r1/8/2N5/8/8/4K3 w - - 0 1");

    let mut stack = BoardStack::with_board(board.clone());
    let (ext_score, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, 0,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    // Ne5 forks both rooks; after one retreats, Nxother wins a rook
    // The improvement should be positive (fork finds material gain)
    assert!(
        ext_score > orig_score,
        "Knight fork should improve score: ext={ext_score}, orig={orig_score}"
    );
}

// ======== Test 3: Pawn fork position ========
// White pawn d4 can advance to d5 forking Nc6 and Ne6

#[test]
fn test_ext_pawn_fork_improves_score() {
    let (move_gen, pesto) = setup();
    // Two black knights on c6 and e6, white pawn on d4 can fork them with d5
    let board = Board::new_from_fen("4k3/8/2n1n3/8/3P4/8/8/4K3 w - - 0 1");

    let mut stack = BoardStack::with_board(board.clone());
    let (ext_score, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, 0,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    // d5 forks both knights; after one retreats, dxother wins a knight (~300 cp)
    assert!(
        ext_score > orig_score,
        "Pawn fork should improve score: ext={ext_score}, orig={orig_score}"
    );
}

// ======== Test 4: Non-capture check winning material ========
// Nc7+ checks king and forks rook on a8; after king moves, Nxa8

#[test]
fn test_ext_check_wins_material() {
    let (move_gen, pesto) = setup();
    let board = Board::new_from_fen("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1");

    let mut stack = BoardStack::with_board(board.clone());
    let (ext_score, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, 0,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    // Nc7+ forces king move, then Nxa8 wins rook
    assert!(
        ext_score > orig_score,
        "Check winning material should improve score: ext={ext_score}, orig={orig_score}"
    );
}

// ======== Test 5: Budget exhaustion ========
// With white_tactic_used=true, only captures should be generated for white

#[test]
fn test_ext_budget_exhaustion_no_extra_tactics() {
    let (move_gen, pesto) = setup();
    // Same knight fork position, but with white's budget already used
    let board = Board::new_from_fen("7k/8/2r3r1/8/2N5/8/8/4K3 w - - 0 1");

    let mut stack = BoardStack::with_board(board.clone());
    let (exhausted_score, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack,
        &move_gen,
        &pesto,
        -100_000,
        100_000,
        20,
        true,  // white budget used
        false, // black budget available
        0,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    // With budget exhausted, should match pure capture qsearch (no fork found)
    assert_eq!(
        exhausted_score, orig_score,
        "Exhausted budget should match pure capture qsearch: exhausted={exhausted_score}, orig={orig_score}"
    );
}

// ======== Test 6: In-check evasion ========
// When in check, generates all legal evasions (not just captures), no stand-pat

#[test]
fn test_ext_in_check_generates_evasions() {
    let (move_gen, pesto) = setup();
    // White king on e1 in check from black rook on a1 (far away, can't capture)
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/r3K3 w - - 0 1");
    assert!(board.is_check(&move_gen), "White should be in check");

    let mut stack = BoardStack::with_board(board);
    let (score, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, 0,
    );

    // In check with only a king vs king+rook, score should be very negative
    // but NOT -1000000 (checkmate) since evasions exist (Kd2, Ke2, Kf2)
    assert!(
        score > -1_000_000,
        "Should find evasions, not report checkmate: score={score}"
    );
    assert!(score < 0, "Down a rook, score should be negative: {score}");
}

#[test]
fn test_ext_in_check_detects_checkmate() {
    let (move_gen, pesto) = setup();
    // Back-rank mate: white king on h1 boxed in by own pawns, black rook checks
    let board = Board::new_from_fen("6k1/8/8/8/8/8/5PPP/r6K w - - 0 1");
    assert!(board.is_check(&move_gen), "White should be in check");

    let mut stack = BoardStack::with_board(board);
    let (score, completed, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, 0,
    );

    assert_eq!(score, -1_000_000, "Should detect checkmate: score={score}");
    assert!(completed, "Checkmate is a definitive result");
}

// ======== Test 7: Forked piece retreat ========
// Position where forked_pieces is set, those pieces should be able to retreat

#[test]
fn test_ext_forked_piece_retreat_generated() {
    let (move_gen, pesto) = setup();
    // After Ne5 forking Rc6 and Rg6, it's black to move
    // Black should generate retreats for the forked rooks
    let board = Board::new_from_fen("7k/8/2r3r1/4N3/8/8/8/4K3 b - - 0 1");

    // c6 = sq 42, g6 = sq 46
    let forked_pieces: u64 = (1u64 << 42) | (1u64 << 46);

    let mut stack = BoardStack::with_board(board.clone());
    let (with_retreat, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack,
        &move_gen,
        &pesto,
        -100_000,
        100_000,
        20,
        false,
        false,
        forked_pieces,
    );

    // Without forked_pieces, only captures are tried
    let mut stack2 = BoardStack::with_board(board);
    let (without_retreat, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack2,
        &move_gen,
        &pesto,
        -100_000,
        100_000,
        20,
        false,
        false,
        0, // no forked pieces
    );

    // With retreat, black should be able to save at least one rook
    // Score from black's perspective should be better (less negative or more positive)
    assert!(
        with_retreat >= without_retreat,
        "Forked piece retreat should help: with={with_retreat}, without={without_retreat}"
    );
}

// ======== Test 8: Regression — existing positions same or better ========

#[test]
fn test_ext_regression_free_piece_same_or_better() {
    let (move_gen, pesto) = setup();
    // White can capture undefended black knight with pawn
    let board = Board::new_from_fen("4k3/8/8/8/4n3/3P4/8/4K3 w - - 0 1");

    let mut stack = BoardStack::with_board(board.clone());
    let (ext_score, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, 0,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    assert!(
        ext_score >= orig_score,
        "Extended should be >= original on capture position: ext={ext_score}, orig={orig_score}"
    );
}

#[test]
fn test_ext_regression_queen_advantage() {
    let (move_gen, pesto) = setup();
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/4K2Q w - - 0 1");

    let mut stack = BoardStack::with_board(board.clone());
    let (ext_score, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, 0,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    assert!(
        ext_score >= orig_score,
        "Extended should be >= original: ext={ext_score}, orig={orig_score}"
    );
}

// ======== forced_ext_pesto_balance wrapper tests ========

#[test]
fn test_forced_ext_pesto_balance_starting_position() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    let (result, completed) = forced_ext_pesto_balance(&mut stack, &move_gen, &pesto);

    assert!(
        result.abs() < 0.5,
        "Starting position should be near 0, got {result}"
    );
    assert!(completed, "Starting position should complete");
}

#[test]
fn test_forced_ext_pesto_balance_counted_returns_stats() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    let (result, completed, nodes, depth) =
        forced_ext_pesto_balance_counted(&mut stack, &move_gen, &pesto);

    assert!(result.abs() < 0.5);
    assert!(completed);
    assert!(nodes >= 1, "Should visit at least 1 node");
    assert_eq!(depth, 0, "Quiet position should have depth 0");
}
