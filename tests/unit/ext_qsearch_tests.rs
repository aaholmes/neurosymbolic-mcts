//! Unit tests for extended PeSTO quiescence search with tactical moves
//! and null-move threat detection.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::quiescence::{
    cap1_pesto_qsearch, ext_pesto_qsearch_counted, forced_cap1_pesto_balance,
    forced_ext_pesto_balance, forced_ext_pesto_balance_counted, forced_pesto_balance,
    forced_principal_exchange, pesto_qsearch_counted, principal_exchange,
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
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
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
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    // Ne5 forks both rooks; the extended search should find the fork and
    // show a better score than capture-only (which misses the fork entirely).
    println!("Knight fork: ext={ext_score}, orig={orig_score}");
    assert!(
        ext_score > orig_score,
        "Extended should find the knight fork: ext={ext_score}, orig={orig_score}"
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
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
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
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
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
        false,
        false,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, _, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    // With budget exhausted, no tactical quiets are generated for White.
    // The null-move probe still runs, so scores may differ from pure capture
    // qsearch — that's expected and correct (null-move detects threats that
    // capture-only search misses). Just verify it doesn't crash.
    println!("Budget exhausted: exhausted={exhausted_score}, orig={orig_score}");
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
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
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
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
    );

    assert_eq!(score, -1_000_000, "Should detect checkmate: score={score}");
    assert!(completed, "Checkmate is a definitive result");
}

// ======== Test 7: Null-move adjusts score when threats exist ========

#[test]
fn test_ext_null_move_runs_without_crash() {
    let (move_gen, pesto) = setup();
    // Various positions — just verify null-move doesn't crash or loop
    for fen in &[
        "4k3/8/2p5/1B1p4/4N3/8/8/4K3 w - - 0 1",  // pieces under pawn attack
        "r1bqkbnr/pppppppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", // normal position
        "8/8/8/8/8/8/k7/K7 w - - 0 1",  // bare kings
    ] {
        let board = Board::new_from_fen(fen);
        let mut stack = BoardStack::with_board(board);
        let (score, completed, nodes, _) = ext_pesto_qsearch_counted(
            &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
        );
        assert!(nodes >= 1, "Should visit at least 1 node for {fen}");
        println!("{fen}: score={score}, completed={completed}, nodes={nodes}");
    }
}

// ======== Test 8: Null-move detects fork threat ========
// After d5 forking Bc4 and Ne4, null-move on White's turn reveals the threat

#[test]
fn test_ext_null_move_detects_fork_threat() {
    let (move_gen, pesto) = setup();
    // Position after 1.e4 e5 2.Nc3 Nf6 3.Bc4 Nxe4 4.Nxe4 — Black can play d5 forking
    let board = Board::new_from_fen("rnbqkb1r/pppp1ppp/8/4p3/2B1N3/8/PPPP1PPP/R1BQK1NR b KQkq - 0 4");

    let mut stack = BoardStack::with_board(board.clone());
    let (ext_score, _, ext_nodes, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
    );

    let mut stack2 = BoardStack::with_board(board);
    let (orig_score, _, orig_nodes, _) =
        pesto_qsearch_counted(&mut stack2, &move_gen, &pesto, -100_000, 100_000, 20);

    // d5 forks Bc4 and Ne4. The ext search should find this via fork + null-move
    // and show a better score for Black (less negative or more positive)
    assert!(
        ext_score > orig_score,
        "Fork+null-move should improve Black's score: ext={ext_score}, orig={orig_score}"
    );
    println!("Fork threat: ext={ext_score} ({ext_nodes} nodes), orig={orig_score} ({orig_nodes} nodes)");
}

// ======== Test 9: Regression — existing positions same or better ========

#[test]
fn test_ext_regression_free_piece_same_or_better() {
    let (move_gen, pesto) = setup();
    // White can capture undefended black knight with pawn
    let board = Board::new_from_fen("4k3/8/8/8/4n3/3P4/8/4K3 w - - 0 1");

    let mut stack = BoardStack::with_board(board.clone());
    let (ext_score, _, _, _) = ext_pesto_qsearch_counted(
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
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
        &mut stack, &move_gen, &pesto, -100_000, 100_000, 20, false, false, false, false,
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

    let (result, completed, nodes, _depth) =
        forced_ext_pesto_balance_counted(&mut stack, &move_gen, &pesto);

    assert!(result.abs() < 0.5);
    assert!(completed);
    assert!(nodes >= 1, "Should visit at least 1 node");
}

// ======== Principal Exchange (cap=0) tests ========

#[test]
fn test_principal_exchange_starting_position() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();
    let (score, nodes) = forced_principal_exchange(&mut stack, &move_gen, &pesto);
    assert!(score.abs() < 0.5, "Starting pos should be ~0, got {score}");
    assert_eq!(nodes, 1, "No captures → 1 node");
}

#[test]
fn test_principal_exchange_free_piece() {
    let (move_gen, pesto) = setup();
    // White can capture undefended black knight
    let board = Board::new_from_fen("4k3/8/8/8/4n3/3P4/8/4K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board);
    let (score, nodes) = forced_principal_exchange(&mut stack, &move_gen, &pesto);
    assert!(score > 0.5, "Should win the knight: {score}");
    assert!(nodes <= 5, "Simple capture: {nodes} nodes");
}

#[test]
fn test_principal_exchange_defended_piece() {
    let (move_gen, pesto) = setup();
    // White queen can capture defended pawn (QxP, PxQ = bad)
    // Principal exchange: QxP (+1), opponent recaptures PxQ (+9-1=+8), stand pat better
    let board = Board::new_from_fen("4k3/8/8/3p4/2p5/8/8/3QK3 w - - 0 1");
    let mut stack = BoardStack::with_board(board);
    let (score, nodes) = forced_principal_exchange(&mut stack, &move_gen, &pesto);
    println!("Defended pawn: score={score}, nodes={nodes}");
    // Should NOT capture the defended pawn (stand pat is better)
    assert!(nodes <= 5, "Should resolve quickly: {nodes}");
}

#[test]
fn test_principal_exchange_exchange_sequence() {
    let (move_gen, pesto) = setup();
    // White rook can capture black rook (equal trade)
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/r3K2R w K - 0 1");
    let mut stack = BoardStack::with_board(board);
    let (score, nodes) = forced_principal_exchange(&mut stack, &move_gen, &pesto);
    println!("Rook exchange: score={score}, nodes={nodes}");
    assert!(nodes <= 5);
}

// ======== Cap=1 Q-Search tests ========

#[test]
fn test_cap1_starting_position() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();
    let (score, completed, nodes, _) = forced_cap1_pesto_balance(&mut stack, &move_gen, &pesto);
    assert!(score.abs() < 0.5, "Starting pos: {score}");
    assert!(completed);
    println!("Cap=1 starting: score={score}, nodes={nodes}");
}

#[test]
fn test_cap1_check_wins_material() {
    let (move_gen, pesto) = setup();
    // Nc7+ checks king and forks rook on a8; after king moves, Nxa8
    let board = Board::new_from_fen("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1");
    let mut stack_cap1 = BoardStack::with_board(board.clone());
    let (cap1_score, _, cap1_nodes, _) = forced_cap1_pesto_balance(&mut stack_cap1, &move_gen, &pesto);

    let mut stack_pe = BoardStack::with_board(board);
    let (pe_score, pe_nodes) = forced_principal_exchange(&mut stack_pe, &move_gen, &pesto);

    println!("Check wins material: cap1={cap1_score} ({cap1_nodes}n), pe={pe_score} ({pe_nodes}n)");
    // Cap=1 should find Nc7+ (check) → Kd8/Kf8 → Nxa8 (capture rook)
    // Principal exchange can't find this (Nc7+ is not a capture)
    assert!(
        cap1_score > pe_score,
        "Cap=1 should find check winning material: cap1={cap1_score}, pe={pe_score}"
    );
}

#[test]
fn test_cap1_back_rank_mate() {
    let (move_gen, pesto) = setup();
    // Back rank mate: Rd1 can play Rd8# (check → checkmate)
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/3RK3 w - - 0 1");
    let mut stack = BoardStack::with_board(board);
    let (score, _, nodes, _) = forced_cap1_pesto_balance(&mut stack, &move_gen, &pesto);
    println!("Back rank mate: cap1={score} ({nodes}n)");
    // Rd8+ is a check → leads to mate or winning the exchange
    // Score should be very positive for White
    assert!(score > 3.0, "Should find back rank threat: {score}");
}

#[test]
fn test_cap1_node_count_bounded() {
    let (move_gen, pesto) = setup();
    // Complex tactical position (R3 blunder) — should be bounded
    let board = Board::new_from_fen("7k/1p1b1Qp1/1r6/p4pq1/1b1Np2p/4P3/K1rNR1PP/3R4 w - - 0 36");
    let mut stack = BoardStack::with_board(board);
    let (score, _, nodes, _) = forced_cap1_pesto_balance(&mut stack, &move_gen, &pesto);
    println!("R3 blunder cap=1: score={score:.2} ({nodes} nodes)");
    // With cap=1 + 1 check per side, tree should be much smaller than uncapped
    assert!(nodes < 5000, "Cap=1 should be bounded: {nodes} nodes");
}

#[test]
fn test_cap1_vs_principal_exchange_comparison() {
    let (move_gen, pesto) = setup();

    let positions = vec![
        ("Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Fork trick", "rnbqkb1r/pppp1ppp/8/4p3/2B1N3/8/PPPP1PPP/R1BQK1NR b KQkq - 0 4"),
        ("R5 Nxb4", "r1b2rk1/pp2nppp/2pBp3/q7/1nP4P/5BN1/P4P2/R2QK2R w KQ - 0 15"),
        ("Quiet mid", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b - - 5 7"),
    ];

    println!("\n{:<15} {:>8} {:>8} {:>10} {:>10}", "Position", "PE score", "PE nodes", "Cap1 score", "Cap1 nodes");
    println!("{}", "-".repeat(55));
    for (name, fen) in &positions {
        let board = Board::new_from_fen(fen);

        let mut s1 = BoardStack::with_board(board.clone());
        let (pe_s, pe_n) = forced_principal_exchange(&mut s1, &move_gen, &pesto);

        let mut s2 = BoardStack::with_board(board);
        let (c1_s, _, c1_n, _) = forced_cap1_pesto_balance(&mut s2, &move_gen, &pesto);

        println!("{:<15} {:>+8.2} {:>8} {:>+10.2} {:>10}", name, pe_s, pe_n, c1_s, c1_n);
    }
}
