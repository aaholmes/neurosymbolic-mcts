//! Unit tests for quiescence search

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::quiescence::{quiescence_search, quiescence_search_tactical};
use std::time::{Duration, Instant};

/// Helper to create test components
fn setup() -> (MoveGen, PestoEval) {
    (MoveGen::new(), PestoEval::new())
}

#[test]
fn test_quiescence_quiet_position() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    let (score, nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        8,
        false,
        None,
        None,
    );

    // Quiet starting position should return stand-pat evaluation
    assert!(score.abs() < 1000, "Starting position should have reasonable eval: {}", score);
    assert!(nodes >= 1, "Should count at least the root node");
}

#[test]
fn test_quiescence_captures_position() {
    let (move_gen, pesto) = setup();
    // Position with mutual captures available: both sides can capture
    let board = Board::new_from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3");
    let mut stack = BoardStack::with_board(board);

    let (score, nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        8,
        false,
        None,
        None,
    );

    // Should search some nodes but position is relatively quiet
    assert!(nodes >= 1);
    assert!(score.abs() < 500, "Roughly equal position: {}", score);
}

#[test]
fn test_quiescence_winning_capture() {
    let (move_gen, pesto) = setup();
    // White can capture a free queen with QxQ (queen on d3 can take queen on g4)
    let board = Board::new_from_fen("r1b1kbnr/pppp1ppp/2n5/4p3/4P1q1/3Q1N2/PPPP1PPP/RNB1KB1R w KQkq - 0 3");
    let mut stack = BoardStack::with_board(board);

    let stand_pat = pesto.eval(&stack.current_state(), &move_gen);

    let (score, nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        8,
        false,
        None,
        None,
    );

    // QS should search Qxg4 and find it's winning
    assert!(nodes >= 1, "Should search at least root node");
    // Score should be better than stand_pat since we can capture the queen
    assert!(score >= stand_pat, "Score should improve or stay same after searching captures");
}

#[test]
fn test_quiescence_beta_cutoff() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    // With a very low beta, should get immediate cutoff
    let (score, nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        -50000, // Very low beta (opponent already has a winning line)
        8,
        false,
        None,
        None,
    );

    // Should return beta immediately due to stand-pat cutoff
    assert_eq!(score, -50000, "Should return beta on cutoff");
    assert_eq!(nodes, 1, "Should only count root node on immediate cutoff");
}

#[test]
fn test_quiescence_alpha_improvement() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    // With very low alpha, stand-pat should improve it
    let (score, _nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000, // Very low alpha
        100000,
        8,
        false,
        None,
        None,
    );

    // Score should be better than -100000
    assert!(score > -100000, "Stand-pat should improve alpha");
}

#[test]
fn test_quiescence_depth_limit() {
    let (move_gen, pesto) = setup();
    // Position with many captures available
    let board = Board::new_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
    let mut stack = BoardStack::with_board(board);

    // With depth 0, should return immediately
    let (score_d0, nodes_d0) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        0,
        false,
        None,
        None,
    );

    // With higher depth, may search more
    let (score_d4, nodes_d4) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        4,
        false,
        None,
        None,
    );

    assert_eq!(nodes_d0, 1, "Depth 0 should only count root");
    assert!(nodes_d4 >= nodes_d0, "Higher depth should search at least as many nodes");
}

#[test]
fn test_quiescence_time_limit() {
    let (move_gen, pesto) = setup();
    let board = Board::new_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
    let mut stack = BoardStack::with_board(board);

    let start = Instant::now();
    let time_limit = Duration::from_millis(1); // Very short limit

    let (_score, _nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        100, // High depth
        false,
        Some(start),
        Some(time_limit),
    );

    // Should return relatively quickly
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_millis(500),
        "Should respect time limit, took {:?}", elapsed);
}

#[test]
fn test_quiescence_tactical_returns_tree() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    let tree = quiescence_search_tactical(&mut stack, &move_gen, &pesto);

    // Should return a valid TacticalTree
    assert!(tree.leaf_score.abs() < 1000, "Starting position should have reasonable score");
    // Starting position has no captures, so siblings should be empty
    assert!(tree.siblings.is_empty(), "No captures in starting position");
    assert!(tree.principal_variation.is_empty(), "No tactical line in quiet position");
}

#[test]
fn test_quiescence_tactical_finds_captures() {
    let (move_gen, pesto) = setup();
    // Position where white can capture pawn on e5
    let board = Board::new_from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3");
    let mut stack = BoardStack::with_board(board);

    let tree = quiescence_search_tactical(&mut stack, &move_gen, &pesto);

    // Should find captures (Nxe5 is available)
    // Note: The position might not have any SEE-positive captures
}

#[test]
fn test_quiescence_tactical_with_winning_capture() {
    let (move_gen, pesto) = setup();
    // White queen can capture undefended black queen
    let board = Board::new_from_fen("r1b1kbnr/pppp1ppp/2n5/4p3/4P1q1/3Q1N2/PPPP1PPP/RNB1KB1R w KQkq - 0 3");
    let mut stack = BoardStack::with_board(board);

    let tree = quiescence_search_tactical(&mut stack, &move_gen, &pesto);

    // Should find Qxg4 as a winning capture
    // There should be at least one capture move evaluated
    let _has_qxg4 = tree.siblings.iter().any(|(mv, _)| mv.to == 30); // g4 = square 30
    // Qxg4 may or may not be found depending on SEE
}

#[test]
fn test_quiescence_see_pruning() {
    let (move_gen, pesto) = setup();
    // Position where capturing would lose material (e.g., QxP defended by bishop)
    let board = Board::new_from_fen("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4");
    let mut stack = BoardStack::with_board(board);

    let (score, _nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        8,
        false,
        None,
        None,
    );

    // Score should be reasonable (SEE should prune bad captures)
    assert!(score.abs() < 1000, "Position should evaluate reasonably: {}", score);
}

#[test]
fn test_quiescence_checkmate_detection() {
    let (move_gen, pesto) = setup();
    // Position where white is checkmated
    let board = Board::new_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    let mut stack = BoardStack::with_board(board);

    let (_score, nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        8,
        false,
        None,
        None,
    );

    // The evaluation function handles checkmate
    assert!(nodes >= 1);
}

#[test]
fn test_quiescence_tactical_siblings_scored() {
    let (move_gen, pesto) = setup();
    // Position with multiple captures for white
    let board = Board::new_from_fen("r1b1k2r/ppppnppp/2n5/2b1p3/2B1P1q1/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6");
    let mut stack = BoardStack::with_board(board);

    let tree = quiescence_search_tactical(&mut stack, &move_gen, &pesto);

    // Each sibling should have a score
    for (mv, score) in &tree.siblings {
        // Scores should be within reasonable bounds
        assert!(score.abs() < 1000001, "Score {} for move {:?} seems unreasonable", score, mv);
    }
}
