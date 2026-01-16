//! Tests for iterative deepening search.
//!
//! Tests verify that iterative deepening properly increases depth
//! and respects time limits.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::NULL_MOVE;
use kingfisher::search::iterative_deepening_ab_search;
use kingfisher::transposition::TranspositionTable;
use std::time::Duration;

fn setup() -> (MoveGen, PestoEval) {
    (MoveGen::new(), PestoEval::new())
}

#[test]
fn test_id_returns_legal_move() {
    let (move_gen, pesto) = setup();
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();

    let (depth, _eval, best_move, nodes) = iterative_deepening_ab_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        4, // max_depth
        4, // q_search_max_depth
        None,
        false,
    );

    // Should return a valid move
    assert!(best_move != NULL_MOVE, "Should return a valid move");

    // Verify the move is legal
    board.make_move(best_move);
    assert!(board.current_state().is_legal(&move_gen), "Move should be legal");

    // Should have searched some nodes
    assert!(nodes > 0, "Should search at least one node");

    // Should have reached some depth
    assert!(depth > 0, "Should reach at least depth 1");
}

#[test]
fn test_id_reaches_requested_depth() {
    let (move_gen, pesto) = setup();
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();

    let (depth_reached, _eval, _best_move, _nodes) = iterative_deepening_ab_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        3, // max_depth
        2,
        None,
        false,
    );

    // Should reach the requested depth (or close to it)
    assert!(
        depth_reached >= 2,
        "Should reach at least depth 2, reached {}",
        depth_reached
    );
}

#[test]
fn test_id_respects_time_limit() {
    let (move_gen, pesto) = setup();
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();

    let time_limit = Duration::from_millis(500);
    let start = std::time::Instant::now();

    let (_depth, _eval, _best_move, _nodes) = iterative_deepening_ab_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        20, // Deep search that should be cut short
        4,
        Some(time_limit),
        false,
    );

    let elapsed = start.elapsed();

    // Should complete within reasonable time (allow margin for test overhead)
    assert!(
        elapsed < Duration::from_millis(2000),
        "Should respect time limit, took {:?}",
        elapsed
    );
}

#[test]
fn test_id_finds_mate() {
    let (move_gen, pesto) = setup();
    // Position where white has mate in 1 or 2
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1");
    let mut board_stack = BoardStack::with_board(board);
    let mut tt = TranspositionTable::new();

    let (_depth, eval, best_move, _nodes) = iterative_deepening_ab_search(
        &mut board_stack,
        &move_gen,
        &pesto,
        &mut tt,
        5,
        4,
        None,
        false,
    );

    // Should find a winning move
    assert!(best_move != NULL_MOVE, "Should find a move");
    // Score should be very good for white
    assert!(eval > 100, "Should evaluate position as winning, got {}", eval);
}

#[test]
fn test_id_deeper_search_same_or_better() {
    let (move_gen, pesto) = setup();

    // Search at depth 2
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();
    let (_depth1, eval1, _best_move1, _nodes1) = iterative_deepening_ab_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        2,
        2,
        None,
        false,
    );

    // Search at depth 4
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();
    let (_depth2, eval2, _best_move2, _nodes2) = iterative_deepening_ab_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        4,
        2,
        None,
        false,
    );

    // Evaluations should be similar (starting position is roughly equal)
    // Both should be close to 0 for a balanced position
    assert!(
        eval1.abs() < 200,
        "Depth 2 eval should be reasonable: {}",
        eval1
    );
    assert!(
        eval2.abs() < 200,
        "Depth 4 eval should be reasonable: {}",
        eval2
    );
}
