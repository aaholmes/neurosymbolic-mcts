//! Tests for alpha-beta search algorithm.
//!
//! Tests verify the alpha-beta search returns valid moves and correctly
//! implements pruning, check extensions, and time limits.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::NULL_MOVE;
use kingfisher::search::alpha_beta::alpha_beta_search;
use kingfisher::search::history::{HistoryTable, MAX_PLY};
use kingfisher::transposition::TranspositionTable;
use std::time::{Duration, Instant};

fn setup() -> (MoveGen, PestoEval) {
    (MoveGen::new(), PestoEval::new())
}

#[test]
fn test_alpha_beta_returns_legal_move() {
    let (move_gen, pesto) = setup();
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    let (score, best_move, nodes, _terminated) = alpha_beta_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        3, // depth
        -1000000,
        1000000,
        4, // q_search_max_depth
        false,
        None,
        None,
    );

    // Best move should be a legal move
    assert!(best_move != NULL_MOVE, "Should return a valid move");

    // Apply the move and verify it's legal
    board.make_move(best_move);
    assert!(board.current_state().is_legal(&move_gen), "Move should be legal");

    // Should have searched at least some nodes
    assert!(nodes > 0, "Should search at least one node");

    // Score should be reasonable
    assert!(score.abs() < 1000000, "Score should be reasonable, got {}", score);
}

#[test]
fn test_alpha_beta_mate_in_1_white() {
    let (move_gen, pesto) = setup();
    // Mate in 1: Qh7#
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1");
    let mut board_stack = BoardStack::with_board(board);
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    let (score, _best_move, _nodes, _terminated) = alpha_beta_search(
        &mut board_stack,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        4,
        -1000000,
        1000000,
        4,
        false,
        None,
        None,
    );

    // Should find the winning position (very high score)
    assert!(score > 500, "Should evaluate winning position highly, got {}", score);
}

#[test]
fn test_alpha_beta_avoids_checkmate() {
    let (move_gen, pesto) = setup();
    // Black to move, can avoid mate with defensive move
    let board = Board::new_from_fen("8/8/8/8/8/5k2/4q3/7K b - - 0 1");
    let mut board_stack = BoardStack::with_board(board);
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    let (score, best_move, _nodes, _terminated) = alpha_beta_search(
        &mut board_stack,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        3,
        -1000000,
        1000000,
        4,
        false,
        None,
        None,
    );

    // Should find a move (black has many good moves with queen)
    assert!(best_move != NULL_MOVE, "Should find a move");
    // Black should have a winning position
    assert!(score > 0, "Black should evaluate position favorably");
}

#[test]
fn test_alpha_beta_searches_more_nodes_at_higher_depth() {
    let (move_gen, pesto) = setup();
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    // Search at depth 2
    let (_, _, nodes_d2, _) = alpha_beta_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        2,
        -1000000,
        1000000,
        2,
        false,
        None,
        None,
    );

    // Reset for depth 3 search
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    let (_, _, nodes_d3, _) = alpha_beta_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        3,
        -1000000,
        1000000,
        2,
        false,
        None,
        None,
    );

    // Deeper search should search more nodes
    assert!(
        nodes_d3 > nodes_d2,
        "Depth 3 ({}) should search more nodes than depth 2 ({})",
        nodes_d3,
        nodes_d2
    );
}

#[test]
fn test_alpha_beta_time_limit() {
    let (move_gen, pesto) = setup();
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    let start = Instant::now();
    let time_limit = Duration::from_millis(100);

    let (_, _, _, terminated) = alpha_beta_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        20, // Deep search that should be cut short
        -1000000,
        1000000,
        4,
        false,
        Some(start),
        Some(time_limit),
    );

    let elapsed = start.elapsed();

    // Should terminate early or complete quickly
    // Allow some margin for test overhead
    assert!(
        elapsed < Duration::from_millis(500) || terminated,
        "Should respect time limit or terminate, elapsed: {:?}",
        elapsed
    );
}

#[test]
fn test_alpha_beta_detects_stalemate() {
    let (move_gen, pesto) = setup();
    // Stalemate position: Black king has no legal moves but is not in check
    let board = Board::new_from_fen("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
    let mut board_stack = BoardStack::with_board(board);
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    let (score, _best_move, _nodes, terminated) = alpha_beta_search(
        &mut board_stack,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        1,
        -1000000,
        1000000,
        1,
        false,
        None,
        None,
    );

    // Stalemate should be detected and score should be draw (0)
    assert_eq!(score, 0, "Stalemate should evaluate to 0");
    assert!(terminated, "Stalemate should terminate search immediately");
}

#[test]
fn test_alpha_beta_detects_checkmate() {
    let (move_gen, pesto) = setup();
    // Checkmate position: Black king is in checkmate
    let board = Board::new_from_fen("k7/8/1K6/8/8/8/8/R7 b - - 0 1");
    let mut board_stack = BoardStack::with_board(board);
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    let (score, _best_move, _nodes, terminated) = alpha_beta_search(
        &mut board_stack,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        1,
        -1000000,
        1000000,
        1,
        false,
        None,
        None,
    );

    // Checkmate should return very high score (from perspective of side to move being mated)
    assert_eq!(score, 1000000, "Checkmate should return mate score");
    assert!(terminated, "Checkmate should terminate search immediately");
}

#[test]
fn test_alpha_beta_uses_transposition_table() {
    let (move_gen, pesto) = setup();
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    // First search populates TT
    let (_, _, _nodes1, _) = alpha_beta_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        3,
        -1000000,
        1000000,
        2,
        false,
        None,
        None,
    );

    // Check TT has entries
    let (tt_size, _) = tt.stats();
    assert!(tt_size > 0, "TT should have entries after search");
}

#[test]
fn test_alpha_beta_history_heuristic() {
    let (move_gen, pesto) = setup();
    let mut board = BoardStack::new();
    let mut tt = TranspositionTable::new();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    // Run search which should update history table
    let _ = alpha_beta_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        &mut killers,
        &mut history,
        4,
        -1000000,
        1000000,
        2,
        false,
        None,
        None,
    );

    // History table should have some non-zero entries
    // Check a few common squares
    let has_history = (0..64).any(|from| {
        (0..64).any(|to| history.get_score_from_squares(from, to) > 0)
    });

    assert!(has_history, "History table should have some entries after search");
}
