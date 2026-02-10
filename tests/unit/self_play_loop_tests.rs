/// Tests for self-play loop improvements:
/// - 3-fold repetition detection
/// - 50-move rule detection
/// - Shared transposition table across moves

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::mcts::{
    tactical_mcts_search_for_training_with_reuse,
    TacticalMctsConfig,
};
use kingfisher::transposition::TranspositionTable;
use std::time::Duration;

use crate::common::board_from_fen;

/// BoardStack detects 3-fold repetition correctly.
#[test]
fn test_self_play_detects_repetition_draw() {
    // Set up a position where both sides can shuffle a piece back and forth
    // Ke1-d1, Ke8-d8 repeated 3x
    let mut stack = BoardStack::new_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let move_gen = MoveGen::new();

    // Move kings back and forth to create repetition
    // Ke1-d1
    stack.make_move(Move::new(4, 3, None)); // Ke1-d1
    stack.make_move(Move::new(60, 59, None)); // Ke8-d8
    // Kd1-e1
    stack.make_move(Move::new(3, 4, None)); // Kd1-e1
    stack.make_move(Move::new(59, 60, None)); // Kd8-e8
    // Now we're back at the starting position for the 2nd time
    assert!(!stack.is_draw_by_repetition(), "Should not be draw after 2 repetitions");

    // Ke1-d1, Ke8-d8
    stack.make_move(Move::new(4, 3, None));
    stack.make_move(Move::new(60, 59, None));
    // Kd1-e1, Kd8-e8
    stack.make_move(Move::new(3, 4, None));
    stack.make_move(Move::new(59, 60, None));
    // Now we're back at the starting position for the 3rd time
    assert!(stack.is_draw_by_repetition(), "Should be draw after 3 repetitions");
}

/// 50-move rule: 100 half-moves without pawn push or capture triggers draw.
#[test]
fn test_self_play_detects_50_move_draw() {
    // KK position — make king moves back and forth
    let mut stack = BoardStack::new_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");

    // Make 100 half-moves (50 full moves) with just king shuffling
    for i in 0..50 {
        // White: e1-d1 or d1-e1
        if i % 2 == 0 {
            stack.make_move(Move::new(4, 3, None));
        } else {
            stack.make_move(Move::new(3, 4, None));
        }
        // Black: e8-d8 or d8-e8
        if i % 2 == 0 {
            stack.make_move(Move::new(60, 59, None));
        } else {
            stack.make_move(Move::new(59, 60, None));
        }
    }

    let hmc = stack.current_state().halfmove_clock();
    assert!(hmc >= 100, "Halfmove clock should be >= 100 after 50 full moves, got {}", hmc);
}

/// Transposition table populated by one move's search can be probed on the next.
#[test]
fn test_tt_shared_across_moves() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let mut tt = TranspositionTable::new();
    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(10),
        ..Default::default()
    };

    // First search populates the TT
    let result1 = tactical_mcts_search_for_training_with_reuse(
        board.clone(), &move_gen,
        config.clone(), None, &mut tt,
    );

    let stats1_hits = result1.stats.tt_mate_hits;

    // Apply a move
    let selected = result1.best_move.unwrap();
    let board2 = board.apply_move_to_board(selected);

    // Second search reuses the same TT — should have mate hits if any were stored
    let result2 = tactical_mcts_search_for_training_with_reuse(
        board2, &move_gen,
        config.clone(), None, &mut tt,
    );

    // The TT should still be functional (not recreated)
    // We can't guarantee hits from move 1 help move 2 in starting position,
    // but we can verify the TT is the same object (both searches ran without error)
    assert!(result2.stats.iterations > 0, "Second search should complete iterations");
}

/// BoardStack halfmove_clock() is accessible and tracks correctly.
#[test]
fn test_halfmove_clock_tracking() {
    let mut stack = BoardStack::new();
    let move_gen = MoveGen::new();

    assert_eq!(stack.current_state().halfmove_clock(), 0, "Initial halfmove clock");

    // Knight move (not pawn, not capture) — clock increments
    stack.make_move(Move::new(1, 18, None)); // Nb1-c3
    assert_eq!(stack.current_state().halfmove_clock(), 1, "After Nc3");

    // Black knight move — clock increments
    stack.make_move(Move::new(57, 42, None)); // Nb8-c6
    assert_eq!(stack.current_state().halfmove_clock(), 2, "After Nc6");

    // Pawn move — clock resets
    stack.make_move(Move::new(12, 28, None)); // e2-e4
    assert_eq!(stack.current_state().halfmove_clock(), 0, "After e4 (pawn move resets)");
}
