//! Tests for transposition table.
//!
//! Tests verify that the transposition table correctly stores and retrieves
//! position evaluations, handles depth-based replacement, and manages mate results.

use kingfisher::board::Board;
use kingfisher::move_types::Move;
use kingfisher::transposition::{Bound, TranspositionTable};

fn create_move(from: usize, to: usize) -> Move {
    Move {
        from,
        to,
        promotion: None,
    }
}

#[test]
fn test_tt_new_is_empty() {
    let tt = TranspositionTable::new();
    let (size, _) = tt.stats();
    assert_eq!(size, 0, "New TT should be empty");
}

#[test]
fn test_tt_store_and_probe() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    tt.store(&board, 5, 100, mv);

    // Probe with same or lower depth should succeed
    let result = tt.probe(&board, 5);
    assert!(result.is_some(), "Should find entry at same depth");
}

#[test]
fn test_tt_probe_lower_depth() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    tt.store(&board, 5, 100, mv);

    // Probe with lower depth should succeed
    let result = tt.probe(&board, 3);
    assert!(
        result.is_some(),
        "Should find entry with lower depth request"
    );
}

#[test]
fn test_tt_probe_higher_depth_fails() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    tt.store(&board, 3, 100, mv);

    // Probe with higher depth should fail
    let result = tt.probe(&board, 5);
    assert!(
        result.is_none(),
        "Should not find entry with higher depth request"
    );
}

#[test]
fn test_tt_depth_replacement() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv1 = create_move(12, 28);
    let mv2 = create_move(6, 21);

    // Store at depth 3
    tt.store(&board, 3, 50, mv1);

    // Store at depth 5 (higher) - should replace
    tt.store(&board, 5, 100, mv2);

    // Should be able to probe at depth 5
    let result = tt.probe(&board, 5);
    assert!(result.is_some(), "Higher depth should replace");
}

#[test]
fn test_tt_no_replacement_lower_depth() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv1 = create_move(12, 28);
    let mv2 = create_move(6, 21);

    // Store at depth 5
    tt.store(&board, 5, 100, mv1);

    // Store at depth 3 (lower) - should NOT replace
    tt.store(&board, 3, 50, mv2);

    // Should still be able to probe at depth 5
    let result = tt.probe(&board, 5);
    assert!(
        result.is_some(),
        "Lower depth should not replace higher depth entry"
    );
}

#[test]
fn test_tt_different_positions() {
    let mut tt = TranspositionTable::new();
    let board1 = Board::new();
    let board2 = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");

    let mv1 = create_move(12, 28);
    let mv2 = create_move(52, 36);

    tt.store(&board1, 3, 0, mv1);
    tt.store(&board2, 3, -20, mv2);

    let result1 = tt.probe(&board1, 3);
    let result2 = tt.probe(&board2, 3);

    assert!(
        result1.is_some() && result2.is_some(),
        "Both positions should be found"
    );
}

#[test]
fn test_tt_store_mate_result() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    // Store mate result
    tt.store_mate_result(&board, 3, mv, 5); // Mate in 3, searched at depth 5

    // Probe mate result
    let result = tt.probe_mate(&board, 5);
    assert!(result.is_some(), "Should find mate result");

    let (mate_depth, _mate_move) = result.unwrap();
    assert_eq!(mate_depth, 3, "Mate depth should be 3");
}

#[test]
fn test_tt_probe_mate_depth_check() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    // Store mate result at depth 3
    tt.store_mate_result(&board, 2, mv, 3);

    // Probe with higher depth should fail
    let result = tt.probe_mate(&board, 5);
    assert!(
        result.is_none(),
        "Should not return mate result for higher depth"
    );

    // Probe with same or lower depth should succeed
    let result = tt.probe_mate(&board, 3);
    assert!(result.is_some(), "Should return mate result for same depth");
}

#[test]
fn test_tt_clear() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    tt.store(&board, 5, 100, mv);

    let (size_before, _) = tt.stats();
    assert!(size_before > 0, "TT should have entries before clear");

    tt.clear();

    let (size_after, _) = tt.stats();
    assert_eq!(size_after, 0, "TT should be empty after clear");
}

#[test]
fn test_tt_stats() {
    let mut tt = TranspositionTable::new();
    let board1 = Board::new();
    let board2 = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let mv = create_move(12, 28);

    tt.store(&board1, 3, 0, mv);
    tt.store(&board2, 3, 0, mv);
    tt.store_mate_result(&board1, 2, mv, 4);

    let (size, entries_with_mate) = tt.stats();
    // Two different positions should create at least 1 entry each
    assert!(size >= 1, "Should have at least 1 entry");
    assert_eq!(entries_with_mate, 1, "Should have 1 entry with mate result");
}

// --- Bound-aware probe semantics ---

#[test]
fn test_probe_value_exact_always_usable() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    tt.store_with_bound(&board, 5, 42, Bound::Exact, mv, None);

    // Exact score is usable for any window at sufficient depth.
    assert_eq!(tt.probe_value(&board, 5, -1000, 1000), Some(42));
    assert_eq!(tt.probe_value(&board, 5, 100, 200), Some(42));
    assert_eq!(tt.probe_value(&board, 5, -200, -100), Some(42));
    // Insufficient depth -> not usable.
    assert_eq!(tt.probe_value(&board, 6, -1000, 1000), None);
}

#[test]
fn test_probe_value_lower_bound_only_on_cutoff() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    // Stored as a fail-high: true value >= 100.
    tt.store_with_bound(&board, 5, 100, Bound::Lower, mv, None);

    // Usable only when it still causes a beta cutoff (score >= beta).
    assert_eq!(tt.probe_value(&board, 5, 0, 100), Some(100));
    assert_eq!(tt.probe_value(&board, 5, 0, 50), Some(100));
    // Window beta above the bound: cannot conclude the exact value.
    assert_eq!(tt.probe_value(&board, 5, 0, 200), None);
}

#[test]
fn test_probe_value_upper_bound_only_on_fail_low() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    // Stored as a fail-low: true value <= 30.
    tt.store_with_bound(&board, 5, 30, Bound::Upper, mv, None);

    // Usable only when it still fails low (score <= alpha).
    assert_eq!(tt.probe_value(&board, 5, 30, 100), Some(30));
    assert_eq!(tt.probe_value(&board, 5, 50, 100), Some(30));
    // Alpha below the bound: cannot conclude the exact value.
    assert_eq!(tt.probe_value(&board, 5, 0, 100), None);
}

#[test]
fn test_probe_value_respects_depth() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    tt.store_with_bound(&board, 3, 77, Bound::Exact, mv, None);
    assert_eq!(tt.probe_value(&board, 3, -1000, 1000), Some(77));
    assert_eq!(tt.probe_value(&board, 2, -1000, 1000), Some(77));
    assert_eq!(tt.probe_value(&board, 4, -1000, 1000), None);
}

#[test]
fn test_legacy_store_is_exact() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    tt.store(&board, 4, 12, mv);
    let entry = tt.probe_entry(&board).unwrap();
    assert_eq!(entry.bound(), Bound::Exact);
    assert_eq!(tt.probe_value(&board, 4, 100, 200), Some(12));
}

#[test]
fn test_tt_store_with_mate_preserves_mate() {
    let mut tt = TranspositionTable::new();
    let board = Board::new();
    let mv = create_move(12, 28);

    // Store regular entry
    tt.store(&board, 3, 50, mv);

    // Store mate result for same position
    tt.store_mate_result(&board, 2, mv, 5);

    // Probe should return mate result
    let mate_result = tt.probe_mate(&board, 5);
    assert!(mate_result.is_some(), "Mate result should be preserved");
}
