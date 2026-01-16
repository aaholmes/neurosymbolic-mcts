//! Tests for history heuristic table.
//!
//! Tests verify that the history table correctly tracks move scores
//! and provides proper move ordering hints.

use kingfisher::move_types::Move;
use kingfisher::search::history::HistoryTable;

fn create_move(from: usize, to: usize) -> Move {
    Move {
        from,
        to,
        promotion: None,
    }
}

#[test]
fn test_history_table_new() {
    let history = HistoryTable::new();

    // All scores should be initialized to 0
    for from in 0..64 {
        for to in 0..64 {
            assert_eq!(
                history.get_score_from_squares(from, to),
                0,
                "All history scores should start at 0"
            );
        }
    }
}

#[test]
fn test_history_table_update() {
    let mut history = HistoryTable::new();
    let mv = create_move(12, 28); // e2-e4

    // Update with depth 3 (bonus = 9)
    history.update(&mv, 3);

    assert_eq!(
        history.get_score(&mv),
        9,
        "Score should be depth^2 = 9"
    );
}

#[test]
fn test_history_table_accumulates() {
    let mut history = HistoryTable::new();
    let mv = create_move(12, 28); // e2-e4

    // Multiple updates accumulate
    history.update(&mv, 2); // +4
    history.update(&mv, 3); // +9
    history.update(&mv, 1); // +1

    assert_eq!(
        history.get_score(&mv),
        14,
        "Scores should accumulate: 4 + 9 + 1 = 14"
    );
}

#[test]
fn test_history_table_different_moves() {
    let mut history = HistoryTable::new();
    let mv1 = create_move(12, 28); // e2-e4
    let mv2 = create_move(6, 21);  // g1-f3

    history.update(&mv1, 3); // +9
    history.update(&mv2, 2); // +4

    assert_eq!(history.get_score(&mv1), 9, "mv1 should have score 9");
    assert_eq!(history.get_score(&mv2), 4, "mv2 should have score 4");
}

#[test]
fn test_history_table_clear() {
    let mut history = HistoryTable::new();
    let mv = create_move(12, 28);

    history.update(&mv, 5);
    assert!(history.get_score(&mv) > 0, "Score should be positive before clear");

    history.clear();

    assert_eq!(
        history.get_score(&mv),
        0,
        "Score should be 0 after clear"
    );
}

#[test]
fn test_history_table_high_depth() {
    let mut history = HistoryTable::new();
    let mv = create_move(0, 63);

    // High depth gives large bonus
    history.update(&mv, 10); // +100

    assert_eq!(
        history.get_score(&mv),
        100,
        "Depth 10 should give score 100"
    );
}

#[test]
fn test_history_table_saturation() {
    let mut history = HistoryTable::new();
    let mv = create_move(0, 0);

    // Many updates with large depth
    for _ in 0..1000 {
        history.update(&mv, 100); // +10000 each
    }

    // Should not overflow (saturating add)
    let score = history.get_score(&mv);
    assert!(
        score > 0,
        "Score should be positive (not overflowed negative)"
    );
}

#[test]
fn test_history_table_get_score_from_squares() {
    let mut history = HistoryTable::new();
    let mv = create_move(4, 36);

    history.update(&mv, 4); // +16

    // Both methods should return same result
    assert_eq!(
        history.get_score(&mv),
        history.get_score_from_squares(4, 36),
        "Both getter methods should return same score"
    );
}
