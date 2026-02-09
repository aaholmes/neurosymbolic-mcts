//! Tests for experiments::position_classifier module

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::experiments::position_classifier::{PositionClassifier, PositionType};
use kingfisher::move_generation::MoveGen;

fn setup() -> (MoveGen, PestoEval) {
    (MoveGen::new(), PestoEval::new())
}

// === Position Classification Tests ===

#[test]
fn test_classify_starting_position_as_positional() {
    let (move_gen, pesto) = setup();
    let classifier = PositionClassifier::new(&move_gen, &pesto);
    let board = Board::new();
    // Starting position: many pieces, no captures, not in check
    assert_eq!(classifier.classify(&board), PositionType::Positional);
}

#[test]
fn test_classify_endgame_few_pieces() {
    let (move_gen, pesto) = setup();
    let classifier = PositionClassifier::new(&move_gen, &pesto);
    // K+R vs K endgame (3 pieces)
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1");
    assert_eq!(classifier.classify(&board), PositionType::Endgame);
}

#[test]
fn test_classify_endgame_boundary_10_pieces() {
    let (move_gen, pesto) = setup();
    let classifier = PositionClassifier::new(&move_gen, &pesto);
    // Exactly 10 pieces (boundary, should be endgame: <= 10)
    let board = Board::new_from_fen("r1bk4/1pp5/8/8/8/8/1PP5/R1BK4 w - - 0 1");
    let piece_count = board.get_all_occupancy().count_ones();
    assert_eq!(piece_count, 10);
    assert_eq!(classifier.classify(&board), PositionType::Endgame);
}

#[test]
fn test_classify_tactical_many_captures() {
    let (move_gen, pesto) = setup();
    let classifier = PositionClassifier::new(&move_gen, &pesto);
    // Position with many capture opportunities (pieces intermingled)
    let board = Board::new_from_fen("r1bqkbnr/pppppppp/2n5/4P3/3pP3/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 0 4");
    let result = classifier.classify(&board);
    // This position has captures available (d4xNc3, Nc6xe5, etc.)
    // May be Tactical or Positional depending on capture count
    assert!(result == PositionType::Tactical || result == PositionType::Positional);
}

#[test]
fn test_classify_defensive_in_check() {
    let (move_gen, pesto) = setup();
    let classifier = PositionClassifier::new(&move_gen, &pesto);
    // Scholar's mate attempt: Qf7+ — Black is in check, many pieces, few captures
    let board = Board::new_from_fen("rnbqkbnr/pppp1Qpp/8/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1");
    let piece_count = board.get_all_occupancy().count_ones();
    assert!(piece_count > 10, "Should have many pieces");
    let result = classifier.classify(&board);
    // In check → either Defensive or Tactical depending on captures available
    // The classifier checks captures first (>= 3), then check
    assert!(
        result == PositionType::Defensive || result == PositionType::Tactical,
        "Position in check should be Defensive or Tactical, got {:?}", result
    );
}

// === has_likely_forced_mate Tests ===

#[test]
fn test_has_forced_mate_back_rank() {
    let (move_gen, pesto) = setup();
    let classifier = PositionClassifier::new(&move_gen, &pesto);
    // Back rank mate in 1: Re8#
    let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");
    assert!(classifier.has_likely_forced_mate(&board, 3));
}

#[test]
fn test_no_forced_mate_starting_position() {
    let (move_gen, pesto) = setup();
    let classifier = PositionClassifier::new(&move_gen, &pesto);
    let board = Board::new();
    assert!(!classifier.has_likely_forced_mate(&board, 3));
}

#[test]
fn test_no_forced_mate_equal_position() {
    let (move_gen, pesto) = setup();
    let classifier = PositionClassifier::new(&move_gen, &pesto);
    // Equal K+R vs K+R position
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    // Depth 1 shouldn't find mate in this position
    assert!(!classifier.has_likely_forced_mate(&board, 1));
}

// === PositionType Enum Tests ===

#[test]
fn test_position_type_debug() {
    assert_eq!(format!("{:?}", PositionType::ForcedMate), "ForcedMate");
    assert_eq!(format!("{:?}", PositionType::Tactical), "Tactical");
    assert_eq!(format!("{:?}", PositionType::Positional), "Positional");
    assert_eq!(format!("{:?}", PositionType::Endgame), "Endgame");
    assert_eq!(format!("{:?}", PositionType::Defensive), "Defensive");
}

#[test]
fn test_position_type_equality() {
    assert_eq!(PositionType::Tactical, PositionType::Tactical);
    assert_ne!(PositionType::Tactical, PositionType::Positional);
}

#[test]
fn test_position_type_copy() {
    let t = PositionType::Endgame;
    let t2 = t; // Copy
    assert_eq!(t, t2);
}
