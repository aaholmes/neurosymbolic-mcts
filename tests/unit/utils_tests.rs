//! Tests for utility functions (print_move, print_bits)

use kingfisher::utils::{print_move, print_bits};
use kingfisher::move_types::Move;
use kingfisher::piece_types::{QUEEN, ROOK, KNIGHT, BISHOP};

#[test]
fn test_print_move_standard() {
    let mv = Move::new(12, 28, None); // e2e4
    assert_eq!(print_move(&mv), "e2e4");
}

#[test]
fn test_print_move_a1_to_a2() {
    let mv = Move::new(0, 8, None); // a1a2
    assert_eq!(print_move(&mv), "a1a2");
}

#[test]
fn test_print_move_h1_to_h8() {
    let mv = Move::new(7, 63, None); // h1h8
    assert_eq!(print_move(&mv), "h1h8");
}

#[test]
fn test_print_move_promotion_queen() {
    let mv = Move::new(48, 56, Some(QUEEN)); // a7a8=Q
    assert_eq!(print_move(&mv), "a7a8=Q");
}

#[test]
fn test_print_move_promotion_rook() {
    let mv = Move::new(48, 56, Some(ROOK)); // a7a8=R
    assert_eq!(print_move(&mv), "a7a8=R");
}

#[test]
fn test_print_move_promotion_knight() {
    let mv = Move::new(48, 56, Some(KNIGHT)); // a7a8=N
    assert_eq!(print_move(&mv), "a7a8=N");
}

#[test]
fn test_print_move_promotion_bishop() {
    let mv = Move::new(48, 56, Some(BISHOP)); // a7a8=B
    assert_eq!(print_move(&mv), "a7a8=B");
}

#[test]
fn test_print_move_castling_kingside() {
    let mv = Move::new(4, 6, None); // e1g1
    assert_eq!(print_move(&mv), "e1g1");
}

#[test]
fn test_print_move_castling_queenside() {
    let mv = Move::new(4, 2, None); // e1c1
    assert_eq!(print_move(&mv), "e1c1");
}

#[test]
fn test_print_move_capture_notation() {
    // Capture is just from-to in UCI
    let mv = Move::new(28, 35, None); // e4d5
    assert_eq!(print_move(&mv), "e4d5");
}

#[test]
fn test_print_move_promotion_capture() {
    let mv = Move::new(48, 57, Some(QUEEN)); // a7b8=Q
    assert_eq!(print_move(&mv), "a7b8=Q");
}

#[test]
fn test_print_bits_does_not_panic() {
    // Just verify it doesn't panic for various inputs
    print_bits(0);
    print_bits(u64::MAX);
    print_bits(1);
    print_bits(1u64 << 63);
    // Starting position pawns (rank 2 + rank 7)
    print_bits(0x00FF00000000FF00);
}
