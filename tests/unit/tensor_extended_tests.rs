//! Extended tests for tensor policy mapping (move_to_index)

use kingfisher::tensor::move_to_index;
use kingfisher::move_types::Move;
use kingfisher::piece_types::{QUEEN, ROOK, KNIGHT, BISHOP};

// === Slide Moves: All 8 Directions ===

#[test]
fn test_slide_north() {
    // e4 (28) -> e5 (36): N, dist 1. Direction=0, plane = 0*7+0 = 0
    let mv = Move::new(28, 36, None);
    assert_eq!(move_to_index(mv), 28 * 73 + 0);
}

#[test]
fn test_slide_north_distance_7() {
    // a1 (0) -> a8 (56): N, dist 7. Direction=0, plane = 0*7+6 = 6
    let mv = Move::new(0, 56, None);
    assert_eq!(move_to_index(mv), 0 * 73 + 6);
}

#[test]
fn test_slide_northeast() {
    // a1 (0) -> b2 (9): NE, dist 1. Direction=1, plane = 1*7+0 = 7
    let mv = Move::new(0, 9, None);
    assert_eq!(move_to_index(mv), 0 * 73 + 7);
}

#[test]
fn test_slide_east() {
    // a1 (0) -> b1 (1): E, dist 1. Direction=2, plane = 2*7+0 = 14
    let mv = Move::new(0, 1, None);
    assert_eq!(move_to_index(mv), 0 * 73 + 14);
}

#[test]
fn test_slide_southeast() {
    // a8 (56) -> b7 (49): SE, dist 1. Direction=3, plane = 3*7+0 = 21
    let mv = Move::new(56, 49, None);
    assert_eq!(move_to_index(mv), 56 * 73 + 21);
}

#[test]
fn test_slide_south() {
    // e5 (36) -> e4 (28): S, dist 1. Direction=4, plane = 4*7+0 = 28
    let mv = Move::new(36, 28, None);
    assert_eq!(move_to_index(mv), 36 * 73 + 28);
}

#[test]
fn test_slide_southwest() {
    // h8 (63) -> g7 (54): SW, dist 1. Direction=5, plane = 5*7+0 = 35
    let mv = Move::new(63, 54, None);
    assert_eq!(move_to_index(mv), 63 * 73 + 35);
}

#[test]
fn test_slide_west() {
    // h1 (7) -> g1 (6): W, dist 1. Direction=6, plane = 6*7+0 = 42
    let mv = Move::new(7, 6, None);
    assert_eq!(move_to_index(mv), 7 * 73 + 42);
}

#[test]
fn test_slide_northwest() {
    // h1 (7) -> g2 (14): NW, dist 1. Direction=7, plane = 7*7+0 = 49
    let mv = Move::new(7, 14, None);
    assert_eq!(move_to_index(mv), 7 * 73 + 49);
}

// === Knight Moves: All 8 L-shapes ===

#[test]
fn test_knight_all_directions() {
    // From d4 (27), test all 8 knight jumps
    let cases = vec![
        (27, 44, 0),  // (1, 2) -> idx 0
        (27, 37, 1),  // (2, 1) -> idx 1
        (27, 21, 2),  // (2, -1) -> idx 2
        (27, 12, 3),  // (1, -2) -> idx 3
        (27, 10, 4),  // (-1, -2) -> idx 4
        (27, 17, 5),  // (-2, -1) -> idx 5
        (27, 33, 6),  // (-2, 1) -> idx 6
        (27, 42, 7),  // (-1, 2) -> idx 7
    ];

    for (from, to, knight_idx) in cases {
        let mv = Move::new(from, to, None);
        let expected = from * 73 + 56 + knight_idx;
        assert_eq!(
            move_to_index(mv), expected,
            "Knight move from {} to {} should map to plane {}",
            from, to, 56 + knight_idx
        );
    }
}

// === Underpromotion: All 9 Combinations (3 directions × 3 pieces) ===

#[test]
fn test_underpromotion_knight_straight() {
    // a7 (48) -> a8 (56), promote to Knight. dx=0, plane = 64 + 0 + 0 = 64
    let mv = Move::new(48, 56, Some(KNIGHT));
    assert_eq!(move_to_index(mv), 48 * 73 + 64);
}

#[test]
fn test_underpromotion_knight_capture_left() {
    // b7 (49) -> a8 (56), promote to Knight. dx=-1, plane = 64 + 1 + 0 = 65
    let mv = Move::new(49, 56, Some(KNIGHT));
    assert_eq!(move_to_index(mv), 49 * 73 + 65);
}

#[test]
fn test_underpromotion_knight_capture_right() {
    // a7 (48) -> b8 (57), promote to Knight. dx=1, plane = 64 + 2 + 0 = 66
    let mv = Move::new(48, 57, Some(KNIGHT));
    assert_eq!(move_to_index(mv), 48 * 73 + 66);
}

#[test]
fn test_underpromotion_bishop_straight() {
    // e7 (52) -> e8 (60), promote to Bishop. dx=0, plane = 64 + 0 + 3 = 67
    let mv = Move::new(52, 60, Some(BISHOP));
    assert_eq!(move_to_index(mv), 52 * 73 + 67);
}

#[test]
fn test_underpromotion_bishop_capture_left() {
    // e7 (52) -> d8 (59), promote to Bishop. dx=-1, plane = 64 + 1 + 3 = 68
    let mv = Move::new(52, 59, Some(BISHOP));
    assert_eq!(move_to_index(mv), 52 * 73 + 68);
}

#[test]
fn test_underpromotion_bishop_capture_right() {
    // e7 (52) -> f8 (61), promote to Bishop. dx=1, plane = 64 + 2 + 3 = 69
    let mv = Move::new(52, 61, Some(BISHOP));
    assert_eq!(move_to_index(mv), 52 * 73 + 69);
}

#[test]
fn test_underpromotion_rook_straight() {
    // a7 (48) -> a8 (56), promote to Rook. dx=0, plane = 64 + 0 + 6 = 70
    let mv = Move::new(48, 56, Some(ROOK));
    assert_eq!(move_to_index(mv), 48 * 73 + 70);
}

#[test]
fn test_underpromotion_rook_capture_left() {
    // b7 (49) -> a8 (56), promote to Rook. dx=-1, plane = 64 + 1 + 6 = 71
    let mv = Move::new(49, 56, Some(ROOK));
    assert_eq!(move_to_index(mv), 49 * 73 + 71);
}

#[test]
fn test_underpromotion_rook_capture_right() {
    // a7 (48) -> b8 (57), promote to Rook. dx=1, plane = 64 + 2 + 6 = 72
    let mv = Move::new(48, 57, Some(ROOK));
    assert_eq!(move_to_index(mv), 48 * 73 + 72);
}

// === Queen Promotion (treated as slide) ===

#[test]
fn test_queen_promotion_straight() {
    // a7 (48) -> a8 (56), promote to Queen. Treated as N slide dist 1.
    // Direction=0, plane = 0*7+0 = 0
    let mv = Move::new(48, 56, Some(QUEEN));
    assert_eq!(move_to_index(mv), 48 * 73 + 0);
}

#[test]
fn test_queen_promotion_capture_right() {
    // a7 (48) -> b8 (57), promote to Queen. Treated as NE slide dist 1.
    // Direction=1, plane = 1*7+0 = 7
    let mv = Move::new(48, 57, Some(QUEEN));
    assert_eq!(move_to_index(mv), 48 * 73 + 7);
}

// === Boundary Tests ===

#[test]
fn test_index_within_policy_range() {
    // All valid indices should be in [0, 4672)
    // 64 squares × 73 planes = 4672

    // Test various moves
    let test_moves = vec![
        Move::new(0, 8, None),        // a1-a2 slide
        Move::new(63, 55, None),       // h8-h7 slide
        Move::new(28, 45, None),       // knight move
        Move::new(48, 56, Some(ROOK)), // underpromotion
        Move::new(55, 63, Some(QUEEN)), // queen promotion
    ];

    for mv in test_moves {
        let idx = move_to_index(mv);
        assert!(idx < 4672, "Index {} for move {}->{} exceeds 4672", idx, mv.from, mv.to);
    }
}

#[test]
fn test_slide_various_distances() {
    // Test distances 1-7 for north slide from a1
    for dist in 1..=7 {
        let mv = Move::new(0, dist * 8, None);
        let expected = 0 * 73 + (dist - 1); // Direction 0 (N), plane = 0*7 + (dist-1)
        assert_eq!(
            move_to_index(mv), expected,
            "N slide dist {} from a1 should be plane {}",
            dist, dist - 1
        );
    }
}
