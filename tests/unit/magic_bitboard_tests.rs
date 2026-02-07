//! Tests for magic bitboard move initialization functions

use kingfisher::magic_bitboard::{
    init_king_moves, init_knight_moves, init_pawn_moves,
    init_pawn_captures_promotions, bishop_attacks, rook_attacks,
    append_promotions,
};
use kingfisher::move_types::Move;
use kingfisher::piece_types::{QUEEN, ROOK, KNIGHT, BISHOP};

// === King Move Tests ===

#[test]
fn test_king_moves_center() {
    // e4 (sq 28) - king in the center should have 8 moves
    let moves = init_king_moves(28);
    assert_eq!(moves.len(), 8, "King on e4 should have 8 moves");
    // Check all expected squares
    let expected: Vec<usize> = vec![27, 35, 36, 37, 29, 21, 20, 19];
    for sq in &expected {
        assert!(moves.contains(sq), "King on e4 should reach sq {}", sq);
    }
}

#[test]
fn test_king_moves_corner_a1() {
    // a1 (sq 0) - corner king has 3 moves
    let moves = init_king_moves(0);
    assert_eq!(moves.len(), 3, "King on a1 should have 3 moves");
    assert!(moves.contains(&1));  // b1
    assert!(moves.contains(&8));  // a2
    assert!(moves.contains(&9));  // b2
}

#[test]
fn test_king_moves_corner_h8() {
    // h8 (sq 63) - corner king has 3 moves
    let moves = init_king_moves(63);
    assert_eq!(moves.len(), 3, "King on h8 should have 3 moves");
    assert!(moves.contains(&62)); // g8
    assert!(moves.contains(&55)); // h7
    assert!(moves.contains(&54)); // g7
}

#[test]
fn test_king_moves_corner_a8() {
    let moves = init_king_moves(56);
    assert_eq!(moves.len(), 3, "King on a8 should have 3 moves");
    assert!(moves.contains(&57)); // b8
    assert!(moves.contains(&48)); // a7
    assert!(moves.contains(&49)); // b7
}

#[test]
fn test_king_moves_corner_h1() {
    let moves = init_king_moves(7);
    assert_eq!(moves.len(), 3, "King on h1 should have 3 moves");
    assert!(moves.contains(&6));  // g1
    assert!(moves.contains(&14)); // g2
    assert!(moves.contains(&15)); // h2
}

#[test]
fn test_king_moves_edge_a4() {
    // a4 (sq 24) - a-file edge has 5 moves
    let moves = init_king_moves(24);
    assert_eq!(moves.len(), 5, "King on a4 should have 5 moves");
}

#[test]
fn test_king_moves_edge_h4() {
    // h4 (sq 31) - h-file edge has 5 moves
    let moves = init_king_moves(31);
    assert_eq!(moves.len(), 5, "King on h4 should have 5 moves");
}

// === Knight Move Tests ===

#[test]
fn test_knight_moves_center() {
    // e4 (sq 28) - knight in center has 8 moves
    let moves = init_knight_moves(28);
    assert_eq!(moves.len(), 8, "Knight on e4 should have 8 moves");
}

#[test]
fn test_knight_moves_corner_a1() {
    // a1 (sq 0) - corner knight has 2 moves
    let moves = init_knight_moves(0);
    assert_eq!(moves.len(), 2, "Knight on a1 should have 2 moves");
    assert!(moves.contains(&10)); // c2
    assert!(moves.contains(&17)); // b3
}

#[test]
fn test_knight_moves_corner_h1() {
    let moves = init_knight_moves(7);
    assert_eq!(moves.len(), 2, "Knight on h1 should have 2 moves");
    assert!(moves.contains(&13)); // f2
    assert!(moves.contains(&22)); // g3
}

#[test]
fn test_knight_moves_b1() {
    // b1 (sq 1) - standard starting position for knight
    let moves = init_knight_moves(1);
    assert_eq!(moves.len(), 3, "Knight on b1 should have 3 moves");
    assert!(moves.contains(&16)); // a3
    assert!(moves.contains(&18)); // c3
    assert!(moves.contains(&11)); // d2
}

#[test]
fn test_knight_moves_edge() {
    // a4 (sq 24) - a-file edge
    let moves = init_knight_moves(24);
    assert_eq!(moves.len(), 4, "Knight on a4 should have 4 moves");
}

// === Pawn Move Tests ===

#[test]
fn test_pawn_moves_rank2_white() {
    // e2 (sq 12) - white pawn on rank 2 has single and double push
    let (white, _black) = init_pawn_moves(12);
    assert_eq!(white.len(), 2, "White pawn on e2 should have 2 moves (single + double push)");
    assert!(white.contains(&20)); // e3
    assert!(white.contains(&28)); // e4 (double push)
}

#[test]
fn test_pawn_moves_rank3_white() {
    // e3 (sq 20) - white pawn on rank 3 has only single push
    let (white, _black) = init_pawn_moves(20);
    assert_eq!(white.len(), 1, "White pawn on e3 should have 1 move");
    assert!(white.contains(&28)); // e4
}

#[test]
fn test_pawn_moves_rank7_black() {
    // e7 (sq 52) - black pawn on rank 7 has single and double push
    let (_white, black) = init_pawn_moves(52);
    assert_eq!(black.len(), 2, "Black pawn on e7 should have 2 moves");
    assert!(black.contains(&44)); // e6
    assert!(black.contains(&36)); // e5 (double push)
}

#[test]
fn test_pawn_moves_rank1_empty() {
    // sq 0 (a1) - rank 1 has no pawn moves
    let (white, black) = init_pawn_moves(0);
    assert_eq!(white.len(), 0, "No pawn moves from rank 1");
    assert_eq!(black.len(), 0, "No pawn moves from rank 1");
}

#[test]
fn test_pawn_moves_rank8_empty() {
    // sq 56 (a8) - rank 8 has no pawn moves
    let (white, black) = init_pawn_moves(56);
    assert_eq!(white.len(), 0, "No pawn moves from rank 8");
    assert_eq!(black.len(), 0, "No pawn moves from rank 8");
}

// === Pawn Capture Tests ===

#[test]
fn test_pawn_captures_center() {
    // e4 (sq 28) - center pawn has two captures for each color
    let (w_cap, _w_prom, b_cap, _b_prom) = init_pawn_captures_promotions(28);
    assert_eq!(w_cap.len(), 2, "White pawn on e4 should have 2 captures");
    assert!(w_cap.contains(&35)); // d5
    assert!(w_cap.contains(&37)); // f5
    assert_eq!(b_cap.len(), 2, "Black pawn on e4 should have 2 captures");
    assert!(b_cap.contains(&19)); // d3
    assert!(b_cap.contains(&21)); // f3
}

#[test]
fn test_pawn_captures_a_file() {
    // a4 (sq 24) - a-file pawn only captures right
    let (w_cap, _, b_cap, _) = init_pawn_captures_promotions(24);
    assert_eq!(w_cap.len(), 1, "White pawn on a-file has 1 capture");
    assert!(w_cap.contains(&33)); // b5
    assert_eq!(b_cap.len(), 1, "Black pawn on a-file has 1 capture");
}

#[test]
fn test_pawn_captures_h_file() {
    // h4 (sq 31) - h-file pawn only captures left
    let (w_cap, _, b_cap, _) = init_pawn_captures_promotions(31);
    assert_eq!(w_cap.len(), 1, "White pawn on h-file has 1 capture");
    assert!(w_cap.contains(&38)); // g5
    assert_eq!(b_cap.len(), 1, "Black pawn on h-file has 1 capture");
}

#[test]
fn test_pawn_promotion_rank7() {
    // e7 (sq 52) - rank 7 pawn has white promotion
    let (_, w_prom, _, _) = init_pawn_captures_promotions(52);
    assert_eq!(w_prom.len(), 1, "White pawn on rank 7 has 1 promotion push");
    assert!(w_prom.contains(&60)); // e8
}

#[test]
fn test_pawn_promotion_rank2() {
    // e2 (sq 12) - rank 2 pawn has black promotion
    let (_, _, _, b_prom) = init_pawn_captures_promotions(12);
    assert_eq!(b_prom.len(), 1, "Black pawn on rank 2 has 1 promotion push");
    assert!(b_prom.contains(&4)); // e1
}

// === Rook Attack Tests ===

#[test]
fn test_rook_attacks_empty_board_center() {
    // Rook on e4 (28), no blockers
    let (captures, moves) = rook_attacks(28, 0);
    assert_eq!(captures.len(), 0, "No captures on empty board");
    assert_eq!(moves.len(), 14, "Rook on e4 should reach 14 squares on empty board");
}

#[test]
fn test_rook_attacks_corner_empty() {
    // Rook on a1 (0), no blockers
    let (captures, moves) = rook_attacks(0, 0);
    assert_eq!(captures.len(), 0);
    assert_eq!(moves.len(), 14, "Rook on a1 should reach 14 squares on empty board");
}

#[test]
fn test_rook_attacks_with_blocker() {
    // Rook on a1 (0), blocker on a4 (24)
    let blockers = 1u64 << 24;
    let (captures, moves) = rook_attacks(0, blockers);
    // North: a2(8), a3(16) are moves, a4(24) is capture
    // East: b1(1)..h1(7) are moves
    assert!(captures.contains(&24), "Should capture blocker on a4");
    assert!(moves.contains(&8));  // a2
    assert!(moves.contains(&16)); // a3
    assert!(!moves.contains(&32), "Should not pass through blocker");
}

#[test]
fn test_rook_attacks_surrounded() {
    // Rook on e4 (28), blockers on all 4 adjacent squares
    let blockers = (1u64 << 27) | (1u64 << 29) | (1u64 << 20) | (1u64 << 36);
    let (captures, moves) = rook_attacks(28, blockers);
    assert_eq!(captures.len(), 4, "Should capture all 4 adjacent blockers");
    assert_eq!(moves.len(), 0, "No moves when completely surrounded");
}

// === Bishop Attack Tests ===

#[test]
fn test_bishop_attacks_empty_board_center() {
    // Bishop on e4 (28), no blockers
    let (captures, moves) = bishop_attacks(28, 0);
    assert_eq!(captures.len(), 0);
    assert_eq!(moves.len(), 13, "Bishop on e4 should reach 13 squares on empty board");
}

#[test]
fn test_bishop_attacks_corner() {
    // Bishop on a1 (0), no blockers
    let (captures, moves) = bishop_attacks(0, 0);
    assert_eq!(captures.len(), 0);
    assert_eq!(moves.len(), 7, "Bishop on a1 should reach 7 squares (a1-h8 diagonal)");
}

#[test]
fn test_bishop_attacks_with_blocker() {
    // Bishop on a1 (0), blocker on c3 (18)
    let blockers = 1u64 << 18;
    let (captures, moves) = bishop_attacks(0, blockers);
    assert!(captures.contains(&18), "Should capture blocker on c3");
    assert!(moves.contains(&9));  // b2
    assert!(!moves.contains(&27), "Should not pass through blocker");
}

// === Append Promotions Tests ===

#[test]
fn test_append_promotions_white() {
    let mut promotions = Vec::new();
    append_promotions(&mut promotions, 48, &56, true);
    assert_eq!(promotions.len(), 4, "Should append 4 promotion moves");
    assert!(promotions.iter().any(|m| m.promotion == Some(QUEEN)));
    assert!(promotions.iter().any(|m| m.promotion == Some(ROOK)));
    assert!(promotions.iter().any(|m| m.promotion == Some(KNIGHT)));
    assert!(promotions.iter().any(|m| m.promotion == Some(BISHOP)));
    // All should be from same square to same square
    for m in &promotions {
        assert_eq!(m.from, 48);
        assert_eq!(m.to, 56);
    }
}

#[test]
fn test_append_promotions_black() {
    let mut promotions = Vec::new();
    append_promotions(&mut promotions, 15, &7, false);
    assert_eq!(promotions.len(), 4, "Should append 4 promotion moves");
    assert!(promotions.iter().any(|m| m.promotion == Some(QUEEN)));
    assert!(promotions.iter().any(|m| m.promotion == Some(ROOK)));
    assert!(promotions.iter().any(|m| m.promotion == Some(KNIGHT)));
    assert!(promotions.iter().any(|m| m.promotion == Some(BISHOP)));
}

// === Systematic Coverage: All 64 squares for King ===

#[test]
fn test_king_moves_all_squares_valid_range() {
    for sq in 0..64 {
        let moves = init_king_moves(sq);
        for &target in &moves {
            assert!(target < 64, "King move target {} from sq {} is out of range", target, sq);
            // King moves should be adjacent (distance 1 in both rank and file)
            let from_rank = sq / 8;
            let from_file = sq % 8;
            let to_rank = target / 8;
            let to_file = target % 8;
            let rank_diff = (from_rank as i32 - to_rank as i32).abs();
            let file_diff = (from_file as i32 - to_file as i32).abs();
            assert!(rank_diff <= 1 && file_diff <= 1, "King move from {} to {} is not adjacent", sq, target);
        }
    }
}

#[test]
fn test_knight_moves_all_squares_valid_range() {
    for sq in 0..64 {
        let moves = init_knight_moves(sq);
        for &target in &moves {
            assert!(target < 64, "Knight move target {} from sq {} is out of range", target, sq);
            let from_rank = sq / 8;
            let from_file = sq % 8;
            let to_rank = target / 8;
            let to_file = target % 8;
            let rank_diff = (from_rank as i32 - to_rank as i32).abs();
            let file_diff = (from_file as i32 - to_file as i32).abs();
            assert!(
                (rank_diff == 2 && file_diff == 1) || (rank_diff == 1 && file_diff == 2),
                "Knight move from {} to {} is not an L-shape", sq, target
            );
        }
    }
}

#[test]
fn test_pawn_moves_all_squares_valid() {
    for sq in 0..64 {
        let (white, black) = init_pawn_moves(sq);
        for &target in white.iter().chain(black.iter()) {
            assert!(target < 64, "Pawn move target {} from sq {} is out of range", target, sq);
        }
    }
}
