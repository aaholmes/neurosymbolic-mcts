//! Comprehensive tests for board_to_planes() STM-perspective encoding.
//!
//! These tests verify the board encoding without requiring the `neural` feature,
//! since board_to_planes() returns a plain Vec<f32>.

use kingfisher::board::Board;
use kingfisher::move_types::Move;
use kingfisher::tensor::{board_to_planes, move_to_index};

/// Get the value at (tensor_row, tensor_col) in a specific plane.
fn plane_value(planes: &[f32], plane_idx: usize, row: usize, col: usize) -> f32 {
    planes[plane_idx * 64 + row * 8 + col]
}

/// Count the number of set (1.0) cells in a plane.
fn plane_count(planes: &[f32], plane_idx: usize) -> usize {
    let start = plane_idx * 64;
    planes[start..start + 64]
        .iter()
        .filter(|&&v| v == 1.0)
        .count()
}

/// Check if every cell in a plane is 1.0.
fn plane_all_ones(planes: &[f32], plane_idx: usize) -> bool {
    plane_count(planes, plane_idx) == 64
}

/// Check if every cell in a plane is 0.0.
fn plane_all_zeros(planes: &[f32], plane_idx: usize) -> bool {
    plane_count(planes, plane_idx) == 0
}

// =========================================================================
// A. Piece plane assignment (STM vs opponent)
// =========================================================================

#[test]
fn test_white_stm_pieces_in_planes_0_5() {
    // Starting position, White to move: White pieces should be in planes 0-5
    let board = Board::new();
    let planes = board_to_planes(&board);

    // White pawns (plane 0): 8 pawns on rank 1 (a2-h2)
    assert_eq!(
        plane_count(&planes, 0),
        8,
        "Expected 8 white pawns in plane 0"
    );
    // White knights (plane 1): 2 knights
    assert_eq!(plane_count(&planes, 1), 2);
    // White bishops (plane 2): 2 bishops
    assert_eq!(plane_count(&planes, 2), 2);
    // White rooks (plane 3): 2 rooks
    assert_eq!(plane_count(&planes, 3), 2);
    // White queen (plane 4): 1 queen
    assert_eq!(plane_count(&planes, 4), 1);
    // White king (plane 5): 1 king
    assert_eq!(plane_count(&planes, 5), 1);
}

#[test]
fn test_white_opponent_pieces_in_planes_6_11() {
    // Starting position, White to move: Black pieces should be in planes 6-11
    let board = Board::new();
    let planes = board_to_planes(&board);

    assert_eq!(
        plane_count(&planes, 6),
        8,
        "Expected 8 black pawns in plane 6"
    );
    assert_eq!(plane_count(&planes, 7), 2);
    assert_eq!(plane_count(&planes, 8), 2);
    assert_eq!(plane_count(&planes, 9), 2);
    assert_eq!(plane_count(&planes, 10), 1);
    assert_eq!(plane_count(&planes, 11), 1);
}

#[test]
fn test_black_stm_pieces_in_planes_0_5() {
    // Black to move: Black pieces should be in planes 0-5 (STM planes)
    let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let planes = board_to_planes(&board);

    // Black pawns in plane 0 (STM pawns)
    assert_eq!(
        plane_count(&planes, 0),
        8,
        "Expected 8 black pawns in plane 0 when Black is STM"
    );
    // Black knights in plane 1
    assert_eq!(plane_count(&planes, 1), 2);
    // Black king in plane 5
    assert_eq!(plane_count(&planes, 5), 1);
}

#[test]
fn test_black_opponent_pieces_in_planes_6_11() {
    // Black to move: White pieces should be in planes 6-11 (opponent planes)
    let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let planes = board_to_planes(&board);

    // White pawns in plane 6 (opponent pawns): 7 pawns (e-pawn moved to e4) + 1 on e4 = 8
    assert_eq!(
        plane_count(&planes, 6),
        8,
        "Expected 8 white pawns in plane 6 when Black is STM"
    );
    assert_eq!(plane_count(&planes, 7), 2); // White knights
    assert_eq!(plane_count(&planes, 11), 1); // White king
}

// =========================================================================
// B. Board flipping (rank inversion when Black to move)
// =========================================================================

#[test]
fn test_white_to_move_rank_mapping() {
    // White pawn on a2 (sq 8, rank 1, file 0)
    // With White to move, tensor_rank = 7 - 1 = 6
    let board = Board::new_from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1");
    let planes = board_to_planes(&board);

    assert_eq!(
        plane_value(&planes, 0, 6, 0),
        1.0,
        "White pawn on a2 should be at tensor row 6, col 0"
    );
    assert_eq!(plane_count(&planes, 0), 1, "Should have exactly 1 pawn");
}

#[test]
fn test_black_to_move_rank_mapping() {
    // Black pawn on a7 (sq 48, rank 6, file 0), Black to move
    // With Black to move, tensor_rank = rank = 6
    let board = Board::new_from_fen("4k3/p7/8/8/8/8/8/4K3 b - - 0 1");
    let planes = board_to_planes(&board);

    // Black pawn is STM pawn → plane 0
    // rank 6, with Black flip: tensor_rank = 6
    assert_eq!(
        plane_value(&planes, 0, 6, 0),
        1.0,
        "Black pawn on a7 should be at tensor row 6, col 0 (same as White's a2 pawn)"
    );
    assert_eq!(plane_count(&planes, 0), 1);
}

#[test]
fn test_stm_symmetry_kings() {
    // White king on e1 (sq 4), White to move → tensor row = 7 - 0 = 7, col 4
    let board_w = Board::new_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let planes_w = board_to_planes(&board_w);

    // Black king on e8 (sq 60), Black to move → tensor row = 7, col 4
    let board_b = Board::new_from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1");
    let planes_b = board_to_planes(&board_b);

    // STM king (plane 5) should be at tensor (7, 4) in both cases
    assert_eq!(
        plane_value(&planes_w, 5, 7, 4),
        1.0,
        "White king on e1 → tensor (7, 4) when White to move"
    );
    assert_eq!(
        plane_value(&planes_b, 5, 7, 4),
        1.0,
        "Black king on e8 → tensor (7, 4) when Black to move (flipped)"
    );
}

#[test]
fn test_mirror_position_produces_identical_planes() {
    // Position: White pawn e2, White king e1, Black king e8, White to move
    let board_w = Board::new_from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    // Mirror: Black pawn e7, Black king e8, White king e1, Black to move
    let board_b = Board::new_from_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1");

    let planes_w = board_to_planes(&board_w);
    let planes_b = board_to_planes(&board_b);

    // Planes 0-11 should be identical (piece placement from STM perspective)
    for plane in 0..12 {
        for sq in 0..64 {
            assert_eq!(
                planes_w[plane * 64 + sq],
                planes_b[plane * 64 + sq],
                "Mirrored positions differ at plane {}, sq {}",
                plane,
                sq
            );
        }
    }
}

// =========================================================================
// C. Castling rights (STM-relative ordering)
// =========================================================================

#[test]
fn test_castling_white_stm_kingside_only() {
    // White to move, only White kingside castling
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
    let planes = board_to_planes(&board);

    assert!(
        plane_all_ones(&planes, 13),
        "Plane 13 (STM KS) should be all 1s"
    );
    assert!(
        plane_all_zeros(&planes, 14),
        "Plane 14 (STM QS) should be all 0s"
    );
    assert!(
        plane_all_zeros(&planes, 15),
        "Plane 15 (Opp KS) should be all 0s"
    );
    assert!(
        plane_all_zeros(&planes, 16),
        "Plane 16 (Opp QS) should be all 0s"
    );
}

#[test]
fn test_castling_black_stm_kingside_only() {
    // Black to move, only Black kingside castling → same as above: plane 13 filled
    let board = Board::new_from_fen("4k2r/8/8/8/8/8/8/4K3 b k - 0 1");
    let planes = board_to_planes(&board);

    assert!(
        plane_all_ones(&planes, 13),
        "Plane 13 (STM KS) should be all 1s for Black KS"
    );
    assert!(
        plane_all_zeros(&planes, 14),
        "Plane 14 (STM QS) should be all 0s"
    );
    assert!(
        plane_all_zeros(&planes, 15),
        "Plane 15 (Opp KS) should be all 0s"
    );
    assert!(
        plane_all_zeros(&planes, 16),
        "Plane 16 (Opp QS) should be all 0s"
    );
}

#[test]
fn test_castling_opponent_rights() {
    // White to move, only Black queenside castling → plane 16 (Opp QS)
    let board = Board::new_from_fen("r3k3/8/8/8/8/8/8/4K3 w q - 0 1");
    let planes = board_to_planes(&board);

    assert!(
        plane_all_zeros(&planes, 13),
        "Plane 13 (STM KS) should be all 0s"
    );
    assert!(
        plane_all_zeros(&planes, 14),
        "Plane 14 (STM QS) should be all 0s"
    );
    assert!(
        plane_all_zeros(&planes, 15),
        "Plane 15 (Opp KS) should be all 0s"
    );
    assert!(
        plane_all_ones(&planes, 16),
        "Plane 16 (Opp QS) should be all 1s"
    );
}

#[test]
fn test_castling_all_rights() {
    // Starting position: all castling rights
    let board = Board::new();
    let planes = board_to_planes(&board);

    assert!(plane_all_ones(&planes, 13), "Plane 13 (STM KS)");
    assert!(plane_all_ones(&planes, 14), "Plane 14 (STM QS)");
    assert!(plane_all_ones(&planes, 15), "Plane 15 (Opp KS)");
    assert!(plane_all_ones(&planes, 16), "Plane 16 (Opp QS)");
}

// =========================================================================
// D. En passant flipping
// =========================================================================

#[test]
fn test_en_passant_white_to_move() {
    // After 1. e4, en passant square is e3 (sq 20, rank 2, file 4)
    // White to move doesn't make sense for e3 EP — use a position where Black just played d5
    // after 1. e4 d5: EP square is d6 (sq 43? no — EP square is d6 = rank 5, file 3 = sq 43)
    // Actually after 1. e4 d5, it's White to move with EP on d6? No — EP is set on the side that can capture.
    // After 1. d4 ... 2. d5 e5: EP square is e6 for White to capture. White to move.
    // Simpler: after 1. e4 e5 2. Nf3 d5: EP on d6 (sq 43, rank 5, file 3), White to move
    // Let's just use a FEN directly
    let board = Board::new_from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
    let planes = board_to_planes(&board);

    // EP square: d6 = sq 43, rank 5, file 3
    // White to move: tensor_rank = 7 - 5 = 2
    assert_eq!(
        plane_value(&planes, 12, 2, 3),
        1.0,
        "EP square d6 should be at tensor (2, 3) when White to move"
    );
    assert_eq!(
        plane_count(&planes, 12),
        1,
        "Should have exactly 1 EP square"
    );
}

#[test]
fn test_en_passant_black_to_move() {
    // After 1. e4: EP square is e3 (sq 20, rank 2, file 4), Black to move
    let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let planes = board_to_planes(&board);

    // EP square: e3 = sq 20, rank 2, file 4
    // Black to move: tensor_rank = rank = 2
    assert_eq!(
        plane_value(&planes, 12, 2, 4),
        1.0,
        "EP square e3 should be at tensor (2, 4) when Black to move"
    );
    assert_eq!(plane_count(&planes, 12), 1);
}

#[test]
fn test_en_passant_symmetry() {
    // White pawn on e5 with EP on d6, White to move
    let board_w = Board::new_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    // Mirror: Black pawn on e4 with EP on d3, Black to move
    let board_b = Board::new_from_fen("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1");

    let planes_w = board_to_planes(&board_w);
    let planes_b = board_to_planes(&board_b);

    // d6: rank 5, file 3 → White: tensor_rank = 7-5 = 2, so (2, 3)
    // d3: rank 2, file 3 → Black: tensor_rank = 2, so (2, 3)
    assert_eq!(plane_value(&planes_w, 12, 2, 3), 1.0);
    assert_eq!(plane_value(&planes_b, 12, 2, 3), 1.0);
}

// =========================================================================
// E. Full position consistency
// =========================================================================

#[test]
fn test_starting_position_plane_counts() {
    let board = Board::new();
    let planes = board_to_planes(&board);

    // Total length
    assert_eq!(planes.len(), 17 * 64);

    // STM pieces (White)
    assert_eq!(plane_count(&planes, 0), 8); // Pawns
    assert_eq!(plane_count(&planes, 1), 2); // Knights
    assert_eq!(plane_count(&planes, 2), 2); // Bishops
    assert_eq!(plane_count(&planes, 3), 2); // Rooks
    assert_eq!(plane_count(&planes, 4), 1); // Queen
    assert_eq!(plane_count(&planes, 5), 1); // King

    // Opponent pieces (Black)
    assert_eq!(plane_count(&planes, 6), 8);
    assert_eq!(plane_count(&planes, 7), 2);
    assert_eq!(plane_count(&planes, 8), 2);
    assert_eq!(plane_count(&planes, 9), 2);
    assert_eq!(plane_count(&planes, 10), 1);
    assert_eq!(plane_count(&planes, 11), 1);

    // No en passant
    assert!(plane_all_zeros(&planes, 12));

    // All castling rights
    for p in 13..=16 {
        assert!(plane_all_ones(&planes, p));
    }
}

#[test]
fn test_empty_board_only_kings() {
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let planes = board_to_planes(&board);

    // Only king planes should have pieces
    for p in 0..5 {
        assert!(plane_all_zeros(&planes, p), "Plane {} should be empty", p);
    }
    assert_eq!(plane_count(&planes, 5), 1, "STM king");

    for p in 6..11 {
        assert!(plane_all_zeros(&planes, p), "Plane {} should be empty", p);
    }
    assert_eq!(plane_count(&planes, 11), 1, "Opponent king");

    // No EP, no castling
    assert!(plane_all_zeros(&planes, 12));
    for p in 13..=16 {
        assert!(plane_all_zeros(&planes, p));
    }
}

#[test]
fn test_specific_piece_positions_white() {
    // White knight on g1 (sq 6, rank 0, file 6), White to move
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1");
    let planes = board_to_planes(&board);

    // Knight on g1: rank 0, file 6 → tensor_rank = 7 - 0 = 7
    assert_eq!(
        plane_value(&planes, 1, 7, 6),
        1.0,
        "White knight on g1 should be at tensor (7, 6)"
    );
}

#[test]
fn test_specific_piece_positions_black_flipped() {
    // Black knight on g8 (sq 62, rank 7, file 6), Black to move
    let board = Board::new_from_fen("4k1n1/8/8/8/8/8/8/4K3 b - - 0 1");
    let planes = board_to_planes(&board);

    // Knight on g8: rank 7, file 6 → Black flip: tensor_rank = 7
    assert_eq!(
        plane_value(&planes, 1, 7, 6),
        1.0,
        "Black knight on g8 should be at tensor (7, 6) when Black is STM (same as White g1)"
    );
}

// =========================================================================
// F. Policy move flipping
// =========================================================================

#[test]
fn test_policy_move_flip_white() {
    // White e2e4 (sq 12 → sq 28): used directly for White
    let mv = Move::new(12, 28, None);
    let idx_w = move_to_index(mv);

    // Should be src=12, N direction, distance 2
    // dy=2, dx=0 → direction 0 (N), distance 2 → plane = 0*7 + 1 = 1
    assert_eq!(idx_w, 12 * 73 + 1);
}

#[test]
fn test_policy_move_flip_black() {
    // Black e7e5 (sq 52 → sq 36): flip_vertical gives e2e4 equivalent
    let mv = Move::new(52, 36, None);
    let flipped = mv.flip_vertical();

    // flip_vertical: 52 → 8*(7-6)+4 = 12, 36 → 8*(7-4)+4 = 28
    assert_eq!(flipped.from(), 12);
    assert_eq!(flipped.to(), 28);

    // So the index should be the same as White's e2e4
    let idx_b = move_to_index(flipped);
    let idx_w = move_to_index(Move::new(12, 28, None));
    assert_eq!(
        idx_b, idx_w,
        "Black e7e5 flipped should produce same index as White e2e4"
    );
}

#[test]
fn test_policy_move_flip_capture() {
    // White: d4 captures e5 (sq 27 → sq 36)
    let mv_w = Move::new(27, 36, None);
    let idx_w = move_to_index(mv_w);

    // Black: d5 captures e4 (sq 35 → sq 28), flipped: d4→e5
    let mv_b = Move::new(35, 28, None);
    let flipped = mv_b.flip_vertical();
    // 35 → 8*(7-4)+3 = 27, 28 → 8*(7-3)+4 = 36
    assert_eq!(flipped.from(), 27);
    assert_eq!(flipped.to(), 36);

    let idx_b = move_to_index(flipped);
    assert_eq!(idx_b, idx_w);
}
