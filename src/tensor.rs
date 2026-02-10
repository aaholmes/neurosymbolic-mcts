//! Tensor mapping logic for AlphaZero policy representation
//!
//! Maps chess moves to a flat index (0..4672) representing the 8x8x73 policy tensor.
//! Also provides board-to-planes encoding for NN input (STM-relative perspective).

use crate::board::Board;
use crate::move_types::Move;
use crate::piece_types::{KNIGHT, BISHOP, ROOK, QUEEN, WHITE, BLACK};

/// Encode a board into 17×8×8 planes from STM's perspective.
/// Returns Vec<f32> of length 1088 (17 * 64).
///
/// Plane layout (STM-relative):
///   0-5:   STM pieces (P, N, B, R, Q, K)
///   6-11:  Opponent pieces (P, N, B, R, Q, K)
///   12:    En passant target square
///   13-16: Castling rights (STM-KS, STM-QS, Opp-KS, Opp-QS)
///
/// When Black to move, ranks are flipped (rank 0 → tensor row 7, rank 7 → tensor row 0)
/// so that STM's pieces always appear at high tensor rows ("bottom").
pub fn board_to_planes(board: &Board) -> Vec<f32> {
    let mut planes = vec![0.0f32; 17 * 8 * 8];
    let stm = if board.w_to_move { WHITE } else { BLACK };
    let opp = if board.w_to_move { BLACK } else { WHITE };

    // Planes 0-5: STM pieces | Planes 6-11: Opponent pieces
    for (p_idx, &color) in [stm, opp].iter().enumerate() {
        for pt in 0..6 {
            let offset = (p_idx * 6 + pt) * 64;
            let bb = board.get_piece_bitboard(color, pt);
            for i in 0..64 {
                if (bb >> i) & 1 == 1 {
                    let rank = i / 8;
                    let file = i % 8;
                    let tensor_rank = if board.w_to_move { 7 - rank } else { rank };
                    planes[offset + tensor_rank * 8 + file] = 1.0;
                }
            }
        }
    }

    // Plane 12: En passant target square
    if let Some(sq) = board.en_passant() {
        let rank = (sq / 8) as usize;
        let file = (sq % 8) as usize;
        let tensor_rank = if board.w_to_move { 7 - rank } else { rank };
        planes[12 * 64 + tensor_rank * 8 + file] = 1.0;
    }

    // Planes 13-16: Castling rights (STM-relative)
    let rights = if board.w_to_move {
        [board.castling_rights.white_kingside, board.castling_rights.white_queenside,
         board.castling_rights.black_kingside, board.castling_rights.black_queenside]
    } else {
        [board.castling_rights.black_kingside, board.castling_rights.black_queenside,
         board.castling_rights.white_kingside, board.castling_rights.white_queenside]
    };

    for (i, &allowed) in rights.iter().enumerate() {
        if allowed {
            let offset = (13 + i) * 64;
            for j in 0..64 {
                planes[offset + j] = 1.0;
            }
        }
    }

    planes
}

/// Converts a move to a flat index in the 8x8x73 policy tensor.
///
/// Formula: Index = (SourceSquare * 73) + PlaneIndex
pub fn move_to_index(mv: Move) -> usize {
    let src = mv.from;
    let dst = mv.to;
    
    let src_rank = (src / 8) as i32;
    let src_file = (src % 8) as i32;
    let dst_rank = (dst / 8) as i32;
    let dst_file = (dst % 8) as i32;
    
    let dx = dst_file - src_file;
    let dy = dst_rank - src_rank;
    
    let plane = if mv.is_promotion() && mv.promotion.unwrap() != QUEEN {
        // Case A: Underpromotion (Promoting to N, B, R)
        let promo_piece = mv.promotion.unwrap();
        
        let direction_offset = match dx {
            0 => 0,  // Straight
            -1 => 1, // Capture Left
            1 => 2,  // Capture Right
            _ => panic!("Invalid promotion move dx: {}", dx),
        };
        
        let piece_offset = match promo_piece {
            KNIGHT => 0,
            BISHOP => 3,
            ROOK => 6,
            _ => panic!("Invalid underpromotion piece: {}", promo_piece),
        };
        
        64 + direction_offset + piece_offset
    } else if (dx * dy).abs() == 2 {
        // Case B: Knight Move
        // Map (dx, dy) to 0..7
        let knight_idx = match (dx, dy) {
            (1, 2) => 0,
            (2, 1) => 1,
            (2, -1) => 2,
            (1, -2) => 3,
            (-1, -2) => 4,
            (-2, -1) => 5,
            (-2, 1) => 6,
            (-1, 2) => 7,
            _ => panic!("Invalid knight move delta: ({}, {})", dx, dy),
        };
        
        56 + knight_idx
    } else {
        // Case C: Queen Move (Slide) - includes Queen promotion
        // Direction: 0..7
        // N(0,1), NE(1,1), E(1,0), SE(1,-1), S(0,-1), SW(-1,-1), W(-1,0), NW(-1,1)
        
        let direction = if dx == 0 && dy > 0 { 0 }      // N
        else if dx > 0 && dy > 0 { 1 }     // NE
        else if dx > 0 && dy == 0 { 2 }    // E
        else if dx > 0 && dy < 0 { 3 }     // SE
        else if dx == 0 && dy < 0 { 4 }    // S
        else if dx < 0 && dy < 0 { 5 }     // SW
        else if dx < 0 && dy == 0 { 6 }    // W
        else if dx < 0 && dy > 0 { 7 }     // NW
        else { panic!("Invalid slide move delta: ({}, {})", dx, dy) };
        
        let distance = std::cmp::max(dx.abs(), dy.abs());
        
        // Plane = (Direction * 7) + (Distance - 1)
        // 0..55
        (direction * 7) + (distance - 1)
    };
    
    src * 73 + plane as usize
}

/// Convert a raw NN policy vector into per-move priors for the given legal moves.
///
/// Flips moves for Black (to match the STM-relative tensor encoding),
/// indexes into the policy vector, and normalizes to sum to 1.
pub fn policy_to_move_priors(policy: &[f32], moves: &[Move], board: &Board) -> Vec<(Move, f32)> {
    let mut result = Vec::with_capacity(moves.len());
    let mut total_prob = 0.0;
    for &mv in moves {
        let relative_mv = if board.w_to_move { mv } else { mv.flip_vertical() };
        let idx = move_to_index(relative_mv);
        if idx < policy.len() {
            let prob = policy[idx];
            result.push((mv, prob));
            total_prob += prob;
        } else {
            result.push((mv, 0.0));
        }
    }
    if total_prob > 0.0 {
        for (_, prob) in result.iter_mut() { *prob /= total_prob; }
    } else {
        let uniform = 1.0 / moves.len() as f32;
        for (_, prob) in result.iter_mut() { *prob = uniform; }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece_types::{QUEEN, ROOK, KNIGHT};

    #[test]
    fn test_queen_slide() {
        // e4 (28) -> e5 (36): N, dist 1. Plane = 0*7 + 0 = 0.
        // Index = 28 * 73 + 0 = 2044
        let mv = Move::new(28, 36, None);
        assert_eq!(move_to_index(mv), 28 * 73 + 0);
        
        // e4 -> h4 (31): E, dist 3. Plane = 2*7 + 2 = 16.
        let mv = Move::new(28, 31, None);
        assert_eq!(move_to_index(mv), 28 * 73 + 16);
    }
    
    #[test]
    fn test_knight_move() {
        // e4 (28) -> f6 (45): (1, 2) -> Idx 0. Plane = 56.
        let mv = Move::new(28, 45, None);
        assert_eq!(move_to_index(mv), 28 * 73 + 56);
    }
    
    #[test]
    fn test_underpromotion() {
        // a7 (48) -> a8 (56) promote to Rook. 
        // dx=0. Plane = 64 + 0 + 6 = 70.
        let mv = Move::new(48, 56, Some(ROOK));
        assert_eq!(move_to_index(mv), 48 * 73 + 70);
    }
}
