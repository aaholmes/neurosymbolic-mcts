//! Static Exchange Evaluation (SEE) for capture analysis.
//!
//! SEE determines whether a capture sequence on a square is likely to win or lose
//! material. It's used extensively for:
//!
//! - **Move ordering**: Prioritize winning captures over losing ones
//! - **Quiescence search pruning**: Skip captures that lose material
//! - **Search extensions/reductions**: Adjust depth for tactical exchanges
//!
//! The algorithm simulates a sequence of captures on the target square,
//! alternating sides, using the least valuable attacker each time. It builds
//! a "swap list" of material gains/losses and determines the optimal stopping
//! point for each side.
//!
//! Uses a lightweight `SeeBoard` struct instead of cloning the full `Board`,
//! avoiding copying zobrist hash, eval, game phase, castling rights, etc.
//!
//! # Example
//!
//! If a queen captures a pawn defended by a knight, SEE would return negative
//! because QxP, NxQ loses 9 pawns worth of material (Q=9, P=1, N=3).

use crate::board::Board;
use crate::board_utils::sq_ind_to_bit;
use crate::move_generation::MoveGen;
use crate::piece_types::{KING, PAWN};
use std::cmp::max;

// Piece values for SEE (simple centipawn values)
// Order: P, N, B, R, Q, K (index 6 is 0)
const SEE_PIECE_VALUES: [i32; 7] = [100, 320, 330, 500, 975, 10000, 0];

/// Lightweight board representation for SEE computation.
/// Only stores piece bitboards and occupancy — avoids copying hash, eval, etc.
struct SeeBoard {
    pieces: [[u64; 6]; 2],
    pieces_occ: [u64; 2],
}

impl SeeBoard {
    /// Create from a full Board by copying only the piece data.
    fn from_board(board: &Board) -> Self {
        let mut pieces = [[0u64; 6]; 2];
        for color in 0..2 {
            for pt in 0..6 {
                pieces[color][pt] = board.get_piece_bitboard(color, pt);
            }
        }
        SeeBoard {
            pieces,
            pieces_occ: [board.get_color_occupancy(0), board.get_color_occupancy(1)],
        }
    }

    fn all_occupancy(&self) -> u64 {
        self.pieces_occ[0] | self.pieces_occ[1]
    }

    fn get_piece_type_on_sq(&self, sq: usize) -> Option<usize> {
        let sq_bb = 1u64 << sq;
        for color in 0..2 {
            for piece_type in PAWN..=KING {
                if (self.pieces[color][piece_type] & sq_bb) != 0 {
                    return Some(piece_type);
                }
            }
        }
        None
    }

    fn clear_square(&mut self, sq: usize) {
        let sq_bb_inv = !(1u64 << sq);
        for color in 0..2 {
            for piece_type in PAWN..=KING {
                self.pieces[color][piece_type] &= sq_bb_inv;
            }
        }
    }

    fn set_square(&mut self, sq: usize, piece_type: usize, side: bool) {
        let sq_bb = 1u64 << sq;
        self.pieces[side as usize][piece_type] |= sq_bb;
    }

    fn update_occupancy(&mut self) {
        for color in 0..2 {
            self.pieces_occ[color] = 0;
            for piece_type in PAWN..=KING {
                self.pieces_occ[color] |= self.pieces[color][piece_type];
            }
        }
    }

    fn find_least_valuable_attacker_sq(&self, attackers_bb: u64, side: bool) -> usize {
        let color_index = side as usize;
        for piece_type_idx in PAWN..=KING {
            let piece_bb = self.pieces[color_index][piece_type_idx];
            let intersection = attackers_bb & piece_bb;
            if intersection != 0 {
                return intersection.trailing_zeros() as usize;
            }
        }
        64
    }

    /// Calculate attackers to a square for a given side, using the MoveGen for
    /// sliding piece attack generation with this board's occupancy.
    fn attackers_to(&self, move_gen: &MoveGen, sq: usize, side: bool) -> u64 {
        let side_idx = side as usize;
        let mut attackers: u64 = 0;
        let all_occ = self.all_occupancy();

        // Pawns
        if side_idx == 0 {
            if sq > 8 && !sq.is_multiple_of(8) {
                attackers |= self.pieces[side_idx][PAWN] & sq_ind_to_bit(sq - 9);
            }
            if sq > 7 && sq % 8 != 7 {
                attackers |= self.pieces[side_idx][PAWN] & sq_ind_to_bit(sq - 7);
            }
        } else {
            if sq < 55 && !sq.is_multiple_of(8) {
                attackers |= self.pieces[side_idx][PAWN] & sq_ind_to_bit(sq + 7);
            }
            if sq < 56 && sq % 8 != 7 {
                attackers |= self.pieces[side_idx][PAWN] & sq_ind_to_bit(sq + 9);
            }
        }

        // Knights
        attackers |= move_gen.n_move_bitboard[sq] & self.pieces[side_idx][1]; // KNIGHT=1

        // King
        attackers |= move_gen.k_move_bitboard[sq] & self.pieces[side_idx][KING];

        // Bishops and Queens (diagonal)
        let bishop_attacks = move_gen.gen_bishop_attacks_occ(sq, all_occ);
        attackers |= bishop_attacks & (self.pieces[side_idx][2] | self.pieces[side_idx][4]); // BISHOP=2, QUEEN=4

        // Rooks and Queens (horizontal/vertical)
        let rook_attacks = move_gen.gen_rook_attacks_occ(sq, all_occ);
        attackers |= rook_attacks & (self.pieces[side_idx][3] | self.pieces[side_idx][4]); // ROOK=3, QUEEN=4

        attackers
    }
}

/// Calculates the Static Exchange Evaluation (SEE) for a move to a target square.
/// Determines if a sequence of captures on `target_sq` initiated by the current side to move
/// is likely to win material.
///
/// # Arguments
/// * `board` - The current board state.
/// * `move_gen` - The move generator (needed for finding attackers).
/// * `target_sq` - The square where the capture sequence occurs.
/// * `initial_attacker_sq` - The square of the piece making the initial capture.
///
/// # Returns
/// The estimated material balance after the exchange sequence. Positive means gain, negative means loss.
/// Returns 0 if the target square is empty or the initial attacker is invalid.
pub fn see(board: &Board, move_gen: &MoveGen, target_sq: usize, initial_attacker_sq: usize) -> i32 {
    let mut gain = [0; 32];
    let mut depth = 0;
    let mut see_board = SeeBoard::from_board(board);
    let mut side_to_move = board.w_to_move;

    // Get initial captured piece type and value
    let captured_piece_type = match see_board.get_piece_type_on_sq(target_sq) {
        Some(pt) => pt,
        None => return 0,
    };
    gain[depth] = SEE_PIECE_VALUES[captured_piece_type];

    // Get initial attacker piece type
    let mut attacker_piece_type = match see_board.get_piece_type_on_sq(initial_attacker_sq) {
        Some(pt) => pt,
        None => return 0,
    };

    // Simulate the initial capture
    see_board.clear_square(initial_attacker_sq);
    see_board.set_square(target_sq, attacker_piece_type, side_to_move);
    see_board.update_occupancy();
    side_to_move = !side_to_move;

    loop {
        depth += 1;
        gain[depth] = SEE_PIECE_VALUES[attacker_piece_type] - gain[depth - 1];

        if max(-gain[depth - 1], gain[depth]) < 0 {
            break;
        }

        let attackers_bb = see_board.attackers_to(move_gen, target_sq, side_to_move);
        if attackers_bb == 0 {
            break;
        }

        let next_attacker_sq =
            see_board.find_least_valuable_attacker_sq(attackers_bb, side_to_move);
        if next_attacker_sq == 64 {
            break;
        }

        attacker_piece_type = see_board.get_piece_type_on_sq(next_attacker_sq).unwrap();

        see_board.clear_square(next_attacker_sq);
        see_board.set_square(target_sq, attacker_piece_type, side_to_move);
        see_board.update_occupancy();
        side_to_move = !side_to_move;
    }

    // Calculate final score by propagating the gains/losses back up the sequence
    while depth > 0 {
        depth -= 1;
        gain[depth] = -max(-gain[depth], gain[depth + 1]);
    }
    gain[0]
}
