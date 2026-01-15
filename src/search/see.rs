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
//! # Example
//!
//! If a queen captures a pawn defended by a knight, SEE would return negative
//! because QxP, NxQ loses 9 pawns worth of material (Q=9, P=1, N=3).

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::piece_types::{KING, PAWN};
use std::cmp::max;

// Piece values for SEE (simple centipawn values)
// Order: P, N, B, R, Q, K (index 6 is 0)
const SEE_PIECE_VALUES: [i32; 7] = [100, 320, 330, 500, 975, 10000, 0];

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
    let mut gain = [0; 32]; // Max possible captures in a sequence
    let mut depth = 0;
    let mut current_board = board.clone(); // Clone board to simulate captures
    let mut side_to_move = board.w_to_move;

    // Get initial captured piece type and value
    let captured_piece_type = match get_piece_type_on_sq(&current_board, target_sq) {
        Some(pt) => pt,
        None => return 0, // Target square is empty, not a capture
    };
    gain[depth] = SEE_PIECE_VALUES[captured_piece_type as usize];

    // Get initial attacker piece type
    let mut attacker_piece_type = match get_piece_type_on_sq(&current_board, initial_attacker_sq) {
        Some(pt) => pt,
        None => return 0, // Should not happen for a valid capture move
    };

    // Simulate the initial capture
    clear_square(&mut current_board, initial_attacker_sq);
    set_square(
        &mut current_board,
        target_sq,
        attacker_piece_type,
        side_to_move,
    );
    update_occupancy(&mut current_board);
    side_to_move = !side_to_move; // Switch sides

    loop {
        depth += 1;
        // Score relative to previous capture. gain[d] = value of attacker - gain[d-1]
        gain[depth] = SEE_PIECE_VALUES[attacker_piece_type as usize] - gain[depth - 1];

        if max(-gain[depth - 1], gain[depth]) < 0 {
            break;
        }

        let attackers_bb = move_gen.attackers_to(&current_board, target_sq, side_to_move);
        if attackers_bb == 0 {
            break; // No more attackers for the current side
        }

        let next_attacker_sq =
            find_least_valuable_attacker_sq(&current_board, attackers_bb, side_to_move);
        if next_attacker_sq == 64 {
            break;
        }

        attacker_piece_type = get_piece_type_on_sq(&current_board, next_attacker_sq).unwrap();

        clear_square(&mut current_board, next_attacker_sq);
        set_square(
            &mut current_board,
            target_sq,
            attacker_piece_type,
            side_to_move,
        );
        update_occupancy(&mut current_board);
        side_to_move = !side_to_move;
    }

    // Calculate final score by propagating the gains/losses back up the sequence
    while depth > 0 {
        depth -= 1;
        gain[depth] = -max(-gain[depth], gain[depth + 1]);
    }
    gain[0]
}

/// Helper function to find the square of the least valuable attacker from a bitboard of attackers.
fn find_least_valuable_attacker_sq(board: &Board, attackers_bb: u64, side: bool) -> usize {
    let color_index = side as usize;
    for piece_type_idx in PAWN..=KING {
        let piece_bb = board.pieces[color_index][piece_type_idx as usize];
        let intersection = attackers_bb & piece_bb;
        if intersection != 0 {
            return intersection.trailing_zeros() as usize;
        }
    }
    64 // Indicate no attacker found (error condition)
}

/// Helper function to determine the piece type (if any) on a given square
fn get_piece_type_on_sq(board: &Board, sq: usize) -> Option<usize> {
    let sq_bb = 1u64 << sq;

    for color in 0..2 {
        for piece_type in PAWN..=KING {
            if (board.pieces[color][piece_type as usize] & sq_bb) != 0 {
                return Some(piece_type as usize);
            }
        }
    }

    None
}

/// Helper function to clear a square on the board
fn clear_square(board: &mut Board, sq: usize) {
    let sq_bb = 1u64 << sq;
    let sq_bb_inv = !sq_bb;

    for color in 0..2 {
        for piece_type in PAWN..=KING {
            board.pieces[color][piece_type as usize] &= sq_bb_inv;
        }
    }
}

/// Helper function to set a piece on a square
fn set_square(board: &mut Board, sq: usize, piece_type: usize, side: bool) {
    let sq_bb = 1u64 << sq;
    let color_index = side as usize;

    board.pieces[color_index][piece_type] |= sq_bb;
}

/// Helper function to update occupancy bitboards
fn update_occupancy(board: &mut Board) {
    for color in 0..2 {
        board.pieces_occ[color] = 0;
        for piece_type in PAWN..=KING {
            board.pieces_occ[color] |= board.pieces[color][piece_type as usize];
        }
    }
}
