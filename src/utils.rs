//! Utility functions for the chess engine
//!
//! This module contains various utility functions used throughout the chess engine,
//! including functions for printing bitboards, moves, and performing performance tests.

use crate::board_utils::{coords_to_sq_ind, sq_ind_to_algebraic, sq_ind_to_bit};
use crate::move_types::Move;
use crate::piece_types::{BISHOP, KNIGHT, QUEEN, ROOK};

/// Print a u64 as an 8x8 chess board representation
///
/// This function takes a 64-bit integer representing a bitboard and prints it
/// as an 8x8 chess board, where 'X' represents a set bit and '.' represents an unset bit.
///
/// # Arguments
///
/// * `bits` - A u64 representing a bitboard
pub fn print_bits(bits: u64) {
    println!("  +-----------------+");
    for rank in (0..8).rev() {
        print!("{} | ", rank + 1);
        for file in 0..8 {
            let sq_ind = coords_to_sq_ind(file, rank);
            let bit = sq_ind_to_bit(sq_ind);
            if bit & bits != 0 {
                print!("X ");
            } else {
                print!(". ");
            }
        }
        println!("|");
    }
    println!("  +-----------------+");
    println!("    a b c d e f g h");
}

/// Convert a Move to a string in algebraic notation
///
/// This function takes a Move and returns a String representing the move
/// in algebraic notation (e.g., "e2e4" or "e7e8=Q" for promotions).
///
/// # Arguments
///
/// * `the_move` - A reference to a Move
///
/// # Returns
///
/// A String representing the move in algebraic notation
pub fn print_move(the_move: &Move) -> String {
    let from = sq_ind_to_algebraic(the_move.from());
    let to = sq_ind_to_algebraic(the_move.to());
    let mut promotion = String::from("");
    if the_move.promotion().is_some() {
        promotion = String::from("=");
        match the_move.promotion().unwrap() {
            n if n == KNIGHT => promotion.push('N'),
            n if n == BISHOP => promotion.push('B'),
            n if n == ROOK => promotion.push('R'),
            n if n == QUEEN => promotion.push('Q'),
            _ => panic!("Invalid promotion piece"),
        }
    }
    format!("{}{}{}", from, to, promotion)
}
