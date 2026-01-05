//! Module for making moves on the chess board
//!
//! This module provides functionality to apply moves on the Bitboard representation of a chess position.

use crate::board::Board;
use crate::board_utils::sq_ind_to_bit;
use crate::move_types::Move;
use crate::piece_types::{BLACK, KING, PAWN, ROOK, WHITE};

impl Board {
    /// Makes a move on the board, returning a new board with the move applied
    ///
    /// This method assumes the move is legal and does not perform any legality checks.
    ///
    /// # Arguments
    ///
    /// * `the_move` - The Move to be applied to the board
    ///
    /// # Returns
    ///
    /// A new Bitboard representing the position after the move has been made
    pub fn apply_move_to_board(&self, the_move: Move) -> Board {
        // Make a move, returning a new board.
        // Assumes the move is legal.

        let mut new_board = self.clone();

        // Null move: remove en passant ability and return new board
        if the_move.from == 0 && the_move.to == 0 {
            new_board.en_passant = None;
            new_board.w_to_move = !new_board.w_to_move;
            if new_board.w_to_move {
                new_board.fullmove_number += 1;
            }
            new_board.zobrist_hash = new_board.compute_zobrist_hash();
            return new_board;
        }

        new_board.halfmove_clock += 1;

        let from_bit = sq_ind_to_bit(the_move.from);
        let to_bit = sq_ind_to_bit(the_move.to);

        let from_piece = self.get_piece(the_move.from);
        if from_piece.is_none() {
            panic!("No piece at from_sq_ind {} ({})", the_move.from, crate::board_utils::sq_ind_to_algebraic(the_move.from));
        }

        let to_piece = self.get_piece(the_move.to);
        if to_piece.is_some() {
            // Capture: Remove the captured piece before moving.
            let (color, piece) = to_piece.unwrap();
            new_board.pieces[color][piece] ^= to_bit;
            new_board.halfmove_clock = 0;
        }

        if from_piece.unwrap().1 == PAWN {
            // En passant
            if new_board.en_passant.is_some()
                && the_move.to == new_board.en_passant.unwrap() as usize
            {
                // Capture the pawn.
                if new_board.w_to_move {
                    new_board.pieces[BLACK][PAWN] ^= sq_ind_to_bit(the_move.to - 8);
                } else {
                    new_board.pieces[WHITE][PAWN] ^= sq_ind_to_bit(the_move.to + 8);
                }
            }
        }
        // Reset the en passant rule.
        new_board.en_passant = None;
        if from_piece.unwrap().1 == PAWN {
            // Pawn move: Reset halfmove clock.
            new_board.halfmove_clock = 0;
            if ((the_move.to as i8) - (the_move.from as i8)).abs() == 16 {
                // Pawn double move: Set en passant square.
                new_board.en_passant = Some(((the_move.from + the_move.to) / 2) as u8);
            }
        }

        // Finally, move the piece.
        let (color, piece) = from_piece.unwrap();
        new_board.pieces[color][piece] ^= from_bit;
        new_board.pieces[color][piece] ^= to_bit;

        // Handle promotions
        if let Some(promotion) = the_move.promotion {
            new_board.pieces[color][piece] ^= to_bit;
            new_board.pieces[color][promotion] ^= to_bit;
        }

        // Handle castling
        if let Some((color, KING)) = from_piece {
            if color == WHITE {
                // White king
                if the_move.from == 4 && the_move.to == 6 {
                    // White king-side castle
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(5);
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(7);
                } else if the_move.from == 4 && the_move.to == 2 {
                    // White queen-side castle
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(3);
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(0);
                }
                new_board.castling_rights.white_kingside = false;
                new_board.castling_rights.white_queenside = false;
            } else {
                // Black king
                if the_move.from == 60 && the_move.to == 62 {
                    // Black king-side castle
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(61);
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(63);
                } else if the_move.from == 60 && the_move.to == 58 {
                    // Black queen-side castle
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(59);
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(56);
                }
                new_board.castling_rights.black_kingside = false;
                new_board.castling_rights.black_queenside = false;
            }
        } else if let Some((color, ROOK)) = from_piece {
            if color == WHITE {
                // White rook
                if the_move.from == 0 {
                    new_board.castling_rights.white_queenside = false;
                } else if the_move.from == 7 {
                    new_board.castling_rights.white_kingside = false;
                }
            } else {
                // Black rook
                if the_move.from == 56 {
                    new_board.castling_rights.black_queenside = false;
                } else if the_move.from == 63 {
                    new_board.castling_rights.black_kingside = false;
                }
            }
        }

        new_board.w_to_move = !new_board.w_to_move;
        if new_board.w_to_move {
            new_board.fullmove_number += 1;
        }

        for color in 0..2 {
            new_board.pieces_occ[color] = new_board.pieces[color][PAWN];
            for piece in 1..6 {
                new_board.pieces_occ[color] |= new_board.pieces[color][piece];
            }
        }

        // Update zobrist hash
        new_board.zobrist_hash = new_board.compute_zobrist_hash();

        new_board
    }
}
