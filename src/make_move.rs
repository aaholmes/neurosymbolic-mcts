//! Module for making moves on the chess board
//!
//! This module provides functionality to apply moves on the Bitboard representation of a chess position.

use crate::board::Board;
use crate::board_utils::sq_ind_to_bit;
use crate::hash::ZOBRIST_KEYS;
use crate::move_types::Move;
use crate::piece_types::{BLACK, KING, PAWN, ROOK, WHITE};

impl Board {
    /// Makes a move on the board, returning a new board with the move applied
    ///
    /// This method assumes the move is legal and does not perform any legality checks.
    /// Uses incremental Zobrist hashing for efficiency.
    ///
    /// # Arguments
    ///
    /// * `the_move` - The Move to be applied to the board
    ///
    /// # Returns
    ///
    /// A new Bitboard representing the position after the move has been made
    pub fn apply_move_to_board(&self, the_move: Move) -> Board {
        let mut new_board = self.clone();
        let mut hash = self.zobrist_hash;

        // Null move: remove en passant ability and return new board
        if the_move.from == 0 && the_move.to == 0 {
            // XOR out old en passant if present
            if let Some(ep_sq) = new_board.en_passant {
                hash ^= ZOBRIST_KEYS.en_passant_key((ep_sq % 8) as usize);
            }
            new_board.en_passant = None;
            // Toggle side to move
            hash ^= ZOBRIST_KEYS.side_to_move_key();
            new_board.w_to_move = !new_board.w_to_move;
            if new_board.w_to_move {
                new_board.fullmove_number += 1;
            }
            new_board.zobrist_hash = hash;
            debug_assert_eq!(
                new_board.zobrist_hash,
                new_board.compute_zobrist_hash(),
                "Incremental hash mismatch for null move"
            );
            return new_board;
        }

        new_board.halfmove_clock += 1;

        let from_bit = sq_ind_to_bit(the_move.from);
        let to_bit = sq_ind_to_bit(the_move.to);

        let from_piece = self.get_piece(the_move.from);
        if from_piece.is_none() {
            panic!(
                "No piece at from_sq_ind {} ({})",
                the_move.from,
                crate::board_utils::sq_ind_to_algebraic(the_move.from)
            );
        }

        let to_piece = self.get_piece(the_move.to);
        if to_piece.is_some() {
            // Capture: Remove the captured piece before moving.
            let (cap_color, cap_piece) = to_piece.unwrap();
            new_board.pieces[cap_color][cap_piece] ^= to_bit;
            new_board.halfmove_clock = 0;
            // XOR out captured piece from hash
            hash ^= ZOBRIST_KEYS.piece_key(cap_color, cap_piece, the_move.to);
        }

        // Save old castling rights for diffing
        let old_castling = new_board.castling_rights;

        if from_piece.unwrap().1 == PAWN {
            // En passant capture
            if new_board.en_passant.is_some()
                && the_move.to == new_board.en_passant.unwrap() as usize
            {
                if new_board.w_to_move {
                    let ep_captured_sq = the_move.to - 8;
                    new_board.pieces[BLACK][PAWN] ^= sq_ind_to_bit(ep_captured_sq);
                    hash ^= ZOBRIST_KEYS.piece_key(BLACK, PAWN, ep_captured_sq);
                } else {
                    let ep_captured_sq = the_move.to + 8;
                    new_board.pieces[WHITE][PAWN] ^= sq_ind_to_bit(ep_captured_sq);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, PAWN, ep_captured_sq);
                }
            }
        }

        // XOR out old en passant key if present
        if let Some(ep_sq) = new_board.en_passant {
            hash ^= ZOBRIST_KEYS.en_passant_key((ep_sq % 8) as usize);
        }

        // Reset the en passant rule.
        new_board.en_passant = None;
        if from_piece.unwrap().1 == PAWN {
            new_board.halfmove_clock = 0;
            if ((the_move.to as i8) - (the_move.from as i8)).abs() == 16 {
                let ep_sq = ((the_move.from + the_move.to) / 2) as u8;
                new_board.en_passant = Some(ep_sq);
                // XOR in new en passant key
                hash ^= ZOBRIST_KEYS.en_passant_key((ep_sq % 8) as usize);
            }
        }

        // Move the piece: XOR out from origin, XOR in at destination
        let (color, piece) = from_piece.unwrap();
        new_board.pieces[color][piece] ^= from_bit;
        new_board.pieces[color][piece] ^= to_bit;
        hash ^= ZOBRIST_KEYS.piece_key(color, piece, the_move.from);
        hash ^= ZOBRIST_KEYS.piece_key(color, piece, the_move.to);

        // Handle promotions
        if let Some(promotion) = the_move.promotion {
            new_board.pieces[color][piece] ^= to_bit;
            new_board.pieces[color][promotion] ^= to_bit;
            // XOR out pawn at destination, XOR in promoted piece
            hash ^= ZOBRIST_KEYS.piece_key(color, piece, the_move.to);
            hash ^= ZOBRIST_KEYS.piece_key(color, promotion, the_move.to);
        }

        // Handle castling
        if let Some((color, KING)) = from_piece {
            if color == WHITE {
                if the_move.from == 4 && the_move.to == 6 {
                    // White king-side castle
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(5);
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(7);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, ROOK, 7);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, ROOK, 5);
                } else if the_move.from == 4 && the_move.to == 2 {
                    // White queen-side castle
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(3);
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(0);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, ROOK, 0);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, ROOK, 3);
                }
                new_board.castling_rights.white_kingside = false;
                new_board.castling_rights.white_queenside = false;
            } else {
                if the_move.from == 60 && the_move.to == 62 {
                    // Black king-side castle
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(61);
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(63);
                    hash ^= ZOBRIST_KEYS.piece_key(BLACK, ROOK, 63);
                    hash ^= ZOBRIST_KEYS.piece_key(BLACK, ROOK, 61);
                } else if the_move.from == 60 && the_move.to == 58 {
                    // Black queen-side castle
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(59);
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(56);
                    hash ^= ZOBRIST_KEYS.piece_key(BLACK, ROOK, 56);
                    hash ^= ZOBRIST_KEYS.piece_key(BLACK, ROOK, 59);
                }
                new_board.castling_rights.black_kingside = false;
                new_board.castling_rights.black_queenside = false;
            }
        } else if let Some((color, ROOK)) = from_piece {
            if color == WHITE {
                if the_move.from == 0 {
                    new_board.castling_rights.white_queenside = false;
                } else if the_move.from == 7 {
                    new_board.castling_rights.white_kingside = false;
                }
            } else if the_move.from == 56 {
                new_board.castling_rights.black_queenside = false;
            } else if the_move.from == 63 {
                new_board.castling_rights.black_kingside = false;
            }
        }

        // Also handle rook captures removing opponent's castling rights
        if to_piece.is_some() {
            match the_move.to {
                0 => new_board.castling_rights.white_queenside = false,
                7 => new_board.castling_rights.white_kingside = false,
                56 => new_board.castling_rights.black_queenside = false,
                63 => new_board.castling_rights.black_kingside = false,
                _ => {}
            }
        }

        // Diff castling rights and XOR changed keys
        if old_castling.white_kingside != new_board.castling_rights.white_kingside {
            hash ^= ZOBRIST_KEYS.castling_key(0);
        }
        if old_castling.white_queenside != new_board.castling_rights.white_queenside {
            hash ^= ZOBRIST_KEYS.castling_key(1);
        }
        if old_castling.black_kingside != new_board.castling_rights.black_kingside {
            hash ^= ZOBRIST_KEYS.castling_key(2);
        }
        if old_castling.black_queenside != new_board.castling_rights.black_queenside {
            hash ^= ZOBRIST_KEYS.castling_key(3);
        }

        // Toggle side to move
        hash ^= ZOBRIST_KEYS.side_to_move_key();
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

        new_board.zobrist_hash = hash;
        debug_assert_eq!(
            new_board.zobrist_hash,
            new_board.compute_zobrist_hash(),
            "Incremental hash mismatch for move {}{}",
            crate::board_utils::sq_ind_to_algebraic(the_move.from),
            crate::board_utils::sq_ind_to_algebraic(the_move.to)
        );

        new_board
    }
}
