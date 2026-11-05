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
        if the_move.from() == 0 && the_move.to() == 0 {
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

        let from_bit = sq_ind_to_bit(the_move.from());
        let to_bit = sq_ind_to_bit(the_move.to());

        let from_piece = self.get_piece(the_move.from());
        if from_piece.is_none() {
            panic!(
                "No piece at from_sq_ind {} ({})",
                the_move.from(),
                crate::board_utils::sq_ind_to_algebraic(the_move.from())
            );
        }

        let to_piece = self.get_piece(the_move.to());
        if to_piece.is_some() {
            // Capture: Remove the captured piece before moving.
            let (cap_color, cap_piece) = to_piece.unwrap();
            new_board.pieces[cap_color][cap_piece] ^= to_bit;
            new_board.halfmove_clock = 0;
            // XOR out captured piece from hash
            hash ^= ZOBRIST_KEYS.piece_key(cap_color, cap_piece, the_move.to());
        }

        // Save old castling rights for diffing
        let old_castling = new_board.castling_rights;

        if from_piece.unwrap().1 == PAWN {
            // En passant capture
            if new_board.en_passant.is_some()
                && the_move.to() == new_board.en_passant.unwrap() as usize
            {
                if new_board.w_to_move {
                    let ep_captured_sq = the_move.to() - 8;
                    new_board.pieces[BLACK][PAWN] ^= sq_ind_to_bit(ep_captured_sq);
                    hash ^= ZOBRIST_KEYS.piece_key(BLACK, PAWN, ep_captured_sq);
                } else {
                    let ep_captured_sq = the_move.to() + 8;
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
            if ((the_move.to() as i8) - (the_move.from() as i8)).abs() == 16 {
                let ep_sq = ((the_move.from() + the_move.to()) / 2) as u8;
                new_board.en_passant = Some(ep_sq);
                // XOR in new en passant key
                hash ^= ZOBRIST_KEYS.en_passant_key((ep_sq % 8) as usize);
            }
        }

        // Move the piece: XOR out from origin, XOR in at destination
        let (color, piece) = from_piece.unwrap();
        new_board.pieces[color][piece] ^= from_bit;
        new_board.pieces[color][piece] ^= to_bit;
        hash ^= ZOBRIST_KEYS.piece_key(color, piece, the_move.from());
        hash ^= ZOBRIST_KEYS.piece_key(color, piece, the_move.to());

        // Handle promotions
        if let Some(promotion) = the_move.promotion() {
            new_board.pieces[color][piece] ^= to_bit;
            new_board.pieces[color][promotion] ^= to_bit;
            // XOR out pawn at destination, XOR in promoted piece
            hash ^= ZOBRIST_KEYS.piece_key(color, piece, the_move.to());
            hash ^= ZOBRIST_KEYS.piece_key(color, promotion, the_move.to());
        }

        // Handle castling
        if let Some((color, KING)) = from_piece {
            if color == WHITE {
                if the_move.from() == 4 && the_move.to() == 6 {
                    // White king-side castle
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(5);
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(7);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, ROOK, 7);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, ROOK, 5);
                } else if the_move.from() == 4 && the_move.to() == 2 {
                    // White queen-side castle
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(3);
                    new_board.pieces[WHITE][ROOK] ^= sq_ind_to_bit(0);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, ROOK, 0);
                    hash ^= ZOBRIST_KEYS.piece_key(WHITE, ROOK, 3);
                }
                new_board.castling_rights.white_kingside = false;
                new_board.castling_rights.white_queenside = false;
            } else {
                if the_move.from() == 60 && the_move.to() == 62 {
                    // Black king-side castle
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(61);
                    new_board.pieces[BLACK][ROOK] ^= sq_ind_to_bit(63);
                    hash ^= ZOBRIST_KEYS.piece_key(BLACK, ROOK, 63);
                    hash ^= ZOBRIST_KEYS.piece_key(BLACK, ROOK, 61);
                } else if the_move.from() == 60 && the_move.to() == 58 {
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
                if the_move.from() == 0 {
                    new_board.castling_rights.white_queenside = false;
                } else if the_move.from() == 7 {
                    new_board.castling_rights.white_kingside = false;
                }
            } else if the_move.from() == 56 {
                new_board.castling_rights.black_queenside = false;
            } else if the_move.from() == 63 {
                new_board.castling_rights.black_kingside = false;
            }
        }

        // Also handle rook captures removing opponent's castling rights
        if to_piece.is_some() {
            match the_move.to() {
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
            crate::board_utils::sq_ind_to_algebraic(the_move.from()),
            crate::board_utils::sq_ind_to_algebraic(the_move.to())
        );

        new_board
    }
}

/// What `Board::make_move_in_place` changed that the move itself cannot
/// recover: the captured piece and the board's counters, flags and hash.
#[derive(Clone, Copy, Debug)]
pub struct UndoInfo {
    /// Colour, piece type and square of the captured piece (for en passant,
    /// not the square moved to).
    captured: Option<(usize, usize, usize)>,
    en_passant: Option<u8>,
    castling_rights: crate::move_types::CastlingRights,
    halfmove_clock: u8,
    fullmove_number: u8,
    zobrist_hash: u64,
}

impl Board {
    fn undo_info(&self) -> UndoInfo {
        UndoInfo {
            captured: None,
            en_passant: self.en_passant,
            castling_rights: self.castling_rights,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
            zobrist_hash: self.zobrist_hash,
        }
    }

    /// The piece type of `color` on `sq`, if any.
    fn piece_of(&self, color: usize, sq: usize) -> Option<usize> {
        let bit = sq_ind_to_bit(sq);
        if self.pieces_occ[color] & bit == 0 {
            return None;
        }
        (0..6).find(|&p| self.pieces[color][p] & bit != 0)
    }

    /// Toggles `piece` of `color` on `sq` in the piece and occupancy boards.
    fn toggle(&mut self, color: usize, piece: usize, sq: usize) {
        let bit = sq_ind_to_bit(sq);
        self.pieces[color][piece] ^= bit;
        self.pieces_occ[color] ^= bit;
    }

    /// Makes `the_move` on this board instead of on a copy: the board becomes
    /// what `apply_move_to_board` returns. Undo it with `unmake_move`.
    pub fn make_move_in_place(&mut self, the_move: Move) -> UndoInfo {
        let mut undo = self.undo_info();
        let (from, to) = (the_move.from(), the_move.to());
        if from == 0 && to == 0 {
            // A null move, as in apply_move_to_board.
            let mut hash = self.zobrist_hash;
            if let Some(ep_sq) = self.en_passant {
                hash ^= ZOBRIST_KEYS.en_passant_key((ep_sq % 8) as usize);
            }
            self.en_passant = None;
            hash ^= ZOBRIST_KEYS.side_to_move_key();
            self.w_to_move = !self.w_to_move;
            if self.w_to_move {
                self.fullmove_number += 1;
            }
            self.zobrist_hash = hash;
            return undo;
        }
        let color = if self.w_to_move { WHITE } else { BLACK };
        let them = 1 - color;
        let piece = self.piece_of(color, from).unwrap_or_else(|| {
            panic!(
                "No piece at from_sq_ind {} ({})",
                from,
                crate::board_utils::sq_ind_to_algebraic(from)
            )
        });
        let mut hash = self.zobrist_hash;
        self.halfmove_clock += 1;

        let captured = self.piece_of(them, to);
        if let Some(cap) = captured {
            self.toggle(them, cap, to);
            self.halfmove_clock = 0;
            hash ^= ZOBRIST_KEYS.piece_key(them, cap, to);
            undo.captured = Some((them, cap, to));
        }
        let old_castling = self.castling_rights;

        if piece == PAWN && self.en_passant == Some(to as u8) {
            let sq = if color == WHITE { to - 8 } else { to + 8 };
            self.toggle(them, PAWN, sq);
            hash ^= ZOBRIST_KEYS.piece_key(them, PAWN, sq);
            undo.captured = Some((them, PAWN, sq));
        }
        if let Some(ep_sq) = self.en_passant {
            hash ^= ZOBRIST_KEYS.en_passant_key((ep_sq % 8) as usize);
        }
        self.en_passant = None;
        if piece == PAWN {
            self.halfmove_clock = 0;
            if from.abs_diff(to) == 16 {
                let ep_sq = ((from + to) / 2) as u8;
                self.en_passant = Some(ep_sq);
                hash ^= ZOBRIST_KEYS.en_passant_key((ep_sq % 8) as usize);
            }
        }

        self.toggle(color, piece, from);
        hash ^= ZOBRIST_KEYS.piece_key(color, piece, from);
        match the_move.promotion() {
            Some(promotion) => {
                self.toggle(color, promotion, to);
                hash ^= ZOBRIST_KEYS.piece_key(color, promotion, to);
            }
            None => {
                self.toggle(color, piece, to);
                hash ^= ZOBRIST_KEYS.piece_key(color, piece, to);
            }
        }

        if piece == KING {
            if let Some((rook_from, rook_to)) = castling_rook(color, from, to) {
                self.toggle(color, ROOK, rook_from);
                self.toggle(color, ROOK, rook_to);
                hash ^= ZOBRIST_KEYS.piece_key(color, ROOK, rook_from);
                hash ^= ZOBRIST_KEYS.piece_key(color, ROOK, rook_to);
            }
            let c = &mut self.castling_rights;
            if color == WHITE {
                c.white_kingside = false;
                c.white_queenside = false;
            } else {
                c.black_kingside = false;
                c.black_queenside = false;
            }
        } else if piece == ROOK {
            let c = &mut self.castling_rights;
            match (color, from) {
                (WHITE, 0) => c.white_queenside = false,
                (WHITE, 7) => c.white_kingside = false,
                (BLACK, 56) => c.black_queenside = false,
                (BLACK, 63) => c.black_kingside = false,
                _ => {}
            }
        }
        if captured.is_some() {
            let c = &mut self.castling_rights;
            match to {
                0 => c.white_queenside = false,
                7 => c.white_kingside = false,
                56 => c.black_queenside = false,
                63 => c.black_kingside = false,
                _ => {}
            }
        }
        let c = self.castling_rights;
        for (i, (old, new)) in [
            (old_castling.white_kingside, c.white_kingside),
            (old_castling.white_queenside, c.white_queenside),
            (old_castling.black_kingside, c.black_kingside),
            (old_castling.black_queenside, c.black_queenside),
        ]
        .into_iter()
        .enumerate()
        {
            if old != new {
                hash ^= ZOBRIST_KEYS.castling_key(i);
            }
        }

        hash ^= ZOBRIST_KEYS.side_to_move_key();
        self.w_to_move = !self.w_to_move;
        if self.w_to_move {
            self.fullmove_number += 1;
        }
        self.zobrist_hash = hash;
        debug_assert_eq!(self.zobrist_hash, self.compute_zobrist_hash());
        undo
    }

    /// Undoes `the_move`, made by `make_move_in_place`, which returned `undo`.
    pub fn unmake_move(&mut self, the_move: Move, undo: UndoInfo) {
        self.w_to_move = !self.w_to_move;
        let (from, to) = (the_move.from(), the_move.to());
        if !(from == 0 && to == 0) {
            let color = if self.w_to_move { WHITE } else { BLACK };
            let moved = match the_move.promotion() {
                Some(promotion) => {
                    self.toggle(color, promotion, to);
                    PAWN
                }
                None => {
                    let p = self.piece_of(color, to).expect("the moved piece");
                    self.toggle(color, p, to);
                    p
                }
            };
            self.toggle(color, moved, from);
            if moved == KING {
                if let Some((rook_from, rook_to)) = castling_rook(color, from, to) {
                    self.toggle(color, ROOK, rook_to);
                    self.toggle(color, ROOK, rook_from);
                }
            }
            if let Some((c, p, sq)) = undo.captured {
                self.toggle(c, p, sq);
            }
        }
        self.en_passant = undo.en_passant;
        self.castling_rights = undo.castling_rights;
        self.halfmove_clock = undo.halfmove_clock;
        self.fullmove_number = undo.fullmove_number;
        self.zobrist_hash = undo.zobrist_hash;
    }

    /// Passes the move, as `BoardStack::make_null_move_without_history` does
    /// (the move number is not advanced). Undo it with `unmake_null_move`.
    pub fn make_null_move_in_place(&mut self) -> UndoInfo {
        let undo = self.undo_info();
        let mut hash = self.zobrist_hash ^ ZOBRIST_KEYS.side_to_move_key();
        if let Some(ep_sq) = self.en_passant {
            hash ^= ZOBRIST_KEYS.en_passant_key((ep_sq % 8) as usize);
        }
        self.w_to_move = !self.w_to_move;
        self.en_passant = None;
        self.zobrist_hash = hash;
        debug_assert_eq!(self.zobrist_hash, self.compute_zobrist_hash());
        undo
    }

    /// Undoes `make_null_move_in_place`.
    pub fn unmake_null_move(&mut self, undo: UndoInfo) {
        self.w_to_move = !self.w_to_move;
        self.en_passant = undo.en_passant;
        self.zobrist_hash = undo.zobrist_hash;
    }
}

/// The rook's squares (from, to) if a king move from `from` to `to` castles.
fn castling_rook(color: usize, from: usize, to: usize) -> Option<(usize, usize)> {
    match (color, from, to) {
        (WHITE, 4, 6) => Some((7, 5)),
        (WHITE, 4, 2) => Some((0, 3)),
        (BLACK, 60, 62) => Some((63, 61)),
        (BLACK, 60, 58) => Some((56, 59)),
        _ => None,
    }
}
