//! Move generation module.
//!
//! This module provides functions for generating pseudo-legal moves for a given
//! chess position. The moves are generated using a combination of bitboards and
//! precomputed tables.
//!
//! The main entry point for move generation is the `gen_pseudo_legal_moves`
//! function, which generates all pseudo-legal moves for a given position.
//!
//! The `gen_pseudo_legal_captures` function generates only the capture moves.
//!
//! The `gen_pawn_moves`, `gen_knight_moves`, `gen_bishop_moves`, `gen_rook_moves`,
//! `gen_queen_moves`, and `gen_king_moves` functions generate moves for specific
//! piece types.

use crate::bits::bits;
use crate::board::Board;
use crate::board_utils::sq_ind_to_bit;
use crate::magic_bitboard::{
    append_promotions, init_bishop_moves, init_king_moves, init_knight_moves,
    init_pawn_captures_promotions, init_pawn_moves, init_rook_moves,
};
use crate::magic_constants::{B_BITS, B_MAGICS, B_MASKS, R_BITS, R_MAGICS, R_MASKS};
use crate::move_types::Move;

use crate::eval::PestoEval;
use crate::piece_types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use crate::search::HistoryTable;

// Define mask constants for mobility calculations
const FILE_MASKS: [u64; 64] = generate_file_masks();
const RANK_MASKS: [u64; 64] = generate_rank_masks();
const DIAG_MASKS: [u64; 64] = generate_diag_masks();
const ANTI_DIAG_MASKS: [u64; 64] = generate_anti_diag_masks();

// Helper function to generate file masks
const fn generate_file_masks() -> [u64; 64] {
    let mut masks = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let file = sq % 8;
        let mut mask = 0;
        let mut i = 0;
        while i < 8 {
            mask |= 1u64 << (file + i * 8);
            i += 1;
        }
        masks[sq] = mask & !(1u64 << sq); // Exclude the square itself
        sq += 1;
    }
    masks
}

// Helper function to generate rank masks
const fn generate_rank_masks() -> [u64; 64] {
    let mut masks = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let rank = sq / 8;
        let mask = 0xFF_u64 << (rank * 8);
        masks[sq] = mask & !(1u64 << sq); // Exclude the square itself
        sq += 1;
    }
    masks
}

// Helper function to generate diagonal masks
const fn generate_diag_masks() -> [u64; 64] {
    let mut masks = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let mut mask = 0_u64;

        // Add squares to the NW
        let mut f = file;
        let mut r = rank;
        while f > 0 && r > 0 {
            f -= 1;
            r -= 1;
            mask |= 1_u64 << (f + r * 8);
        }

        // Add squares to the SE
        f = file;
        r = rank;
        while f < 7 && r < 7 {
            f += 1;
            r += 1;
            mask |= 1_u64 << (f + r * 8);
        }

        masks[sq] = mask;
        sq += 1;
    }
    masks
}

// Helper function to generate anti-diagonal masks
const fn generate_anti_diag_masks() -> [u64; 64] {
    let mut masks = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let mut mask = 0_u64;

        // Add squares to the NE
        let mut f = file;
        let mut r = rank;
        while f < 7 && r > 0 {
            f += 1;
            r -= 1;
            mask |= 1_u64 << (f + r * 8);
        }

        // Add squares to the SW
        f = file;
        r = rank;
        while f > 0 && r < 7 {
            f -= 1;
            r += 1;
            mask |= 1_u64 << (f + r * 8);
        }

        masks[sq] = mask;
        sq += 1;
    }
    masks
}

/// Represents the move generator, which generates pseudo-legal moves.
///
/// This struct contains precomputed tables and bitboards used for move generation.
pub struct MoveGen {
    /// Precomputed tables for pawn captures.
    pub wp_captures: Vec<Vec<usize>>,
    /// Precomputed tables for pawn captures.
    pub bp_captures: Vec<Vec<usize>>,
    /// Bitboards for pawn captures.
    pub wp_capture_bitboard: [u64; 64],
    /// Bitboards for pawn captures.
    pub bp_capture_bitboard: [u64; 64],
    /// Precomputed tables for pawn promotions.
    _wp_promotions: Vec<Vec<usize>>,
    /// Precomputed tables for pawn promotions.
    _bp_promotions: Vec<Vec<usize>>,
    /// Precomputed tables for knight moves.
    pub n_moves: Vec<Vec<usize>>,
    /// Precomputed tables for king moves.
    pub k_moves: Vec<Vec<usize>>,
    /// Bitboards for knight moves.
    pub n_move_bitboard: [u64; 64],
    /// Bitboards for king moves.
    pub k_move_bitboard: [u64; 64],
    /// Precomputed tables for pawn moves.
    wp_moves: Vec<Vec<usize>>,
    /// Precomputed tables for pawn moves.
    bp_moves: Vec<Vec<usize>>,
    /// Bitboards for rook moves.
    r_move_bitboard: Vec<Vec<u64>>,
    /// Bitboards for bishop moves.
    b_move_bitboard: Vec<Vec<u64>>,
    /// Magic numbers for bishop moves.
    b_magics: [u64; 64],
    /// Magic numbers for rook moves.
    r_magics: [u64; 64],
}

impl Default for MoveGen {
    fn default() -> Self {
        Self::new()
    }
}

/// Room for every move of a position: the most found in any chess position is 218.
pub const MAX_MOVES: usize = 256;

/// Starting capacity of each move list: `MAX_MOVES` when `KF_RESERVE_MOVES=1`, else 0
/// (lists grow as needed). Reserving was 5.8% slower in self-play (282 versus 299
/// moves per second at 480 threads), so it is off by default.
pub fn move_list_capacity(setting: Option<&str>) -> usize {
    match setting {
        Some("1") => MAX_MOVES,
        Some("0") | None => 0,
        Some(v) => panic!("KF_RESERVE_MOVES must be 0 or 1, not {v:?}"),
    }
}

fn reserved_moves() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| move_list_capacity(std::env::var("KF_RESERVE_MOVES").ok().as_deref()))
}

impl MoveGen {
    /// Creates a new `MoveGen` instance.
    ///
    /// This function initializes the precomputed tables and bitboards used for
    /// move generation.
    ///
    /// # Returns
    ///
    /// A new `MoveGen` instance.
    pub fn new() -> MoveGen {
        // Initialize the move generator by creating the iterators for Pawn, Knight, and King moves.
        let mut wp_captures: Vec<Vec<usize>> = Vec::new();
        let mut bp_captures: Vec<Vec<usize>> = Vec::new();
        let mut wp_capture_bitboard: [u64; 64] = [0; 64];
        let mut bp_capture_bitboard: [u64; 64] = [0; 64];
        let mut wp_promotions: Vec<Vec<usize>> = Vec::new();
        let mut bp_promotions: Vec<Vec<usize>> = Vec::new();
        let mut n_moves: Vec<Vec<usize>> = Vec::new();
        let mut k_moves: Vec<Vec<usize>> = Vec::new();
        let mut n_move_bitboard: [u64; 64] = [0; 64];
        let mut k_move_bitboard: [u64; 64] = [0; 64];
        let mut wp_moves: Vec<Vec<usize>> = Vec::new();
        let mut bp_moves: Vec<Vec<usize>> = Vec::new();
        let mut _wp: Vec<usize>;
        let mut _bp: Vec<usize>;
        let mut _wp_cap: Vec<usize>;
        let mut _bp_cap: Vec<usize>;
        let mut _wp_prom: Vec<usize>;
        let mut _bp_prom: Vec<usize>;
        for from_sq_ind in 0..64 {
            let (wp_cap, wp_prom, bp_cap, bp_prom) = init_pawn_captures_promotions(from_sq_ind);
            wp_captures.push(wp_cap.clone());
            bp_captures.push(bp_cap.clone());
            for i in &wp_captures[from_sq_ind] {
                wp_capture_bitboard[from_sq_ind] |= sq_ind_to_bit(*i);
            }
            // Also add pawn moves from pawns on the first rank, as this is used in reverse to determine checks by pawns
            if from_sq_ind < 8 {
                if from_sq_ind < 7 {
                    wp_capture_bitboard[from_sq_ind] |= sq_ind_to_bit(from_sq_ind + 9);
                }
                if from_sq_ind > 0 {
                    wp_capture_bitboard[from_sq_ind] |= sq_ind_to_bit(from_sq_ind + 7);
                }
            }
            for i in &bp_captures[from_sq_ind] {
                bp_capture_bitboard[from_sq_ind] |= sq_ind_to_bit(*i);
            }
            // Also add pawn moves from pawns on the first rank, as this is used in reverse to determine checks by pawns
            if from_sq_ind > 55 {
                if from_sq_ind > 56 {
                    bp_capture_bitboard[from_sq_ind] |= sq_ind_to_bit(from_sq_ind - 9);
                }
                if from_sq_ind < 63 {
                    bp_capture_bitboard[from_sq_ind] |= sq_ind_to_bit(from_sq_ind - 7);
                }
            }
            wp_promotions.push(wp_prom.clone());
            bp_promotions.push(bp_prom.clone());
            n_moves.push(init_knight_moves(from_sq_ind));
            k_moves.push(init_king_moves(from_sq_ind));
            for i in &n_moves[from_sq_ind] {
                n_move_bitboard[from_sq_ind] |= sq_ind_to_bit(*i);
            }
            for i in &k_moves[from_sq_ind] {
                k_move_bitboard[from_sq_ind] |= sq_ind_to_bit(*i);
            }
            let (wp, bp) = init_pawn_moves(from_sq_ind);
            wp_moves.push(wp.clone());
            bp_moves.push(bp.clone());
        }
        let mut move_gen = MoveGen {
            wp_captures,
            bp_captures,
            wp_capture_bitboard,
            bp_capture_bitboard,
            _wp_promotions: Vec::new(),
            _bp_promotions: Vec::new(),
            n_moves,
            k_moves,
            n_move_bitboard,
            k_move_bitboard,
            wp_moves,
            bp_moves,
            r_move_bitboard: vec![],
            b_move_bitboard: vec![],
            b_magics: [0; 64],
            r_magics: [0; 64],
        };
        // Generate magic numbers for sliding pieces
        // let (b_magics, r_magics) = find_magic_numbers();
        move_gen.b_magics = B_MAGICS;
        move_gen.r_magics = R_MAGICS;
        move_gen.b_move_bitboard = init_bishop_moves(move_gen.b_magics);
        move_gen.r_move_bitboard = init_rook_moves(move_gen.r_magics);
        move_gen
    }

    /// Access the rook attack bitboard table (for GPU table export).
    /// Returns a reference to the jagged array: [square][magic_key] -> attack bitboard.
    pub fn rook_attack_table(&self) -> &Vec<Vec<u64>> {
        &self.r_move_bitboard
    }

    /// Access the bishop attack bitboard table (for GPU table export).
    pub fn bishop_attack_table(&self) -> &Vec<Vec<u64>> {
        &self.b_move_bitboard
    }

    /// Internal helper to generate all pseudo-legal moves, separated.
    fn _generate_all_moves(&self, board: &Board) -> (Vec<Move>, Vec<Move>, Vec<Move>) {
        // Returns (captures, promotions, quiet_moves); each list keeps piece order
        // pawn, knight, bishop, rook, queen, king.
        // Promotions are rare and are appended to `captures` by callers.
        let cap = reserved_moves();
        let mut captures = Vec::with_capacity(cap);
        let mut promotions = Vec::new();
        let mut moves = Vec::with_capacity(cap);
        self.gen_all_moves_into(board, &mut captures, &mut promotions, &mut moves);
        (captures, promotions, moves)
    }

    /// Appends every pseudo-legal move to `captures`, `promotions` and `moves`,
    /// in the order `_generate_all_moves` returns them.
    pub(crate) fn gen_all_moves_into(
        &self,
        board: &Board,
        captures: &mut Vec<Move>,
        promotions: &mut Vec<Move>,
        moves: &mut Vec<Move>,
    ) {
        self.gen_pawn_moves_into(board, captures, promotions, moves);
        self.gen_knight_moves_into(board, captures, moves);
        self.gen_bishop_moves_into(board, captures, moves);
        self.gen_rook_moves_into(board, captures, moves);
        self.gen_queen_moves_into(board, captures, moves);
        self.gen_king_moves_into(board, captures, moves);
    }

    /// Appends every pseudo-legal move that gives check to `out`: exactly the
    /// moves `Board::gives_check` accepts, without generating the others.
    ///
    /// Direct checks are found by intersecting each piece's targets with the
    /// squares from which that piece type attacks the enemy king. Moves by a
    /// piece that may uncover a check from one of our sliders, promotions,
    /// en passant and castling are rare, so those are generated and tested
    /// with `gives_check`.
    pub fn gen_checking_moves_into(&self, board: &Board, out: &mut Vec<Move>) {
        let us = if board.w_to_move { WHITE } else { BLACK };
        let them = 1 - us;
        let their_king = board.pieces[them][KING];
        if their_king == 0 {
            return;
        }
        let ksq = their_king.trailing_zeros() as usize;
        let occ = board.pieces_occ[WHITE] | board.pieces_occ[BLACK];
        let own = board.pieces_occ[us];
        let ours = &board.pieces[us];

        // Our pieces that block one of our sliders from the king.
        let mut uncovers = 0u64;
        let rook_line = self.gen_rook_attacks_occ(ksq, occ);
        let bishop_line = self.gen_bishop_attacks_occ(ksq, occ);
        for (line, sliders, attacks) in [
            (
                rook_line,
                ours[ROOK] | ours[QUEEN],
                Self::gen_rook_attacks_occ as fn(&Self, usize, u64) -> u64,
            ),
            (
                bishop_line,
                ours[BISHOP] | ours[QUEEN],
                Self::gen_bishop_attacks_occ,
            ),
        ] {
            if sliders == 0 {
                continue;
            }
            for sq in bits(&(line & own)) {
                let bit = 1u64 << sq;
                if attacks(self, ksq, occ & !bit) & sliders & !bit != 0 {
                    uncovers |= bit;
                }
            }
        }

        let push_tested = |out: &mut Vec<Move>, from: usize, targets: u64| {
            for to in bits(&targets) {
                let m = Move::new(from, to, None);
                if board.gives_check(m, self) {
                    out.push(m);
                }
            }
        };
        let push_all = |out: &mut Vec<Move>, from: usize, targets: u64| {
            for to in bits(&targets) {
                out.push(Move::new(from, to, None));
            }
        };

        // Knights, bishops, rooks and queens.
        let knight_checks = self.n_move_bitboard[ksq];
        for from in bits(&ours[KNIGHT]) {
            let t = self.n_move_bitboard[from] & !own;
            if uncovers & (1u64 << from) != 0 {
                push_tested(out, from, t);
            } else {
                push_all(out, from, t & knight_checks);
            }
        }
        for (piece, checks) in [
            (BISHOP, bishop_line),
            (ROOK, rook_line),
            (QUEEN, bishop_line | rook_line),
        ] {
            for from in bits(&ours[piece]) {
                let reach = match piece {
                    BISHOP => self.gen_bishop_attacks_occ(from, occ),
                    ROOK => self.gen_rook_attacks_occ(from, occ),
                    _ => {
                        self.gen_bishop_attacks_occ(from, occ)
                            | self.gen_rook_attacks_occ(from, occ)
                    }
                };
                let t = reach & !own;
                if uncovers & (1u64 << from) != 0 {
                    push_tested(out, from, t);
                } else {
                    push_all(out, from, t & checks);
                }
            }
        }

        // King: only uncovered checks, and castling (below).
        for from in bits(&ours[KING]) {
            if uncovers & (1u64 << from) != 0 {
                push_tested(out, from, self.k_move_bitboard[from] & !own);
            }
        }

        // Pawns not about to promote: pushes and ordinary captures.
        let (last_rank, start_rank, pawn_checks) = if us == WHITE {
            (
                0xFF00_0000_0000_0000u64 >> 8,
                0xFF00u64,
                self.bp_capture_bitboard[ksq],
            )
        } else {
            (
                0xFF00u64,
                0xFF00_0000_0000_0000u64 >> 8,
                self.wp_capture_bitboard[ksq],
            )
        };
        for from in bits(&(ours[PAWN] & !last_rank)) {
            let (one, two, caps) = if us == WHITE {
                (from + 8, from + 16, self.wp_capture_bitboard[from])
            } else {
                (
                    from - 8,
                    from.wrapping_sub(16),
                    self.bp_capture_bitboard[from],
                )
            };
            let mut t = caps & board.pieces_occ[them];
            if occ & (1u64 << one) == 0 {
                t |= 1u64 << one;
                if (1u64 << from) & start_rank != 0 && occ & (1u64 << two) == 0 {
                    t |= 1u64 << two;
                }
            }
            if uncovers & (1u64 << from) != 0 {
                push_tested(out, from, t);
            } else {
                push_all(out, from, t & pawn_checks);
            }
        }

        // Promotions, en passant and castling, tested one by one.
        let castling = &board.castling_rights;
        let may_castle = if us == WHITE {
            castling.white_kingside || castling.white_queenside
        } else {
            castling.black_kingside || castling.black_queenside
        };
        if ours[PAWN] & last_rank != 0 || board.en_passant.is_some() || may_castle {
            let (mut captures, mut promotions, mut quiet) = (Vec::new(), Vec::new(), Vec::new());
            if ours[PAWN] & last_rank != 0 || board.en_passant.is_some() {
                self.gen_pawn_moves_into(board, &mut captures, &mut promotions, &mut quiet);
            }
            if may_castle {
                self.gen_king_moves_into(board, &mut captures, &mut quiet);
            }
            let ep = board.en_passant.map(|e| e as usize);
            for m in captures.into_iter().chain(promotions).chain(quiet) {
                let special = m.promotion().is_some()
                    || m.is_castle()
                    || (Some(m.to()) == ep && ours[PAWN] & (1u64 << m.from()) != 0);
                if special && board.gives_check(m, self) {
                    out.push(m);
                }
            }
        }
    }

    /// Appends the candidate moves out of check for the side to move, which
    /// must be in check: king steps, and with a single checker, moves that
    /// capture it or land between it and the king. Filtered for legality,
    /// they are exactly the legal moves. Castling is never legal in check.
    /// Captures come first, then blocks, then king steps, pawns first within
    /// each, so a refutation of a mating attempt tends to come early.
    pub fn gen_evasions_into(&self, board: &Board, out: &mut Vec<Move>) {
        self.visit_evasions(board, false, &mut |m| {
            out.push(m);
            false
        });
    }

    /// True if the side to move, which must be in check, has a legal move.
    /// Tests each candidate as it is generated, king steps first, and stops
    /// at the first legal one.
    pub fn has_legal_evasion(&self, board: &Board) -> bool {
        self.visit_evasions(board, true, &mut |m| board.is_legal_after_move(m, self))
    }

    /// Calls `f` on each candidate move out of check until it returns true;
    /// returns whether it did. King steps come first if `king_first`, last
    /// otherwise.
    fn visit_evasions(
        &self,
        board: &Board,
        king_first: bool,
        f: &mut impl FnMut(Move) -> bool,
    ) -> bool {
        let us = if board.w_to_move { WHITE } else { BLACK };
        let them = 1 - us;
        let ours = &board.pieces[us];
        let theirs = &board.pieces[them];
        let king = ours[KING];
        if king == 0 {
            return false;
        }
        let ksq = king.trailing_zeros() as usize;
        let occ = board.pieces_occ[WHITE] | board.pieces_occ[BLACK];
        let own = board.pieces_occ[us];
        let king_steps = self.k_move_bitboard[ksq] & !own;
        let each = |f: &mut dyn FnMut(Move) -> bool, from: usize, targets: u64| {
            bits(&targets).any(|to| f(Move::new(from, to, None)))
        };
        let king_captures = king_steps & board.pieces_occ[them];
        if king_first && each(f, ksq, king_steps) {
            return true;
        }

        let rook_line = self.gen_rook_attacks_occ(ksq, occ);
        let bishop_line = self.gen_bishop_attacks_occ(ksq, occ);
        let pawn_attackers = if us == WHITE {
            self.wp_capture_bitboard[ksq]
        } else {
            self.bp_capture_bitboard[ksq]
        };
        let rook_checkers = rook_line & (theirs[ROOK] | theirs[QUEEN]);
        let bishop_checkers = bishop_line & (theirs[BISHOP] | theirs[QUEEN]);
        let checkers = (self.n_move_bitboard[ksq] & theirs[KNIGHT])
            | (pawn_attackers & theirs[PAWN])
            | rook_checkers
            | bishop_checkers;
        if checkers.count_ones() == 1 {
            let csq = checkers.trailing_zeros() as usize;
            // The squares between the checker and the king.
            let between = if rook_checkers != 0 {
                rook_line & self.gen_rook_attacks_occ(csq, occ)
            } else if bishop_checkers != 0 {
                bishop_line & self.gen_bishop_attacks_occ(csq, occ)
            } else {
                0
            };
            let (last_rank, start_rank) = if us == WHITE {
                (0x00FF_0000_0000_0000u64, 0xFF00u64)
            } else {
                (0xFF00u64, 0x00FF_0000_0000_0000u64)
            };
            let ep = board.en_passant.map(|e| e as usize);
            let pawn_moves = |f: &mut dyn FnMut(Move) -> bool, from: usize, targets: u64| {
                bits(&targets).any(|to| {
                    if (1u64 << from) & last_rank != 0 {
                        [QUEEN, ROOK, KNIGHT, BISHOP]
                            .into_iter()
                            .any(|p| f(Move::new(from, to, Some(p))))
                    } else {
                        f(Move::new(from, to, None))
                    }
                })
            };
            let reach = |piece: usize, from: usize| match piece {
                KNIGHT => self.n_move_bitboard[from],
                BISHOP => self.gen_bishop_attacks_occ(from, occ),
                ROOK => self.gen_rook_attacks_occ(from, occ),
                _ => self.gen_bishop_attacks_occ(from, occ) | self.gen_rook_attacks_occ(from, occ),
            };

            // Captures of the checker, en passant included.
            for from in bits(&ours[PAWN]) {
                let caps = if us == WHITE {
                    self.wp_capture_bitboard[from]
                } else {
                    self.bp_capture_bitboard[from]
                };
                if pawn_moves(f, from, caps & checkers) {
                    return true;
                }
                if let Some(e) = ep {
                    let captured = if us == WHITE { e - 8 } else { e + 8 };
                    if caps & (1u64 << e) != 0
                        && (csq == captured || between & (1u64 << e) != 0)
                        && f(Move::new(from, e, None))
                    {
                        return true;
                    }
                }
            }
            for piece in [KNIGHT, BISHOP, ROOK, QUEEN] {
                for from in bits(&ours[piece]) {
                    if each(f, from, reach(piece, from) & checkers) {
                        return true;
                    }
                }
            }
            if !king_first && each(f, ksq, king_captures) {
                return true;
            }

            // Blocks.
            if between != 0 {
                for from in bits(&ours[PAWN]) {
                    let one = if us == WHITE { from + 8 } else { from - 8 };
                    if occ & (1u64 << one) != 0 {
                        continue;
                    }
                    let mut t = (1u64 << one) & between;
                    if (1u64 << from) & start_rank != 0 {
                        let two = if us == WHITE { from + 16 } else { from - 16 };
                        if occ & (1u64 << two) == 0 {
                            t |= (1u64 << two) & between;
                        }
                    }
                    if pawn_moves(f, from, t) {
                        return true;
                    }
                }
                for piece in [KNIGHT, BISHOP, ROOK, QUEEN] {
                    for from in bits(&ours[piece]) {
                        if each(f, from, reach(piece, from) & between) {
                            return true;
                        }
                    }
                }
            }
        } else if !king_first && each(f, ksq, king_captures) {
            return true;
        }
        !king_first && each(f, ksq, king_steps & !king_captures)
    }

    /// Generates all pseudo-legal moves for a given position.
    ///
    /// This function generates all pseudo-legal moves for the given position,
    /// including captures and non-captures. Promotions are included in the captures list.
    ///
    /// # Arguments
    ///
    /// * `board` - The current chess position.
    ///
    /// # Returns
    ///
    /// A tuple containing the capture moves (including promotions) and non-capture moves.
    pub fn gen_pseudo_legal_moves(&self, board: &Board) -> (Vec<Move>, Vec<Move>) {
        let (mut captures, mut promotions, moves) = self._generate_all_moves(board);
        captures.append(&mut promotions); // Combine promotions with captures
        (captures, moves)
    }

    /// Generates all pseudo-legal moves, sorted for search efficiency.
    ///
    /// Generates captures (including promotions) sorted by MVV-LVA, and
    /// quiet moves sorted first by history score and then by evaluation score.
    ///
    /// # Arguments
    ///
    /// * `board` - The current chess position.
    /// * `pesto` - The evaluation function instance for move scoring.
    /// * `history` - Optional history table for move ordering.
    ///
    /// # Returns
    ///
    /// A tuple containing the sorted capture moves and sorted non-capture moves.
    pub fn gen_pseudo_legal_moves_with_evals(
        &self,
        board: &Board,
        pesto: &PestoEval,
        history: Option<&HistoryTable>,
    ) -> (Vec<Move>, Vec<Move>) {
        // Get all moves
        let (captures, non_captures) = self.gen_pseudo_legal_moves(board);

        // Nothing to do if captures or non_captures are empty
        if captures.is_empty() && non_captures.is_empty() {
            return (captures, non_captures);
        }

        // Sort captures by MVV-LVA
        let mut captures_with_eval: Vec<(Move, i32)> = captures
            .into_iter()
            .map(|m| (m, self.mvv_lva(board, m.from(), m.to())))
            .collect();
        captures_with_eval.sort_by_key(|b| std::cmp::Reverse(b.1));
        let sorted_captures = captures_with_eval.into_iter().map(|(m, _)| m).collect();

        // Sort non-captures by history score if available, then by evaluation
        let mut non_captures_with_eval: Vec<(Move, i32, i32)> = non_captures
            .into_iter()
            .map(|m| {
                // Get history score (if history table provided)
                let history_score =
                    history.map_or(0, |h| h.get_score_from_squares(m.from(), m.to()));
                // Get evaluation score
                let eval_score = pesto.move_eval(board, self, m.from(), m.to());
                (m, history_score, eval_score)
            })
            .collect();

        // First sort by history score (unstable sort)
        if history.is_some() {
            non_captures_with_eval.sort_by_key(|b| std::cmp::Reverse(b.1));
        }

        // Then sort by evaluation (stable sort)
        non_captures_with_eval.sort_by_key(|b| std::cmp::Reverse(b.2));

        let sorted_non_captures = non_captures_with_eval
            .into_iter()
            .map(|(m, _, _)| m)
            .collect();

        (sorted_captures, sorted_non_captures)
    }

    /// Backward-compatible version of gen_pseudo_legal_moves_with_evals that doesn't use history table
    #[deprecated(
        since = "0.2.0",
        note = "Use gen_pseudo_legal_moves_with_evals with history parameter instead"
    )]
    pub fn gen_pseudo_legal_moves_with_evals_no_history(
        &self,
        board: &Board,
        pesto: &PestoEval,
    ) -> (Vec<Move>, Vec<Move>) {
        self.gen_pseudo_legal_moves_with_evals(board, pesto, None)
    }

    /// Generates only the capture moves for a given position.
    ///
    /// This function generates only the capture moves for the given position.
    ///
    /// # Arguments
    ///
    /// * `board` - The current chess position.
    ///
    /// # Returns
    ///
    /// A vector of capture moves.
    /// Generates only the capture and promotion moves, sorted by MVV-LVA.
    ///
    /// # Arguments
    ///
    /// * `board` - The current chess position.
    ///
    /// # Returns
    ///
    /// A vector of capture and promotion moves, sorted by MVV-LVA (descending).
    pub fn gen_pseudo_legal_captures(&self, board: &Board) -> Vec<Move> {
        let (mut captures, mut promotions, _moves) = self._generate_all_moves(board);
        captures.append(&mut promotions); // Combine promotions with captures

        // Sort captures by MVV-LVA (descending)
        captures.sort_unstable_by_key(|m| -self.mvv_lva(board, m.from(), m.to()));

        captures
    }

    pub fn mvv_lva(&self, board: &Board, from_sq_ind: usize, to_sq_ind: usize) -> i32 {
        // Return the MVV-LVA score for a capture move.
        // To enable sorting by MVV, then by LVA, we return the score as 10 * victim - attacker,
        // where value is 012345 for kpnbrq
        if board.get_piece(to_sq_ind).is_none() {
            return 0;
        }
        let victim = board.get_piece(to_sq_ind).unwrap().1;
        let attacker = board.get_piece(from_sq_ind).unwrap().1;
        10 * victim as i32 - attacker as i32
    }

    /// Generates moves for a pawn on a specific square.
    ///
    /// This function generates moves for a pawn on the given square, including
    /// captures and non-captures.
    ///
    /// # Arguments
    ///
    /// * `board` - The current chess position.
    /// * `from_sq_ind` - The index of the square the pawn is on (0-63).
    ///
    /// # Returns
    ///
    /// A tuple containing the capture moves and non-capture moves for the pawn.
    pub fn gen_pawn_moves(&self, board: &Board) -> (Vec<Move>, Vec<Move>, Vec<Move>) {
        let (mut captures, mut promotions, mut moves) = (Vec::new(), Vec::new(), Vec::new());
        self.gen_pawn_moves_into(board, &mut captures, &mut promotions, &mut moves);
        (captures, promotions, moves)
    }

    /// As `gen_pawn_moves`, appending to the given vectors.
    pub(crate) fn gen_pawn_moves_into(
        &self,
        board: &Board,
        captures: &mut Vec<Move>,
        promotions: &mut Vec<Move>,
        moves: &mut Vec<Move>,
    ) {
        if board.w_to_move {
            // White to move
            for from_sq_ind in bits(&board.pieces[WHITE][PAWN]) {
                let is_promotion_rank = from_sq_ind > 47 && from_sq_ind < 56;

                // Handle captures and en passant
                for to_sq_ind in &self.wp_captures[from_sq_ind] {
                    if board.pieces_occ[BLACK] & (1u64 << to_sq_ind) != 0
                        || board.en_passant == Some(*to_sq_ind as u8)
                    {
                        if is_promotion_rank {
                            append_promotions(promotions, from_sq_ind, to_sq_ind, board.w_to_move);
                        } else {
                            captures.push(Move::new(from_sq_ind, *to_sq_ind, None));
                        }
                    }
                }

                // Handle regular moves and promotions
                for to_sq_ind in &self.wp_moves[from_sq_ind] {
                    if board.get_piece(*to_sq_ind).is_none() {
                        if is_promotion_rank {
                            append_promotions(promotions, from_sq_ind, to_sq_ind, board.w_to_move);
                        } else if from_sq_ind > 7 && from_sq_ind < 16 {
                            // Double pawn push
                            if (board.pieces_occ[BLACK] + board.pieces_occ[WHITE])
                                & (1u64 << (from_sq_ind + 8))
                                == 0
                            {
                                moves.push(Move::new(from_sq_ind, *to_sq_ind, None));
                            }
                        } else {
                            moves.push(Move::new(from_sq_ind, *to_sq_ind, None));
                        }
                    }
                }
            }
        } else {
            // Black to move (similar logic, but for black pawns)
            for from_sq_ind in bits(&board.pieces[BLACK][PAWN]) {
                let is_promotion_rank = from_sq_ind > 7 && from_sq_ind < 16;

                // Handle captures and en passant
                for to_sq_ind in &self.bp_captures[from_sq_ind] {
                    if board.pieces_occ[WHITE] & (1u64 << to_sq_ind) != 0
                        || board.en_passant == Some(*to_sq_ind as u8)
                    {
                        if is_promotion_rank {
                            append_promotions(promotions, from_sq_ind, to_sq_ind, board.w_to_move);
                        } else {
                            captures.push(Move::new(from_sq_ind, *to_sq_ind, None));
                        }
                    }
                }

                // Handle regular moves and promotions
                for to_sq_ind in &self.bp_moves[from_sq_ind] {
                    if board.get_piece(*to_sq_ind).is_none() {
                        if is_promotion_rank {
                            append_promotions(promotions, from_sq_ind, to_sq_ind, board.w_to_move);
                        } else if from_sq_ind > 47 && from_sq_ind < 56 {
                            // Double pawn push
                            if (board.pieces_occ[WHITE] + board.pieces_occ[BLACK])
                                & (1u64 << (from_sq_ind - 8))
                                == 0
                            {
                                moves.push(Move::new(from_sq_ind, *to_sq_ind, None));
                            }
                        } else {
                            moves.push(Move::new(from_sq_ind, *to_sq_ind, None));
                        }
                    }
                }
            }
        }
    }

    /// Appends the pseudo-legal knight moves: captures to `captures`, others to `moves`.
    pub(crate) fn gen_knight_moves_into(
        &self,
        board: &Board,
        captures: &mut Vec<Move>,
        moves: &mut Vec<Move>,
    ) {
        // Generate all possible knight moves for the current position.
        // Returns a vector of captures and a vector of non-captures.
        if board.w_to_move {
            // White to move
            for from_sq_ind in bits(&board.pieces[WHITE][KNIGHT]) {
                for to_sq_ind in &self.n_moves[from_sq_ind] {
                    if board.pieces_occ[BLACK] & (1u64 << to_sq_ind) != 0 {
                        captures.push(Move::new(from_sq_ind, *to_sq_ind, None));
                    } else if board.pieces_occ[WHITE] & (1u64 << to_sq_ind) == 0 {
                        moves.push(Move::new(from_sq_ind, *to_sq_ind, None));
                    }
                }
            }
        } else {
            // Black to move
            for from_sq_ind in bits(&board.pieces[BLACK][KNIGHT]) {
                for to_sq_ind in &self.n_moves[from_sq_ind] {
                    if board.pieces_occ[WHITE] & (1u64 << to_sq_ind) != 0 {
                        captures.push(Move::new(from_sq_ind, *to_sq_ind, None));
                    } else if board.pieces_occ[BLACK] & (1u64 << to_sq_ind) == 0 {
                        moves.push(Move::new(from_sq_ind, *to_sq_ind, None));
                    }
                }
            }
        }
    }

    /// Generates moves for a king on a specific square.
    ///
    /// This function generates moves for a king on the given square, including
    /// captures and non-captures.
    ///
    /// # Arguments
    ///
    /// * `board` - The current chess position.
    ///
    /// # Returns
    ///
    /// A tuple containing the capture moves and non-capture moves for the king.
    pub(crate) fn gen_king_moves_into(
        &self,
        board: &Board,
        captures: &mut Vec<Move>,
        moves: &mut Vec<Move>,
    ) {
        // Generate all possible king moves for the current position.
        // For castling, checks whether in check and whether the king moves through check.
        // Returns a vector of captures and a vector of non-captures.
        if board.w_to_move {
            // White to move
            if board.castling_rights.white_kingside && board.pieces[WHITE][KING] & (1u64 << 4) != 0
            {
                // Make sure a rook is there because it could have been captured
                if board.pieces[WHITE][ROOK] & (1u64 << 7) != 0
                    && (board.pieces_occ[WHITE] | board.pieces_occ[BLACK])
                        & ((1u64 << 5) | (1u64 << 6))
                        == 0
                    && !board.is_square_attacked(4, false, self)
                    && !board.is_square_attacked(5, false, self)
                    && !board.is_square_attacked(6, false, self)
                {
                    moves.push(Move::new(4, 6, None));
                }
            }
            if board.castling_rights.white_queenside && board.pieces[WHITE][KING] & (1u64 << 4) != 0
            {
                // Make sure a rook is there because it could have been captured
                if board.pieces[WHITE][ROOK] & (1u64 << 0) != 0
                    && (board.pieces_occ[WHITE] | board.pieces_occ[BLACK])
                        & ((1u64 << 1) | (1u64 << 2) | (1u64 << 3))
                        == 0
                    && !board.is_square_attacked(4, false, self)
                    && !board.is_square_attacked(3, false, self)
                    && !board.is_square_attacked(2, false, self)
                {
                    moves.push(Move::new(4, 2, None));
                }
            }
            for from_sq_ind in bits(&board.pieces[WHITE][KING]) {
                for to_sq_ind in &self.k_moves[from_sq_ind] {
                    if board.pieces_occ[BLACK] & (1u64 << to_sq_ind) != 0 {
                        captures.push(Move::new(from_sq_ind, *to_sq_ind, None));
                    } else if board.pieces_occ[WHITE] & (1u64 << to_sq_ind) == 0 {
                        moves.push(Move::new(from_sq_ind, *to_sq_ind, None));
                    }
                }
            }
        } else {
            // Black to move
            if board.castling_rights.black_kingside && board.pieces[BLACK][KING] & (1u64 << 60) != 0
            {
                // Make sure a rook is there because it could have been captured
                if board.pieces[BLACK][ROOK] & (1u64 << 63) != 0
                    && (board.pieces_occ[WHITE] | board.pieces_occ[BLACK])
                        & ((1u64 << 61) | (1u64 << 62))
                        == 0
                    && !board.is_square_attacked(60, true, self)
                    && !board.is_square_attacked(61, true, self)
                    && !board.is_square_attacked(62, true, self)
                {
                    moves.push(Move::new(60, 62, None));
                }
            }
            if board.castling_rights.black_queenside
                && board.pieces[BLACK][KING] & (1u64 << 60) != 0
            {
                // Make sure a rook is there because it could have been captured
                if board.pieces[BLACK][ROOK] & (1u64 << 56) != 0
                    && (board.pieces_occ[BLACK] | board.pieces_occ[WHITE])
                        & ((1u64 << 57) | (1u64 << 58) | (1u64 << 59))
                        == 0
                    && !board.is_square_attacked(60, true, self)
                    && !board.is_square_attacked(59, true, self)
                    && !board.is_square_attacked(58, true, self)
                {
                    moves.push(Move::new(60, 58, None));
                }
            }
            for from_sq_ind in bits(&board.pieces[BLACK][KING]) {
                for to_sq_ind in &self.k_moves[from_sq_ind] {
                    if board.pieces_occ[WHITE] & (1u64 << to_sq_ind) != 0 {
                        captures.push(Move::new(from_sq_ind, *to_sq_ind, None));
                    } else if board.pieces_occ[BLACK] & (1u64 << to_sq_ind) == 0 {
                        moves.push(Move::new(from_sq_ind, *to_sq_ind, None));
                    }
                }
            }
        }
    }

    /// Appends the pseudo-legal rook moves: captures to `captures`, others to `moves`.
    pub(crate) fn gen_rook_moves_into(
        &self,
        board: &Board,
        captures: &mut Vec<Move>,
        moves: &mut Vec<Move>,
    ) {
        // Generate all possible rook moves for the current position.
        // Returns a vector of captures and a vector of non-captures.
        // Uses magic bitboards.
        let mut blockers: u64;
        let mut key: usize;
        if board.w_to_move {
            // White to move
            for from_sq_ind in bits(&board.pieces[WHITE][ROOK]) {
                // Mask blockers
                blockers =
                    (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & R_MASKS[from_sq_ind];

                // Generate the key using a multiplication and right shift
                key = ((blockers.wrapping_mul(self.r_magics[from_sq_ind]))
                    >> (64 - R_BITS[from_sq_ind])) as usize;

                // Bit-iterate the attack bitboard, classifying each target by occupancy.
                for to_sq_ind in bits(&self.r_move_bitboard[from_sq_ind][key]) {
                    let sq_bit = 1u64 << to_sq_ind;
                    if board.pieces_occ[WHITE] & sq_bit != 0 {
                        // Own piece — not a legal target.
                    } else if board.pieces_occ[BLACK] & sq_bit != 0 {
                        captures.push(Move::new(from_sq_ind, to_sq_ind, None));
                    } else {
                        moves.push(Move::new(from_sq_ind, to_sq_ind, None));
                    }
                }
            }
        } else {
            // Black to move
            for from_sq_ind in bits(&board.pieces[BLACK][ROOK]) {
                // Mask blockers
                blockers =
                    (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & R_MASKS[from_sq_ind];

                // Generate the key using a multiplication and right shift
                key = ((blockers.wrapping_mul(self.r_magics[from_sq_ind]))
                    >> (64 - R_BITS[from_sq_ind])) as usize;

                // Bit-iterate the attack bitboard, classifying each target by occupancy.
                for to_sq_ind in bits(&self.r_move_bitboard[from_sq_ind][key]) {
                    let sq_bit = 1u64 << to_sq_ind;
                    if board.pieces_occ[BLACK] & sq_bit != 0 {
                        // Own piece — not a legal target.
                    } else if board.pieces_occ[WHITE] & sq_bit != 0 {
                        captures.push(Move::new(from_sq_ind, to_sq_ind, None));
                    } else {
                        moves.push(Move::new(from_sq_ind, to_sq_ind, None));
                    }
                }
            }
        }
    }

    pub fn gen_bishop_potential_captures(&self, board: &Board, from_sq_ind: usize) -> u64 {
        // Generate potential bishop captures from the given square.
        // Used to determine whether a king is in check.

        // Mask blockers
        let blockers: u64 =
            (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & B_MASKS[from_sq_ind];

        // Generate the key using a multiplication and right shift
        let key: usize = ((blockers.wrapping_mul(self.b_magics[from_sq_ind]))
            >> (64 - B_BITS[from_sq_ind])) as usize;

        // Return the preinitialized capture set bitboard from the table
        self.b_move_bitboard[from_sq_ind][key]
    }

    pub fn gen_rook_potential_captures(&self, board: &Board, from_sq_ind: usize) -> u64 {
        // Generate potential rook captures from the given square.
        // Used to determine whether a king is in check.

        // Mask blockers
        let blockers: u64 =
            (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & R_MASKS[from_sq_ind];

        // Generate the key using a multiplication and right shift
        let key: usize = ((blockers.wrapping_mul(self.r_magics[from_sq_ind]))
            >> (64 - R_BITS[from_sq_ind])) as usize;

        // Return the preinitialized capture set bitboard from the table
        self.r_move_bitboard[from_sq_ind][key]
    }

    /// Generate bishop attack bitboard given raw occupancy.
    pub fn gen_bishop_attacks_occ(&self, from_sq_ind: usize, occupancy: u64) -> u64 {
        let blockers = occupancy & B_MASKS[from_sq_ind];
        let key = ((blockers.wrapping_mul(self.b_magics[from_sq_ind]))
            >> (64 - B_BITS[from_sq_ind])) as usize;
        self.b_move_bitboard[from_sq_ind][key]
    }

    /// Generate rook attack bitboard given raw occupancy.
    pub fn gen_rook_attacks_occ(&self, from_sq_ind: usize, occupancy: u64) -> u64 {
        let blockers = occupancy & R_MASKS[from_sq_ind];
        let key = ((blockers.wrapping_mul(self.r_magics[from_sq_ind]))
            >> (64 - R_BITS[from_sq_ind])) as usize;
        self.r_move_bitboard[from_sq_ind][key]
    }

    /// Appends the pseudo-legal bishop moves: captures to `captures`, others to `moves`.
    pub(crate) fn gen_bishop_moves_into(
        &self,
        board: &Board,
        captures: &mut Vec<Move>,
        moves: &mut Vec<Move>,
    ) {
        // Generate all possible bishop moves for the current position.
        // Returns a vector of captures and a vector of non-captures.
        // Uses magic bitboards.
        let mut blockers: u64;
        let mut key: usize;
        if board.w_to_move {
            // White to move
            for from_sq_ind in bits(&board.pieces[WHITE][BISHOP]) {
                // Mask blockers
                blockers =
                    (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & B_MASKS[from_sq_ind];

                // Generate the key using a multiplication and right shift
                key = ((blockers.wrapping_mul(self.b_magics[from_sq_ind]))
                    >> (64 - B_BITS[from_sq_ind])) as usize;

                // Bit-iterate the attack bitboard, classifying each target by occupancy.
                for to_sq_ind in bits(&self.b_move_bitboard[from_sq_ind][key]) {
                    let sq_bit = 1u64 << to_sq_ind;
                    if board.pieces_occ[WHITE] & sq_bit != 0 {
                        // Own piece — not a legal target.
                    } else if board.pieces_occ[BLACK] & sq_bit != 0 {
                        captures.push(Move::new(from_sq_ind, to_sq_ind, None));
                    } else {
                        moves.push(Move::new(from_sq_ind, to_sq_ind, None));
                    }
                }
            }
        } else {
            // Black to move
            for from_sq_ind in bits(&board.pieces[BLACK][BISHOP]) {
                // Mask blockers
                blockers =
                    (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & B_MASKS[from_sq_ind];

                // Generate the key using a multiplication and right shift
                key = ((blockers.wrapping_mul(self.b_magics[from_sq_ind]))
                    >> (64 - B_BITS[from_sq_ind])) as usize;

                // Bit-iterate the attack bitboard, classifying each target by occupancy.
                for to_sq_ind in bits(&self.b_move_bitboard[from_sq_ind][key]) {
                    let sq_bit = 1u64 << to_sq_ind;
                    if board.pieces_occ[BLACK] & sq_bit != 0 {
                        // Own piece — not a legal target.
                    } else if board.pieces_occ[WHITE] & sq_bit != 0 {
                        captures.push(Move::new(from_sq_ind, to_sq_ind, None));
                    } else {
                        moves.push(Move::new(from_sq_ind, to_sq_ind, None));
                    }
                }
            }
        }
    }

    /// Appends the pseudo-legal queen moves: captures to `captures`, others to `moves`.
    pub(crate) fn gen_queen_moves_into(
        &self,
        board: &Board,
        captures: &mut Vec<Move>,
        moves: &mut Vec<Move>,
    ) {
        // Generate all possible queen moves for the current position.
        // Returns a vector of captures and a vector of non-captures.
        // Uses magic bitboards.
        let mut blockers: u64;
        let mut key: usize;
        if board.w_to_move {
            // White to move
            for from_sq_ind in bits(&board.pieces[WHITE][QUEEN]) {
                // Mask blockers
                blockers =
                    (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & R_MASKS[from_sq_ind];

                // Generate the key using a multiplication and right shift
                key = ((blockers.wrapping_mul(self.r_magics[from_sq_ind]))
                    >> (64 - R_BITS[from_sq_ind])) as usize;

                // Bit-iterate the rook-component attack bitboard.
                for to_sq_ind in bits(&self.r_move_bitboard[from_sq_ind][key]) {
                    let sq_bit = 1u64 << to_sq_ind;
                    if board.pieces_occ[WHITE] & sq_bit != 0 {
                    } else if board.pieces_occ[BLACK] & sq_bit != 0 {
                        captures.push(Move::new(from_sq_ind, to_sq_ind, None));
                    } else {
                        moves.push(Move::new(from_sq_ind, to_sq_ind, None));
                    }
                }
                // Mask blockers
                blockers =
                    (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & B_MASKS[from_sq_ind];

                // Generate the key using a multiplication and right shift
                key = ((blockers.wrapping_mul(self.b_magics[from_sq_ind]))
                    >> (64 - B_BITS[from_sq_ind])) as usize;

                // Bit-iterate the bishop-component attack bitboard.
                for to_sq_ind in bits(&self.b_move_bitboard[from_sq_ind][key]) {
                    let sq_bit = 1u64 << to_sq_ind;
                    if board.pieces_occ[WHITE] & sq_bit != 0 {
                    } else if board.pieces_occ[BLACK] & sq_bit != 0 {
                        captures.push(Move::new(from_sq_ind, to_sq_ind, None));
                    } else {
                        moves.push(Move::new(from_sq_ind, to_sq_ind, None));
                    }
                }
            }
        } else {
            // Black to move
            for from_sq_ind in bits(&board.pieces[BLACK][QUEEN]) {
                // Mask blockers
                blockers =
                    (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & R_MASKS[from_sq_ind];

                // Generate the key using a multiplication and right shift
                key = ((blockers.wrapping_mul(self.r_magics[from_sq_ind]))
                    >> (64 - R_BITS[from_sq_ind])) as usize;

                // Bit-iterate the rook-component attack bitboard.
                for to_sq_ind in bits(&self.r_move_bitboard[from_sq_ind][key]) {
                    let sq_bit = 1u64 << to_sq_ind;
                    if board.pieces_occ[BLACK] & sq_bit != 0 {
                    } else if board.pieces_occ[WHITE] & sq_bit != 0 {
                        captures.push(Move::new(from_sq_ind, to_sq_ind, None));
                    } else {
                        moves.push(Move::new(from_sq_ind, to_sq_ind, None));
                    }
                }
                // Mask blockers
                blockers =
                    (board.pieces_occ[BLACK] | board.pieces_occ[WHITE]) & B_MASKS[from_sq_ind];

                // Generate the key using a multiplication and right shift
                key = ((blockers.wrapping_mul(self.b_magics[from_sq_ind]))
                    >> (64 - B_BITS[from_sq_ind])) as usize;

                // Bit-iterate the bishop-component attack bitboard.
                for to_sq_ind in bits(&self.b_move_bitboard[from_sq_ind][key]) {
                    let sq_bit = 1u64 << to_sq_ind;
                    if board.pieces_occ[BLACK] & sq_bit != 0 {
                    } else if board.pieces_occ[WHITE] & sq_bit != 0 {
                        captures.push(Move::new(from_sq_ind, to_sq_ind, None));
                    } else {
                        moves.push(Move::new(from_sq_ind, to_sq_ind, None));
                    }
                }
            }
        }
    }

    /// Calculates a bitboard of all pieces of a given side attacking a specific square.
    ///
    /// # Arguments
    /// * `board` - The current board state.
    /// * `sq` - The target square index (0-63).
    /// * `side` - The attacking side (true for white, false for black).
    ///
    /// # Returns
    /// A bitboard containing the locations of all pieces of `side` attacking `sq`.
    pub fn attackers_to(&self, board: &Board, sq: usize, side: bool) -> u64 {
        let side_idx = side as usize;
        let mut attackers: u64 = 0;
        let _enemy_color_index = !side as usize; // Needed for pawn captures

        // Pawns (check captures from the target square's perspective)
        if side_idx == 0 {
            // White attackers
            if sq > 8 && !sq.is_multiple_of(8) {
                // Can be attacked from sq - 9 (black pawn on sq-9 attacks sq)
                attackers |= board.pieces[side_idx][PAWN] & sq_ind_to_bit(sq - 9);
            }
            if sq > 7 && sq % 8 != 7 {
                // Can be attacked from sq - 7 (black pawn on sq-7 attacks sq)
                attackers |= board.pieces[side_idx][PAWN] & sq_ind_to_bit(sq - 7);
            }
        } else {
            // Black attackers
            if sq < 55 && !sq.is_multiple_of(8) {
                // Can be attacked from sq + 7 (white pawn on sq+7 attacks sq)
                attackers |= board.pieces[side_idx][PAWN] & sq_ind_to_bit(sq + 7);
            }
            if sq < 56 && sq % 8 != 7 {
                // Can be attacked from sq + 9 (white pawn on sq+9 attacks sq)
                attackers |= board.pieces[side_idx][PAWN] & sq_ind_to_bit(sq + 9);
            }
        }

        // Knights
        attackers |= self.n_move_bitboard[sq] & board.pieces[side_idx][KNIGHT];

        // King
        attackers |= self.k_move_bitboard[sq] & board.pieces[side_idx][KING];

        // Bishops and Queens (diagonal)
        let bishop_attacks = self.gen_bishop_potential_captures(board, sq);
        attackers |=
            bishop_attacks & (board.pieces[side_idx][BISHOP] | board.pieces[side_idx][QUEEN]);

        // Rooks and Queens (horizontal/vertical)
        let rook_attacks = self.gen_rook_potential_captures(board, sq);
        attackers |= rook_attacks & (board.pieces[side_idx][ROOK] | board.pieces[side_idx][QUEEN]);

        attackers
    }

    /// Finds the square index of the least valuable attacker from a bitboard of attackers.
    /// Returns 64 if no attacker is found (should not happen if attackers_bb > 0).
    ///
    /// # Arguments
    /// * `board` - The current board state.
    /// * `attackers_bb` - A bitboard of pieces attacking a square.
    /// * `side` - The side whose attackers we are considering.
    ///
    /// # Returns
    /// The square index (0-63) of the least valuable attacker, or 64 if none found.
    pub fn least_valuable_attacker_sq(
        &self,
        board: &Board,
        attackers_bb: u64,
        side: bool,
    ) -> usize {
        let color_index = side as usize;
        for piece_type_idx in PAWN..=KING {
            // Iterate from Pawn (least valuable) to King
            let piece_bb = board.pieces[color_index][piece_type_idx];
            let intersection = attackers_bb & piece_bb;
            if intersection != 0 {
                return intersection.trailing_zeros() as usize; // Return square of the first found attacker of this type
            }
        }
        64 // Indicate no attacker found (error condition)
    }

    /// Get diagonal moves from a specific square, given an occupied bitboard.
    /// Helper method for get_bishop_moves.
    fn get_diag_moves(&self, from_sq_ind: usize, occupied: u64) -> u64 {
        // Get blocker mask for diagonals
        let blockers = occupied & DIAG_MASKS[from_sq_ind];

        // Calculate index for magic bitboard lookup
        let key = ((blockers.wrapping_mul(self.b_magics[from_sq_ind]))
            >> (64 - B_BITS[from_sq_ind])) as usize;

        // Get the diagonal component from the bishop moves
        let diag_mask = DIAG_MASKS[from_sq_ind];
        self.b_move_bitboard[from_sq_ind][key] & diag_mask
    }

    /// Get anti-diagonal moves from a specific square, given an occupied bitboard.
    /// Helper method for get_bishop_moves.
    fn get_anti_diag_moves(&self, from_sq_ind: usize, occupied: u64) -> u64 {
        // Get blocker mask for anti-diagonals
        let blockers = occupied & ANTI_DIAG_MASKS[from_sq_ind];

        // Calculate index for magic bitboard lookup
        let key = ((blockers.wrapping_mul(self.b_magics[from_sq_ind]))
            >> (64 - B_BITS[from_sq_ind])) as usize;

        // Get the anti-diagonal component from the bishop moves
        let anti_diag_mask = ANTI_DIAG_MASKS[from_sq_ind];
        self.b_move_bitboard[from_sq_ind][key] & anti_diag_mask
    }

    /// Get file moves from a specific square, given an occupied bitboard.
    /// Helper method for get_rook_moves.
    fn get_file_moves(&self, from_sq_ind: usize, occupied: u64) -> u64 {
        // Get blocker mask for files
        let blockers = occupied & FILE_MASKS[from_sq_ind];

        // Calculate index for magic bitboard lookup
        let key = ((blockers.wrapping_mul(self.r_magics[from_sq_ind]))
            >> (64 - R_BITS[from_sq_ind])) as usize;

        // Get the file component from the rook moves
        let file_mask = FILE_MASKS[from_sq_ind];
        self.r_move_bitboard[from_sq_ind][key] & file_mask
    }

    /// Get rank moves from a specific square, given an occupied bitboard.
    /// Helper method for get_rook_moves.
    fn get_rank_moves(&self, from_sq_ind: usize, occupied: u64) -> u64 {
        // Get blocker mask for ranks
        let blockers = occupied & RANK_MASKS[from_sq_ind];

        // Calculate index for magic bitboard lookup
        let key = ((blockers.wrapping_mul(self.r_magics[from_sq_ind]))
            >> (64 - R_BITS[from_sq_ind])) as usize;

        // Get the rank component from the rook moves
        let rank_mask = RANK_MASKS[from_sq_ind];
        self.r_move_bitboard[from_sq_ind][key] & rank_mask
    }

    /// Get bishop moves from a specific square, given an occupied bitboard.
    /// This is primarily used for mobility evaluation.
    pub fn get_bishop_moves(&self, from_sq_ind: usize, occupied: u64) -> u64 {
        let diag_moves = self.get_diag_moves(from_sq_ind, occupied);
        let anti_diag_moves = self.get_anti_diag_moves(from_sq_ind, occupied);
        diag_moves | anti_diag_moves
    }

    /// Get rook moves from a specific square, given an occupied bitboard.
    /// This is primarily used for mobility evaluation.
    pub fn get_rook_moves(&self, from_sq_ind: usize, occupied: u64) -> u64 {
        let file_moves = self.get_file_moves(from_sq_ind, occupied);
        let rank_moves = self.get_rank_moves(from_sq_ind, occupied);
        file_moves | rank_moves
    }

    /// Get queen moves from a specific square, given an occupied bitboard.
    /// This is primarily used for mobility evaluation.
    pub fn get_queen_moves(&self, from_sq_ind: usize, occupied: u64) -> u64 {
        self.get_bishop_moves(from_sq_ind, occupied) | self.get_rook_moves(from_sq_ind, occupied)
    }
}
