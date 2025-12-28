//! Magic bitboard implementation for chess move generation.
//!
//! This module provides functions for initializing and using magic bitboards,
//! which are an efficient method for generating sliding piece moves in chess.

// Generate all possible moves for a given position
// We divide it up into two functions: one for generating moves for a single piece, and one for generating moves for all pieces.
// The latter will call the former for each piece.
// For this version, we generate moves in the following order:
// 1. Captures/promotions in MVV-LVA order
// 2. Knight forks
// 3. All other moves, ordered by change in pesto eval (precomputed)
// I think this is relatively easy to do, and may even be close to optimal, even if we later change the eval function, e.g. to NNUE.
// We can also use quiescence search to only generate captures and promotions.
// For capture move orders, we should use 10*victim_value - attacker_value, where PNBRQK = 123456 e.g., PxQ = 10*5 - 1 = 49, KxN = 10*3 - 6 = 24, etc.
// For promotions, we should use 10*promoted_piece_value - attacker_value, so Q promotion = 10*5 - 1 = 49, etc.
// We want to use two heaps, one for captures and one for non-captures, with each element of the heap being a vector of sorted moves.
// Note that non-captures can be pre-sorted, but captures require the piece value of the captured piece and so they have to be internally sorted, which may be just as fast as sorting the entire vector.
// Note also that the pesto eval has 25 game modes, ranging from opening to endgame, so our non-capture move ordering should be different for each game mode.

use crate::bits::bits;
use crate::board_utils::sq_ind_to_bit;
use crate::magic_constants::{B_BITS, B_MASKS, R_BITS, R_MASKS};
use crate::move_types::Move;
use crate::piece_types::{BISHOP, KNIGHT, QUEEN, ROOK};

const NOT_A_FILE: u64 = 0xfefefefefefefefe;
const NOT_H_FILE: u64 = 0x7f7f7f7f7f7f7f7f;
const NOT_AB_FILE: u64 = 0xfcfcfcfcfcfcfcfc;
const NOT_GH_FILE: u64 = 0x3f3f3f3f3f3f3f3f;
const NOT_1_RANK: u64 = 0xffffffffffffff00;
const NOT_8_RANK: u64 = 0x00ffffffffffffff;
const NOT_12_RANK: u64 = 0xffffffffffff0000;
const NOT_78_RANK: u64 = 0x0000ffffffffffff;
const RANK_2: u64 = 0x000000000000ff00;
const RANK_7: u64 = 0x00ff000000000000;

pub fn find_magic_numbers() -> ([u64; 64], [u64; 64]) {
    // Find magic numbers for magic bitboards.
    // (blockers * magic) >> (64 - n_bits) should give a unique key for each blocker combination.

    let mut blockers: u64;
    let mut blocker_squares: Vec<usize> = Vec::new();
    let mut key: usize;
    let mut keys: Vec<usize> = Vec::new();
    let mut magic: u64;
    let mut b_magics: [u64; 64] = [0; 64];
    let mut r_magics: [u64; 64] = [0; 64];
    for is_bishop in [true, false].iter() {
        for from_sq_ind in 0..64 {
            // Mask blockers
            if *is_bishop {
                blockers = B_MASKS[from_sq_ind];
            } else {
                blockers = R_MASKS[from_sq_ind];
            }
            blocker_squares.clear();
            for i in bits(&blockers) {
                blocker_squares.push(i);
            }
            for _i_magic in 0..1000000 {
                magic = rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>();
                // Iterate over all possible blocker combinations
                // Require that they are all unique
                keys.clear();
                for blocker_ind in 0..(1 << blocker_squares.len()) {
                    if *is_bishop {
                        blockers = B_MASKS[from_sq_ind];
                    } else {
                        blockers = R_MASKS[from_sq_ind];
                    }
                    for i in 0..blocker_squares.len() {
                        if (blocker_ind & (1 << i)) != 0 {
                            blockers &= !sq_ind_to_bit(blocker_squares[i]);
                        }
                    }
                    if *is_bishop {
                        key = ((blockers * magic) >> (64 - B_BITS[from_sq_ind])) as usize;
                    } else {
                        key = ((blockers * magic) >> (64 - R_BITS[from_sq_ind])) as usize;
                    }
                    if keys.contains(&key) {
                        break;
                    } else {
                        keys.push(key);
                    }
                }
                if *is_bishop {
                    if keys.len() == (1 << B_BITS[from_sq_ind]) {
                        println!(
                            "Found bishop magic number for square {} with {} bits: {}",
                            from_sq_ind, B_BITS[from_sq_ind], magic
                        );
                        b_magics[from_sq_ind] = magic;
                        break;
                    }
                } else if keys.len() == (1 << R_BITS[from_sq_ind]) {
                    println!(
                        "Found rook magic number for square {} with {} bits: {}",
                        from_sq_ind, R_BITS[from_sq_ind], magic
                    );
                    r_magics[from_sq_ind] = magic;
                    break;
                }
            }
        }
    }
    (b_magics, r_magics)
}

/// Initializes king moves for a given square.
///
/// # Arguments
///
/// * `from_sq_ind` - The square index (0-63) of the king.
///
/// # Returns
///
/// A vector of usize representing possible king move destinations.
pub fn init_king_moves(from_sq_ind: usize) -> Vec<usize> {
    // Initialize the king moves for a given square, ignoring castling
    let from_bit: u64 = sq_ind_to_bit(from_sq_ind);
    let mut out: Vec<usize> = Vec::new();
    if from_bit & NOT_A_FILE != 0 {
        out.push(from_sq_ind - 1);
    }
    if from_bit & NOT_8_RANK != 0 {
        if from_bit & NOT_A_FILE != 0 {
            out.push(from_sq_ind + 7);
        }
        out.push(from_sq_ind + 8);
        if from_bit & NOT_H_FILE != 0 {
            out.push(from_sq_ind + 9);
        }
    }
    if from_bit & NOT_H_FILE != 0 {
        out.push(from_sq_ind + 1);
    }
    if from_bit & NOT_1_RANK != 0 {
        if from_bit & NOT_H_FILE != 0 {
            out.push(from_sq_ind - 7);
        }
        out.push(from_sq_ind - 8);
        if from_bit & NOT_A_FILE != 0 {
            out.push(from_sq_ind - 9);
        }
    }
    out
}

/// Initializes knight moves for a given square.
///
/// # Arguments
///
/// * `from_sq_ind` - The square index (0-63) of the knight.
///
/// # Returns
///
/// A vector of usize representing possible knight move destinations.
pub fn init_knight_moves(from_sq_ind: usize) -> Vec<usize> {
    // Initialize the knight moves a given square.
    let from_bit: u64 = sq_ind_to_bit(from_sq_ind);
    let mut out: Vec<usize> = Vec::new();
    if from_bit & NOT_A_FILE & NOT_78_RANK != 0 {
        out.push(from_sq_ind + 15);
    }
    if from_bit & NOT_H_FILE & NOT_78_RANK != 0 {
        out.push(from_sq_ind + 17);
    }
    if from_bit & NOT_AB_FILE & NOT_8_RANK != 0 {
        out.push(from_sq_ind + 6);
    }
    if from_bit & NOT_GH_FILE & NOT_8_RANK != 0 {
        out.push(from_sq_ind + 10);
    }
    if from_bit & NOT_AB_FILE & NOT_1_RANK != 0 {
        out.push(from_sq_ind - 10);
    }
    if from_bit & NOT_GH_FILE & NOT_1_RANK != 0 {
        out.push(from_sq_ind - 6);
    }
    if from_bit & NOT_A_FILE & NOT_12_RANK != 0 {
        out.push(from_sq_ind - 17);
    }
    if from_bit & NOT_H_FILE & NOT_12_RANK != 0 {
        out.push(from_sq_ind - 15);
    }
    out
}

/// Initializes pawn captures and promotions for a given square.
///
/// # Arguments
///
/// * `from_sq_ind` - The square index (0-63) of the pawn.
///
/// # Returns
///
/// A tuple containing four vectors of usize:
/// - White pawn captures
/// - White pawn promotions
/// - Black pawn captures
/// - Black pawn promotions
pub fn init_pawn_captures_promotions(
    from_sq_ind: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    // Initialize the pawn captures and promotions for a given square.
    // Separate for white and black.
    let from_bit: u64 = sq_ind_to_bit(from_sq_ind);
    let mut w_cap: Vec<usize> = Vec::new();
    let mut b_cap: Vec<usize> = Vec::new();
    let mut w_prom: Vec<usize> = Vec::new();
    let mut b_prom: Vec<usize> = Vec::new();
    if from_bit & NOT_1_RANK & NOT_8_RANK != 0 {
        if from_bit & NOT_A_FILE != 0 {
            w_cap.push(from_sq_ind + 7);
        }
        if from_bit & NOT_H_FILE != 0 {
            w_cap.push(from_sq_ind + 9);
        }
        if from_bit & RANK_7 != 0 {
            w_prom.push(from_sq_ind + 8);
        }
        if from_bit & NOT_A_FILE != 0 {
            b_cap.push(from_sq_ind - 9);
        }
        if from_bit & NOT_H_FILE != 0 {
            b_cap.push(from_sq_ind - 7);
        }
        if from_bit & RANK_2 != 0 {
            b_prom.push(from_sq_ind - 8);
        }
    }
    (w_cap, w_prom, b_cap, b_prom)
}

/// Initializes pawn moves for a given square.
///
/// # Arguments
///
/// * `from_sq_ind` - The square index (0-63) of the pawn.
///
/// # Returns
///
/// A tuple containing two vectors of usize:
/// - The first vector represents white pawn moves.
/// - The second vector represents black pawn moves.
pub fn init_pawn_moves(from_sq_ind: usize) -> (Vec<usize>, Vec<usize>) {
    // Initialize the pawn moves for a given square.
    // Separate for white and black.
    let from_bit: u64 = sq_ind_to_bit(from_sq_ind);
    let mut white: Vec<usize> = Vec::new();
    let mut black: Vec<usize> = Vec::new();
    if from_bit & NOT_1_RANK & NOT_8_RANK != 0 {
        if from_bit & RANK_2 != 0 {
            white.push(from_sq_ind + 16);
        }
        white.push(from_sq_ind + 8);
        if from_bit & RANK_7 != 0 {
            black.push(from_sq_ind - 16);
        }
        black.push(from_sq_ind - 8);
    }
    (white, black)
}

/// Initializes the magic bitboards for bishop moves.
///
/// # Arguments
///
/// * `b_magics` - An array of 64 pre-computed magic numbers for bishops.
///
/// # Returns
///
/// A tuple containing:
/// - A vector of vectors of tuples, where each tuple contains two vectors of usize (for captures and moves).
/// - A vector of vectors of u64 (bitboards).
pub fn init_bishop_moves(
    b_magics: [u64; 64],
) -> (Vec<Vec<(Vec<usize>, Vec<usize>)>>, Vec<Vec<u64>>) {
    let mut out1: Vec<Vec<(Vec<usize>, Vec<usize>)>> = Vec::new();
    let mut out2: Vec<Vec<u64>> = Vec::new();
    let mut blockers: u64;
    let mut key: usize;
    let mut blocker_squares: Vec<usize> = Vec::new();
    for from_sq_ind in 0..64 {
        out1.push(vec![(vec![], vec![]); 4096]);
        out2.push(vec![0; 4096]);
        // Mask blockers
        blockers = B_MASKS[from_sq_ind];
        blocker_squares.clear();
        for i in bits(&blockers) {
            blocker_squares.push(i);
        }

        // Iterate over all possible blocker combinations
        for blocker_ind in 0..(1 << blocker_squares.len()) {
            let mut current_blockers = 0u64;
            for i in 0..blocker_squares.len() {
                if (blocker_ind & (1 << i)) != 0 {
                    current_blockers |= sq_ind_to_bit(blocker_squares[i]);
                }
            }

            // Generate the key using a multiplication and right shift
            key = ((current_blockers.wrapping_mul(b_magics[from_sq_ind])) >> (64 - B_BITS[from_sq_ind]))
                as usize;

            // Assign the captures and moves for this blocker combination
            out1[from_sq_ind][key] = bishop_attacks(from_sq_ind, current_blockers);
            for i in out1[from_sq_ind][key].0.iter() {
                out2[from_sq_ind][key] |= sq_ind_to_bit(*i);
            }
        }
    }
    (out1, out2)
}

/// Initializes the magic bitboards for rook moves.
///
/// # Arguments
///
/// * `r_magics` - An array of 64 pre-computed magic numbers for rooks.
///
/// # Returns
///
/// A tuple containing:
/// - A vector of vectors of tuples, where each tuple contains two vectors of usize (for captures and moves).
/// - A vector of vectors of u64 (bitboards).
pub fn init_rook_moves(r_magics: [u64; 64]) -> (Vec<Vec<(Vec<usize>, Vec<usize>)>>, Vec<Vec<u64>>) {
    let mut out1: Vec<Vec<(Vec<usize>, Vec<usize>)>> = Vec::new();
    let mut out2: Vec<Vec<u64>> = Vec::new();
    let mut key: usize;
    let mut blocker_squares: Vec<usize> = Vec::new();
    for from_sq_ind in 0..64 {
        out1.push(vec![(vec![], vec![]); 4096]);
        out2.push(vec![0; 4096]);
        // Mask blockers
        let mask = R_MASKS[from_sq_ind];
        blocker_squares.clear();
        for i in bits(&mask) {
            blocker_squares.push(i);
        }

        // Iterate over all possible blocker combinations
        for blocker_ind in 0..(1 << blocker_squares.len()) {
            let mut current_blockers = 0u64;
            for i in 0..blocker_squares.len() {
                if (blocker_ind & (1 << i)) != 0 {
                    current_blockers |= sq_ind_to_bit(blocker_squares[i]);
                }
            }

            // Generate the key using a multiplication and right shift
            key = ((current_blockers.wrapping_mul(r_magics[from_sq_ind])) >> (64 - R_BITS[from_sq_ind]))
                as usize;

            // Assign the captures and moves for this blocker combination
            out1[from_sq_ind][key] = rook_attacks(from_sq_ind, current_blockers);
            for i in out1[from_sq_ind][key].0.iter() {
                out2[from_sq_ind][key] |= sq_ind_to_bit(*i);
            }
        }
    }
    (out1, out2)
}

/// Generates rook attacks for a given square and blocking bitboard.
///
/// # Arguments
///
/// * `sq` - The square index (0-63) of the rook.
/// * `block` - A bitboard representing the blocking pieces.
///
/// # Returns
///
/// A tuple containing two vectors of usize:
/// - The first vector represents capture squares.
/// - The second vector represents move squares.
pub fn rook_attacks(sq: usize, block: u64) -> (Vec<usize>, Vec<usize>) {
    let rk = (sq / 8) as i32;
    let fl = (sq % 8) as i32;
    let mut captures: Vec<usize> = Vec::new();
    let mut moves: Vec<usize> = Vec::new();

    let dr = [1, -1, 0, 0];
    let df = [0, 0, 1, -1];

    for i in 0..4 {
        for d in 1..8 {
            let r = rk + dr[i] * d;
            let f = fl + df[i] * d;
            if r >= 0 && r < 8 && f >= 0 && f < 8 {
                let s = (r * 8 + f) as usize;
                if (block & (1u64 << s)) != 0 {
                    captures.push(s);
                    break;
                } else {
                    moves.push(s);
                }
            } else {
                break;
            }
        }
    }
    (captures, moves)
}

/// Generates bishop attacks for a given square and blocking bitboard.
///
/// # Arguments
///
/// * `sq` - The square index (0-63) of the bishop.
/// * `block` - A bitboard representing the blocking pieces.
///
/// # Returns
///
/// A tuple containing two vectors of usize:
/// - The first vector represents capture squares.
/// - The second vector represents move squares.
pub fn bishop_attacks(sq: usize, block: u64) -> (Vec<usize>, Vec<usize>) {
    let rk = (sq / 8) as i32;
    let fl = (sq % 8) as i32;
    let mut captures: Vec<usize> = Vec::new();
    let mut moves: Vec<usize> = Vec::new();

    let dr = [1, 1, -1, -1];
    let df = [1, -1, 1, -1];

    for i in 0..4 {
        for d in 1..8 {
            let r = rk + dr[i] * d;
            let f = fl + df[i] * d;
            if r >= 0 && r < 8 && f >= 0 && f < 8 {
                let s = (r * 8 + f) as usize;
                if (block & (1u64 << s)) != 0 {
                    captures.push(s);
                    break;
                } else {
                    moves.push(s);
                }
            } else {
                break;
            }
        }
    }
    (captures, moves)
}

/// Appends promotion moves to a vector of moves.
///
/// # Arguments
///
/// * `moves` - A mutable reference to a vector of Move structs.
/// * `from` - The starting square of the pawn.
/// * `to` - The destination square of the pawn.
///
/// # Returns
///
/// The number of promotion moves added.
pub fn append_promotions(
    promotions: &mut Vec<Move>,
    from_sq_ind: usize,
    to_sq_ind: &usize,
    w_to_move: bool,
) {
    if w_to_move {
        promotions.push(Move::new(from_sq_ind, *to_sq_ind, Some(QUEEN)));
        promotions.push(Move::new(from_sq_ind, *to_sq_ind, Some(ROOK)));
        promotions.push(Move::new(from_sq_ind, *to_sq_ind, Some(KNIGHT)));
        promotions.push(Move::new(from_sq_ind, *to_sq_ind, Some(BISHOP)));
    } else {
        promotions.push(Move::new(from_sq_ind, *to_sq_ind, Some(QUEEN)));
        promotions.push(Move::new(from_sq_ind, *to_sq_ind, Some(ROOK)));
        promotions.push(Move::new(from_sq_ind, *to_sq_ind, Some(KNIGHT)));
        promotions.push(Move::new(from_sq_ind, *to_sq_ind, Some(BISHOP)));
    }
}
