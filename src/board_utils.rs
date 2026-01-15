//! Board utility functions for coordinate conversion and square manipulation.
//!
//! Provides helper functions for converting between different board representations:
//! - Square indices (0-63) to file/rank coordinates
//! - Algebraic notation (e.g., "e4") to square indices
//! - Bitboard operations for individual squares
//!
//! # Square Indexing Convention
//!
//! Uses Little-Endian Rank-File Mapping (LERF):
//! - `index = 8 * rank + file`
//! - a1 = 0, h1 = 7, a8 = 56, h8 = 63
//! - File 0 = a-file, File 7 = h-file
//! - Rank 0 = 1st rank (white's back rank), Rank 7 = 8th rank

/// Converts file and rank coordinates to a square index.
///
/// # Arguments
///
/// * `file` - The file (0-7, where 0 is the a-file)
/// * `rank` - The rank (0-7, where 0 is the first rank)
///
/// # Returns
///
/// The square index (0-63)
pub fn coords_to_sq_ind(file: usize, rank: usize) -> usize {
    8 * rank + file
}

/// Converts a square index to file and rank coordinates.
///
/// # Arguments
///
/// * `sq_ind` - The square index (0-63)
///
/// # Returns
///
/// A tuple (file, rank) where file and rank are 0-7
pub fn sq_ind_to_coords(sq_ind: usize) -> (usize, usize) {
    (sq_ind % 8, sq_ind / 8)
}

/// Converts a square index to a bitboard representation.
///
/// # Arguments
///
/// * `sq_ind` - The square index (0-63)
///
/// # Returns
///
/// A 64-bit integer with only the bit at the given square index set
pub fn sq_ind_to_bit(sq_ind: usize) -> u64 {
    1 << sq_ind
}

/// Converts a bitboard with a single bit set to the index of that bit.
///
/// # Arguments
///
/// * `bit` - A 64-bit integer with (hopefully) one bit set
///
/// # Returns
///
/// The index of the set bit (0-63)
pub fn bit_to_sq_ind(bit: u64) -> usize {
    if bit == 0 {
        // Return a default value or handle specially
        // This is safer than returning 64 which would cause index out of bounds
        0 // Default to a0 square for now, caller should handle this case specially
    } else {
        bit.trailing_zeros() as usize
    }
}

/// Converts a square index to algebraic notation.
///
/// # Arguments
///
/// * `sq_ind` - The square index (0-63)
///
/// # Returns
///
/// A string representing the square in algebraic notation (e.g., "e4")
pub fn sq_ind_to_algebraic(sq_ind: usize) -> String {
    let (file, rank) = sq_ind_to_coords(sq_ind);
    let file = (file + 97) as u8 as char;
    let rank = (rank + 49) as u8 as char;
    format!("{}{}", file, rank)
}

/// Converts algebraic notation to a square index.
///
/// # Arguments
///
/// * `algebraic` - A string representing a square in algebraic notation (e.g., "e4")
///
/// # Returns
///
/// The corresponding square index (0-63)
pub fn algebraic_to_sq_ind(algebraic: &str) -> usize {
    let mut chars = algebraic.chars();
    let file = chars.next().unwrap() as usize - 97;
    let rank = chars.next().unwrap() as usize - 49;
    coords_to_sq_ind(file, rank)
}

/// Converts algebraic notation to a bitboard representation.
///
/// # Arguments
///
/// * `algebraic` - A string representing a square in algebraic notation (e.g., "e4")
///
/// # Returns
///
/// A 64-bit integer with only the bit at the given square set
pub fn algebraic_to_bit(algebraic: &str) -> u64 {
    let sq_ind = algebraic_to_sq_ind(algebraic);
    sq_ind_to_bit(sq_ind)
}

/// Converts a bitboard with a single bit set to algebraic notation.
///
/// # Arguments
///
/// * `bit` - A 64-bit integer with only one bit set
///
/// # Returns
///
/// A string representing the square in algebraic notation (e.g., "e4")
pub fn bit_to_algebraic(bit: u64) -> String {
    let sq_ind = bit_to_sq_ind(bit);
    sq_ind_to_algebraic(sq_ind)
}

/// Flips a square index vertically on the board.
///
/// # Arguments
///
/// * `sq_ind` - The square index to flip (0-63)
///
/// # Returns
///
/// The vertically flipped square index (0-63)
pub fn flip_sq_ind_vertically(sq_ind: usize) -> usize {
    8 * (7 - sq_ind / 8) + sq_ind % 8
}

/// Flips a bitboard vertically.
///
/// # Arguments
///
/// * `bit` - The bitboard to flip
///
/// # Returns
///
/// The vertically flipped bitboard
pub fn flip_vertically(bit: u64) -> u64 {
    (bit << 56)
        | ((bit << 40) & (0x00ff000000000000))
        | ((bit << 24) & (0x0000ff0000000000))
        | ((bit << 8) & (0x000000ff00000000))
        | ((bit >> 8) & (0x00000000ff000000))
        | ((bit >> 24) & (0x0000000000ff0000))
        | ((bit >> 40) & (0x000000000000ff00))
        | (bit >> 56)
}

/// Returns the rank (0-7) of a given square index.
pub fn sq_to_rank(sq_ind: usize) -> usize {
    sq_ind / 8
}

/// Returns the file (0-7) of a given square index.
pub fn sq_to_file(sq_ind: usize) -> usize {
    sq_ind % 8
}

// --- Masks for Evaluation ---
use crate::piece_types::{BLACK, WHITE};
use lazy_static::lazy_static;

// Precomputed masks for determining passed pawns.
// For a given square `sq`, `PASSED_MASKS[color][sq]` contains a mask of squares
// in front of the pawn on the same file and adjacent files, in the direction
// of pawn movement for that color. If this mask AND the opponent's pawn bitboard
// is zero, the pawn is passed.
lazy_static! {
    static ref PASSED_MASKS: [[u64; 64]; 2] = {
        let mut masks = [[0u64; 64]; 2];
        for sq in 0..64 {
            let file = sq_to_file(sq);
            let rank = sq_to_rank(sq);

            // White passed pawn mask
            let mut white_mask: u64 = 0;
            for r in (rank + 1)..8 {
                // Same file
                white_mask |= sq_ind_to_bit(coords_to_sq_ind(file, r));
                // Adjacent files
                if file > 0 {
                    white_mask |= sq_ind_to_bit(coords_to_sq_ind(file - 1, r));
                }
                if file < 7 {
                    white_mask |= sq_ind_to_bit(coords_to_sq_ind(file + 1, r));
                }
            }
            masks[WHITE][sq] = white_mask;

            // Black passed pawn mask
            let mut black_mask: u64 = 0;
            for r in 0..rank { // Iterate ranks below the pawn
                 // Same file
                 black_mask |= sq_ind_to_bit(coords_to_sq_ind(file, r));
                 // Adjacent files
                 if file > 0 {
                     black_mask |= sq_ind_to_bit(coords_to_sq_ind(file - 1, r));
                 }
                 if file < 7 {
                     black_mask |= sq_ind_to_bit(coords_to_sq_ind(file + 1, r));
                 }
            }
             masks[BLACK][sq] = black_mask;
        }
        masks
    };

    // Precomputed masks for the king shield zone (squares directly and diagonally in front).
    // Index is [color][king_square]
    static ref KING_SHIELD_ZONES: [[u64; 64]; 2] = {
        let mut masks = [[0u64; 64]; 2];
        for sq in 0..64 {
            let file = sq_to_file(sq);
            let rank = sq_to_rank(sq);

            // White king shield zone (rank + 1)
            if rank < 7 {
                let front_rank = rank + 1;
                // Directly in front
                masks[WHITE][sq] |= sq_ind_to_bit(coords_to_sq_ind(file, front_rank));
                // Diagonally in front
                if file > 0 {
                    masks[WHITE][sq] |= sq_ind_to_bit(coords_to_sq_ind(file - 1, front_rank));
                }
                if file < 7 {
                    masks[WHITE][sq] |= sq_ind_to_bit(coords_to_sq_ind(file + 1, front_rank));
                }
            }

            // Black king shield zone (rank - 1)
            if rank > 0 {
                let front_rank = rank - 1;
                 // Directly in front
                 masks[BLACK][sq] |= sq_ind_to_bit(coords_to_sq_ind(file, front_rank));
                 // Diagonally in front
                 if file > 0 {
                     masks[BLACK][sq] |= sq_ind_to_bit(coords_to_sq_ind(file - 1, front_rank));
                 }
                 if file < 7 {
                     masks[BLACK][sq] |= sq_ind_to_bit(coords_to_sq_ind(file + 1, front_rank));
                 }
            }
        }
        masks
    };

    // Precomputed file masks (A-H)
    static ref FILE_MASKS: [u64; 8] = {
        let mut masks = [0u64; 8];
        for file in 0..8 {
            for rank in 0..8 {
                masks[file] |= sq_ind_to_bit(coords_to_sq_ind(file, rank));
            }
        }
        masks
    };

    // Precomputed masks for files adjacent to a given square's file
    static ref ADJACENT_FILE_MASKS: [u64; 64] = {
        let mut masks = [0u64; 64];
        for sq in 0..64 {
            let file = sq_to_file(sq);
            if file > 0 {
                masks[sq] |= FILE_MASKS[file - 1];
            }
            if file < 7 {
                masks[sq] |= FILE_MASKS[file + 1];
            }
        }
        masks
    };

    // Precomputed masks for the square directly in front of a pawn.
    static ref PAWN_FRONT_SQUARE_MASKS: [[u64; 64]; 2] = {
        let mut masks = [[0u64; 64]; 2];
        for sq in 0..64 {
            let rank = sq_to_rank(sq);
            // White pawn front square (rank + 1)
            if rank < 7 {
                masks[WHITE][sq] = sq_ind_to_bit(sq + 8);
            }
            // Black pawn front square (rank - 1)
            if rank > 0 {
                 masks[BLACK][sq] = sq_ind_to_bit(sq - 8);
            }
        }
        masks
    };

    // Precomputed rank masks (1-8)
    static ref RANK_MASKS: [u64; 8] = {
        let mut masks = [0u64; 8];
        for rank in 0..8 {
            for file in 0..8 {
                masks[rank] |= sq_ind_to_bit(coords_to_sq_ind(file, rank));
            }
        }
        masks
    };

}

/// Returns a bitmask representing the path and adjacent files in front of a pawn.
/// Used to check if a pawn is passed.
pub fn get_passed_pawn_mask(color: usize, sq_ind: usize) -> u64 {
    PASSED_MASKS[color][sq_ind]
}

/// Returns a bitmask representing the king shield zone (squares directly/diagonally in front).
pub fn get_king_shield_zone_mask(color: usize, king_sq_ind: usize) -> u64 {
    KING_SHIELD_ZONES[color][king_sq_ind]
}

/// Returns a bitmask for a given file (0-7).
pub fn get_file_mask(file: usize) -> u64 {
    FILE_MASKS[file]
}

/// Returns a bitmask for files adjacent to the given square's file.
pub fn get_adjacent_files_mask(sq_ind: usize) -> u64 {
    ADJACENT_FILE_MASKS[sq_ind]
}

/// Returns a bitmask for the square directly in front of a pawn.
pub fn get_pawn_front_square_mask(color: usize, sq_ind: usize) -> u64 {
    PAWN_FRONT_SQUARE_MASKS[color][sq_ind]
}

/// Returns a bitmask for a given rank (0-7).
pub fn get_rank_mask(rank: usize) -> u64 {
    RANK_MASKS[rank]
}

/// Get a bitboard containing all squares in front of a square on the same and adjacent files.
/// Used for backward pawn detection and similar evaluations.
pub fn get_front_span_mask(color: usize, sq: usize) -> u64 {
    let file = sq_to_file(sq);
    let rank = sq_to_rank(sq);

    let mut mask = 0;

    // Add the file mask
    mask |= get_file_mask(file);

    // Add adjacent files if they exist
    if file > 0 {
        mask |= get_file_mask(file - 1);
    }
    if file < 7 {
        mask |= get_file_mask(file + 1);
    }

    // Filter to only include squares in front of the given square
    if color == 0 {
        // WHITE
        // Keep only ranks higher than the current rank (in front for White)
        for r in (rank + 1)..8 {
            mask &= !(0xFFu64 << (r * 8));
        }
    } else {
        // BLACK
        // Keep only ranks lower than the current rank (in front for Black)
        for r in 0..rank {
            mask &= !(0xFFu64 << (r * 8));
        }
    }

    mask
}

/// Get a bitboard containing the king's attack zone (squares around the king).
/// Used for king safety evaluations and detecting attacks against the king.
pub fn get_king_attack_zone_mask(_color: usize, king_sq: usize) -> u64 {
    // The attack zone is all squares within distance 2 of the king
    let mut mask = 0;

    let rank = sq_to_rank(king_sq);
    let file = sq_to_file(king_sq);

    // Iterate over a 5x5 square centered on the king
    for r in rank.saturating_sub(2)..=rank.saturating_add(2).min(7) {
        for f in file.saturating_sub(2)..=file.saturating_add(2).min(7) {
            mask |= 1u64 << (r * 8 + f);
        }
    }

    // Remove the king's square itself from the mask
    mask &= !(1u64 << king_sq);

    mask
}

// TODO: Add functions to calculate attacks to a square if not using MoveGen directly in eval.
// e.g., pub fn knight_attacks(sq: usize) -> u64 { ... }
// pub fn bishop_attacks(sq: usize, occupied: u64) -> u64 { ... } // Using magic bitboards
// pub fn rook_attacks(sq: usize, occupied: u64) -> u64 { ... } // Using magic bitboards
