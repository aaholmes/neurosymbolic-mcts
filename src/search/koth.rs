//! King of the Hill (KOTH) variant win detection.
//!
//! In King of the Hill chess, a player wins instantly by moving their king
//! to one of the four center squares (d4, e4, d5, e5). This module provides
//! efficient detection of forced KOTH wins within a small number of moves.
//!
//! The `koth_center_in_3` function is used as a **Tier 1 Safety Gate** in the
//! three-tier MCTS architecture, allowing the engine to detect instant wins
//! before committing to expensive neural network evaluation.
//!
//! # Algorithm
//!
//! Uses a depth-limited minimax search with geometric pruning based on
//! king distance from the center squares. Kings more than 3 moves away
//! are pruned early.

use crate::board::{Board, KOTH_CENTER};
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::piece_types::{BLACK, KING, WHITE};

/// Distance rings for KOTH geometric pruning (Chebyshev distance from center set)
const RING_1: u64 = 0x00003C24243C0000 & !KOTH_CENTER; // Squares 1 distance from center
const RING_2: u64 = 0x007E424242427E00 & !(RING_1 | KOTH_CENTER); // Squares 2 distance from center
const RING_3: u64 = 0xFF818181818181FF; // Squares 3 distance from center (board border)

/// Returns the cumulative ring mask for a given number of remaining root moves.
/// The king must be within this mask to have any chance of reaching center.
fn cumulative_ring_mask(remaining_moves: u8) -> u64 {
    match remaining_moves {
        0 => KOTH_CENTER,
        1 => KOTH_CENTER | RING_1,
        2 => KOTH_CENTER | RING_1 | RING_2,
        _ => KOTH_CENTER | RING_1 | RING_2 | RING_3, // whole board for ≥3
    }
}

/// Checks if the side to move can reach the KOTH center within `max_n` of their own moves,
/// assuming the opponent cannot block them or reach the center themselves first.
/// Returns `Some(n)` where n is the minimum number of side-to-move moves needed (0–max_n),
/// or `None` if unreachable within max_n moves.
/// This is a "safety gate" to detect rapid KOTH wins.
pub fn koth_center_in_n(board: &Board, move_gen: &MoveGen, max_n: u8) -> Option<u8> {
    let side_to_move = if board.w_to_move { WHITE } else { BLACK };
    let king_bit = board.get_piece_bitboard(side_to_move, KING);

    if king_bit == 0 {
        return None;
    }

    // Check if already on center (0 moves)
    let (w_won, b_won) = board.is_koth_win();
    if (board.w_to_move && w_won) || (!board.w_to_move && b_won) {
        return Some(0);
    }

    // Try 1..=max_n root-side moves (max_ply = 1, 3, 5, ...)
    for n in 1..=max_n {
        let max_ply = (n as i32) * 2 - 1;
        if solve_koth(board, move_gen, 0, max_ply, max_n) {
            return Some(n);
        }
    }
    None
}

/// Wrapper for backward compatibility: checks KOTH center within 3 moves.
pub fn koth_center_in_3(board: &Board, move_gen: &MoveGen) -> Option<u8> {
    koth_center_in_n(board, move_gen, 3)
}

/// Returns the best move for the side to move to force a KOTH win,
/// or `None` if there is no forced KOTH win (or the king is already on center).
///
/// This is used for KOTH play-outs: the winning side plays `koth_best_move()` moves
/// to march its king to the center.
pub fn koth_best_move(board: &Board, move_gen: &MoveGen) -> Option<Move> {
    // If already on center (n=0), no move needed
    let (w_won, b_won) = board.is_koth_win();
    if (board.w_to_move && w_won) || (!board.w_to_move && b_won) {
        return None;
    }

    // Try n=1, 2, 3 — return the first winning move at minimum n
    let max_n = 3u8;
    let side_to_move = if board.w_to_move { WHITE } else { BLACK };
    let king_bit = board.get_piece_bitboard(side_to_move, KING);
    let king_sq = king_bit.trailing_zeros() as usize;

    for n in 1..=max_n {
        let max_ply = (n as i32) * 2 - 1;

        let target_mask = cumulative_ring_mask(n.saturating_sub(1));
        let king_must_advance = (king_bit & target_mask) == 0;

        if king_must_advance {
            // Direct king-move generation: skip full movegen
            let friendly_occ = board.get_color_occupancy(side_to_move);
            let candidates = move_gen.k_move_bitboard[king_sq] & target_mask & !friendly_occ;
            let mut bits = candidates;
            while bits != 0 {
                let to_sq = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let m = Move::new(king_sq, to_sq, None);
                let next_board = board.apply_move_to_board(m);
                if !next_board.is_legal(move_gen) {
                    continue;
                }

                // Check if this move immediately wins (king on center)
                let (nw, nb) = next_board.is_koth_win();
                if (board.w_to_move && nw) || (!board.w_to_move && nb) {
                    return Some(m);
                }

                // For n >= 2, check if the opponent can't prevent the win
                if n >= 2 && solve_koth(&next_board, move_gen, 1, max_ply, max_n) {
                    return Some(m);
                }
            }
        } else {
            // King already in ring — full movegen
            let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
            for m in captures.iter().chain(moves.iter()) {
                let next_board = board.apply_move_to_board(*m);
                if !next_board.is_legal(move_gen) {
                    continue;
                }

                // Check if this move immediately wins (king on center)
                let (nw, nb) = next_board.is_koth_win();
                if (board.w_to_move && nw) || (!board.w_to_move && nb) {
                    return Some(*m);
                }

                // For n >= 2, check if the opponent can't prevent the win
                if n >= 2 && solve_koth(&next_board, move_gen, 1, max_ply, max_n) {
                    return Some(*m);
                }
            }
        }
    }
    None
}

/// Like `koth_center_in_n` but also returns the number of nodes visited.
pub fn koth_center_in_n_counted(board: &Board, move_gen: &MoveGen, max_n: u8) -> (Option<u8>, u32) {
    let side_to_move = if board.w_to_move { WHITE } else { BLACK };
    let king_bit = board.get_piece_bitboard(side_to_move, KING);

    if king_bit == 0 {
        return (None, 1);
    }

    let (w_won, b_won) = board.is_koth_win();
    if (board.w_to_move && w_won) || (!board.w_to_move && b_won) {
        return (Some(0), 1);
    }

    let mut total_nodes: u32 = 1;
    for n in 1..=max_n {
        let max_ply = (n as i32) * 2 - 1;
        let (result, nodes) = solve_koth_counted(board, move_gen, 0, max_ply, max_n);
        total_nodes += nodes;
        if result {
            return (Some(n), total_nodes);
        }
    }
    (None, total_nodes)
}

/// Wrapper for backward compatibility: counted version with max_n=3.
pub fn koth_center_in_3_counted(board: &Board, move_gen: &MoveGen) -> (Option<u8>, u32) {
    koth_center_in_n_counted(board, move_gen, 3)
}

fn solve_koth_counted(
    board: &Board,
    move_gen: &MoveGen,
    ply: i32,
    max_ply: i32,
    max_n: u8,
) -> (bool, u32) {
    let mut nodes: u32 = 1;
    let (white_won, black_won) = board.is_koth_win();

    let am_i_root_side = ply % 2 == 0;
    let current_side_is_white = board.w_to_move;
    let root_side_is_white = if am_i_root_side {
        current_side_is_white
    } else {
        !current_side_is_white
    };

    if root_side_is_white {
        if white_won {
            return (true, nodes);
        }
        if black_won {
            return (false, nodes);
        }
    } else {
        if black_won {
            return (true, nodes);
        }
        if white_won {
            return (false, nodes);
        }
    }

    if ply > max_ply {
        return (false, nodes);
    }

    let is_root_turn = ply % 2 == 0;

    if is_root_turn {
        let root_side = if root_side_is_white { WHITE } else { BLACK };
        let king_bit = board.get_piece_bitboard(root_side, KING);
        let king_sq = king_bit.trailing_zeros() as usize;
        // remaining root moves after this one = max_n - ply/2 - 1
        let remaining_after = max_n as i32 - ply / 2 - 1;
        let target_mask = cumulative_ring_mask(remaining_after.max(0) as u8);
        let king_must_advance = (king_bit & target_mask) == 0;

        if king_must_advance {
            let friendly_occ = board.get_color_occupancy(root_side);
            let candidates = move_gen.k_move_bitboard[king_sq] & target_mask & !friendly_occ;
            let mut bits = candidates;
            while bits != 0 {
                let to_sq = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let m = Move::new(king_sq, to_sq, None);
                let next_board = board.apply_move_to_board(m);
                if !next_board.is_legal(move_gen) {
                    continue;
                }
                let (result, child_nodes) =
                    solve_koth_counted(&next_board, move_gen, ply + 1, max_ply, max_n);
                nodes += child_nodes;
                if result {
                    return (true, nodes);
                }
            }
            (false, nodes)
        } else {
            let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
            for m in captures.iter().chain(moves.iter()) {
                let next_board = board.apply_move_to_board(*m);
                if !next_board.is_legal(move_gen) {
                    continue;
                }

                let king_bit = next_board.get_piece_bitboard(root_side, KING);
                if (king_bit & target_mask) == 0 {
                    continue;
                }

                let (result, child_nodes) =
                    solve_koth_counted(&next_board, move_gen, ply + 1, max_ply, max_n);
                nodes += child_nodes;
                if result {
                    return (true, nodes);
                }
            }
            (false, nodes)
        }
    } else {
        // Opponent turn
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
        let mut legal_move_found = false;
        for m in captures.iter().chain(moves.iter()) {
            let next_board = board.apply_move_to_board(*m);
            if !next_board.is_legal(move_gen) {
                continue;
            }
            legal_move_found = true;
            let (result, child_nodes) =
                solve_koth_counted(&next_board, move_gen, ply + 1, max_ply, max_n);
            nodes += child_nodes;
            if !result {
                return (false, nodes);
            }
        }
        if !legal_move_found {
            return (false, nodes);
        }
        (true, nodes)
    }
}

fn solve_koth(board: &Board, move_gen: &MoveGen, ply: i32, max_ply: i32, max_n: u8) -> bool {
    let (white_won, black_won) = board.is_koth_win();

    // Determine who is the Root Side (the side that moved at ply 0)
    let am_i_root_side = ply % 2 == 0;
    let current_side_is_white = board.w_to_move;
    let root_side_is_white = if am_i_root_side {
        current_side_is_white
    } else {
        !current_side_is_white
    };

    // Check for win/loss conditions
    if root_side_is_white {
        if white_won {
            return true;
        } // Root (White) won
        if black_won {
            return false;
        } // Opponent (Black) won
    } else {
        if black_won {
            return true;
        } // Root (Black) won
        if white_won {
            return false;
        } // Opponent (White) won
    }

    if ply > max_ply {
        return false;
    }

    let is_root_turn = ply % 2 == 0;

    if is_root_turn {
        let root_side = if root_side_is_white { WHITE } else { BLACK };
        let king_bit = board.get_piece_bitboard(root_side, KING);
        let king_sq = king_bit.trailing_zeros() as usize;
        // remaining root moves after this one = max_n - ply/2 - 1
        let remaining_after = max_n as i32 - ply / 2 - 1;
        let target_mask = cumulative_ring_mask(remaining_after.max(0) as u8);
        let king_must_advance = (king_bit & target_mask) == 0;

        if king_must_advance {
            // Direct king-move generation: only enumerate king destinations
            // that land in the target ring, skipping full movegen entirely.
            let friendly_occ = board.get_color_occupancy(root_side);
            let candidates = move_gen.k_move_bitboard[king_sq] & target_mask & !friendly_occ;
            let mut bits = candidates;
            while bits != 0 {
                let to_sq = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let m = Move::new(king_sq, to_sq, None);
                let next_board = board.apply_move_to_board(m);
                if !next_board.is_legal(move_gen) {
                    continue;
                }
                if solve_koth(&next_board, move_gen, ply + 1, max_ply, max_n) {
                    return true;
                }
            }
            false
        } else {
            // King already in ring — need full movegen
            let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
            for m in captures.iter().chain(moves.iter()) {
                let next_board = board.apply_move_to_board(*m);
                if !next_board.is_legal(move_gen) {
                    continue;
                }

                // Post-apply ring check: king must still be close enough
                let king_bit = next_board.get_piece_bitboard(root_side, KING);
                if (king_bit & target_mask) == 0 {
                    continue;
                }

                if solve_koth(&next_board, move_gen, ply + 1, max_ply, max_n) {
                    return true;
                }
            }
            false
        }
    } else {
        // Opponent turn: try all moves, if ANY prevents Root from winning, branch fails.
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
        let mut legal_move_found = false;
        for m in captures.iter().chain(moves.iter()) {
            let next_board = board.apply_move_to_board(*m);
            if !next_board.is_legal(move_gen) {
                continue;
            }
            legal_move_found = true;
            if !solve_koth(&next_board, move_gen, ply + 1, max_ply, max_n) {
                return false;
            }
        }
        if !legal_move_found {
            return false; // Mate or stalemate
        }
        true
    }
}
