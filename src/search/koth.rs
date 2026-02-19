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

/// Distance rings for KOTH-in-3 geometric pruning
const RING_1: u64 = 0x00003C24243C0000 & !KOTH_CENTER; // Squares 1 distance from center
const RING_2: u64 = 0x007E424242427E00 & !(RING_1 | KOTH_CENTER); // Squares 2 distance from center

/// Checks if the side to move can reach the KOTH center within 3 of their own moves,
/// assuming the opponent cannot block them or reach the center themselves first.
/// Returns `Some(n)` where n is the minimum number of side-to-move moves needed (0–3),
/// or `None` if unreachable within 3 moves.
/// This is a "safety gate" to detect rapid KOTH wins.
pub fn koth_center_in_3(board: &Board, move_gen: &MoveGen) -> Option<u8> {
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

    // Try 1, 2, 3 root-side moves (max_ply = 1, 3, 5)
    for n in 1..=3u8 {
        let max_ply = (n as i32) * 2 - 1;
        if solve_koth(board, move_gen, 0, max_ply) {
            return Some(n);
        }
    }
    None
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
    let side_to_move = if board.w_to_move { WHITE } else { BLACK };
    let king_bit = board.get_piece_bitboard(side_to_move, KING);
    let king_sq = king_bit.trailing_zeros() as usize;

    for n in 1..=3u8 {
        let max_ply = (n as i32) * 2 - 1;

        // Pre-filter: target ring for root move (ply 0) matches solve_koth's ply-0 mask
        let target_mask = match n {
            1 => KOTH_CENTER,
            2 => KOTH_CENTER | RING_1,
            3 => KOTH_CENTER | RING_1 | RING_2,
            _ => unreachable!(),
        };
        let king_must_advance = (king_bit & target_mask) == 0;

        let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
        for m in captures.iter().chain(moves.iter()) {
            // Pre-filter: skip non-king and wrong-direction king moves
            if king_must_advance {
                if m.from != king_sq {
                    continue;
                }
                if (1u64 << m.to) & target_mask == 0 {
                    continue;
                }
            }

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
            if n >= 2 && solve_koth(&next_board, move_gen, 1, max_ply) {
                return Some(*m);
            }
        }
    }
    None
}

/// Like `koth_center_in_3` but also returns the number of nodes visited.
pub fn koth_center_in_3_counted(board: &Board, move_gen: &MoveGen) -> (Option<u8>, u32) {
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
    for n in 1..=3u8 {
        let max_ply = (n as i32) * 2 - 1;
        let (result, nodes) = solve_koth_counted(board, move_gen, 0, max_ply);
        total_nodes += nodes;
        if result {
            return (Some(n), total_nodes);
        }
    }
    (None, total_nodes)
}

fn solve_koth_counted(board: &Board, move_gen: &MoveGen, ply: i32, max_ply: i32) -> (bool, u32) {
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

    // Pre-filter: on root-side turns, if king isn't already in the target ring,
    // only try king moves whose destination is in the ring.
    let (king_must_advance, king_sq, target_mask) = if is_root_turn {
        let root_side = if root_side_is_white { WHITE } else { BLACK };
        let king_bit = board.get_piece_bitboard(root_side, KING);
        let mask = match ply {
            0 => KOTH_CENTER | RING_1 | RING_2,
            2 => KOTH_CENTER | RING_1,
            4 => KOTH_CENTER,
            _ => unreachable!(),
        };
        let must_advance = (king_bit & mask) == 0;
        (must_advance, king_bit.trailing_zeros() as usize, mask)
    } else {
        (false, 0, 0)
    };

    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
    let mut legal_move_found = false;

    for m in captures.iter().chain(moves.iter()) {
        // Pre-filter: skip non-king and wrong-direction king moves
        if king_must_advance {
            if m.from != king_sq {
                continue;
            }
            if (1u64 << m.to) & target_mask == 0 {
                continue;
            }
        }

        let next_board = board.apply_move_to_board(*m);
        if !next_board.is_legal(move_gen) {
            continue;
        }

        legal_move_found = true;

        if is_root_turn {
            let root_side = if root_side_is_white { WHITE } else { BLACK };
            let king_bit = next_board.get_piece_bitboard(root_side, KING);

            let allowed = match ply {
                0 => (king_bit & (KOTH_CENTER | RING_1 | RING_2)) != 0,
                2 => (king_bit & (KOTH_CENTER | RING_1)) != 0,
                4 => (king_bit & KOTH_CENTER) != 0,
                _ => unreachable!(),
            };

            if !allowed {
                continue;
            }

            let (result, child_nodes) = solve_koth_counted(&next_board, move_gen, ply + 1, max_ply);
            nodes += child_nodes;
            if result {
                return (true, nodes);
            }
        } else {
            let (result, child_nodes) = solve_koth_counted(&next_board, move_gen, ply + 1, max_ply);
            nodes += child_nodes;
            if !result {
                return (false, nodes);
            }
        }
    }

    if !legal_move_found {
        return (false, nodes);
    }

    (!is_root_turn, nodes)
}

fn solve_koth(board: &Board, move_gen: &MoveGen, ply: i32, max_ply: i32) -> bool {
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

    // Pre-filter: on root-side turns, if king isn't already in the target ring,
    // only try king moves whose destination is in the ring (skip all non-king moves
    // and wrong-direction king moves BEFORE the expensive apply_move_to_board).
    let (king_must_advance, king_sq, target_mask) = if is_root_turn {
        let root_side = if root_side_is_white { WHITE } else { BLACK };
        let king_bit = board.get_piece_bitboard(root_side, KING);
        let mask = match ply {
            0 => KOTH_CENTER | RING_1 | RING_2,
            2 => KOTH_CENTER | RING_1,
            4 => KOTH_CENTER,
            _ => unreachable!(),
        };
        let must_advance = (king_bit & mask) == 0;
        (must_advance, king_bit.trailing_zeros() as usize, mask)
    } else {
        (false, 0, 0)
    };

    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
    let mut legal_move_found = false;

    for m in captures.iter().chain(moves.iter()) {
        // Pre-filter: skip non-king and wrong-direction king moves
        if king_must_advance {
            if m.from != king_sq {
                continue;
            }
            if (1u64 << m.to) & target_mask == 0 {
                continue;
            }
        }

        let next_board = board.apply_move_to_board(*m);
        if !next_board.is_legal(move_gen) {
            continue;
        }

        legal_move_found = true;

        if is_root_turn {
            // After Root Side's move, their King MUST be close enough to win in remaining plies.
            let root_side = if root_side_is_white { WHITE } else { BLACK };
            let king_bit = next_board.get_piece_bitboard(root_side, KING);

            let allowed = match ply {
                0 => (king_bit & (KOTH_CENTER | RING_1 | RING_2)) != 0, // After Move 1: Dist <= 2
                2 => (king_bit & (KOTH_CENTER | RING_1)) != 0,          // After Move 2: Dist <= 1
                4 => (king_bit & KOTH_CENTER) != 0,                     // After Move 3: Dist == 0
                _ => unreachable!(),
            };

            if !allowed {
                continue; // Pruned: King didn't make enough progress or was moved away
            }

            if solve_koth(&next_board, move_gen, ply + 1, max_ply) {
                return true;
            }
        } else {
            // Opponent turn: If ANY move prevents Root from winning, this branch fails.
            if !solve_koth(&next_board, move_gen, ply + 1, max_ply) {
                return false;
            }
        }
    }

    if !legal_move_found {
        return false; // Mate or stalemate
    }

    // If it's Root's turn and no winning move found: false.
    // If it's Opponent's turn and all moves were checked without failure: true.
    !is_root_turn
}
