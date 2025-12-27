use crate::board::{Board, KOTH_CENTER};
use crate::move_generation::MoveGen;
use crate::piece_types::{KING, WHITE, BLACK};
use crate::board_utils::bit_to_sq_ind;

/// Distance rings for KOTH-in-3 geometric pruning
const RING_1: u64 = 0x00003C24243C0000 & !KOTH_CENTER; // Squares 1 distance from center
const RING_2: u64 = 0x007E424242427E00 & !(RING_1 | KOTH_CENTER); // Squares 2 distance from center

/// Checks if the side to move can reach the KOTH center within 3 of their own moves (ply 5),
/// assuming the opponent cannot block them or reach the center themselves first.
/// This is a "safety gate" to detect rapid KOTH wins.
pub fn koth_center_in_3(board: &Board, move_gen: &MoveGen) -> bool {
    let side_to_move = if board.w_to_move { WHITE } else { BLACK };
    let king_bit = board.get_piece_bitboard(side_to_move, KING);
    
    if king_bit == 0 { return false; }
    
    // We use a simplified recursive search (DFS) with pruning
    solve_koth_in_3(board, move_gen, 0)
}

fn solve_koth_in_3(board: &Board, move_gen: &MoveGen, ply: i32) -> bool {
    let (white_won, black_won) = board.is_koth_win();

    // Determine who is the Root Side (the side that moved at ply 0)
    // If ply is even, it's Root Side's turn.
    // If ply is odd, it's Opponent's turn.
    let am_i_root_side = ply % 2 == 0;
    let current_side_is_white = board.w_to_move;
    let root_side_is_white = if am_i_root_side { current_side_is_white } else { !current_side_is_white };

    // Check for win/loss conditions
    if root_side_is_white {
        if white_won { return true; }  // Root (White) won
        if black_won { return false; } // Opponent (Black) won
    } else {
        if black_won { return true; }  // Root (Black) won
        if white_won { return false; } // Opponent (White) won
    }

    if ply >= 5 { return false; } // Limit to 3 our moves (ply 0, 2, 4)

    let side_to_move = if board.w_to_move { WHITE } else { BLACK };
    let king_bit = board.get_piece_bitboard(side_to_move, KING);
    if king_bit == 0 { return false; }
    let king_sq = bit_to_sq_ind(king_bit);

    // PRUNING: Geometric Ring Constraints
    // Ply 0 (1st move): Must land on R2, R1, or Center
    // Ply 2 (2nd move): Must land on R1, or Center
    // Ply 4 (3rd move): Must land on Center
    /*
    let allowed_mask = match ply {
        0 => RING_2 | RING_1 | KOTH_CENTER, 
        2 => RING_1 | KOTH_CENTER,          
        4 => KOTH_CENTER,                   
        _ => !0,                            
    };
    */

    // If we (side to move) are making a move, apply pruning
    let is_our_move = ply % 2 == 0;

    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
    let mut legal_move_found = false;

    for m in captures.iter().chain(moves.iter()) {
        // If it's our move and it's a King move, apply ring pruning
        if is_our_move && m.from == king_sq {
            let bit_to = 1u64 << m.to;
            
            // RELAXED PRUNING:
            // At ply 0, we can land on Center, R1, or R2.
            // At ply 2, we can land on Center, R1.
            // At ply 4, we must land on Center.
            let is_allowed = match ply {
                0 => (bit_to & (KOTH_CENTER | RING_1 | RING_2)) != 0,
                2 => (bit_to & (KOTH_CENTER | RING_1)) != 0,
                4 => (bit_to & KOTH_CENTER) != 0,
                _ => true,
            };

            if !is_allowed {
                continue; // Pruned
            }
        }

        let next_board = board.apply_move_to_board(*m);
        if !next_board.is_legal(move_gen) { continue; }
        
        legal_move_found = true;

        if is_our_move {
            // We want to find AT LEAST ONE move that leads to a win
            if solve_koth_in_3(&next_board, move_gen, ply + 1) {
                return true;
            }
        } else {
            // Opponent wants to find AT LEAST ONE move that prevents our win
            // So if ANY opponent move prevents the win, this branch fails
            if !solve_koth_in_3(&next_board, move_gen, ply + 1) {
                return false;
            }
        }
    }

    if !legal_move_found {
        return false; // Mate or stalemate
    }

    // If it's our move and we checked all moves without returning true, we fail.
    // If it's opponent's move and we checked all moves without returning false, we succeed!
    !is_our_move
}