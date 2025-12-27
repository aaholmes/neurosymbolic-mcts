use crate::board::{Board, KOTH_CENTER};
use crate::move_generation::MoveGen;
use crate::piece_types::{KING, WHITE, BLACK};

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

    let is_root_turn = ply % 2 == 0;
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
    let mut legal_move_found = false;

    for m in captures.iter().chain(moves.iter()) {
        let next_board = board.apply_move_to_board(*m);
        if !next_board.is_legal(move_gen) { continue; }
        
        legal_move_found = true;

        if is_root_turn {
            // After Root Side's move, their King MUST be close enough to win in remaining plies.
            let root_side = if root_side_is_white { WHITE } else { BLACK };
            let king_bit = next_board.get_piece_bitboard(root_side, KING);
            
            let allowed = match ply {
                0 => (king_bit & (KOTH_CENTER | RING_1 | RING_2)) != 0, // After Move 1: Dist <= 2
                2 => (king_bit & (KOTH_CENTER | RING_1)) != 0,          // After Move 2: Dist <= 1
                4 => (king_bit & KOTH_CENTER) != 0,                    // After Move 3: Dist == 0
                _ => unreachable!(),
            };

            if !allowed {
                continue; // Pruned: King didn't make enough progress or was moved away
            }

            if solve_koth_in_3(&next_board, move_gen, ply + 1) {
                return true;
            }
        } else {
            // Opponent turn: If ANY move prevents Root from winning, this branch fails.
            if !solve_koth_in_3(&next_board, move_gen, ply + 1) {
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
