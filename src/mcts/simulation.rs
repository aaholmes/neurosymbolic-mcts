//! Implements the Simulation (Rollout) phase of MCTS.

use crate::board::Board;
use crate::move_generation::MoveGen;
use rand::seq::SliceRandom;

const MAX_PLAYOUT_DEPTH: u32 = 100; // Prevent infinite loops

/// Simulates a random playout from the given board state
/// Returns the result from perspective of the player to move:
/// 1.0 = win, 0.5 = draw, 0.0 = loss
pub fn simulate_random_playout(state: &Board, move_gen: &MoveGen) -> f64 {
    let mut current_state = state.clone();
    let side_to_move = current_state.w_to_move; // Remember whose perspective we're evaluating

    // Playout until terminal state or max depth
    for _ in 0..MAX_PLAYOUT_DEPTH {
        // Check if game is over using checkmate/stalemate detection
        let (is_checkmate, is_stalemate) = current_state.is_checkmate_or_stalemate(move_gen);

        if is_checkmate {
            // If the side that's in checkmate is the same as our original side, we lost
            return if current_state.w_to_move == side_to_move {
                0.0
            } else {
                1.0
            };
        } else if is_stalemate {
            return 0.5; // Draw
        }

        // Get pseudo-legal moves
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(&current_state);

        // Combine and filter for legal moves
        let mut legal_moves = Vec::with_capacity(captures.len() + moves.len());

        // Check captures for legality
        for m in captures {
            if current_state.get_piece(m.from).is_none() {
                continue;
            }
            let new_board = current_state.apply_move_to_board(m);
            if new_board.is_legal(move_gen) {
                legal_moves.push(m);
            }
        }

        // Check moves for legality
        for m in moves {
            if current_state.get_piece(m.from).is_none() {
                continue;
            }
            let new_board = current_state.apply_move_to_board(m);
            if new_board.is_legal(move_gen) {
                legal_moves.push(m);
            }
        }

        // No legal moves, must be a draw (should be caught by stalemate check above)
        if legal_moves.is_empty() {
            return 0.5;
        }

        // Select random move
        let random_move = legal_moves
            .choose(&mut rand::thread_rng())
            .expect("Legal moves should not be empty here");

        // Apply move
        current_state = current_state.apply_move_to_board(*random_move);
    }

    // If we reach max depth, use a simple material evaluation to determine result
    simple_material_eval(&current_state, side_to_move)
}

/// Simple material evaluation for when we reach max playout depth
/// Returns a score in [0.0, 1.0] range from perspective of side_to_move
fn simple_material_eval(state: &Board, side_to_move: bool) -> f64 {
    use crate::piece_types::*;

    // Count material (pawns=1, knights/bishops=3, rooks=5, queens=9)
    let piece_values = [1, 3, 3, 5, 9, 0]; // P, N, B, R, Q, K

    let mut white_material = 0;
    let mut black_material = 0;

    // Count white material
    for piece_type in 0..6 {
        let pieces = state.pieces[WHITE][piece_type];
        let count = pieces.count_ones();
        white_material += count as i32 * piece_values[piece_type];
    }

    // Count black material
    for piece_type in 0..6 {
        let pieces = state.pieces[BLACK][piece_type];
        let count = pieces.count_ones();
        black_material += count as i32 * piece_values[piece_type];
    }

    // Calculate total material
    let total_material = white_material + black_material;
    if total_material == 0 {
        return 0.5; // Draw if no material
    }

    // Material advantage normalized to [0.0, 1.0] range with sigmoid
    let advantage = if side_to_move {
        white_material as f64 - black_material as f64
    } else {
        black_material as f64 - white_material as f64
    };

    // Simple sigmoid function to map material advantage to [0.0, 1.0]
    0.5 + 0.5 * (advantage / 10.0).tanh()
}
