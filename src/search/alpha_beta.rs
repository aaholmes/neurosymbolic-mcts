//! Alpha-beta search with modern pruning enhancements.
//!
//! This module implements the core alpha-beta minimax search algorithm with several
//! enhancements to improve search efficiency:
//!
//! # Pruning Techniques
//!
//! - **Alpha-Beta Pruning**: The fundamental optimization that eliminates branches
//!   that cannot affect the final result.
//!
//! - **Check Extensions**: Extends search depth by 1 ply when a move gives check,
//!   ensuring tactical sequences are fully explored.
//!
//! - **History Heuristic**: Prioritizes moves that have caused beta cutoffs in
//!   previous searches, improving move ordering.
//!
//! - **Transposition Table**: Caches search results to avoid re-searching identical
//!   positions reached via different move orders.
//!
//! # Search Flow
//!
//! ```text
//! alpha_beta_search (root)
//!     │
//!     ├── Generate moves with eval-based ordering
//!     │
//!     ├── For each legal move:
//!     │   ├── Make move
//!     │   ├── alpha_beta_recursive (child)
//!     │   │   ├── Check extensions if in check
//!     │   │   ├── Recurse or call quiescence_search at depth 0
//!     │   │   └── Update alpha/beta bounds
//!     │   ├── Undo move
//!     │   └── Update best move if improved
//!     │
//!     └── Store result in transposition table
//! ```
//!
//! # Integration
//!
//! This search is primarily used by the `SimpleAgent` for pure alpha-beta play.
//! The MCTS system uses [`crate::search::mate_search`] and [`crate::search::quiescence_search`]
//! directly for tactical analysis.
//!
//! # Example
//!
//! ```ignore
//! use kingfisher::search::alpha_beta::alpha_beta_search;
//! use kingfisher::boardstack::BoardStack;
//! use kingfisher::move_generation::MoveGen;
//! use kingfisher::eval::PestoEval;
//!
//! let mut board = BoardStack::new();
//! let move_gen = MoveGen::new();
//! let pesto = PestoEval::new();
//! let mut tt = TranspositionTable::new();
//! let mut killers = [[Move::null(); 2]; MAX_PLY];
//! let mut history = HistoryTable::new();
//!
//! let (score, best_move, nodes, _) = alpha_beta_search(
//!     &mut board, &move_gen, &pesto, &mut tt,
//!     &mut killers, &mut history,
//!     6,           // depth
//!     -1000000,    // alpha
//!     1000000,     // beta
//!     4,           // q_search_max_depth
//!     false,       // verbose
//!     None, None,  // time limits
//! );
//! ```

use super::history::HistoryTable;
use super::history::MAX_PLY;
use super::quiescence::quiescence_search;
use crate::board::Board;
use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::{Move, NULL_MOVE};
use crate::transposition::TranspositionTable;
use std::time::{Duration, Instant};

/// Perform alpha-beta search from the given position
///
/// This function performs an exhaustive search to the given depth, using alpha-beta pruning
/// to optimize the search process. Includes NMP, Killers, History Heuristic, LMR.
///
/// # Arguments
///
/// * `board` - A mutable reference to the current board state
/// * `move_gen` - A reference to the move generator
/// * `pesto` - A reference to the Pesto evaluation function
/// * `tt` - Transposition Table
/// * `killers` - Killer move table
/// * `history` - History heuristic table
/// * `depth` - Remaining depth to search
/// * `alpha` - Alpha bound
/// * `beta` - Beta bound
/// * `q_search_max_depth` - Max depth for quiescence search
/// * `verbose` - Verbosity flag
/// * `start_time` - Start time for time limit checks
/// * `time_limit` - Optional time limit duration
///
/// # Returns
///
/// A tuple containing: (score, best_move, nodes_searched, terminated_early)
pub fn alpha_beta_search(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    tt: &mut TranspositionTable,
    killers: &mut [[Move; 2]; MAX_PLY],
    history: &mut HistoryTable,
    depth: i32,
    alpha_init: i32,
    beta_init: i32,
    q_search_max_depth: i32,
    verbose: bool,
    start_time: Option<Instant>,
    time_limit: Option<Duration>,
) -> (i32, Move, i32, bool) {
    let mut best_move: Move = NULL_MOVE;
    let mut alpha: i32 = alpha_init;
    let beta: i32 = beta_init;
    let mut n: i32 = 0;
    let mut eval: i32 = 0;

    // Check for checkmate and stalemate
    if verbose {
        println!("Checking for checkmate and stalemate");
    }
    let (checkmate, stalemate) = board.current_state().is_checkmate_or_stalemate(move_gen);
    if verbose {
        println!("Checkmate and stalemate checked");
        println!("Checkmate: {} Stalemate: {}", checkmate, stalemate);
    }

    // Handle checkmate and stalemate cases
    if checkmate {
        if verbose {
            println!("AB search: Checkmate!");
        }
        return (1000000, best_move, 1, true);
    } else if stalemate {
        if verbose {
            println!("AB search: Stalemate!");
        }
        return (0, best_move, 1, true);
    }

    // Generate and combine captures and regular moves
    let (mut captures, moves) = move_gen.gen_pseudo_legal_moves_with_evals(
        &mut board.current_state(),
        pesto,
        Some(history),
    );
    captures.extend(moves);

    // Print the list of captures
    if verbose {
        println!("Generated {} moves", captures.len());
        for m in &captures {
            println!("Move: {}", m);
        }
    }

    for m in captures {
        if verbose {
            println!("Considering move {} at root of search tree", m);
        }
        board.make_move(m);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }
        let (search_eval, nodes, _terminated) = alpha_beta_recursive(
            board,
            move_gen,
            pesto,
            tt,
            killers,
            history,
            depth - 1,
            -beta,
            -alpha,
            q_search_max_depth,
            verbose,
            None, // start_time
            None, // time_limit
        );
        eval = -search_eval;
        n += nodes;
        if eval > alpha {
            alpha = eval;
            best_move = m;
        }

        if verbose {
            println!(
                "Just checked move {}, current best move is {}",
                m, best_move
            );
            if let Some(start_time) = start_time {
                println!(
                    "Current time: {:?}, time limit: {:?}",
                    start_time.elapsed(),
                    time_limit
                );
            }
        }

        // Check time limit
        if let Some(start_time) = start_time {
            if let Some(time_limit) = time_limit {
                if start_time.elapsed() > time_limit {
                    board.undo_move();
                    if verbose {
                        println!("Time limit reached. Stopping search.");
                    }
                    return (alpha, best_move, nodes, true);
                }
            }
        }

        // Undo the move
        board.undo_move();

        // Prune if necessary
        if alpha >= beta {
            // Update history table for the cutoff move
            if !is_capture(&board.current_state(), &best_move) {
                history.update(&best_move, depth);
            }
            break;
        }
    }

    if verbose {
        println!(
            "Alpha beta search at depth {} searched {} nodes. Best eval and move are {} {}",
            depth, n, alpha, best_move
        );
    }

    // Store the result in the transposition table
    tt.store(board.current_state(), depth, eval, best_move);

    (alpha, best_move, n, false)
}

/// Recursive helper function for alpha-beta search
/// Returns (score, nodes_searched, terminated_early)
fn alpha_beta_recursive(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    tt: &mut TranspositionTable,
    killers: &mut [[Move; 2]; MAX_PLY],
    history: &mut HistoryTable,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    q_search_max_depth: i32,
    verbose: bool,
    start_time: Option<Instant>, // Added
    time_limit: Option<Duration>, // Added
) -> (i32, i32, bool) { // Added bool for termination flag
    // --- Time Check (Periodic) ---
    // Check every N nodes (e.g., 2048) to balance overhead and responsiveness
    // Note: 'n' is the node count for *this* call, not total. Need total node count passed down or a shared counter.
    // Let's simplify: Check time at the start of each recursive call for now. Less efficient but simpler.
    if let (Some(start), Some(limit)) = (start_time, time_limit) {
        if start.elapsed() >= limit {
            return (0, 0, true); // Return 0 score, 0 nodes, terminated=true
        }
    }
    // --- Depth Check ---
    if depth <= 0 {
        let (score, nodes) = quiescence_search(
            board,
            move_gen,
            pesto,
            alpha,
            beta,
            q_search_max_depth,
            verbose,
            start_time, // Pass time info down
            time_limit, // Pass time info down
        );
        // Check if quiescence search terminated due to time
        let terminated = if let (Some(start), Some(limit)) = (start_time, time_limit) {
             start.elapsed() >= limit
        } else { false };
        return (score, nodes, terminated);
    }

    let mut best_eval: i32 = -1000000;
    let mut n: i32 = 1; // Count current node

    // Generate and combine captures and regular moves
    let (mut captures, moves) = move_gen.gen_pseudo_legal_moves_with_evals(
        &mut board.current_state(),
        pesto,
        Some(history),
    );
    captures.extend(moves);

    // Iterate through all moves
    for m in captures {
        board.make_move(m);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }

        // --- Check Extension ---
        // Extend search if the move 'm' results in check for the opponent
        let mut extension = 0;
        if board.is_check(move_gen) {
            extension = 1;
            // Optional: Limit extension depth or based on move type (e.g., only for non-quiet moves?)
        }
        let new_depth = depth - 1 + extension;

        let (mut eval, nodes, terminated) = alpha_beta_recursive(
            board,
            move_gen,
            pesto,
            tt,
            killers,
            history,
            new_depth,
            -beta,
            -alpha,
            q_search_max_depth,
            verbose,
            start_time, // Pass down
            time_limit, // Pass down
        );
        n += nodes; // Accumulate nodes searched

        // If a sub-search terminated due to time, propagate the termination upwards
        if terminated {
            board.undo_move();
            return (0, n, true); // Return dummy score, accumulated nodes, terminated=true
        }

        eval = -eval;
        // n += nodes; // Moved up

        // Update best evaluation
        best_eval = best_eval.max(eval);
        if eval > alpha {
            alpha = eval;
            if alpha >= beta {
                // Update history table for the cutoff move
                if !is_capture(&board.current_state(), &m) {
                    history.update(&m, depth);
                }
                board.undo_move();
                return (beta, n, false); // Not terminated here
            }
        }

        // Undo the move
        board.undo_move();
    }

    (best_eval, n, false) // Not terminated if loop completes normally
}

/// Helper function to check if a move is a capture
fn is_capture(board: &Board, mv: &Move) -> bool {
    let target_square_bb = 1u64 << mv.to;
    let opponent_color = !board.w_to_move as usize;

    // Check if the target square is occupied by an opponent's piece
    (board.pieces_occ[opponent_color] & target_square_bb) != 0
}
