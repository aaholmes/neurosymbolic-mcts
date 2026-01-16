//! Iterative deepening search wrapper.
//!
//! Provides iterative deepening around alpha-beta search for better time management
//! and move ordering via previous iterations. Each iteration searches one ply deeper,
//! allowing the engine to return the best move found within a time limit.
//!
//! # Benefits
//!
//! - **Time Management**: Can stop at any depth and return the best move found
//! - **Move Ordering**: Previous iteration's best move searched first
//! - **Aspiration Windows**: Future optimization opportunity for narrower search windows
//!
//! # Usage
//!
//! ```ignore
//! let (depth, eval, best_move, nodes) = iterative_deepening_ab_search(
//!     &mut board, &move_gen, &pesto, &mut tt,
//!     max_depth, q_depth, Some(time_limit), verbose
//! );
//! ```

use super::alpha_beta::alpha_beta_search;
use super::history::HistoryTable;
use super::history::MAX_PLY;
use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::{Move, NULL_MOVE};
use crate::transposition::TranspositionTable;
use crate::utils::print_move;
use std::time::{Duration, Instant};

/// Perform iterative deepening alpha-beta search from the given position
pub fn iterative_deepening_ab_search(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    tt: &mut TranspositionTable,
    max_depth: i32,
    q_search_max_depth: i32,
    time_limit: Option<Duration>,
    verbose: bool,
) -> (i32, i32, Move, i32) {
    let mut eval: i32 = 0;
    let mut best_move: Move = NULL_MOVE;
    let mut nodes: i32 = 0;
    let mut last_fully_searched_depth: i32 = 0;

    let start_time = Instant::now();
    let mut killers = [[NULL_MOVE; 2]; MAX_PLY];
    let mut history = HistoryTable::new();

    // Iterate over increasing depths
    for depth in 1..=max_depth {
        if verbose {
            println!("info depth {}", depth);
        }

        // Perform alpha-beta search for the current depth
        let (new_eval, new_best_move, new_nodes, terminated) = alpha_beta_search(
            board,
            move_gen,
            pesto,
            tt,
            &mut killers,
            &mut history,
            depth,
            -1000000,
            1000000,
            q_search_max_depth,
            false,
            Some(start_time),
            time_limit,
        );

        nodes += new_nodes;

        if terminated {
            if verbose {
                println!("info string Search terminated at depth {}", depth);
            }
            break;
        }

        eval = new_eval;
        if new_best_move != NULL_MOVE {
            best_move = new_best_move;
        }
        last_fully_searched_depth = depth;

        if verbose {
            let elapsed = start_time.elapsed().as_millis();
            let nps = if elapsed > 0 {
                (nodes as u128 * 1000) / elapsed
            } else {
                0
            };
            println!(
                "info depth {} score cp {} time {} nodes {} nps {} pv {}",
                depth,
                eval,
                elapsed,
                nodes,
                nps,
                print_move(&best_move)
            );
        }

        if let Some(limit) = time_limit {
            if start_time.elapsed() >= limit {
                if verbose {
                    println!("info string Time limit reached after depth {}", depth);
                }
                break;
            }
        }

        if eval.abs() > 900000 {
            if verbose {
                println!("info string Mate found at depth {}", depth);
            }
            break;
        }
    }

    (last_fully_searched_depth, eval, best_move, nodes)
}
