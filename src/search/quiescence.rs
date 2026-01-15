//! Quiescence search to resolve tactical positions.
//!
//! Quiescence search solves the "horizon effect" - the problem where a fixed-depth
//! search might stop in the middle of a capture sequence and return a misleading
//! evaluation. By searching captures until the position is "quiet", we get more
//! accurate evaluations.
//!
//! # Algorithm
//!
//! 1. **Stand-Pat**: Evaluate the current position. If it's already good enough
//!    to cause a beta cutoff, return immediately.
//!
//! 2. **Generate Captures**: Only consider capture moves (and possibly checks).
//!
//! 3. **SEE Pruning**: Skip captures that lose material according to Static
//!    Exchange Evaluation (e.g., QxP when the pawn is defended).
//!
//! 4. **Recurse**: Search remaining captures recursively until no captures remain
//!    or depth limit is reached.
//!
//! # Two Variants
//!
//! This module provides two quiescence search functions:
//!
//! - [`quiescence_search`]: Standard version returning just the score, used by
//!   alpha-beta search.
//!
//! - [`quiescence_search_tactical`]: Extended version returning a [`TacticalTree`]
//!   with the principal variation and sibling moves. Used by MCTS Tier 2 for
//!   tactical move grafting.
//!
//! # Integration with MCTS
//!
//! The `quiescence_search_tactical` function provides the **Tier 2 Tactical Grafting**
//! in the three-tier MCTS architecture:
//!
//! - Identifies forcing tactical sequences from any position
//! - Returns Q-value estimates for tactical moves to initialize MCTS nodes
//! - Helps MCTS avoid wasting simulations on obviously bad captures
//!
//! # Example
//!
//! ```ignore
//! use kingfisher::search::quiescence_search_tactical;
//! use kingfisher::boardstack::BoardStack;
//! use kingfisher::move_generation::MoveGen;
//! use kingfisher::eval::PestoEval;
//!
//! let mut board = BoardStack::new();
//! let move_gen = MoveGen::new();
//! let pesto = PestoEval::new();
//!
//! let tactical_tree = quiescence_search_tactical(&mut board, &move_gen, &pesto);
//!
//! println!("Best tactical line: {:?}", tactical_tree.principal_variation);
//! println!("Leaf evaluation: {}", tactical_tree.leaf_score);
//! for (mv, score) in &tactical_tree.siblings {
//!     println!("  Alternative: {:?} -> {}", mv, score);
//! }
//! ```

use super::see::see;
use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use std::time::{Duration, Instant};

/// Tactical result from Quiescence Search for MCTS grafting
#[derive(Clone, Debug)]
pub struct TacticalTree {
    pub principal_variation: Vec<Move>,
    pub leaf_score: i32,
    pub siblings: Vec<(Move, i32)>, // Other tactical moves at root and their scores
}

/// Performs a quiescence search to evaluate tactical sequences and avoid the horizon effect.
pub fn quiescence_search(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    mut alpha: i32,
    beta: i32,
    max_depth: i32, // Remaining q-search depth
    _verbose: bool,
    start_time: Option<Instant>,
    time_limit: Option<Duration>,
) -> (i32, i32) {
    let mut nodes = 1;

    if let (Some(start), Some(limit)) = (start_time, time_limit) {
        if start.elapsed() >= limit {
             let stand_pat = pesto.eval(&board.current_state(), move_gen);
             return (stand_pat, nodes);
        }
    }

    let stand_pat = pesto.eval(&board.current_state(), move_gen);

    if stand_pat >= beta {
        return (beta, nodes);
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    if max_depth <= 0 {
        return (alpha, nodes);
    }

    let captures = move_gen.gen_pseudo_legal_captures(&board.current_state());
    if captures.is_empty() && !board.is_check(move_gen) {
        return (alpha, nodes);
    }

    for capture in captures {
        if see(&board.current_state(), move_gen, capture.to, capture.from) < 0 {
            continue;
        }

        board.make_move(capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }

        let (mut score, n) = quiescence_search(
            board,
            move_gen,
            pesto,
            -beta,
            -alpha,
            max_depth - 1,
            _verbose,
            start_time,
            time_limit,
        );
        score = -score;
        nodes += n;

        board.undo_move();

        if score >= beta {
            return (beta, nodes);
        }
        if score > alpha {
            alpha = score;
        }
    }

    (alpha, nodes)
}

/// Specialized Quiescence Search for Tier 2 MCTS Integration
/// Returns the full TacticalTree instead of just a score.
pub fn quiescence_search_tactical(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> TacticalTree {
    let mut siblings = Vec::new();
    let stand_pat = pesto.eval(&board.current_state(), move_gen);
    let mut best_score = stand_pat;
    let mut best_pv = Vec::new();

    let captures = move_gen.gen_pseudo_legal_captures(&board.current_state());
    
    for capture in captures {
        board.make_move(capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }

        // Full search for the first level to find best tactical line
        let (score, _nodes) = quiescence_search(
            board,
            move_gen,
            pesto,
            -1000001,
            1000001,
            8,
            false,
            None,
            None,
        );
        let score = -score;
        board.undo_move();

        siblings.push((capture, score));

        if score > best_score {
            best_score = score;
            best_pv = vec![capture];
        }
    }

    TacticalTree {
        principal_variation: best_pv,
        leaf_score: best_score,
        siblings,
    }
}