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
use crate::board::Board;
use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::piece_types::{BLACK, BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use std::time::{Duration, Instant};

/// Material-only quiescence search for MCTS value function.
///
/// Uses `board.material_imbalance()` (pawn units, STM perspective) as eval —
/// no Pesto, no positional terms. Searches captures + promotions to find the
/// best material balance achievable through forced tactical exchanges.
///
/// Returns `(score, completed)` where:
/// - `score`: best material balance (pawn units) from STM perspective
/// - `completed`: `true` if search resolved naturally (ran out of captures or
///   stand-pat cutoff), `false` if depth limit was hit with captures remaining
pub fn material_qsearch(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    alpha: i32,
    beta: i32,
    max_depth: u8,
) -> (i32, bool) {
    let (score, completed, _nodes, _depth) =
        material_qsearch_counted(board, move_gen, alpha, beta, max_depth);
    (score, completed)
}

/// Material-only quiescence search that also returns the number of nodes visited
/// and the maximum depth actually used.
///
/// Returns `(score, completed, nodes, depth_used)` where `depth_used` is the
/// maximum depth reached relative to the initial call (0 = stand-pat, 1 = one
/// capture deep, etc.).
pub fn material_qsearch_counted(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    mut alpha: i32,
    beta: i32,
    max_depth: u8,
) -> (i32, bool, u32, u8) {
    let mut nodes: u32 = 1;
    let stand_pat = board.current_state().material_imbalance();
    if stand_pat >= beta {
        return (beta, true, nodes, 0);
    }
    alpha = alpha.max(stand_pat);

    let captures = move_gen.gen_pseudo_legal_captures(board.current_state());

    if max_depth == 0 {
        let has_legal_capture = captures.iter().any(|&cap| {
            board.make_move(cap);
            let legal = board.current_state().is_legal(move_gen);
            board.undo_move();
            legal
        });
        return (alpha, !has_legal_capture, nodes, 0);
    }

    let mut all_completed = true;
    let mut max_child_depth: u8 = 0;
    for capture in captures {
        board.make_move(capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }
        let (score, child_completed, child_nodes, child_depth) =
            material_qsearch_counted(board, move_gen, -beta, -alpha, max_depth - 1);
        nodes += child_nodes;
        max_child_depth = max_child_depth.max(child_depth + 1);
        let score = -score;
        if !child_completed {
            all_completed = false;
        }
        board.undo_move();
        if score >= beta {
            return (beta, all_completed, nodes, max_child_depth);
        }
        if score > alpha {
            alpha = score;
        }
    }
    (alpha, all_completed, nodes, max_child_depth)
}

/// Convenience wrapper: returns `(material_balance, completed)` where material_balance
/// is in pawn units (STM perspective) after optimal forced captures/promotions.
/// `completed` is true if the Q-search resolved all captures within its depth limit.
pub fn forced_material_balance(board: &mut BoardStack, move_gen: &MoveGen) -> (i32, bool) {
    material_qsearch(board, move_gen, -1000, 1000, 20)
}

/// Like `forced_material_balance` but also returns the number of nodes visited
/// and the maximum depth actually used.
///
/// Returns `(score, completed, nodes, depth_used)`.
pub fn forced_material_balance_counted(
    board: &mut BoardStack,
    move_gen: &MoveGen,
) -> (i32, bool, u32, u8) {
    material_qsearch_counted(board, move_gen, -1000, 1000, 20)
}

/// PeSTO-based quiescence search for MCTS value function.
///
/// Uses `pesto.pst_eval_cp(board)` (centipawns, STM perspective) as eval —
/// pure piece-square tables, no additional bonuses. Searches captures +
/// promotions to find the best positional+material balance through forced exchanges.
///
/// Returns `(score_cp, completed)` where:
/// - `score_cp`: best evaluation in centipawns from STM perspective
/// - `completed`: `true` if search resolved naturally, `false` if depth limit hit
pub fn pesto_qsearch(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    alpha: i32,
    beta: i32,
    max_depth: u8,
) -> (i32, bool) {
    let (score, completed, _nodes, _depth) =
        pesto_qsearch_counted(board, move_gen, pesto, alpha, beta, max_depth);
    (score, completed)
}

/// PeSTO-based quiescence search that also returns the number of nodes visited
/// and the maximum depth actually used.
///
/// Returns `(score_cp, completed, nodes, depth_used)`.
pub fn pesto_qsearch_counted(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    mut alpha: i32,
    beta: i32,
    max_depth: u8,
) -> (i32, bool, u32, u8) {
    let mut nodes: u32 = 1;
    let stand_pat = pesto.pst_eval_cp(board.current_state());
    if stand_pat >= beta {
        return (beta, true, nodes, 0);
    }
    alpha = alpha.max(stand_pat);

    let captures = move_gen.gen_pseudo_legal_captures(board.current_state());

    if max_depth == 0 {
        let has_legal_capture = captures.iter().any(|&cap| {
            board.make_move(cap);
            let legal = board.current_state().is_legal(move_gen);
            board.undo_move();
            legal
        });
        return (alpha, !has_legal_capture, nodes, 0);
    }

    let mut all_completed = true;
    let mut max_child_depth: u8 = 0;
    for capture in captures {
        board.make_move(capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }
        let (score, child_completed, child_nodes, child_depth) =
            pesto_qsearch_counted(board, move_gen, pesto, -beta, -alpha, max_depth - 1);
        nodes += child_nodes;
        max_child_depth = max_child_depth.max(child_depth + 1);
        let score = -score;
        if !child_completed {
            all_completed = false;
        }
        board.undo_move();
        if score >= beta {
            return (beta, all_completed, nodes, max_child_depth);
        }
        if score > alpha {
            alpha = score;
        }
    }
    (alpha, all_completed, nodes, max_child_depth)
}

/// Convenience wrapper: returns `(pawn_units, completed)` where pawn_units
/// is a float (centipawns / 100) from STM perspective after optimal forced
/// captures/promotions using PeSTO piece-square table evaluation.
pub fn forced_pesto_balance(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> (f32, bool) {
    let (score_cp, completed) = pesto_qsearch(board, move_gen, pesto, -100_000, 100_000, 20);
    (score_cp as f32 / 100.0, completed)
}

/// Like `forced_pesto_balance` but also returns the number of nodes visited
/// and the maximum depth actually used.
///
/// Returns `(pawn_units, completed, nodes, depth_used)`.
pub fn forced_pesto_balance_counted(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> (f32, bool, u32, u8) {
    let (score_cp, completed, nodes, depth) =
        pesto_qsearch_counted(board, move_gen, pesto, -100_000, 100_000, 20);
    (score_cp as f32 / 100.0, completed, nodes, depth)
}

// ======== Extended PeSTO Quiescence Search ========
// Extends capture-only qsearch with one non-capture tactical move per side:
// non-capture checks, pawn forks (2+ valuable pieces), knight forks (2+ high-value pieces).
// Check evasions and forked piece retreats are free (don't consume budget).

/// Checks if a quiet (non-capture) move is tactical: gives check, creates a pawn fork,
/// or creates a knight fork.
fn is_tactical_quiet(board: &Board, mv: Move, move_gen: &MoveGen) -> bool {
    if board.gives_check(mv, move_gen) {
        return true;
    }
    compute_fork_targets(board, mv, move_gen) != 0
}

/// For a non-capture pawn or knight move, returns a bitboard of enemy pieces
/// being forked (attacked by the piece at its destination). Returns 0 if no fork.
///
/// Pawn fork: pawn attacks at dest hit 2+ enemy pieces in {N, B, R, Q, K}.
/// Knight fork: knight attacks at dest hit 2+ enemy pieces in {R, Q, K}.
fn compute_fork_targets(board: &Board, mv: Move, move_gen: &MoveGen) -> u64 {
    let piece_type = match board.get_piece(mv.from) {
        Some((_, pt)) => pt,
        None => return 0,
    };
    let enemy_color = if board.w_to_move { BLACK } else { WHITE };

    match piece_type {
        PAWN => {
            let attack_bb = if board.w_to_move {
                move_gen.wp_capture_bitboard[mv.to]
            } else {
                move_gen.bp_capture_bitboard[mv.to]
            };
            let enemy_valuable = board.pieces[enemy_color][KNIGHT]
                | board.pieces[enemy_color][BISHOP]
                | board.pieces[enemy_color][ROOK]
                | board.pieces[enemy_color][QUEEN]
                | board.pieces[enemy_color][KING];
            let forked = attack_bb & enemy_valuable;
            if forked.count_ones() >= 2 {
                forked
            } else {
                0
            }
        }
        KNIGHT => {
            let attack_bb = move_gen.n_move_bitboard[mv.to];
            let enemy_high_value = board.pieces[enemy_color][ROOK]
                | board.pieces[enemy_color][QUEEN]
                | board.pieces[enemy_color][KING];
            let forked = attack_bb & enemy_high_value;
            if forked.count_ones() >= 2 {
                forked
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// Extended PeSTO-based quiescence search with tactical moves and null-move
/// threat detection.
///
/// Beyond captures+promotions, this search also considers:
/// - **Check evasions**: When in check, all legal moves are searched (no stand-pat)
/// - **Non-capture checks**: Quiet moves that give check (consume tactical budget)
/// - **Pawn forks**: Pawn advances attacking 2+ enemy pieces worth > pawn
/// - **Knight forks**: Knight moves attacking 2+ enemy pieces in {R, Q, K}
/// - **Null-move stand-pat**: Instead of trusting the static eval, pass the turn
///   and let the opponent capture. This detects hanging pieces, fork threats, and
///   any position where "doing nothing" loses material. Each side gets at most one
///   null move per branch.
///
/// Each side gets one non-capture tactical move per search.
///
/// Returns `(score_cp, completed, nodes, depth_used)`.
pub fn ext_pesto_qsearch_counted(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    mut alpha: i32,
    beta: i32,
    max_depth: u8,
    white_tactic_used: bool,
    black_tactic_used: bool,
    white_null_used: bool,
    black_null_used: bool,
) -> (i32, bool, u32, u8) {
    let mut nodes: u32 = 1;
    let mut all_completed = true;
    let mut max_child_depth: u8 = 0;
    let in_check = board.current_state().is_check(move_gen);

    // Stand-pat with null-move threat detection (only when not in check)
    let mut threatened_pieces: u64 = 0;
    if !in_check {
        let stand_pat = pesto.pst_eval_cp(board.current_state());

        let stm_is_white = board.current_state().w_to_move;
        let stm_null_used = if stm_is_white { white_null_used } else { black_null_used };

        // Null-move probe: pass the turn and see what opponent captures do.
        // This catches threats that stand-pat misses (forks, hanging pieces).
        let adjusted_stand_pat = if !stm_null_used && max_depth > 1 {
            board.make_null_move();
            // One null move per branch total — prevent opponent from also passing
            let new_w_null = true;
            let new_b_null = true;
            let (null_score, null_completed, null_nodes, null_depth) =
                ext_pesto_qsearch_counted(
                    board,
                    move_gen,
                    pesto,
                    -beta,
                    -alpha,
                    max_depth - 1,
                    white_tactic_used,
                    black_tactic_used,
                    new_w_null,
                    new_b_null,
                );
            nodes += null_nodes;
            max_child_depth = max_child_depth.max(null_depth + 1);
            if !null_completed {
                all_completed = false;
            }
            board.undo_null_move();
            let null_threat = -null_score;

            // If threat exists, identify which pieces are under attack
            if null_threat < stand_pat {
                board.make_null_move();
                let opp_captures = move_gen.gen_pseudo_legal_captures(board.current_state());
                board.undo_null_move();
                let stm_color = if stm_is_white { WHITE } else { BLACK };
                let stm_valuable = board.current_state().get_piece_bitboard(stm_color, KNIGHT)
                    | board.current_state().get_piece_bitboard(stm_color, BISHOP)
                    | board.current_state().get_piece_bitboard(stm_color, ROOK)
                    | board.current_state().get_piece_bitboard(stm_color, QUEEN);
                for cap in &opp_captures {
                    threatened_pieces |= 1u64 << cap.to;
                }
                threatened_pieces &= stm_valuable;
            }

            // Can't be better than what happens when opponent punishes a pass
            stand_pat.min(null_threat)
        } else {
            stand_pat
        };

        if adjusted_stand_pat >= beta {
            return (beta, all_completed, nodes, max_child_depth);
        }
        if adjusted_stand_pat > alpha {
            alpha = adjusted_stand_pat;
        }
    }

    if max_depth == 0 {
        if in_check {
            // Can't resolve check at depth 0
            return (alpha, false, nodes, 0);
        }
        let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
        let has_legal_capture = captures.iter().any(|&cap| {
            board.make_move(cap);
            let legal = board.current_state().is_legal(move_gen);
            board.undo_move();
            legal
        });
        return (alpha, !has_legal_capture, nodes, 0);
    }

    let stm_is_white = board.current_state().w_to_move;
    let stm_tactic_used = if stm_is_white {
        white_tactic_used
    } else {
        black_tactic_used
    };

    let mut any_legal = false;

    if in_check {
        // In check: generate ALL pseudo-legal moves as evasions (free, no budget consumed)
        let (caps, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
        for mv in caps.iter().chain(quiets.iter()) {
            board.make_move(*mv);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            any_legal = true;
            let (score, child_completed, child_nodes, child_depth) =
                ext_pesto_qsearch_counted(
                    board,
                    move_gen,
                    pesto,
                    -beta,
                    -alpha,
                    max_depth - 1,
                    white_tactic_used,
                    black_tactic_used,
                    white_null_used,
                    black_null_used,
                );
            nodes += child_nodes;
            max_child_depth = max_child_depth.max(child_depth + 1);
            let score = -score;
            if !child_completed {
                all_completed = false;
            }
            board.undo_move();
            if score >= beta {
                return (beta, all_completed, nodes, max_child_depth);
            }
            if score > alpha {
                alpha = score;
            }
        }
        if !any_legal {
            // Checkmate: no legal evasions
            return (-1_000_000, true, nodes, 0);
        }
    } else {
        // Not in check: captures first, then tactical quiets

        // 1. Captures (always, MVV-LVA sorted)
        let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
        for capture in &captures {
            board.make_move(*capture);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            any_legal = true;
            let (score, child_completed, child_nodes, child_depth) =
                ext_pesto_qsearch_counted(
                    board,
                    move_gen,
                    pesto,
                    -beta,
                    -alpha,
                    max_depth - 1,
                    white_tactic_used,
                    black_tactic_used,
                    white_null_used,
                    black_null_used,
                );
            nodes += child_nodes;
            max_child_depth = max_child_depth.max(child_depth + 1);
            let score = -score;
            if !child_completed {
                all_completed = false;
            }
            board.undo_move();
            if score >= beta {
                return (beta, all_completed, nodes, max_child_depth);
            }
            if score > alpha {
                alpha = score;
            }
        }

        // 2. Tactical quiets: non-capture checks and forks (when budget allows)
        if !stm_tactic_used {
            let (_, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
            for mv in &quiets {
                if !is_tactical_quiet(board.current_state(), *mv, move_gen) {
                    continue;
                }

                board.make_move(*mv);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move();
                    continue;
                }
                any_legal = true;

                // Tactical move consumes the moving side's budget
                let new_w_used = if stm_is_white {
                    true
                } else {
                    white_tactic_used
                };
                let new_b_used = if !stm_is_white {
                    true
                } else {
                    black_tactic_used
                };

                let (score, child_completed, child_nodes, child_depth) =
                    ext_pesto_qsearch_counted(
                        board,
                        move_gen,
                        pesto,
                        -beta,
                        -alpha,
                        max_depth - 1,
                        new_w_used,
                        new_b_used,
                        white_null_used,
                        black_null_used,
                    );
                nodes += child_nodes;
                max_child_depth = max_child_depth.max(child_depth + 1);
                let score = -score;
                if !child_completed {
                    all_completed = false;
                }
                board.undo_move();
                if score >= beta {
                    return (beta, all_completed, nodes, max_child_depth);
                }
                if score > alpha {
                    alpha = score;
                }
            }
        }

        // 3. Retreat moves for threatened pieces (when null-move revealed threats)
        if threatened_pieces != 0 {
            let (_, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
            for mv in &quiets {
                if threatened_pieces & (1u64 << mv.from) == 0 {
                    continue;
                }
                board.make_move(*mv);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move();
                    continue;
                }
                any_legal = true;
                // Retreats don't consume tactical budget
                let (score, child_completed, child_nodes, child_depth) =
                    ext_pesto_qsearch_counted(
                        board,
                        move_gen,
                        pesto,
                        -beta,
                        -alpha,
                        max_depth - 1,
                        white_tactic_used,
                        black_tactic_used,
                        white_null_used,
                        black_null_used,
                    );
                nodes += child_nodes;
                max_child_depth = max_child_depth.max(child_depth + 1);
                let score = -score;
                if !child_completed {
                    all_completed = false;
                }
                board.undo_move();
                if score >= beta {
                    return (beta, all_completed, nodes, max_child_depth);
                }
                if score > alpha {
                    alpha = score;
                }
            }
        }
    }

    (alpha, all_completed, nodes, max_child_depth)
}

/// Convenience wrapper: returns `(pawn_units, completed)` using extended quiescence.
pub fn forced_ext_pesto_balance(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> (f32, bool) {
    let (score_cp, completed, _nodes, _depth) =
        ext_pesto_qsearch_counted(board, move_gen, pesto, -100_000, 100_000, 20, false, false, false, false);
    (score_cp as f32 / 100.0, completed)
}

/// Like `forced_ext_pesto_balance` but also returns nodes visited and max depth.
///
/// Returns `(pawn_units, completed, nodes, depth_used)`.
pub fn forced_ext_pesto_balance_counted(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> (f32, bool, u32, u8) {
    let (score_cp, completed, nodes, depth) =
        ext_pesto_qsearch_counted(board, move_gen, pesto, -100_000, 100_000, 20, false, false, false, false);
    (score_cp as f32 / 100.0, completed, nodes, depth)
}

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
            board, move_gen, pesto, -1000001, 1000001, 8, false, None, None,
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
