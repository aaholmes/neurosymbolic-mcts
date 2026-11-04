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
use crate::piece_types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
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
    let piece_type = match board.get_piece(mv.from()) {
        Some((_, pt)) => pt,
        None => return 0,
    };
    let enemy_color = if board.w_to_move { BLACK } else { WHITE };

    match piece_type {
        PAWN => {
            let attack_bb = if board.w_to_move {
                move_gen.wp_capture_bitboard[mv.to()]
            } else {
                move_gen.bp_capture_bitboard[mv.to()]
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
            let attack_bb = move_gen.n_move_bitboard[mv.to()];
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
/// Pruning for the extended quiescence search. The default prunes (one threat
/// probe per side per line, delta pruning with a 200 cp margin, losing captures
/// skipped), which is about 25 times faster on self-play leaves and changes the
/// value of about 4% of positions by more than 0.1; `exact` searches as the
/// original did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtConfig {
    /// A side that has run the threat probe does not run it again further down
    /// the same line of captures.
    pub probe_once: bool,
    /// Skip a capture (not a promotion) when the static evaluation plus the
    /// captured piece and this margin, in centipawns, cannot reach alpha.
    pub delta_margin_cp: Option<i32>,
    /// Skip captures (not promotions) that lose material by static exchange
    /// evaluation, when not in check.
    pub see_prune: bool,
    /// Plies searched at most.
    pub max_depth: u8,
}

impl ExtConfig {
    /// No pruning: the same score as the original search on every position.
    pub const fn exact() -> Self {
        Self {
            probe_once: false,
            delta_margin_cp: None,
            see_prune: false,
            max_depth: 20,
        }
    }
}

impl Default for ExtConfig {
    fn default() -> Self {
        Self {
            probe_once: true,
            delta_margin_cp: Some(200),
            see_prune: true,
            max_depth: 20,
        }
    }
}

/// Piece values for delta pruning, in centipawns (pawn to king).
const DELTA_VALUES: [i32; 6] = [100, 320, 330, 500, 975, 0];

pub fn ext_pesto_qsearch_counted(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    alpha: i32,
    beta: i32,
    max_depth: u8,
    white_tactic_used: bool,
    black_tactic_used: bool,
    white_null_used: bool,
    black_null_used: bool,
) -> (i32, bool, u32, u8) {
    let mut pool = Vec::new();
    let flags = ExtFlags {
        white_tactic_used,
        black_tactic_used,
        white_null_used,
        black_null_used,
    };
    let cfg = ExtConfig::default();
    ext_node(
        &mut pool, board, move_gen, pesto, alpha, beta, max_depth, flags, &cfg,
    )
}

/// Which side has used its one quiet tactical move and its one null move.
#[derive(Clone, Copy)]
struct ExtFlags {
    white_tactic_used: bool,
    black_tactic_used: bool,
    white_null_used: bool,
    black_null_used: bool,
}

/// Move lists for one search node, reused across nodes so that a search
/// allocates them once per level rather than at every position.
#[derive(Default)]
struct MoveLists {
    captures: Vec<Move>,
    promotions: Vec<Move>,
    quiets: Vec<Move>,
}

impl MoveLists {
    /// Every pseudo-legal move, as `gen_pseudo_legal_moves` returns them:
    /// captures then promotions in `captures`, the rest in `quiets`.
    fn fill(&mut self, board: &Board, move_gen: &MoveGen) {
        self.captures.clear();
        self.promotions.clear();
        self.quiets.clear();
        move_gen.gen_all_moves_into(
            board,
            &mut self.captures,
            &mut self.promotions,
            &mut self.quiets,
        );
        self.captures.append(&mut self.promotions);
    }

    /// Only the captures and promotions `fill` would put in `captures`, in the
    /// same order; `quiets` is left empty.
    fn fill_captures(&mut self, board: &Board, move_gen: &MoveGen) {
        self.captures.clear();
        self.promotions.clear();
        self.quiets.clear();
        move_gen.gen_captures_into(board, &mut self.captures, &mut self.promotions);
        self.captures.append(&mut self.promotions);
    }

    /// Captures and promotions in the order `gen_pseudo_legal_captures` returns them.
    fn sort_captures(&mut self, board: &Board, move_gen: &MoveGen) {
        self.captures
            .sort_unstable_by_key(|m| -move_gen.mvv_lva(board, m.from(), m.to()));
    }
}

#[allow(clippy::too_many_arguments)]
fn ext_node(
    pool: &mut Vec<MoveLists>,
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    alpha: i32,
    beta: i32,
    max_depth: u8,
    flags: ExtFlags,
    cfg: &ExtConfig,
) -> (i32, bool, u32, u8) {
    let mut lists = pool.pop().unwrap_or_default();
    let result = ext_body(
        &mut lists, pool, board, move_gen, pesto, alpha, beta, max_depth, flags, cfg,
    );
    pool.push(lists);
    result
}

#[allow(clippy::too_many_arguments)]
fn ext_body(
    lists: &mut MoveLists,
    pool: &mut Vec<MoveLists>,
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    mut alpha: i32,
    beta: i32,
    max_depth: u8,
    flags: ExtFlags,
    cfg: &ExtConfig,
) -> (i32, bool, u32, u8) {
    let ExtFlags {
        white_tactic_used,
        black_tactic_used,
        white_null_used,
        black_null_used,
    } = flags;
    let mut nodes: u32 = 1;
    let mut all_completed = true;
    let mut max_child_depth: u8 = 0;
    let in_check = board.current_state().is_check(move_gen);
    // Flags for this side's own moves below: with `probe_once`, a side that
    // probed here does not probe again further down.
    let mut main_flags = flags;
    let mut static_eval = 0;

    // Stand-pat with null-move threat detection (only when not in check)
    if !in_check {
        let stand_pat = pesto.pst_eval_cp(board.current_state());
        static_eval = stand_pat;

        let stm_is_white = board.current_state().w_to_move;
        let stm_null_used = if stm_is_white {
            white_null_used
        } else {
            black_null_used
        };

        // Null-move probe with "mystery square recapture" for forks: pass the
        // turn and search the opponent's best two captures and one quiet
        // tactical move. With two threats, the passer saves the more valuable
        // piece (moving it to a "mystery square"), the opponent takes the
        // other, and the saved piece recaptures the attacker: a fork costs
        // the lesser piece minus the forking piece. One threat: passing
        // answers it. No threats: stand-pat holds.
        // The probe can only lower stand-pat, so it is skipped when stand-pat
        // cannot raise alpha.
        let adjusted_stand_pat = if !stm_null_used && max_depth > 1 && stand_pat > alpha {
            if cfg.probe_once {
                if stm_is_white {
                    main_flags.white_null_used = true;
                } else {
                    main_flags.black_null_used = true;
                }
            }
            board.make_null_move_without_history();
            let probe_flags = ExtFlags {
                white_null_used: true,
                black_null_used: true,
                ..flags
            };

            // Up to two captures and one quiet tactical move, as (score, move).
            let mut scored = [(0i32, Move::null()); 3];
            let mut n_scored = 0;
            // Quiet moves are needed only if the opponent may still play a quiet tactic
            let opp_tactic_used = if stm_is_white {
                black_tactic_used
            } else {
                white_tactic_used
            };
            if opp_tactic_used {
                lists.fill_captures(board.current_state(), move_gen);
            } else {
                lists.fill(board.current_state(), move_gen);
            }
            lists.sort_captures(board.current_state(), move_gen);
            let mut legal_cap_count = 0;
            for &cap in &lists.captures {
                board.make_move_without_history(cap);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move_without_history();
                    continue;
                }
                let (score, child_completed, child_nodes, child_depth) = ext_node(
                    pool,
                    board,
                    move_gen,
                    pesto,
                    -beta,
                    -alpha,
                    max_depth - 2,
                    probe_flags,
                    cfg,
                );
                nodes += child_nodes;
                max_child_depth = max_child_depth.max(child_depth + 2);
                if !child_completed {
                    all_completed = false;
                }
                board.undo_move_without_history();
                scored[n_scored] = (-score, cap);
                n_scored += 1;
                legal_cap_count += 1;
                if legal_cap_count >= 2 {
                    break;
                }
            }

            // One opponent quiet tactical move (check or fork) if its budget allows
            if !opp_tactic_used {
                for &mv in &lists.quiets {
                    if !is_tactical_quiet(board.current_state(), mv, move_gen) {
                        continue;
                    }
                    board.make_move_without_history(mv);
                    if !board.current_state().is_legal(move_gen) {
                        board.undo_move_without_history();
                        continue;
                    }
                    let child_flags = ExtFlags {
                        white_tactic_used: !stm_is_white || white_tactic_used,
                        black_tactic_used: stm_is_white || black_tactic_used,
                        ..probe_flags
                    };
                    let (score, child_completed, child_nodes, child_depth) = ext_node(
                        pool,
                        board,
                        move_gen,
                        pesto,
                        -beta,
                        -alpha,
                        max_depth - 2,
                        child_flags,
                        cfg,
                    );
                    nodes += child_nodes;
                    max_child_depth = max_child_depth.max(child_depth + 2);
                    if !child_completed {
                        all_completed = false;
                    }
                    board.undo_move_without_history();
                    scored[n_scored] = (-score, mv);
                    n_scored += 1;
                    break;
                }
            }

            board.undo_move_without_history();

            // Worst for the passer (the opponent's best) first
            let scored = &mut scored[..n_scored];
            scored.sort_unstable_by_key(|(s, _)| *s);

            // Captures have a victim to recapture; quiet tactical moves do not
            let mut capture_scores = [(0i32, Move::null()); 3];
            let mut n_caps = 0;
            let mut all_scores = [0i32; 3];
            for (k, &(score, mv)) in scored.iter().enumerate() {
                all_scores[k] = score;
                if board.current_state().get_piece(mv.to()).is_some() {
                    capture_scores[n_caps] = (score, mv);
                    n_caps += 1;
                }
            }
            let all_scores = &mut all_scores[..n_scored];
            all_scores.sort_unstable();

            let null_threat = if all_scores.len() >= 2 {
                let second_score = all_scores[1]; // second-worst for passer

                let capture_scores = &mut capture_scores[..n_caps];
                capture_scores.sort_unstable_by_key(|(s, _)| *s);
                if capture_scores.len() >= 2 {
                    // Mystery-square recapture: deny the first capture, let the
                    // second happen, recapture the attacker with the saved piece.
                    let best_victim_sq = capture_scores[0].1.to();
                    let second_cap = capture_scores[1].1;
                    let second_cap_score = capture_scores[1].0;

                    board.make_null_move_without_history();
                    board.make_move_without_history(second_cap);
                    let recapture_threat = if board.current_state().is_legal(move_gen) {
                        let recapture = Move::new(best_victim_sq, second_cap.to(), None);
                        let recaptured_board = board.current_state().apply_move_to_board(recapture);
                        // From the opponent's side (to move); negate for the passer.
                        let recapture_eval = pesto.pst_eval_cp(&recaptured_board);
                        second_cap_score.max(-recapture_eval)
                    } else {
                        second_cap_score
                    };
                    board.undo_move_without_history();
                    board.undo_move_without_history();

                    second_score.max(recapture_threat)
                } else {
                    second_score
                }
            } else {
                stand_pat // one threat (passing answers it) or none
            };

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
        lists.fill_captures(board.current_state(), move_gen);
        let has_legal_capture = lists.captures.iter().any(|&cap| {
            board.make_move_without_history(cap);
            let legal = board.current_state().is_legal(move_gen);
            board.undo_move_without_history();
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

    // Quiet moves are needed when in check (every evasion) or for a quiet tactic
    if in_check || !stm_tactic_used {
        lists.fill(board.current_state(), move_gen);
    } else {
        lists.fill_captures(board.current_state(), move_gen);
    }
    if in_check {
        // In check: every pseudo-legal move is an evasion (free, no budget consumed)
        let mut any_legal = false;
        for &mv in lists.captures.iter().chain(lists.quiets.iter()) {
            board.make_move_without_history(mv);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move_without_history();
                continue;
            }
            any_legal = true;
            let (score, child_completed, child_nodes, child_depth) = ext_node(
                pool,
                board,
                move_gen,
                pesto,
                -beta,
                -alpha,
                max_depth - 1,
                flags,
                cfg,
            );
            nodes += child_nodes;
            max_child_depth = max_child_depth.max(child_depth + 1);
            let score = -score;
            if !child_completed {
                all_completed = false;
            }
            board.undo_move_without_history();
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
        // 1. Captures (MVV-LVA sorted)
        lists.sort_captures(board.current_state(), move_gen);
        for &capture in &lists.captures {
            if capture.promotion().is_none() {
                let victim = board.current_state().get_piece(capture.to());
                if let Some(margin) = cfg.delta_margin_cp {
                    // En passant: the square is empty and the victim is a pawn.
                    let value = victim.map_or(DELTA_VALUES[0], |(_, pt)| DELTA_VALUES[pt]);
                    if static_eval + value + margin <= alpha {
                        continue;
                    }
                }
                if cfg.see_prune
                    && victim.is_some()
                    && see(
                        board.current_state(),
                        move_gen,
                        capture.to(),
                        capture.from(),
                    ) < 0
                {
                    continue;
                }
            }
            board.make_move_without_history(capture);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move_without_history();
                continue;
            }
            let (score, child_completed, child_nodes, child_depth) = ext_node(
                pool,
                board,
                move_gen,
                pesto,
                -beta,
                -alpha,
                max_depth - 1,
                main_flags,
                cfg,
            );
            nodes += child_nodes;
            max_child_depth = max_child_depth.max(child_depth + 1);
            let score = -score;
            if !child_completed {
                all_completed = false;
            }
            board.undo_move_without_history();
            if score >= beta {
                return (beta, all_completed, nodes, max_child_depth);
            }
            if score > alpha {
                alpha = score;
            }
        }

        // 2. Quiet checks and forks, which use the moving side's budget
        if !stm_tactic_used {
            for &mv in &lists.quiets {
                if !is_tactical_quiet(board.current_state(), mv, move_gen) {
                    continue;
                }
                board.make_move_without_history(mv);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move_without_history();
                    continue;
                }
                let child_flags = ExtFlags {
                    white_tactic_used: stm_is_white || white_tactic_used,
                    black_tactic_used: !stm_is_white || black_tactic_used,
                    ..main_flags
                };
                let (score, child_completed, child_nodes, child_depth) = ext_node(
                    pool,
                    board,
                    move_gen,
                    pesto,
                    -beta,
                    -alpha,
                    max_depth - 1,
                    child_flags,
                    cfg,
                );
                nodes += child_nodes;
                max_child_depth = max_child_depth.max(child_depth + 1);
                let score = -score;
                if !child_completed {
                    all_completed = false;
                }
                board.undo_move_without_history();
                if score >= beta {
                    return (beta, all_completed, nodes, max_child_depth);
                }
                if score > alpha {
                    alpha = score;
                }
            }
        }
        // Threats are handled by the null-move probe above, so no retreats are searched.
    }

    (alpha, all_completed, nodes, max_child_depth)
}

/// Convenience wrapper: returns `(pawn_units, completed)` using extended quiescence.
pub fn forced_ext_pesto_balance(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> (f32, bool) {
    let (score_cp, completed, _nodes, _depth) = ext_pesto_qsearch_counted(
        board, move_gen, pesto, -100_000, 100_000, 20, false, false, false, false,
    );
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
    forced_ext_pesto_balance_with(board, move_gen, pesto, &ExtConfig::default())
}

/// `forced_ext_pesto_balance_counted` with the given pruning.
pub fn forced_ext_pesto_balance_with(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    cfg: &ExtConfig,
) -> (f32, bool, u32, u8) {
    let flags = ExtFlags {
        white_tactic_used: false,
        black_tactic_used: false,
        white_null_used: false,
        black_null_used: false,
    };
    let mut pool = Vec::new();
    let (score_cp, completed, nodes, depth) = ext_node(
        &mut pool,
        board,
        move_gen,
        pesto,
        -100_000,
        100_000,
        cfg.max_depth,
        flags,
        cfg,
    );
    (score_cp as f32 / 100.0, completed, nodes, depth)
}

// ======== Iterative Widening Q-Search Levels ========

/// Principal exchange evaluation (cap=0): follow the single best MVV-LVA capture
/// at each node. Produces a straight line (not a tree). Answers: "what's the
/// material outcome of the most forcing exchange sequence?"
///
/// Returns `(score_cp, nodes)`.
pub fn principal_exchange(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    mut alpha: i32,
    beta: i32,
    max_depth: u8,
) -> (i32, u32) {
    let mut nodes: u32 = 1;
    let stand_pat = pesto.pst_eval_cp(board.current_state());
    if stand_pat >= beta {
        return (beta, nodes);
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }
    if max_depth == 0 {
        return (alpha, nodes);
    }

    let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
    for capture in captures {
        board.make_move(capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }
        let (score, child_nodes) =
            principal_exchange(board, move_gen, pesto, -beta, -alpha, max_depth - 1);
        nodes += child_nodes;
        let score = -score;
        board.undo_move();
        if score >= beta {
            return (beta, nodes);
        }
        if score > alpha {
            alpha = score;
        }
        break; // only the first legal capture (top MVV-LVA)
    }
    (alpha, nodes)
}

/// Convenience wrapper for principal exchange.
pub fn forced_principal_exchange(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> (f32, u32) {
    let (score_cp, nodes) = principal_exchange(board, move_gen, pesto, -100_000, 100_000, 20);
    (score_cp as f32 / 100.0, nodes)
}

/// Cap=1 quiescence search: single best MVV-LVA capture per node, plus one
/// checking move per side per branch, plus full check evasions. No null-move
/// probe, no fork detection (those require cap>=2).
///
/// This is the first "real" level of iterative widening — it produces a narrow
/// tree (up to 2 children per node: 1 capture + 1 check) that resolves the
/// principal exchange AND one tactical check per side.
///
/// Returns `(score_cp, completed, nodes, depth_used)`.
pub fn cap1_pesto_qsearch(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    mut alpha: i32,
    beta: i32,
    max_depth: u8,
    white_check_used: bool,
    black_check_used: bool,
) -> (i32, bool, u32, u8) {
    let mut nodes: u32 = 1;
    let mut all_completed = true;
    let mut max_child_depth: u8 = 0;
    let in_check = board.current_state().is_check(move_gen);

    // Stand-pat (no null-move at cap=1)
    if !in_check {
        let stand_pat = pesto.pst_eval_cp(board.current_state());
        if stand_pat >= beta {
            return (beta, true, nodes, 0);
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    if max_depth == 0 {
        if in_check {
            return (alpha, false, nodes, 0);
        }
        return (alpha, true, nodes, 0);
    }

    let stm_is_white = board.current_state().w_to_move;

    if in_check {
        // In check: try all legal evasions (uncapped)
        let (caps, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
        let mut any_legal = false;
        for mv in caps.iter().chain(quiets.iter()) {
            board.make_move(*mv);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            any_legal = true;
            let (score, child_completed, child_nodes, child_depth) = cap1_pesto_qsearch(
                board,
                move_gen,
                pesto,
                -beta,
                -alpha,
                max_depth - 1,
                white_check_used,
                black_check_used,
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
            return (-1_000_000, true, nodes, 0); // checkmate
        }
    } else {
        // Not in check: 1 best capture + 1 checking move

        // 1. Best MVV-LVA capture (cap=1)
        let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
        let mut best_capture_to: Option<usize> = None;
        for capture in &captures {
            board.make_move(*capture);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            best_capture_to = Some(capture.to());
            let (score, child_completed, child_nodes, child_depth) = cap1_pesto_qsearch(
                board,
                move_gen,
                pesto,
                -beta,
                -alpha,
                max_depth - 1,
                white_check_used,
                black_check_used,
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
            break; // only the first legal capture
        }

        // 2. One checking move (if check budget not spent)
        let stm_check_used = if stm_is_white {
            white_check_used
        } else {
            black_check_used
        };
        if !stm_check_used {
            // Try all moves, find the first legal one that gives check
            // (that isn't the same as the capture we already tried)
            let (cap_moves, quiet_moves) = move_gen.gen_pseudo_legal_moves(board.current_state());
            for mv in cap_moves.iter().chain(quiet_moves.iter()) {
                // Skip if this is the same as the capture we already explored
                if let Some(best_to) = best_capture_to {
                    if mv.from()
                        == captures
                            .iter()
                            .find(|c| c.to() == best_to)
                            .map_or(usize::MAX, |c| c.from())
                        && mv.to() == best_to
                    {
                        continue;
                    }
                }
                if !board.current_state().gives_check(*mv, move_gen) {
                    continue;
                }
                board.make_move(*mv);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move();
                    continue;
                }
                // This side used their check budget
                let new_w_check = if stm_is_white { true } else { white_check_used };
                let new_b_check = if !stm_is_white {
                    true
                } else {
                    black_check_used
                };
                let (score, child_completed, child_nodes, child_depth) = cap1_pesto_qsearch(
                    board,
                    move_gen,
                    pesto,
                    -beta,
                    -alpha,
                    max_depth - 1,
                    new_w_check,
                    new_b_check,
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
                break; // only one check
            }
        }
    }

    (alpha, all_completed, nodes, max_child_depth)
}

/// Convenience wrapper for cap=1 q-search.
pub fn forced_cap1_pesto_balance(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> (f32, bool, u32, u8) {
    let (score_cp, completed, nodes, depth) =
        cap1_pesto_qsearch(board, move_gen, pesto, -100_000, 100_000, 20, false, false);
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
            let stand_pat = pesto.eval(board.current_state(), move_gen);
            return (stand_pat, nodes);
        }
    }

    let stand_pat = pesto.eval(board.current_state(), move_gen);

    if stand_pat >= beta {
        return (beta, nodes);
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    if max_depth <= 0 {
        return (alpha, nodes);
    }

    let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
    if captures.is_empty() && !board.is_check(move_gen) {
        return (alpha, nodes);
    }

    for capture in captures {
        if see(
            board.current_state(),
            move_gen,
            capture.to(),
            capture.from(),
        ) < 0
        {
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
    let stand_pat = pesto.eval(board.current_state(), move_gen);
    let mut best_score = stand_pat;
    let mut best_pv = Vec::new();

    let captures = move_gen.gen_pseudo_legal_captures(board.current_state());

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
