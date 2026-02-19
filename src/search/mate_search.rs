//! Mate search algorithm with configurable exhaustive depth.
//!
//! Searches for forced mates using iterative deepening with alpha-beta pruning.
//! The `exhaustive_depth` parameter controls which depths use exhaustive search
//! (all legal moves) vs checks-only search (only checking moves):
//!
//! - depth ≤ exhaustive_depth: **exhaustive** — all legal attacker moves tried
//! - depth > exhaustive_depth: **checks-only** — only checking moves on attacker plies
//!
//! With the default exhaustive_depth=3, this gives:
//! - Mate-in-1 (depth 1): exhaustive
//! - Mate-in-2 (depth 3): exhaustive — catches quiet-first mates like 1.Qg7! Kh8 2.Qh7#
//! - Mate-in-3 (depth 5): checks-only — keeps branching manageable
//!
//! On the defender's plies all legal moves are always tried.
//!
//! # Integration with MCTS
//!
//! This module provides the **Tier 1 Safety Gate** in the three-tier MCTS
//! architecture. Before expanding any MCTS node, the engine checks for forced
//! mates:
//!
//! - If a forced win is found, the move is played immediately (no MCTS needed)
//! - Results are cached in the transposition table to avoid redundant searches
//!
//! # Score Convention
//!
//! - `1_000_000 + depth`: Forced mate in `depth` plies (winning)
//! - `-1_000_000 - depth`: Being mated in `depth` plies (losing)
//! - `0`: No forced mate found, or draw
//!
//! # Example
//!
//! ```ignore
//! use kingfisher::search::mate_search;
//! use kingfisher::board::Board;
//! use kingfisher::move_generation::MoveGen;
//!
//! let board = Board::new();
//! let move_gen = MoveGen::new();
//!
//! // Search for mate up to depth 6 (3 moves each side)
//! let (score, best_move, nodes) = mate_search(&board, &move_gen, 6, false, 3);
//!
//! if score >= 1_000_000 {
//!     println!("Forced mate found! Play: {:?}", best_move);
//! }
//! ```

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Result of a mate search
#[derive(Clone, Debug)]
struct MateResult {
    score: i32, // 1000000 for mate, -1000000 for mated
    best_move: Move,
    depth: i32, // Ply depth where mate was found
}

/// Shared context for all parallel searches
struct SearchContext {
    nodes_remaining: AtomicUsize,
    stop_signal: AtomicBool,
    best_result: Mutex<Option<MateResult>>,
}

impl SearchContext {
    fn new(node_budget: usize) -> Self {
        Self {
            nodes_remaining: AtomicUsize::new(node_budget),
            stop_signal: AtomicBool::new(false),
            best_result: Mutex::new(None),
        }
    }

    fn should_stop(&self) -> bool {
        self.stop_signal.load(Ordering::Relaxed)
            || self.nodes_remaining.load(Ordering::Relaxed) == 0
    }

    fn decrement_nodes(&self) {
        if self.nodes_remaining.load(Ordering::Relaxed) > 0 {
            self.nodes_remaining.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn report_mate(&self, result: MateResult) {
        let mut best = self.best_result.lock().unwrap();

        let is_improvement = match &*best {
            None => true,
            Some(old) => {
                if result.score >= 1_000_000 && old.score >= 1_000_000 {
                    result.depth < old.depth
                } else {
                    result.score > old.score
                }
            }
        };

        if is_improvement {
            *best = Some(result);
            if best.as_ref().unwrap().score >= 1_000_000 {
                self.stop_signal.store(true, Ordering::Relaxed);
            }
        }
    }
}

/// Public API: Mate search with configurable exhaustive depth.
///
/// Searches for forced mates using iterative deepening. Depths ≤ `exhaustive_depth`
/// use exhaustive search (all legal attacker moves), while deeper levels use
/// checks-only search. Default exhaustive_depth=3 makes mate-in-1 and mate-in-2
/// exhaustive, and mate-in-3 checks-only.
///
/// Stateless: operates on immutable `&Board` references (like KOTH search),
/// avoiding BoardStack overhead. Repetition detection is unnecessary — forced
/// mates exist regardless of position history.
pub fn mate_search(
    board: &Board,
    move_gen: &MoveGen,
    max_depth: i32,
    _verbose: bool,
    exhaustive_depth: i32,
) -> (i32, Move, i32) {
    let total_budget = 100_000;
    let context = SearchContext::new(total_budget);

    iterative_deepening_wrapper(&context, board, move_gen, max_depth, exhaustive_depth);

    let final_res = context.best_result.lock().unwrap().clone();
    let nodes_searched = total_budget - context.nodes_remaining.load(Ordering::Relaxed);

    if let Some(res) = final_res {
        // Double check legality in the ACTUAL root position
        if board.is_pseudo_legal(res.best_move, move_gen)
            && board.is_legal_after_move(res.best_move, move_gen)
        {
            return (res.score, res.best_move, nodes_searched as i32);
        }
    }

    (0, Move::null(), nodes_searched as i32)
}

fn iterative_deepening_wrapper(
    ctx: &SearchContext,
    board: &Board,
    move_gen: &MoveGen,
    max_depth: i32,
    exhaustive_depth: i32,
) {
    for d in 1..=max_depth {
        if ctx.should_stop() {
            break;
        }

        let depth = 2 * d - 1; // Only check odd depths (mate for us)
        let checks_only = depth > exhaustive_depth;

        let (score, best_move) = mate_search_recursive(
            ctx,
            board,
            move_gen,
            depth,
            -1_000_001,
            1_000_001,
            true,
            checks_only,
        );

        // If we found a forced mate at the root, report it
        if score >= 1_000_000 && best_move != Move::null() {
            if board.is_legal_after_move(best_move, move_gen) {
                ctx.report_mate(MateResult {
                    score,
                    best_move,
                    depth: d * 2 - 1,
                });
                break;
            }
        }
    }
}

/// Recursive mate search with alpha-beta pruning.
///
/// When `checks_only` is true, only checking moves are considered on the
/// attacker's plies. When false, all legal moves are tried (exhaustive).
/// On the defender's plies, all legal moves are always searched.
///
/// Stateless design: uses immutable `&Board` and `apply_move_to_board` (like
/// KOTH search), avoiding BoardStack make/undo overhead and repetition checks.
fn mate_search_recursive(
    ctx: &SearchContext,
    board: &Board,
    move_gen: &MoveGen,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    is_attackers_turn: bool,
    checks_only: bool,
) -> (i32, Move) {
    ctx.decrement_nodes();
    if ctx.should_stop() {
        return (0, Move::null());
    }

    // Depth 0: no mate found on this path, but still detect terminal positions.
    // Only checkmate matters here (stalemate = 0, same as default return).
    if depth <= 0 {
        if board.is_check(move_gen) {
            // Might be checkmate — verify no legal escape exists
            let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
            let has_legal = captures.iter().chain(moves.iter()).any(|m| {
                board.get_piece(m.from).is_some() && board.is_legal_after_move(*m, move_gen)
            });
            if !has_legal {
                return (-1_000_000 - depth, Move::null()); // Checkmate
            }
        }
        return (0, Move::null());
    }

    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
    let mut best_move = Move::null();
    let mut best_score = -1_000_001;
    let mut has_legal_move = false;

    for m in captures.iter().chain(moves.iter()) {
        if board.get_piece(m.from).is_none() {
            continue;
        }

        // On attacker turns with checks_only: filter BEFORE apply_move_to_board
        // gives_check() uses magic lookups on the pre-move board — much cheaper
        // than apply + is_check for the ~30 non-checking moves
        if is_attackers_turn && checks_only {
            if !board.gives_check(*m, move_gen) {
                continue;
            }
        }

        let next_board = board.apply_move_to_board(*m);

        if !next_board.is_legal(move_gen) {
            continue;
        }

        has_legal_move = true;

        // Recurse
        let (mut score, _) = mate_search_recursive(
            ctx,
            &next_board,
            move_gen,
            depth - 1,
            -beta,
            -alpha,
            !is_attackers_turn,
            checks_only,
        );
        score = -score;

        if ctx.should_stop() {
            return (0, Move::null());
        }

        if score > best_score {
            best_score = score;
            best_move = *m;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }

    // No legal moves found in this loop
    if !has_legal_move {
        if is_attackers_turn && checks_only {
            // No checking moves available — not necessarily terminal.
            // Non-checking legal moves may exist but were skipped by the filter.
            return (0, Move::null());
        }
        // Defender turn or exhaustive: truly no legal moves
        let in_check = board.is_check(move_gen);
        if in_check {
            return (-1_000_000 - depth, Move::null()); // Checkmate
        } else {
            return (0, Move::null()); // Stalemate
        }
    }

    // Legal moves exist but none improved alpha (or no improvement found)
    if best_score == -1_000_001 {
        return (0, Move::null());
    }

    (best_score, best_move)
}
