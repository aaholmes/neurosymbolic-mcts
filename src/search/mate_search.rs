//! Mate search algorithm with configurable exhaustive depth.
//!
//! Searches for forced mates using iterative deepening with pure minimax.
//! The `exhaustive_depth` parameter controls which depths use exhaustive search
//! (all legal moves) vs checks-only search (only checking moves):
//!
//! - mate-in-N where N ≤ exhaustive_depth: **exhaustive** — all legal attacker moves tried
//! - mate-in-N where N > exhaustive_depth: **checks-only** — only checking moves on attacker plies
//!
//! With the default exhaustive_depth=2, this gives:
//! - Mate-in-1: exhaustive
//! - Mate-in-2: exhaustive — catches quiet-first mates like 1.Qg7! Kh8 2.Qh7#
//! - Mate-in-3+: checks-only — keeps branching manageable
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
//! let (score, best_move, nodes) = mate_search(&board, &move_gen, 6, false, 2);
//!
//! if score >= 1_000_000 {
//!     println!("Forced mate found! Play: {:?}", best_move);
//! }
//! ```

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::search::mate_order::{order_attacker_moves, order_defender_moves, MoveOrder};

/// Public API: Mate search with configurable exhaustive depth.
///
/// Searches for forced mates using iterative deepening. Mate-in-N where
/// N ≤ `exhaustive_depth` uses exhaustive search (all legal attacker moves),
/// while deeper levels use checks-only search. Default exhaustive_depth=2
/// makes mate-in-1 and mate-in-2 exhaustive, and mate-in-3+ checks-only.
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
    let mut nodes: i32 = 0;

    // Convert exhaustive_depth from mate-in-N to plies for internal comparison
    let exhaustive_plies = exhaustive_depth * 2 - 1;

    for d in 1..=max_depth {
        let depth = 2 * d - 1; // Only check odd depths (mate for us)
        let checks_only = depth > exhaustive_plies;

        let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);

        for m in captures.iter().chain(moves.iter()) {
            // On attacker turn with checks_only: filter before legality/clone
            if checks_only && !board.gives_check(*m, move_gen) {
                continue;
            }

            if !board.is_legal_after_move(*m, move_gen) {
                continue;
            }

            let next_board = board.apply_move_to_board(*m);

            let quiet = if checks_only { 0 } else { EVERY_MOVE };
            let found = with_ply_moves(depth, |lists| {
                solve_mate(
                    &next_board,
                    move_gen,
                    depth - 1,
                    false,
                    quiet,
                    MoveOrder::GENERATED,
                    &mut nodes,
                    lists,
                )
            });
            if found {
                return (1_000_000 + depth, *m, nodes);
            }
        }
    }

    (0, Move::null(), nodes)
}

/// Move lists for one ply, kept between searches so that no position allocates.
#[derive(Default)]
struct PlyMoves {
    captures: Vec<Move>,
    promotions: Vec<Move>,
    quiet: Vec<Move>,
}

thread_local! {
    static PLY_MOVES: std::cell::RefCell<Vec<PlyMoves>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Runs `f` with at least `plies` move lists for this thread.
fn with_ply_moves<R>(plies: i32, f: impl FnOnce(&mut [PlyMoves]) -> R) -> R {
    PLY_MOVES.with(|cell| {
        let mut lists = cell.borrow_mut();
        let plies = plies.max(1) as usize + 1;
        if lists.len() < plies {
            lists.resize_with(plies, PlyMoves::default);
        }
        f(&mut lists)
    })
}

/// A quiet-move budget that never runs out: every legal attacker move is searched.
const EVERY_MOVE: i32 = i32::MAX;

/// True if the side to move can force mate: in up to `exhaustive` moves
/// searching every legal move, or in up to `depth` moves playing checking
/// moves only. Unlike `mate_search` it neither finds the move nor the
/// shortest mate, so it runs at most two searches at full depth instead of
/// one per depth. A deeper search still finds shorter mates, because a
/// defender with no legal move while in check is mated.
pub fn has_forced_mate(board: &Board, move_gen: &MoveGen, depth: i32, exhaustive: i32) -> bool {
    has_forced_mate_counting(board, move_gen, depth, exhaustive).0
}

/// `has_forced_mate`, with the number of positions searched.
pub fn has_forced_mate_counting(
    board: &Board,
    move_gen: &MoveGen,
    depth: i32,
    exhaustive: i32,
) -> (bool, i32) {
    has_forced_mate_quiet(board, move_gen, depth, exhaustive, 0)
}

/// `has_forced_mate_counting`, except that the search to `depth` may also
/// play up to `quiet` non-checking attacker moves in each line. 0 is checking
/// moves only; `depth − 1` or more is every move, since the mating move
/// always gives check.
pub fn has_forced_mate_quiet(
    board: &Board,
    move_gen: &MoveGen,
    depth: i32,
    exhaustive: i32,
    quiet: i32,
) -> (bool, i32) {
    has_forced_mate_ordered(
        board,
        move_gen,
        depth,
        exhaustive,
        quiet,
        MoveOrder::DEFAULT,
    )
}

/// `has_forced_mate_quiet`, trying moves in `order`. The answer never depends
/// on the order; the number of positions searched does.
pub fn has_forced_mate_ordered(
    board: &Board,
    move_gen: &MoveGen,
    depth: i32,
    exhaustive: i32,
    quiet: i32,
    order: MoveOrder,
) -> (bool, i32) {
    let mut nodes = 0;
    let exhaustive = exhaustive.min(depth);
    with_ply_moves(2 * depth, |lists| {
        // The search with fewer quiet moves first: usually the smaller tree.
        if depth > exhaustive
            && solve_mate(
                board,
                move_gen,
                2 * depth - 1,
                true,
                quiet.max(0),
                order,
                &mut nodes,
                lists,
            )
        {
            return (true, nodes);
        }
        let found = exhaustive > 0
            && solve_mate(
                board,
                move_gen,
                2 * exhaustive - 1,
                true,
                EVERY_MOVE,
                order,
                &mut nodes,
                lists,
            );
        (found, nodes)
    })
}

/// Recursive pure minimax mate solver.
///
/// Returns true if the position is a forced mate for the attacker.
/// - Attacker: returns true if ANY child leads to mate (short-circuits on first success)
/// - Defender: returns true only if ALL children lead to mate (short-circuits on first refutation)
fn solve_mate(
    board: &Board,
    move_gen: &MoveGen,
    depth: i32,
    is_attackers_turn: bool,
    quiet: i32,
    order: MoveOrder,
    nodes: &mut i32,
    lists: &mut [PlyMoves],
) -> bool {
    *nodes += 1;
    // A limited budget is no use on the attacker's last move, which must mate
    // and so give check.
    let quiet = if quiet != EVERY_MOVE && is_attackers_turn && depth == 1 {
        0
    } else {
        quiet
    };

    let (here, deeper) = lists.split_first_mut().expect("one move list per ply");
    here.captures.clear();
    here.promotions.clear();
    here.quiet.clear();

    // Depth 0: checkmate if no move out of check is legal.
    if depth <= 0 {
        // Unless every move is searched, the attacker's last move gave check,
        // so the defender is in check at depth 0. Skip the is_check() call.
        if quiet == EVERY_MOVE && !board.is_check(move_gen) {
            return false;
        }
        return !move_gen.has_legal_evasion(board);
    }

    // With no quiet moves left, the attacker's moves are its checking moves.
    let checks_only = is_attackers_turn && quiet == 0;
    if checks_only {
        move_gen.gen_checking_moves_into(board, &mut here.captures);
    } else if !is_attackers_turn && board.is_check(move_gen) {
        move_gen.gen_evasions_into(board, &mut here.captures);
    } else {
        move_gen.gen_all_moves_into(
            board,
            &mut here.captures,
            &mut here.promotions,
            &mut here.quiet,
        );
    }
    let generated = if is_attackers_turn {
        order.attacker == MoveOrder::GENERATED.attacker
            || (order.attacker.checking_moves_only() && (!checks_only || depth == 1))
    } else {
        order.defender == MoveOrder::GENERATED.defender
    };
    if !generated {
        here.captures.extend_from_slice(&here.promotions);
        here.captures.extend_from_slice(&here.quiet);
        here.promotions.clear();
        here.quiet.clear();
        if is_attackers_turn {
            order_attacker_moves(board, move_gen, order.attacker, &mut here.captures);
        } else {
            order_defender_moves(board, move_gen, order.defender, &mut here.captures);
        }
    }
    let mut has_legal_move = false;

    for m in here
        .captures
        .iter()
        .chain(here.promotions.iter())
        .chain(here.quiet.iter())
    {
        // A quiet move by the attacker uses up one of its quiet moves.
        let mut child_quiet = quiet;
        if is_attackers_turn
            && !checks_only
            && quiet != EVERY_MOVE
            && !board.gives_check(*m, move_gen)
        {
            child_quiet = quiet - 1;
        }

        if !board.is_legal_after_move(*m, move_gen) {
            continue;
        }

        has_legal_move = true;
        let next_board = board.apply_move_to_board(*m);

        let child_result = solve_mate(
            &next_board,
            move_gen,
            depth - 1,
            !is_attackers_turn,
            child_quiet,
            order,
            nodes,
            deeper,
        );

        if is_attackers_turn {
            if child_result {
                return true; // Found a mating line
            }
        } else {
            // Defender
            if !child_result {
                return false; // Found a refutation
            }
        }
    }

    // No legal moves found
    if !has_legal_move {
        if is_attackers_turn {
            // No checking move (checks only), or the attacker is mated or
            // stalemated: either way no mate for the attacker.
            return false;
        }
        // The defender has no legal move: checkmate or stalemate
        return board.is_check(move_gen); // true = checkmate, false = stalemate
    }

    // Attacker: no child succeeded → no mate
    // Defender: all children led to mate → mate is forced
    !is_attackers_turn
}
