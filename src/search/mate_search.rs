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
use crate::piece_types::{BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};

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

            if solve_mate(
                &next_board,
                move_gen,
                depth - 1,
                false,
                checks_only,
                &mut nodes,
            ) {
                return (1_000_000 + depth, *m, nodes);
            }
        }
    }

    (0, Move::null(), nodes)
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
    checks_only: bool,
    nodes: &mut i32,
) -> bool {
    *nodes += 1;

    // Depth 0: check for checkmate using batched per-piece generation with early abort.
    // Generate moves by piece type in priority order, aborting as soon as any legal
    // evasion is found. This avoids generating all moves for all piece types in the
    // common case where a king move provides the evasion.
    if depth <= 0 {
        // In checks_only mode, the attacker only plays checking moves, so the
        // defender is always in check at depth 0. Skip the is_check() call.
        if !checks_only && !board.is_check(move_gen) {
            return false; // Not in check = not checkmate
        }

        // 1. King moves (direct bitboard — no vectors, no castling since in check)
        let stm = if board.w_to_move { WHITE } else { 1 };
        let king_sq = board.pieces[stm][KING].trailing_zeros() as usize;
        let friendly_occ = board.get_color_occupancy(stm);
        let mut king_bits = move_gen.k_move_bitboard[king_sq] & !friendly_occ;
        while king_bits != 0 {
            let to_sq = king_bits.trailing_zeros() as usize;
            king_bits &= king_bits - 1;
            if board.is_legal_after_move(Move::new(king_sq, to_sq, None), move_gen) {
                return false;
            }
        }

        // 2. Double check detection — if 2+ pieces attack king, only king moves
        //    work. Since king moves already failed above, it's checkmate.
        let opp = 1 - stm;
        let mut num_slider_attackers = 0u32;
        if (move_gen.gen_bishop_potential_captures(board, king_sq)
            & (board.pieces[opp][BISHOP] | board.pieces[opp][QUEEN]))
            != 0
        {
            num_slider_attackers += 1;
        }
        if (move_gen.gen_rook_potential_captures(board, king_sq)
            & (board.pieces[opp][ROOK] | board.pieces[opp][QUEEN]))
            != 0
        {
            num_slider_attackers += 1;
        }
        if num_slider_attackers >= 2 {
            return true; // Double slider check + no king escape = checkmate
        }
        if num_slider_attackers == 1 {
            // Check if a knight or pawn also attacks → double check
            let has_knight_attacker =
                (move_gen.n_move_bitboard[king_sq] & board.pieces[opp][KNIGHT]) != 0;
            if has_knight_attacker {
                return true; // Slider + knight double check, no king escape
            }
            let pawn_attack_bb = if stm == WHITE {
                move_gen.bp_capture_bitboard[king_sq]
            } else {
                move_gen.wp_capture_bitboard[king_sq]
            };
            if (pawn_attack_bb & board.pieces[opp][PAWN]) != 0 {
                return true; // Slider + pawn double check, no king escape
            }
        }

        // 3. Knight moves
        let (caps, moves) = move_gen.gen_knight_moves(board);
        if caps
            .iter()
            .chain(moves.iter())
            .any(|m| board.is_legal_after_move(*m, move_gen))
        {
            return false;
        }

        // 4. Bishop moves
        let (caps, moves) = move_gen.gen_bishop_moves(board);
        if caps
            .iter()
            .chain(moves.iter())
            .any(|m| board.is_legal_after_move(*m, move_gen))
        {
            return false;
        }

        // 5. Rook moves
        let (caps, moves) = move_gen.gen_rook_moves(board);
        if caps
            .iter()
            .chain(moves.iter())
            .any(|m| board.is_legal_after_move(*m, move_gen))
        {
            return false;
        }

        // 6. Queen moves
        let (caps, moves) = move_gen.gen_queen_moves(board);
        if caps
            .iter()
            .chain(moves.iter())
            .any(|m| board.is_legal_after_move(*m, move_gen))
        {
            return false;
        }

        // 7. Pawn moves (most complex generation, checked last)
        let (caps, promos, moves) = move_gen.gen_pawn_moves(board);
        if caps
            .iter()
            .chain(promos.iter())
            .chain(moves.iter())
            .any(|m| board.is_legal_after_move(*m, move_gen))
        {
            return false;
        }

        return true; // No legal moves = checkmate
    }

    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
    let mut has_legal_move = false;

    for m in captures.iter().chain(moves.iter()) {
        // On attacker turns with checks_only: filter before legality/clone
        if is_attackers_turn && checks_only && !board.gives_check(*m, move_gen) {
            continue;
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
            checks_only,
            nodes,
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
        if is_attackers_turn && checks_only {
            // No checking moves available — non-checking moves may exist
            return false;
        }
        // Truly no legal moves: checkmate or stalemate
        return board.is_check(move_gen); // true = checkmate, false = stalemate
    }

    // Attacker: no child succeeded → no mate
    // Defender: all children led to mate → mate is forced
    !is_attackers_turn
}
