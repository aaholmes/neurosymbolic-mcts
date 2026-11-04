//! A frozen copy of the extended quiescence search as it was before it was made
//! faster. Exact speed-ups must return the same score and completed flag on
//! every position; the tests and `qsearch_bench` compare against this copy.
//! Do not change it.

use crate::board::Board;
use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::piece_types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};

/// Checks if a quiet (non-capture) move is tactical: gives check, creates a pawn fork,
/// or creates a knight fork.
fn is_tactical_quiet_ref(board: &Board, mv: Move, move_gen: &MoveGen) -> bool {
    if board.gives_check(mv, move_gen) {
        return true;
    }
    compute_fork_targets_ref(board, mv, move_gen) != 0
}

/// For a non-capture pawn or knight move, returns a bitboard of enemy pieces
/// being forked (attacked by the piece at its destination). Returns 0 if no fork.
///
/// Pawn fork: pawn attacks at dest hit 2+ enemy pieces in {N, B, R, Q, K}.
/// Knight fork: knight attacks at dest hit 2+ enemy pieces in {R, Q, K}.
fn compute_fork_targets_ref(board: &Board, mv: Move, move_gen: &MoveGen) -> u64 {
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
pub fn ext_pesto_qsearch_reference(
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
    if !in_check {
        let stand_pat = pesto.pst_eval_cp(board.current_state());

        let stm_is_white = board.current_state().w_to_move;
        let stm_null_used = if stm_is_white {
            white_null_used
        } else {
            black_null_used
        };

        // Null-move probe with "mystery square recapture" for forks.
        //
        // Pass the turn and evaluate each opponent response. When there are 2+
        // threats, model the fork resolution: saved piece moves to a "mystery
        // square," opponent captures the other piece, saved piece recaptures the
        // attacker. This bends chess rules slightly but correctly evaluates forks:
        //   Fork cost = lesser_piece - forking_piece (recapture recovers material)
        //
        // For single threats: pass fully addresses it (no second threat).
        // For no threats: stand-pat holds.
        let adjusted_stand_pat = if !stm_null_used && max_depth > 1 {
            board.make_null_move();
            let new_w_null = true;
            let new_b_null = true;

            // Evaluate the top 2 opponent captures (MVV-LVA sorted) and up to 1 tactical quiet.
            // We only need first and second choice for deny-first-choice + recapture.
            let opp_captures = move_gen.gen_pseudo_legal_captures(board.current_state());
            let mut scored_moves: Vec<(i32, Move)> = Vec::new();
            let mut legal_cap_count = 0;

            for cap in &opp_captures {
                board.make_move(*cap);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move();
                    continue;
                }
                let (score, child_completed, child_nodes, child_depth) =
                    ext_pesto_qsearch_reference(
                        board,
                        move_gen,
                        pesto,
                        -beta,
                        -alpha,
                        max_depth - 2,
                        white_tactic_used,
                        black_tactic_used,
                        new_w_null,
                        new_b_null,
                    );
                nodes += child_nodes;
                max_child_depth = max_child_depth.max(child_depth + 2);
                if !child_completed {
                    all_completed = false;
                }
                board.undo_move();
                scored_moves.push((-score, *cap));
                legal_cap_count += 1;
                if legal_cap_count >= 2 {
                    break;
                } // only need top 2
            }

            // Also evaluate one opponent tactical quiet (check or fork) if budget allows
            let opp_tactic_used = if stm_is_white {
                black_tactic_used
            } else {
                white_tactic_used
            };
            if !opp_tactic_used {
                let (_, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
                for mv in &quiets {
                    if !is_tactical_quiet_ref(board.current_state(), *mv, move_gen) {
                        continue;
                    }
                    board.make_move(*mv);
                    if !board.current_state().is_legal(move_gen) {
                        board.undo_move();
                        continue;
                    }
                    let opp_w_used = if !stm_is_white {
                        true
                    } else {
                        white_tactic_used
                    };
                    let opp_b_used = if stm_is_white {
                        true
                    } else {
                        black_tactic_used
                    };
                    let (score, child_completed, child_nodes, child_depth) =
                        ext_pesto_qsearch_reference(
                            board,
                            move_gen,
                            pesto,
                            -beta,
                            -alpha,
                            max_depth - 2,
                            opp_w_used,
                            opp_b_used,
                            new_w_null,
                            new_b_null,
                        );
                    nodes += child_nodes;
                    max_child_depth = max_child_depth.max(child_depth + 2);
                    if !child_completed {
                        all_completed = false;
                    }
                    board.undo_move();
                    scored_moves.push((-score, *mv));
                    break; // only need one tactical quiet
                }
            }

            board.undo_null_move();

            // Sort ascending (worst for passer first = opponent's best)
            scored_moves.sort_unstable_by_key(|(s, _)| *s);

            // Separate captures (have victims, eligible for recapture) from
            // tactical quiets (checks/forks, no victim to recapture)
            let mut capture_scores: Vec<(i32, Move)> = Vec::new();
            let mut all_scores: Vec<i32> = Vec::new();
            for &(score, mv) in &scored_moves {
                all_scores.push(score);
                // A move is a capture if there's a piece on the target square
                // (in the current board state, before null-move)
                if board.current_state().get_piece(mv.to()).is_some() {
                    capture_scores.push((score, mv));
                }
            }

            all_scores.sort_unstable();

            let null_threat = if all_scores.len() >= 2 {
                let second_score = all_scores[1]; // second-worst for passer

                // Try mystery-square recapture if there are 2+ captures with victims
                capture_scores.sort_unstable_by_key(|(s, _)| *s);
                if capture_scores.len() >= 2 {
                    // Fork resolution with mystery-square recapture:
                    // 1. Saved piece moves to "mystery square" (deny first capture)
                    // 2. Opponent captures the other piece (second capture)
                    // 3. Saved piece recaptures the attacker from mystery square
                    let best_victim_sq = capture_scores[0].1.to();
                    let second_cap = capture_scores[1].1;
                    let second_cap_score = capture_scores[1].0;

                    board.make_null_move();
                    board.make_move(second_cap);
                    let recapture_threat = if board.current_state().is_legal(move_gen) {
                        // Mystery recapture: saved piece teleports to attacker's landing
                        let recapture = Move::new(best_victim_sq, second_cap.to(), None);
                        let recaptured_board = board.current_state().apply_move_to_board(recapture);
                        let recapture_eval = pesto.pst_eval_cp(&recaptured_board);
                        // recaptured_board has opponent to move (passer just recaptured).
                        // pst_eval_cp returns from STM perspective = opponent's perspective.
                        // Negate for passer's perspective.
                        second_cap_score.max(-recapture_eval)
                    } else {
                        second_cap_score
                    };
                    board.undo_move();
                    board.undo_null_move();

                    // Use better of: overall second-best or recapture-adjusted capture
                    second_score.max(recapture_threat)
                } else {
                    second_score
                }
            } else if all_scores.len() == 1 {
                stand_pat // single threat — pass fully saves
            } else {
                stand_pat // no threats
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
            let (score, child_completed, child_nodes, child_depth) = ext_pesto_qsearch_reference(
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

        // 1. Captures (MVV-LVA sorted)
        let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
        for capture in &captures {
            board.make_move(*capture);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            let (score, child_completed, child_nodes, child_depth) = ext_pesto_qsearch_reference(
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
                if !is_tactical_quiet_ref(board.current_state(), *mv, move_gen) {
                    continue;
                }

                board.make_move(*mv);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move();
                    continue;
                }

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
                    ext_pesto_qsearch_reference(
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

        // No retreat moves needed — the null-move "deny first choice" mechanism
        // handles threats implicitly by assuming the passer saves the most
        // valuable piece and the opponent captures the second-most valuable.
    }

    (alpha, all_completed, nodes, max_child_depth)
}

/// The reference search from the root, as `forced_ext_pesto_balance_counted`:
/// `(pawn_units, completed, nodes, depth_used)`.
pub fn forced_ext_pesto_balance_reference(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> (f32, bool, u32, u8) {
    let (score_cp, completed, nodes, depth) = ext_pesto_qsearch_reference(
        board, move_gen, pesto, -100_000, 100_000, 20, false, false, false, false,
    );
    (score_cp as f32 / 100.0, completed, nodes, depth)
}
