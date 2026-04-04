#include "quiescence.cuh"
#include "pesto_tables.cuh"
#include "movegen.cuh"
#include "apply_move.cuh"

// ============================================================
// PeSTO tapered evaluation
// ============================================================

__device__ int32_t pesto_eval_cp(const BoardState* bs) {
    int32_t mg[2] = {0, 0};
    int32_t eg[2] = {0, 0};
    int32_t game_phase = 0;

    for (int color = 0; color < 2; color++) {
        for (int piece = 0; piece < 6; piece++) {
            uint64_t bb = bs->pieces[color * 6 + piece];
            while (bb) {
                int sq = pop_lsb(bb);
                mg[color] += c_MG_PESTO[color * 6 + piece][sq];
                eg[color] += c_EG_PESTO[color * 6 + piece][sq];
                game_phase += c_PHASE_WEIGHT[piece];
            }
        }
    }

    int32_t mg_score = mg[0] - mg[1];  // white - black
    int32_t eg_score = eg[0] - eg[1];
    int32_t mg_phase = game_phase < 24 ? game_phase : 24;
    int32_t eg_phase = 24 - mg_phase;
    int32_t score = (mg_score * mg_phase + eg_score * eg_phase) / 24;

    // Return from STM perspective
    return bs->w_to_move ? score : -score;
}

// ============================================================
// Helper functions
// ============================================================

__device__ bool is_in_check(const BoardState* bs) {
    int stm = bs->w_to_move ? WHITE : BLACK;
    int enemy = 1 - stm;
    uint64_t king_bb = bs->pieces[stm * 6 + KING];
    if (king_bb == 0) return false;
    int king_sq = __ffsll(king_bb) - 1;
    return is_square_attacked(bs, king_sq, enemy);
}

__device__ bool gives_check(const BoardState* bs, GPUMove mv) {
    BoardState child = *bs;
    apply_move(&child, mv);
    // After apply_move, side to move has flipped.
    // Check if the new STM's king is in check (i.e., the side that just moved attacks the new STM's king).
    // Actually: we want to know if the move gives check to the opponent.
    // After apply_move, child.w_to_move is the opponent. Check if their king is attacked.
    int opp = child.w_to_move ? WHITE : BLACK;
    uint64_t opp_king_bb = child.pieces[opp * 6 + KING];
    if (opp_king_bb == 0) return false;
    int opp_king_sq = __ffsll(opp_king_bb) - 1;
    int attacker = 1 - opp;
    return is_square_attacked(&child, opp_king_sq, attacker);
}

__device__ uint64_t compute_fork_targets(const BoardState* bs, GPUMove mv) {
    int from = GPU_MOVE_FROM(mv);
    int to = GPU_MOVE_TO(mv);
    int stm = bs->w_to_move ? WHITE : BLACK;
    int enemy = 1 - stm;

    // Determine piece type at 'from'
    int piece_type = -1;
    uint64_t from_bit = 1ULL << from;
    for (int p = 0; p < 6; p++) {
        if (bs->pieces[stm * 6 + p] & from_bit) {
            piece_type = p;
            break;
        }
    }

    if (piece_type == PAWN) {
        // Pawn fork: attacks at destination hit 2+ enemy valuable pieces
        uint64_t attack_bb;
        if (stm == WHITE) {
            // Use extern constant from movegen_tables.cuh
            extern __constant__ uint64_t c_wp_cap_bb[64];
            attack_bb = c_wp_cap_bb[to];
        } else {
            extern __constant__ uint64_t c_bp_cap_bb[64];
            attack_bb = c_bp_cap_bb[to];
        }
        uint64_t enemy_valuable = bs->pieces[enemy * 6 + KNIGHT]
            | bs->pieces[enemy * 6 + BISHOP]
            | bs->pieces[enemy * 6 + ROOK]
            | bs->pieces[enemy * 6 + QUEEN]
            | bs->pieces[enemy * 6 + KING];
        uint64_t forked = attack_bb & enemy_valuable;
        return popcount(forked) >= 2 ? forked : 0;
    } else if (piece_type == KNIGHT) {
        // Knight fork: attacks at destination hit 2+ enemy high-value pieces
        extern __constant__ uint64_t c_n_move_bb[64];
        uint64_t attack_bb = c_n_move_bb[to];
        uint64_t enemy_high = bs->pieces[enemy * 6 + ROOK]
            | bs->pieces[enemy * 6 + QUEEN]
            | bs->pieces[enemy * 6 + KING];
        uint64_t forked = attack_bb & enemy_high;
        return popcount(forked) >= 2 ? forked : 0;
    }

    return 0;
}

__device__ bool is_tactical_quiet(const BoardState* bs, GPUMove mv) {
    if (gives_check(bs, mv)) return true;
    return compute_fork_targets(bs, mv) != 0;
}

// ============================================================
// Extended quiescence search
// Port of ext_pesto_qsearch_counted (src/search/quiescence.rs:350-549)
// ============================================================

__device__ int32_t gpu_ext_qsearch(
    const BoardState* bs,
    int32_t alpha, int32_t beta,
    int max_depth,
    bool white_tactic_used,
    bool black_tactic_used,
    uint64_t forked_pieces_bb,
    int* completed
) {
    bool in_check = is_in_check(bs);

    // Stand-pat (only when not in check)
    if (!in_check) {
        int32_t stand_pat = pesto_eval_cp(bs);
        if (stand_pat >= beta) {
            *completed = 1;
            return beta;
        }
        if (stand_pat > alpha) {
            alpha = stand_pat;
        }
    }

    if (max_depth == 0) {
        if (in_check) {
            *completed = 0;
            return alpha;
        }
        // Check if there are legal captures remaining
        MoveList caps;
        caps.clear();
        gen_pseudo_legal_captures(bs, &caps);
        bool has_legal_capture = false;
        for (int i = 0; i < caps.count; i++) {
            BoardState child = *bs;
            apply_move(&child, caps.moves[i]);
            if (is_legal(&child)) {
                has_legal_capture = true;
                break;
            }
        }
        *completed = has_legal_capture ? 0 : 1;
        return alpha;
    }

    bool stm_is_white = bs->w_to_move != 0;
    bool stm_tactic_used = stm_is_white ? white_tactic_used : black_tactic_used;
    bool all_completed = true;

    if (in_check) {
        // In check: generate ALL moves as evasions (free, no budget consumed)
        MoveList caps, quiets;
        caps.clear();
        quiets.clear();
        gen_pseudo_legal_moves(bs, &caps, &quiets);

        bool any_legal = false;
        for (int list_idx = 0; list_idx < 2; list_idx++) {
            MoveList* ml = (list_idx == 0) ? &caps : &quiets;
            for (int i = 0; i < ml->count; i++) {
                BoardState child = *bs;
                apply_move(&child, ml->moves[i]);
                if (!is_legal(&child)) continue;
                any_legal = true;

                int child_completed = 0;
                int32_t score = -gpu_ext_qsearch(
                    &child, -beta, -alpha, max_depth - 1,
                    white_tactic_used, black_tactic_used,
                    0,  // fork state cleared after evasion
                    &child_completed
                );
                if (!child_completed) all_completed = false;

                if (score >= beta) {
                    *completed = all_completed ? 1 : 0;
                    return beta;
                }
                if (score > alpha) alpha = score;
            }
        }

        if (!any_legal) {
            // Checkmate
            *completed = 1;
            return -1000000;
        }
    } else {
        // Not in check: captures first, then tactical quiets + forked retreats

        // 1. Captures (always)
        MoveList caps;
        caps.clear();
        gen_pseudo_legal_captures(bs, &caps);

        for (int i = 0; i < caps.count; i++) {
            BoardState child = *bs;
            apply_move(&child, caps.moves[i]);
            if (!is_legal(&child)) continue;

            int child_completed = 0;
            int32_t score = -gpu_ext_qsearch(
                &child, -beta, -alpha, max_depth - 1,
                white_tactic_used, black_tactic_used,
                0,
                &child_completed
            );
            if (!child_completed) all_completed = false;

            if (score >= beta) {
                *completed = all_completed ? 1 : 0;
                return beta;
            }
            if (score > alpha) alpha = score;
        }

        // 2. Tactical quiets + forked piece retreats
        if (!stm_tactic_used || forked_pieces_bb != 0) {
            MoveList all_caps, quiets;
            all_caps.clear();
            quiets.clear();
            gen_pseudo_legal_moves(bs, &all_caps, &quiets);

            for (int i = 0; i < quiets.count; i++) {
                GPUMove mv = quiets.moves[i];
                int from = GPU_MOVE_FROM(mv);
                uint64_t from_bit = 1ULL << from;

                bool is_forked_retreat = (forked_pieces_bb & from_bit) != 0;
                bool is_tactical = !stm_tactic_used && is_tactical_quiet(bs, mv);

                if (!is_tactical && !is_forked_retreat) continue;

                // Compute fork targets before applying
                uint64_t new_forked = is_tactical ? compute_fork_targets(bs, mv) : 0;

                BoardState child = *bs;
                apply_move(&child, mv);
                if (!is_legal(&child)) continue;

                // Update budget
                bool new_w_used = (is_tactical && stm_is_white) ? true : white_tactic_used;
                bool new_b_used = (is_tactical && !stm_is_white) ? true : black_tactic_used;

                int child_completed = 0;
                int32_t score = -gpu_ext_qsearch(
                    &child, -beta, -alpha, max_depth - 1,
                    new_w_used, new_b_used,
                    new_forked,
                    &child_completed
                );
                if (!child_completed) all_completed = false;

                if (score >= beta) {
                    *completed = all_completed ? 1 : 0;
                    return beta;
                }
                if (score > alpha) alpha = score;
            }
        }
    }

    *completed = all_completed ? 1 : 0;
    return alpha;
}

__device__ float gpu_forced_pesto_balance(const BoardState* bs, int* completed) {
    int32_t score_cp = gpu_ext_qsearch(bs, -100000, 100000, 20, false, false, 0, completed);
    return (float)score_cp / 100.0f;
}

// ============================================================
// Principal Exchange (PE) q-search
// Follow the single best MVV-LVA capture at each node.
// A straight line, not a tree. ~1–5 nodes. GPU-friendly.
// ============================================================

__device__ int32_t gpu_principal_exchange_search(
    const BoardState* bs,
    int32_t alpha, int32_t beta,
    int max_depth
) {
    int32_t stand_pat = pesto_eval_cp(bs);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;
    if (max_depth <= 0) return alpha;

    // Generate captures (MVV-LVA sorted)
    MoveList caps;
    caps.clear();
    gen_pseudo_legal_captures(bs, &caps);

    // Try only the FIRST legal capture (top MVV-LVA)
    for (int i = 0; i < caps.count; i++) {
        BoardState child = *bs;
        apply_move(&child, caps.moves[i]);
        if (!is_legal(&child)) continue;

        int32_t score = -gpu_principal_exchange_search(&child, -beta, -alpha, max_depth - 1);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
        break; // only the first legal capture
    }
    return alpha;
}

__device__ float gpu_principal_exchange(const BoardState* bs) {
    int32_t score_cp = gpu_principal_exchange_search(bs, -100000, 100000, 20);
    return (float)score_cp / 100.0f;
}
