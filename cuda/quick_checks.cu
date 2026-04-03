#include "quick_checks.cuh"
#include "movegen.cuh"
#include "apply_move.cuh"

// ============================================================
// Mate-in-1 detection
// ============================================================

__device__ bool check_mate_in_1(const BoardState* bs) {
    MoveList caps, quiets;
    caps.clear();
    quiets.clear();
    gen_pseudo_legal_moves(bs, &caps, &quiets);

    // Try each move: apply, check legality, then check if opponent is checkmated
    for (int list_idx = 0; list_idx < 2; list_idx++) {
        MoveList* ml = (list_idx == 0) ? &caps : &quiets;
        for (int i = 0; i < ml->count; i++) {
            BoardState child = *bs;
            apply_move(&child, ml->moves[i]);
            if (!is_legal(&child)) continue;

            // After apply_move, child.w_to_move is the opponent.
            // Check if opponent's king is in check.
            int opp = child.w_to_move ? WHITE : BLACK;
            uint64_t opp_king_bb = child.pieces[opp * 6 + KING];
            if (opp_king_bb == 0) continue;
            int opp_king_sq = __ffsll(opp_king_bb) - 1;
            int attacker = 1 - opp;

            if (!is_square_attacked(&child, opp_king_sq, attacker)) continue;

            // Opponent is in check. Check if they have any legal move.
            MoveList opp_caps, opp_quiets;
            opp_caps.clear();
            opp_quiets.clear();
            gen_pseudo_legal_moves(&child, &opp_caps, &opp_quiets);

            bool has_legal = false;
            for (int j = 0; j < opp_caps.count && !has_legal; j++) {
                BoardState grandchild = child;
                apply_move(&grandchild, opp_caps.moves[j]);
                if (is_legal(&grandchild)) has_legal = true;
            }
            for (int j = 0; j < opp_quiets.count && !has_legal; j++) {
                BoardState grandchild = child;
                apply_move(&grandchild, opp_quiets.moves[j]);
                if (is_legal(&grandchild)) has_legal = true;
            }

            if (!has_legal) return true;  // Checkmate found!
        }
    }
    return false;
}

// ============================================================
// KOTH-in-1 detection
// ============================================================

__device__ bool check_koth_in_1(const BoardState* bs) {
    int stm = bs->w_to_move ? WHITE : BLACK;
    uint64_t king_bb = bs->pieces[stm * 6 + KING];
    if (king_bb == 0) return false;
    int king_sq = __ffsll(king_bb) - 1;

    // 1. King already on center
    if ((1ULL << king_sq) & KOTH_CENTER) return true;

    // 2. King can move to center in one move
    extern __constant__ uint64_t c_k_move_bb[64];
    uint64_t friendly_occ = bs->pieces_occ[stm];
    uint64_t king_moves = c_k_move_bb[king_sq] & KOTH_CENTER & ~friendly_occ;

    while (king_moves) {
        int to = pop_lsb(king_moves);
        GPUMove mv = MAKE_GPU_MOVE(king_sq, to, 0);
        BoardState child = *bs;
        apply_move(&child, mv);
        if (is_legal(&child)) return true;
    }

    return false;
}
