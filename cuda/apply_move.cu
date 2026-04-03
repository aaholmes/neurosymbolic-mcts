#include "apply_move.cuh"
#include "movegen.cuh"

// ============================================================
// Helper: find which piece (color, type) occupies a square
// Returns piece_type (0-5) or -1 if empty. Sets *color.
// ============================================================

__device__ int find_piece(const BoardState* bs, int sq, int* color) {
    uint64_t bit = 1ULL << sq;
    for (int c = 0; c < 2; c++) {
        if (bs->pieces_occ[c] & bit) {
            for (int p = 0; p < 6; p++) {
                if (bs->pieces[c * 6 + p] & bit) {
                    *color = c;
                    return p;
                }
            }
        }
    }
    return -1;
}

// ============================================================
// apply_move: port of Rust apply_move_to_board (make_move.rs)
// Modifies bs in-place. No Zobrist hash update.
// ============================================================

__device__ void apply_move(BoardState* bs, GPUMove move) {
    int from = GPU_MOVE_FROM(move);
    int to   = GPU_MOVE_TO(move);
    int promo = GPU_MOVE_PROMO(move);

    uint64_t from_bit = 1ULL << from;
    uint64_t to_bit   = 1ULL << to;

    int stm = bs->w_to_move ? WHITE : BLACK;
    int opp = 1 - stm;

    // Find moving piece
    int move_piece = -1;
    for (int p = 0; p < 6; p++) {
        if (bs->pieces[stm * 6 + p] & from_bit) {
            move_piece = p;
            break;
        }
    }

    bs->halfmove++;

    // Handle captures: remove captured piece at destination
    int cap_piece = -1;
    if (bs->pieces_occ[opp] & to_bit) {
        for (int p = 0; p < 6; p++) {
            if (bs->pieces[opp * 6 + p] & to_bit) {
                cap_piece = p;
                bs->pieces[opp * 6 + p] ^= to_bit;
                bs->halfmove = 0;
                break;
            }
        }
    }

    // En passant capture
    if (move_piece == PAWN && bs->en_passant != EN_PASSANT_NONE &&
        to == (int)bs->en_passant) {
        if (stm == WHITE) {
            int ep_captured_sq = to - 8;
            bs->pieces[BLACK * 6 + PAWN] ^= (1ULL << ep_captured_sq);
        } else {
            int ep_captured_sq = to + 8;
            bs->pieces[WHITE * 6 + PAWN] ^= (1ULL << ep_captured_sq);
        }
    }

    // Clear en passant
    bs->en_passant = EN_PASSANT_NONE;

    // Set en passant for double pawn push
    if (move_piece == PAWN) {
        bs->halfmove = 0;
        int diff = to - from;
        if (diff == 16 || diff == -16) {
            bs->en_passant = (uint8_t)((from + to) / 2);
        }
    }

    // Move the piece: remove from origin, place at destination
    bs->pieces[stm * 6 + move_piece] ^= from_bit;
    bs->pieces[stm * 6 + move_piece] ^= to_bit;

    // Handle promotion: replace pawn with promoted piece
    if (promo != 0) {
        bs->pieces[stm * 6 + move_piece] ^= to_bit;  // remove pawn
        bs->pieces[stm * 6 + promo] ^= to_bit;        // add promoted piece
    }

    // Handle castling: move rook too
    if (move_piece == KING) {
        if (stm == WHITE) {
            if (from == 4 && to == 6) {
                // White kingside
                bs->pieces[WHITE * 6 + ROOK] ^= (1ULL << 7);  // remove from h1
                bs->pieces[WHITE * 6 + ROOK] ^= (1ULL << 5);  // place on f1
            } else if (from == 4 && to == 2) {
                // White queenside
                bs->pieces[WHITE * 6 + ROOK] ^= (1ULL << 0);  // remove from a1
                bs->pieces[WHITE * 6 + ROOK] ^= (1ULL << 3);  // place on d1
            }
            bs->castling &= ~(CASTLE_WK | CASTLE_WQ);
        } else {
            if (from == 60 && to == 62) {
                // Black kingside
                bs->pieces[BLACK * 6 + ROOK] ^= (1ULL << 63);
                bs->pieces[BLACK * 6 + ROOK] ^= (1ULL << 61);
            } else if (from == 60 && to == 58) {
                // Black queenside
                bs->pieces[BLACK * 6 + ROOK] ^= (1ULL << 56);
                bs->pieces[BLACK * 6 + ROOK] ^= (1ULL << 59);
            }
            bs->castling &= ~(CASTLE_BK | CASTLE_BQ);
        }
    } else if (move_piece == ROOK) {
        // Rook moves revoke castling rights for that side
        if (stm == WHITE) {
            if (from == 0)  bs->castling &= ~CASTLE_WQ;
            if (from == 7)  bs->castling &= ~CASTLE_WK;
        } else {
            if (from == 56) bs->castling &= ~CASTLE_BQ;
            if (from == 63) bs->castling &= ~CASTLE_BK;
        }
    }

    // Rook captures also revoke opponent's castling rights
    if (cap_piece >= 0) {
        if (to == 0)  bs->castling &= ~CASTLE_WQ;
        if (to == 7)  bs->castling &= ~CASTLE_WK;
        if (to == 56) bs->castling &= ~CASTLE_BQ;
        if (to == 63) bs->castling &= ~CASTLE_BK;
    }

    // Toggle side to move
    bs->w_to_move = !bs->w_to_move;

    // Recompute occupancy
    for (int c = 0; c < 2; c++) {
        bs->pieces_occ[c] = 0;
        for (int p = 0; p < 6; p++) {
            bs->pieces_occ[c] |= bs->pieces[c * 6 + p];
        }
    }
}

// ============================================================
// is_legal: check that the king of the side that just moved
// (i.e., the side NOT to move) is not in check.
// ============================================================

__device__ bool is_legal(const BoardState* bs) {
    // The side that just moved is the opposite of bs->w_to_move
    int just_moved = bs->w_to_move ? BLACK : WHITE;
    int opponent = 1 - just_moved;

    // Find king square of the side that just moved
    uint64_t king_bb = bs->pieces[just_moved * 6 + KING];
    if (!king_bb) return false;  // no king = illegal
    int king_sq = __ffsll(king_bb) - 1;

    // Check if opponent attacks the king square
    return !is_square_attacked(bs, king_sq, opponent);
}
