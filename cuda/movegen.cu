#define MOVEGEN_IMPL
#include "movegen.cuh"
#include "movegen_tables.cuh"
#include <cstdio>
#include <cstring>

// ============================================================
// Sliding piece attacks via magic bitboards
// ============================================================

__device__ uint64_t bishop_attacks(int sq, uint64_t occ) {
    uint64_t blockers = occ & c_B_MASKS[sq];
    int key = (int)((blockers * c_B_MAGICS[sq]) >> (64 - c_B_BITS[sq]));
    return g_bishop_attacks[sq * 4096 + key];
}

__device__ uint64_t rook_attacks(int sq, uint64_t occ) {
    uint64_t blockers = occ & c_R_MASKS[sq];
    int key = (int)((blockers * c_R_MAGICS[sq]) >> (64 - c_R_BITS[sq]));
    return g_rook_attacks[sq * 4096 + key];
}

// ============================================================
// Square attack detection
// ============================================================

__device__ bool is_square_attacked(const BoardState* bs, int sq, int by_color) {
    uint64_t occ = bs->pieces_occ[0] | bs->pieces_occ[1];

    // Bishop/Queen diagonal attacks
    uint64_t diag_attackers = bs->pieces[by_color * 6 + BISHOP] | bs->pieces[by_color * 6 + QUEEN];
    if (bishop_attacks(sq, occ) & diag_attackers) return true;

    // Rook/Queen straight attacks
    uint64_t line_attackers = bs->pieces[by_color * 6 + ROOK] | bs->pieces[by_color * 6 + QUEEN];
    if (rook_attacks(sq, occ) & line_attackers) return true;

    // Knight attacks
    if (c_n_move_bb[sq] & bs->pieces[by_color * 6 + KNIGHT]) return true;

    // Pawn attacks (reverse direction: to check if sq is attacked BY white pawns,
    // we check if a black pawn at sq would capture a white pawn — i.e. use bp_cap_bb)
    if (by_color == WHITE) {
        if (c_bp_cap_bb[sq] & bs->pieces[WHITE * 6 + PAWN]) return true;
    } else {
        if (c_wp_cap_bb[sq] & bs->pieces[BLACK * 6 + PAWN]) return true;
    }

    // King attacks
    if (c_k_move_bb[sq] & bs->pieces[by_color * 6 + KING]) return true;

    return false;
}

// ============================================================
// Helper: add promotion moves (4 per source/dest pair)
// ============================================================

__device__ void add_promotions(MoveList* list, int from, int to) {
    list->add(MAKE_GPU_MOVE(from, to, KNIGHT));
    list->add(MAKE_GPU_MOVE(from, to, BISHOP));
    list->add(MAKE_GPU_MOVE(from, to, ROOK));
    list->add(MAKE_GPU_MOVE(from, to, QUEEN));
}

// ============================================================
// Pawn move generation
// ============================================================

__device__ void gen_pawn_moves(const BoardState* bs, MoveList* caps, MoveList* quiets) {
    int stm = bs->w_to_move ? WHITE : BLACK;
    int opp = 1 - stm;
    uint64_t pawns = bs->pieces[stm * 6 + PAWN];
    uint64_t own_occ = bs->pieces_occ[stm];
    uint64_t opp_occ = bs->pieces_occ[opp];
    uint64_t all_occ = own_occ | opp_occ;

    if (stm == WHITE) {
        // White pawns push north
        while (pawns) {
            int from = pop_lsb(pawns);
            int rank = from / 8;
            bool is_promo = (rank == 6);  // rank 7 in 0-indexed = about to promote

            // Captures (using capture bitboard table)
            uint64_t cap_targets = c_wp_cap_bb[from] & opp_occ;
            // En passant
            if (bs->en_passant != EN_PASSANT_NONE) {
                cap_targets |= c_wp_cap_bb[from] & (1ULL << bs->en_passant);
            }
            while (cap_targets) {
                int to = pop_lsb(cap_targets);
                if (is_promo) add_promotions(caps, from, to);
                else caps->add(MAKE_GPU_MOVE(from, to, 0));
            }

            // Single push
            int push1 = from + 8;
            if (push1 < 64 && !(all_occ & (1ULL << push1))) {
                if (is_promo) {
                    add_promotions(caps, from, push1);  // promotions go to caps
                } else {
                    quiets->add(MAKE_GPU_MOVE(from, push1, 0));
                    // Double push from rank 2 (rank index 1)
                    if (rank == 1) {
                        int push2 = from + 16;
                        if (!(all_occ & (1ULL << push2))) {
                            quiets->add(MAKE_GPU_MOVE(from, push2, 0));
                        }
                    }
                }
            }
        }
    } else {
        // Black pawns push south
        while (pawns) {
            int from = pop_lsb(pawns);
            int rank = from / 8;
            bool is_promo = (rank == 1);  // rank 2 in 0-indexed = about to promote

            // Captures
            uint64_t cap_targets = c_bp_cap_bb[from] & opp_occ;
            if (bs->en_passant != EN_PASSANT_NONE) {
                cap_targets |= c_bp_cap_bb[from] & (1ULL << bs->en_passant);
            }
            while (cap_targets) {
                int to = pop_lsb(cap_targets);
                if (is_promo) add_promotions(caps, from, to);
                else caps->add(MAKE_GPU_MOVE(from, to, 0));
            }

            // Single push
            int push1 = from - 8;
            if (push1 >= 0 && !(all_occ & (1ULL << push1))) {
                if (is_promo) {
                    add_promotions(caps, from, push1);
                } else {
                    quiets->add(MAKE_GPU_MOVE(from, push1, 0));
                    // Double push from rank 7 (rank index 6)
                    if (rank == 6) {
                        int push2 = from - 16;
                        if (!(all_occ & (1ULL << push2))) {
                            quiets->add(MAKE_GPU_MOVE(from, push2, 0));
                        }
                    }
                }
            }
        }
    }
}

// ============================================================
// Knight move generation
// ============================================================

__device__ void gen_knight_moves(const BoardState* bs, MoveList* caps, MoveList* quiets) {
    int stm = bs->w_to_move ? WHITE : BLACK;
    uint64_t knights = bs->pieces[stm * 6 + KNIGHT];
    uint64_t own_occ = bs->pieces_occ[stm];
    uint64_t opp_occ = bs->pieces_occ[1 - stm];

    while (knights) {
        int sq = pop_lsb(knights);
        uint64_t targets = c_n_move_bb[sq] & ~own_occ;
        uint64_t captures = targets & opp_occ;
        uint64_t quiet = targets & ~opp_occ;
        while (captures) { int to = pop_lsb(captures); caps->add(MAKE_GPU_MOVE(sq, to, 0)); }
        while (quiet)    { int to = pop_lsb(quiet);    quiets->add(MAKE_GPU_MOVE(sq, to, 0)); }
    }
}

// ============================================================
// Bishop move generation
// ============================================================

__device__ void gen_bishop_moves(const BoardState* bs, MoveList* caps, MoveList* quiets) {
    int stm = bs->w_to_move ? WHITE : BLACK;
    uint64_t bishops = bs->pieces[stm * 6 + BISHOP];
    uint64_t own_occ = bs->pieces_occ[stm];
    uint64_t opp_occ = bs->pieces_occ[1 - stm];
    uint64_t occ = own_occ | opp_occ;

    while (bishops) {
        int sq = pop_lsb(bishops);
        uint64_t targets = bishop_attacks(sq, occ) & ~own_occ;
        uint64_t captures = targets & opp_occ;
        uint64_t quiet = targets & ~opp_occ;
        while (captures) { int to = pop_lsb(captures); caps->add(MAKE_GPU_MOVE(sq, to, 0)); }
        while (quiet)    { int to = pop_lsb(quiet);    quiets->add(MAKE_GPU_MOVE(sq, to, 0)); }
    }
}

// ============================================================
// Rook move generation
// ============================================================

__device__ void gen_rook_moves(const BoardState* bs, MoveList* caps, MoveList* quiets) {
    int stm = bs->w_to_move ? WHITE : BLACK;
    uint64_t rooks = bs->pieces[stm * 6 + ROOK];
    uint64_t own_occ = bs->pieces_occ[stm];
    uint64_t opp_occ = bs->pieces_occ[1 - stm];
    uint64_t occ = own_occ | opp_occ;

    while (rooks) {
        int sq = pop_lsb(rooks);
        uint64_t targets = rook_attacks(sq, occ) & ~own_occ;
        uint64_t captures = targets & opp_occ;
        uint64_t quiet = targets & ~opp_occ;
        while (captures) { int to = pop_lsb(captures); caps->add(MAKE_GPU_MOVE(sq, to, 0)); }
        while (quiet)    { int to = pop_lsb(quiet);    quiets->add(MAKE_GPU_MOVE(sq, to, 0)); }
    }
}

// ============================================================
// Queen move generation
// ============================================================

__device__ void gen_queen_moves(const BoardState* bs, MoveList* caps, MoveList* quiets) {
    int stm = bs->w_to_move ? WHITE : BLACK;
    uint64_t queens = bs->pieces[stm * 6 + QUEEN];
    uint64_t own_occ = bs->pieces_occ[stm];
    uint64_t opp_occ = bs->pieces_occ[1 - stm];
    uint64_t occ = own_occ | opp_occ;

    while (queens) {
        int sq = pop_lsb(queens);
        uint64_t targets = queen_attacks(sq, occ) & ~own_occ;
        uint64_t captures = targets & opp_occ;
        uint64_t quiet = targets & ~opp_occ;
        while (captures) { int to = pop_lsb(captures); caps->add(MAKE_GPU_MOVE(sq, to, 0)); }
        while (quiet)    { int to = pop_lsb(quiet);    quiets->add(MAKE_GPU_MOVE(sq, to, 0)); }
    }
}

// ============================================================
// King move generation (including castling)
// ============================================================

__device__ void gen_king_moves(const BoardState* bs, MoveList* caps, MoveList* quiets) {
    int stm = bs->w_to_move ? WHITE : BLACK;
    int opp = 1 - stm;
    uint64_t king = bs->pieces[stm * 6 + KING];
    if (!king) return;
    int sq = lsb(king);
    uint64_t own_occ = bs->pieces_occ[stm];
    uint64_t opp_occ = bs->pieces_occ[opp];

    // Normal king moves
    uint64_t targets = c_k_move_bb[sq] & ~own_occ;
    uint64_t captures = targets & opp_occ;
    uint64_t quiet = targets & ~opp_occ;
    while (captures) { int to = pop_lsb(captures); caps->add(MAKE_GPU_MOVE(sq, to, 0)); }
    while (quiet)    { int to = pop_lsb(quiet);    quiets->add(MAKE_GPU_MOVE(sq, to, 0)); }

    // Castling
    uint64_t all_occ = own_occ | opp_occ;

    if (stm == WHITE && sq == 4) {  // e1
        // White kingside: e1-g1, rook on h1, f1+g1 empty, e1+f1 not attacked
        if ((bs->castling & CASTLE_WK) &&
            (bs->pieces[WHITE * 6 + ROOK] & (1ULL << 7)) &&  // rook on h1
            !(all_occ & ((1ULL << 5) | (1ULL << 6))) &&      // f1, g1 empty
            !is_square_attacked(bs, 4, BLACK) &&              // e1 not attacked
            !is_square_attacked(bs, 5, BLACK))                // f1 not attacked
        {
            quiets->add(MAKE_GPU_MOVE(4, 6, 0));  // e1-g1
        }
        // White queenside: e1-c1, rook on a1, b1+c1+d1 empty, e1+d1 not attacked
        if ((bs->castling & CASTLE_WQ) &&
            (bs->pieces[WHITE * 6 + ROOK] & (1ULL << 0)) &&  // rook on a1
            !(all_occ & ((1ULL << 1) | (1ULL << 2) | (1ULL << 3))) &&  // b1,c1,d1 empty
            !is_square_attacked(bs, 4, BLACK) &&              // e1 not attacked
            !is_square_attacked(bs, 3, BLACK))                // d1 not attacked
        {
            quiets->add(MAKE_GPU_MOVE(4, 2, 0));  // e1-c1
        }
    } else if (stm == BLACK && sq == 60) {  // e8
        // Black kingside: e8-g8
        if ((bs->castling & CASTLE_BK) &&
            (bs->pieces[BLACK * 6 + ROOK] & (1ULL << 63)) &&
            !(all_occ & ((1ULL << 61) | (1ULL << 62))) &&
            !is_square_attacked(bs, 60, WHITE) &&
            !is_square_attacked(bs, 61, WHITE))
        {
            quiets->add(MAKE_GPU_MOVE(60, 62, 0));
        }
        // Black queenside: e8-c8
        if ((bs->castling & CASTLE_BQ) &&
            (bs->pieces[BLACK * 6 + ROOK] & (1ULL << 56)) &&
            !(all_occ & ((1ULL << 57) | (1ULL << 58) | (1ULL << 59))) &&
            !is_square_attacked(bs, 60, WHITE) &&
            !is_square_attacked(bs, 59, WHITE))
        {
            quiets->add(MAKE_GPU_MOVE(60, 58, 0));
        }
    }
}

// ============================================================
// Full move generation
// ============================================================

__device__ void gen_pseudo_legal_moves(const BoardState* bs, MoveList* caps, MoveList* quiets) {
    caps->clear();
    quiets->clear();
    gen_pawn_moves(bs, caps, quiets);
    gen_knight_moves(bs, caps, quiets);
    gen_bishop_moves(bs, caps, quiets);
    gen_rook_moves(bs, caps, quiets);
    gen_queen_moves(bs, caps, quiets);
    gen_king_moves(bs, caps, quiets);
}

__device__ void gen_pseudo_legal_captures(const BoardState* bs, MoveList* caps) {
    MoveList dummy;
    dummy.clear();
    caps->clear();
    gen_pawn_moves(bs, caps, &dummy);    // captures + promotions go to caps
    gen_knight_moves(bs, caps, &dummy);
    gen_bishop_moves(bs, caps, &dummy);
    gen_rook_moves(bs, caps, &dummy);
    gen_queen_moves(bs, caps, &dummy);
    gen_king_moves(bs, caps, &dummy);
}

// ============================================================
// Host-side: table initialization from binary files
// ============================================================

#include <cstdlib>
#include <vector>

static bool load_binary_file(const char* path, void* dst, size_t expected_bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        fprintf(stderr, "Run 'cargo run --bin export_tables' first.\n");
        return false;
    }
    size_t read = fread(dst, 1, expected_bytes, f);
    fclose(f);
    if (read != expected_bytes) {
        fprintf(stderr, "ERROR: %s: expected %zu bytes, got %zu\n", path, expected_bytes, read);
        return false;
    }
    return true;
}

void init_movegen_tables() {
    // The static tables (magics, masks, bits) are already in __constant__ memory
    // via the initializers in movegen_tables.cuh.
    //
    // Load computed tables from binary files generated by:
    //   cargo run --bin export_tables

    const char* table_dir = "cuda/tables";
    char path[256];
    bool ok = true;

    // Rook attack table: 64 * 4096 u64 = 2 MB
    {
        std::vector<uint64_t> data(64 * 4096);
        snprintf(path, sizeof(path), "%s/rook_attacks.bin", table_dir);
        ok = ok && load_binary_file(path, data.data(), data.size() * sizeof(uint64_t));
        if (ok) cudaMemcpyToSymbol(g_rook_attacks, data.data(), data.size() * sizeof(uint64_t));
    }

    // Bishop attack table: 64 * 4096 u64 = 2 MB
    {
        std::vector<uint64_t> data(64 * 4096);
        snprintf(path, sizeof(path), "%s/bishop_attacks.bin", table_dir);
        ok = ok && load_binary_file(path, data.data(), data.size() * sizeof(uint64_t));
        if (ok) cudaMemcpyToSymbol(g_bishop_attacks, data.data(), data.size() * sizeof(uint64_t));
    }

    // Knight move table: 64 u64
    {
        uint64_t data[64];
        snprintf(path, sizeof(path), "%s/knight_moves.bin", table_dir);
        ok = ok && load_binary_file(path, data, sizeof(data));
        if (ok) cudaMemcpyToSymbol(c_n_move_bb, data, sizeof(data));
    }

    // King move table: 64 u64
    {
        uint64_t data[64];
        snprintf(path, sizeof(path), "%s/king_moves.bin", table_dir);
        ok = ok && load_binary_file(path, data, sizeof(data));
        if (ok) cudaMemcpyToSymbol(c_k_move_bb, data, sizeof(data));
    }

    // White pawn captures: 64 u64
    {
        uint64_t data[64];
        snprintf(path, sizeof(path), "%s/wp_captures.bin", table_dir);
        ok = ok && load_binary_file(path, data, sizeof(data));
        if (ok) cudaMemcpyToSymbol(c_wp_cap_bb, data, sizeof(data));
    }

    // Black pawn captures: 64 u64
    {
        uint64_t data[64];
        snprintf(path, sizeof(path), "%s/bp_captures.bin", table_dir);
        ok = ok && load_binary_file(path, data, sizeof(data));
        if (ok) cudaMemcpyToSymbol(c_bp_cap_bb, data, sizeof(data));
    }

    if (ok) {
        printf("Movegen tables loaded successfully (4.0 MB attack tables + jump/pawn tables)\n");
    } else {
        fprintf(stderr, "FATAL: Failed to load movegen tables. Aborting.\n");
        exit(1);
    }
}
