#include "selfplay.cuh"
#include "tree_store.cuh"
#include "movegen.cuh"
#include "apply_move.cuh"
#include "quiescence.cuh"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <sys/stat.h>

// ============================================================
// Host-side board encoding: BoardState → [17, 64] float
// Mirrors block_board_to_planes / tf_board_to_tokens logic
// ============================================================

static void board_to_planes_host(const BoardState& bs, float* planes) {
    memset(planes, 0, 17 * 64 * sizeof(float));
    int stm = bs.w_to_move ? 0 : 1;
    int opp = 1 - stm;
    bool flip = !bs.w_to_move;

    // Piece planes (0-11): STM pieces, then opponent pieces
    for (int piece = 0; piece < 6; piece++) {
        uint64_t stm_bb = bs.pieces[stm * 6 + piece];
        uint64_t opp_bb = bs.pieces[opp * 6 + piece];
        for (int sq = 0; sq < 64; sq++) {
            int mapped = flip ? (sq ^ 56) : sq;
            if ((stm_bb >> sq) & 1) planes[piece * 64 + mapped] = 1.0f;
            if ((opp_bb >> sq) & 1) planes[(6 + piece) * 64 + mapped] = 1.0f;
        }
    }

    // Plane 12: en passant
    if (bs.en_passant != EN_PASSANT_NONE) {
        int mapped = flip ? (bs.en_passant ^ 56) : bs.en_passant;
        planes[12 * 64 + mapped] = 1.0f;
    }

    // Planes 13-16: castling rights
    uint8_t stm_ks, stm_qs, opp_ks, opp_qs;
    if (bs.w_to_move) {
        stm_ks = bs.castling & CASTLE_WK; stm_qs = bs.castling & CASTLE_WQ;
        opp_ks = bs.castling & CASTLE_BK; opp_qs = bs.castling & CASTLE_BQ;
    } else {
        stm_ks = bs.castling & CASTLE_BK; stm_qs = bs.castling & CASTLE_BQ;
        opp_ks = bs.castling & CASTLE_WK; opp_qs = bs.castling & CASTLE_WQ;
    }
    if (stm_ks) for (int i = 0; i < 64; i++) planes[13 * 64 + i] = 1.0f;
    if (stm_qs) for (int i = 0; i < 64; i++) planes[14 * 64 + i] = 1.0f;
    if (opp_ks) for (int i = 0; i < 64; i++) planes[15 * 64 + i] = 1.0f;
    if (opp_qs) for (int i = 0; i < 64; i++) planes[16 * 64 + i] = 1.0f;
}

// ============================================================
// Host-side move application (mirrors GPU apply_move)
// We use a small kernel to apply moves on the GPU, then read back
// ============================================================

__global__ void kernel_apply_move(BoardState* bs, GPUMove move) {
    apply_move(bs, move);
}

__global__ void kernel_gen_legal_moves(BoardState* bs, GPUMove* out_moves, int* out_count) {
    MoveList caps, quiets;
    caps.clear(); quiets.clear();
    gen_pseudo_legal_moves(bs, &caps, &quiets);

    int count = 0;
    for (int i = 0; i < caps.count; i++) {
        BoardState t = *bs;
        apply_move(&t, caps.moves[i]);
        if (is_legal(&t)) out_moves[count++] = caps.moves[i];
    }
    for (int i = 0; i < quiets.count; i++) {
        BoardState t = *bs;
        apply_move(&t, quiets.moves[i]);
        if (is_legal(&t)) out_moves[count++] = quiets.moves[i];
    }
    *out_count = count;
}

__global__ void kernel_is_in_check(BoardState* bs, int* result) {
    int stm = bs->w_to_move ? 0 : 1;
    int king_sq = __ffsll(bs->pieces[stm * 6 + KING]) - 1;
    *result = is_square_attacked(bs, king_sq, 1 - stm) ? 1 : 0;
}

__global__ void kernel_pe(BoardState* bs, float* result) {
    *result = gpu_principal_exchange(bs);
}

// ============================================================
// Simple Zobrist hash for repetition detection
// ============================================================

static uint64_t zobrist_keys[12][64];  // [piece][square]
static uint64_t zobrist_side;
static bool zobrist_initialized = false;

static BoardState make_start_pos() {
    BoardState bs;
    memset(&bs, 0, sizeof(bs));
    bs.pieces[WHITE * 6 + PAWN]   = 0x000000000000FF00ULL;
    bs.pieces[WHITE * 6 + KNIGHT] = 0x0000000000000042ULL;
    bs.pieces[WHITE * 6 + BISHOP] = 0x0000000000000024ULL;
    bs.pieces[WHITE * 6 + ROOK]   = 0x0000000000000081ULL;
    bs.pieces[WHITE * 6 + QUEEN]  = 0x0000000000000008ULL;
    bs.pieces[WHITE * 6 + KING]   = 0x0000000000000010ULL;
    bs.pieces[BLACK * 6 + PAWN]   = 0x00FF000000000000ULL;
    bs.pieces[BLACK * 6 + KNIGHT] = 0x4200000000000000ULL;
    bs.pieces[BLACK * 6 + BISHOP] = 0x2400000000000000ULL;
    bs.pieces[BLACK * 6 + ROOK]   = 0x8100000000000000ULL;
    bs.pieces[BLACK * 6 + QUEEN]  = 0x0800000000000000ULL;
    bs.pieces[BLACK * 6 + KING]   = 0x1000000000000000ULL;
    for (int c = 0; c < 2; c++) {
        bs.pieces_occ[c] = 0;
        for (int p = 0; p < 6; p++) bs.pieces_occ[c] |= bs.pieces[c * 6 + p];
    }
    bs.w_to_move = 1;
    bs.castling = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;
    bs.en_passant = EN_PASSANT_NONE;
    bs.halfmove = 0;
    return bs;
}

static void init_zobrist() {
    if (zobrist_initialized) return;
    srand(42);  // deterministic
    for (int p = 0; p < 12; p++)
        for (int sq = 0; sq < 64; sq++)
            zobrist_keys[p][sq] = ((uint64_t)rand() << 32) | rand();
    zobrist_side = ((uint64_t)rand() << 32) | rand();
    zobrist_initialized = true;
}

static uint64_t compute_hash(const BoardState& bs) {
    uint64_t h = 0;
    for (int p = 0; p < 12; p++) {
        uint64_t bb = bs.pieces[p];
        while (bb) {
            int sq = __builtin_ctzll(bb);
            h ^= zobrist_keys[p][sq];
            bb &= bb - 1;
        }
    }
    if (bs.w_to_move) h ^= zobrist_side;
    return h;
}

static bool is_threefold(const uint64_t* history, int count, uint64_t current) {
    int matches = 0;
    for (int i = 0; i < count; i++)
        if (history[i] == current) matches++;
    return matches >= 3;  // current is already in history
}

// ============================================================
// Per-game deterministic RNG (xorshift64)
// ============================================================

struct GameRng {
    uint64_t state;

    void seed(int game_idx, int generation) {
        // Deterministic seed from game index and generation
        // Mixing via hash to avoid correlated sequences between nearby indices
        uint64_t s = (uint64_t)game_idx * 2654435761ULL + (uint64_t)generation * 6364136223846793005ULL;
        s ^= s >> 33;
        s *= 0xff51afd7ed558ccdULL;
        s ^= s >> 33;
        state = s ? s : 1;  // must be nonzero
    }

    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    // Uniform float in [0, 1)
    float uniform() {
        return (float)(next() >> 11) / (float)(1ULL << 53);
    }

    // Uniform int in [0, n)
    int randint(int n) {
        return (int)(next() % (uint64_t)n);
    }
};

// ============================================================
// Move sampling
// ============================================================

static GPUMove sample_move(
    const TreeEvalResult& result,
    const BoardState& bs,
    int move_number,
    float explore_base,
    int* h_visits,
    GPUMove* h_moves,
    int num_children,
    GameRng& rng
) {
    if (num_children == 0) return 0;

    // Proportional probability decays with move number
    float p = powf(explore_base, (float)(move_number - 1));
    bool use_proportional = rng.uniform() < p;

    if (!use_proportional || num_children == 1) {
        // Greedy: return most-visited move
        int best_idx = 0, best_visits = -1;
        for (int i = 0; i < num_children; i++) {
            if (h_visits[i] > best_visits) {
                best_visits = h_visits[i]; best_idx = i;
            }
        }
        return h_moves[best_idx];
    }

    // Proportional sampling from (visits - 1)
    int total = 0;
    for (int i = 0; i < num_children; i++) {
        int v = h_visits[i] > 1 ? h_visits[i] - 1 : 0;
        total += v;
    }
    if (total == 0) {
        // All moves visited ≤1, pick random
        return h_moves[rng.randint(num_children)];
    }

    int r = rng.randint(total);
    int cumul = 0;
    for (int i = 0; i < num_children; i++) {
        int v = h_visits[i] > 1 ? h_visits[i] - 1 : 0;
        cumul += v;
        if (cumul > r) return h_moves[i];
    }
    return h_moves[num_children - 1];
}

// ============================================================
// Policy extraction from MCTS visit counts
// ============================================================

// Host-side move_to_policy_index (mirrors nn_ops.cu)
static int move_to_policy_index_host(GPUMove move, int w_to_move) {
    int from_sq = GPU_MOVE_FROM(move);
    int to_sq = GPU_MOVE_TO(move);
    int promo = GPU_MOVE_PROMO(move);

    // Flip for black (STM-relative encoding)
    if (!w_to_move) {
        from_sq ^= 56;
        to_sq ^= 56;
    }

    int from_row = from_sq / 8, from_col = from_sq % 8;
    int to_row = to_sq / 8, to_col = to_sq % 8;
    int dr = to_row - from_row;
    int dc = to_col - from_col;

    int plane = -1;

    // Underpromotion
    if (promo >= 1 && promo <= 3) {
        // promo: 1=knight, 2=bishop, 3=rook
        int promo_base = (promo - 1) * 3;  // 0, 3, 6
        if (dc == -1) plane = 64 + promo_base;
        else if (dc == 0) plane = 64 + promo_base + 1;
        else if (dc == 1) plane = 64 + promo_base + 2;
    }
    // Knight moves
    else if (abs(dr) + abs(dc) == 3 && dr != 0 && dc != 0) {
        // 8 knight moves encoded as planes 56-63
        int ki = -1;
        if      (dr ==  2 && dc == -1) ki = 0;
        else if (dr ==  2 && dc ==  1) ki = 1;
        else if (dr ==  1 && dc == -2) ki = 2;
        else if (dr ==  1 && dc ==  2) ki = 3;
        else if (dr == -1 && dc == -2) ki = 4;
        else if (dr == -1 && dc ==  2) ki = 5;
        else if (dr == -2 && dc == -1) ki = 6;
        else if (dr == -2 && dc ==  1) ki = 7;
        if (ki >= 0) plane = 56 + ki;
    }
    // Queen moves (sliding): encode as direction × distance
    else {
        int dir = -1, dist = 0;
        if (dr > 0 && dc == 0)      { dir = 0; dist = dr; }
        else if (dr > 0 && dc > 0 && dr == dc)  { dir = 1; dist = dr; }
        else if (dr == 0 && dc > 0)  { dir = 2; dist = dc; }
        else if (dr < 0 && dc > 0 && -dr == dc) { dir = 3; dist = dc; }
        else if (dr < 0 && dc == 0)  { dir = 4; dist = -dr; }
        else if (dr < 0 && dc < 0 && dr == dc)  { dir = 5; dist = -dr; }
        else if (dr == 0 && dc < 0)  { dir = 6; dist = -dc; }
        else if (dr > 0 && dc < 0 && dr == -dc) { dir = 7; dist = dr; }
        if (dir >= 0 && dist >= 1 && dist <= 7) {
            plane = dir * 7 + (dist - 1);
        }
    }

    if (plane < 0 || plane >= 73) return -1;
    return from_sq * 73 + plane;  // spatial × planes (matches CUDA encoding)
}

// ============================================================
// Core self-play driver
// ============================================================

int run_selfplay_games(
    TransformerWeights* d_weights,
    const SelfPlayConfig& config,
    GameRecord* records,
    int num_games
) {
    init_zobrist();

    int concurrent = config.max_concurrent;
    if (concurrent > num_games) concurrent = num_games;
    if (concurrent > SP_MAX_CONCURRENT) concurrent = SP_MAX_CONCURRENT;

    // Allocate GPU resources for MCTS
    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, concurrent * NN_POLICY_SIZE * sizeof(float));

    // GPU buffers for move generation and board operations
    BoardState* d_bs = nullptr;
    cudaMalloc(&d_bs, sizeof(BoardState));
    GPUMove* d_moves = nullptr;
    cudaMalloc(&d_moves, 256 * sizeof(GPUMove));
    int* d_count = nullptr;
    cudaMalloc(&d_count, sizeof(int));
    int* d_check = nullptr;
    cudaMalloc(&d_check, sizeof(int));
    float* d_pe = nullptr;
    cudaMalloc(&d_pe, sizeof(float));

    // Per-game state
    struct ActiveGame {
        BoardState board;
        uint64_t hash_history[SP_MAX_MOVES_PER_GAME];
        int hash_count;
        int move_number;  // 1-based (for explore_base decay)
        int game_idx;     // index into records[]
        bool white_started;  // for value assignment
        GameRng rng;       // per-game deterministic RNG
    };

    ActiveGame* games = new ActiveGame[concurrent];
    int games_started = 0;
    int games_completed = 0;
    int total_samples = 0;

    // Initialize first batch
    auto init_game = [&](ActiveGame& g, int game_idx) {
        g.board = make_start_pos();
        g.hash_count = 0;
        g.move_number = 1;
        g.game_idx = game_idx;
        g.white_started = true;
        g.rng.seed(game_idx, config.seed);
        uint64_t h = compute_hash(g.board);
        g.hash_history[g.hash_count++] = h;
        if (!records[game_idx].samples) records[game_idx].alloc();
        records[game_idx].num_samples = 0;
        records[game_idx].result = 0;
    };

    int active_count = 0;
    for (int i = 0; i < concurrent && games_started < num_games; i++) {
        init_game(games[i], games_started++);
        active_count++;
    }

    // Main game loop
    while (active_count > 0) {
        // Collect positions of active games
        BoardState positions[SP_MAX_CONCURRENT];
        for (int i = 0; i < active_count; i++)
            positions[i] = games[i].board;

        // Run MCTS for all active positions
        TreeEvalResult results[SP_MAX_CONCURRENT];
        memset(results, 0, sizeof(results));

        gpu_mcts_eval_trees_transformer(
            positions, active_count,
            config.sims_per_move, config.max_nodes_per_tree,
            config.enable_koth, config.c_puct,
            d_weights, d_policy_bufs, results
        );

        // Process each active game
        for (int i = active_count - 1; i >= 0; i--) {
            ActiveGame& g = games[i];
            GameRecord& rec = records[g.game_idx];
            TreeEvalResult& res = results[i];

            // --- Record training sample ---
            if (rec.num_samples < SP_MAX_MOVES_PER_GAME) {
                float* sample = rec.samples + rec.num_samples * SP_SAMPLE_FLOATS;

                // Board encoding [17, 64]
                board_to_planes_host(g.board, sample);

                // Material (principal exchange)
                cudaMemcpy(d_bs, &g.board, sizeof(BoardState), cudaMemcpyHostToDevice);
                kernel_pe<<<1, 1>>>(d_bs, d_pe);
                float pe_result;
                cudaMemcpy(&pe_result, d_pe, sizeof(float), cudaMemcpyDeviceToHost);
                sample[SP_BOARD_FLOATS] = pe_result;

                // Q-search flag
                sample[SP_BOARD_FLOATS + 1] = 1.0f;

                // Value target (placeholder, filled after game ends)
                sample[SP_BOARD_FLOATS + 2] = 0.0f;

                // Policy from visit counts
                float* policy = sample + SP_BOARD_FLOATS + 3;
                memset(policy, 0, NN_POLICY_SIZE * sizeof(float));

                // Read children from GPU to get visit counts
                void* d_pool_ptr = nullptr;
                cudaGetSymbolAddress(&d_pool_ptr, g_node_pool);

                // Find the root for this tree
                // The eval kernel uses tree offsets: tree i starts at i * max_nodes_per_tree
                int tree_offset = i * config.max_nodes_per_tree;
                MCTSNode h_root;
                cudaMemcpy(&h_root, (char*)d_pool_ptr + tree_offset * sizeof(MCTSNode),
                           sizeof(MCTSNode), cudaMemcpyDeviceToHost);

                int total_visits = 0;
                int h_visits[256];
                GPUMove h_moves[256];
                int num_children = h_root.num_children;
                if (num_children > 256) num_children = 256;

                for (int c = 0; c < num_children; c++) {
                    MCTSNode h_child;
                    cudaMemcpy(&h_child, (char*)d_pool_ptr + (h_root.first_child_idx + c) * sizeof(MCTSNode),
                               sizeof(MCTSNode), cudaMemcpyDeviceToHost);
                    h_visits[c] = h_child.visit_count;
                    h_moves[c] = (GPUMove)h_child.move_from_parent;
                    total_visits += h_child.visit_count;
                }

                // Fill policy vector from visit counts
                if (total_visits > 0) {
                    for (int c = 0; c < num_children; c++) {
                        int idx = move_to_policy_index_host(h_moves[c], g.board.w_to_move);
                        if (idx >= 0 && idx < NN_POLICY_SIZE)
                            policy[idx] = (float)h_visits[c] / (float)total_visits;
                    }
                } else if (num_children > 0) {
                    // MCTS ran but no visits on children (node pool exhausted or all sims hit terminals)
                    // Fall back to uniform policy
                    float uniform = 1.0f / (float)num_children;
                    for (int c = 0; c < num_children; c++) {
                        int idx = move_to_policy_index_host(h_moves[c], g.board.w_to_move);
                        if (idx >= 0 && idx < NN_POLICY_SIZE)
                            policy[idx] = uniform;
                    }
                }

                rec.num_samples++;

                // --- Sample and apply move ---
                GPUMove chosen;
                if (num_children == 0 || total_visits == 0) {
                    // No legal moves or no visits — game should end
                    chosen = 0;
                } else {
                    chosen = sample_move(res, g.board, g.move_number, config.explore_base,
                                         h_visits, h_moves, num_children, g.rng);
                }

                if (chosen != 0 && num_children > 0) {
                    // Apply move on host via GPU kernel
                    cudaMemcpy(d_bs, &g.board, sizeof(BoardState), cudaMemcpyHostToDevice);
                    kernel_apply_move<<<1, 1>>>(d_bs, chosen);
                    cudaMemcpy(&g.board, d_bs, sizeof(BoardState), cudaMemcpyDeviceToHost);
                    g.move_number++;

                    // Update hash history
                    uint64_t h = compute_hash(g.board);
                    if (g.hash_count < SP_MAX_MOVES_PER_GAME)
                        g.hash_history[g.hash_count++] = h;

                    // --- Check game termination ---
                    // Generate legal moves for new position
                    cudaMemcpy(d_bs, &g.board, sizeof(BoardState), cudaMemcpyHostToDevice);
                    kernel_gen_legal_moves<<<1, 1>>>(d_bs, d_moves, d_count);
                    int legal_count;
                    cudaMemcpy(&legal_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

                    if (legal_count == 0) {
                        // Check if in check (checkmate vs stalemate)
                        kernel_is_in_check<<<1, 1>>>(d_bs, d_check);
                        int in_check;
                        cudaMemcpy(&in_check, d_check, sizeof(int), cudaMemcpyDeviceToHost);

                        if (in_check) {
                            // Checkmate: STM loses
                            rec.result = g.board.w_to_move ? 2 : 1;  // black wins / white wins
                        } else {
                            rec.result = 3;  // stalemate = draw
                        }
                    } else if (g.board.halfmove >= 100) {
                        rec.result = 3;  // 50-move rule
                    } else if (is_threefold(g.hash_history, g.hash_count, h)) {
                        rec.result = 3;  // threefold repetition
                    } else if (g.move_number > SP_MAX_MOVES_PER_GAME) {
                        rec.result = 3;  // max moves reached
                    }
                } else {
                    // No legal moves from the start
                    rec.result = 3;
                }
            } else {
                rec.result = 3;  // max samples reached
            }

            // --- If game ended, backfill values and maybe start new game ---
            if (rec.result != 0) {
                // Backfill value targets
                float white_value;
                if (rec.result == 1) white_value = 1.0f;       // white wins
                else if (rec.result == 2) white_value = -1.0f;  // black wins
                else white_value = 0.0f;                         // draw

                // Each sample was recorded from the STM's perspective at that point
                // Odd-indexed moves (0-based) are black's moves
                // Value for STM: if STM is white → white_value, if black → -white_value
                // But we need to track who was STM at each sample
                // Since we start from the starting position (white to move),
                // sample 0 = white's move, sample 1 = black's move, etc.
                for (int s = 0; s < rec.num_samples; s++) {
                    float stm_value = (s % 2 == 0) ? white_value : -white_value;
                    rec.samples[s * SP_SAMPLE_FLOATS + SP_BOARD_FLOATS + 2] = stm_value;
                }

                games_completed++;
                total_samples += rec.num_samples;

                // Replace with new game if more needed
                if (games_started < num_games) {
                    init_game(games[i], games_started++);
                } else {
                    // Remove from active list by swapping with last
                    if (i < active_count - 1)
                        games[i] = games[active_count - 1];
                    active_count--;
                }
            }
        }
    }

    // Cleanup
    cudaFree(d_policy_bufs);
    cudaFree(d_bs);
    cudaFree(d_moves);
    cudaFree(d_count);
    cudaFree(d_check);
    cudaFree(d_pe);
    delete[] games;

    return total_samples;
}

// ============================================================
// File-writing self-play driver
// ============================================================

int run_selfplay(
    const char* weights_path,
    const SelfPlayConfig& config,
    const char* output_dir
) {
    TransformerWeights* d_weights = load_transformer_weights(weights_path);
    if (!d_weights) return -1;

    int num_games = config.num_games;
    GameRecord* records = new GameRecord[num_games];
    for (int i = 0; i < num_games; i++) { records[i].samples = nullptr; records[i].num_samples = 0; records[i].result = 0; }

    int total = run_selfplay_games(d_weights, config, records, num_games);

    // Write to output directory
    mkdir(output_dir, 0755);
    char path[512];
    snprintf(path, sizeof(path), "%s/selfplay_%ld.bin", output_dir, (long)time(nullptr));

    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("Failed to open %s for writing\n", path);
        delete[] records;
        free_transformer_weights(d_weights);
        return -1;
    }

    int written = 0;
    for (int g = 0; g < num_games; g++) {
        fwrite(records[g].samples, sizeof(float), records[g].num_samples * SP_SAMPLE_FLOATS, f);
        written += records[g].num_samples;
    }
    fclose(f);

    printf("Wrote %d samples to %s\n", written, path);

    for (int i = 0; i < num_games; i++) records[i].free_buf();
    delete[] records;
    free_transformer_weights(d_weights);
    return written;
}

// ============================================================
// Eval mode: two networks play against each other
// ============================================================

EvalResult run_eval_games(
    TransformerWeights* d_weights_a,
    TransformerWeights* d_weights_b,
    const EvalConfig& config
) {
    init_zobrist();

    EvalResult result = {0, 0, 0};

    int concurrent = config.max_concurrent;
    if (concurrent > config.num_games) concurrent = config.num_games;
    if (concurrent > SP_MAX_CONCURRENT) concurrent = SP_MAX_CONCURRENT;

    // GPU resources
    float* d_policy_bufs = nullptr;
    cudaMalloc(&d_policy_bufs, concurrent * NN_POLICY_SIZE * sizeof(float));

    BoardState* d_bs = nullptr;
    cudaMalloc(&d_bs, sizeof(BoardState));
    GPUMove* d_moves = nullptr;
    cudaMalloc(&d_moves, 256 * sizeof(GPUMove));
    int* d_count = nullptr;
    cudaMalloc(&d_count, sizeof(int));
    int* d_check = nullptr;
    cudaMalloc(&d_check, sizeof(int));

    struct EvalGame {
        BoardState board;
        uint64_t hash_history[SP_MAX_MOVES_PER_GAME];
        int hash_count;
        int move_number;
        int game_idx;
        bool a_is_white;   // true: A plays white, B plays black
        GameRng rng;
    };

    EvalGame* games = new EvalGame[concurrent];
    int games_started = 0;
    int games_completed = 0;

    auto init_eval_game = [&](EvalGame& g, int game_idx) {
        g.board = make_start_pos();
        g.hash_count = 0;
        g.move_number = 1;
        g.game_idx = game_idx;
        g.a_is_white = (game_idx % 2 == 0);  // alternate colors for fairness
        g.rng.seed(game_idx, config.seed);
        uint64_t h = compute_hash(g.board);
        g.hash_history[g.hash_count++] = h;
    };

    int active_count = 0;
    for (int i = 0; i < concurrent && games_started < config.num_games; i++) {
        init_eval_game(games[i], games_started++);
        active_count++;
    }

    // Temp arrays for partitioning
    BoardState positions_a[SP_MAX_CONCURRENT], positions_b[SP_MAX_CONCURRENT];
    int idx_a[SP_MAX_CONCURRENT], idx_b[SP_MAX_CONCURRENT];  // map back to games[]
    TreeEvalResult results_a[SP_MAX_CONCURRENT], results_b[SP_MAX_CONCURRENT];

    while (active_count > 0) {
        // Partition active games by which player is to move
        int count_a = 0, count_b = 0;
        for (int i = 0; i < active_count; i++) {
            bool stm_is_white = games[i].board.w_to_move;
            bool a_to_move = (stm_is_white == games[i].a_is_white);
            if (a_to_move) {
                positions_a[count_a] = games[i].board;
                idx_a[count_a] = i;
                count_a++;
            } else {
                positions_b[count_b] = games[i].board;
                idx_b[count_b] = i;
                count_b++;
            }
        }

        // MCTS for group A (using weights_a)
        if (count_a > 0) {
            memset(results_a, 0, count_a * sizeof(TreeEvalResult));
            gpu_mcts_eval_trees_transformer(
                positions_a, count_a, config.sims_per_move,
                config.max_nodes_per_tree, config.enable_koth, config.c_puct,
                d_weights_a, d_policy_bufs, results_a
            );
        }

        // MCTS for group B (using weights_b)
        if (count_b > 0) {
            memset(results_b, 0, count_b * sizeof(TreeEvalResult));
            gpu_mcts_eval_trees_transformer(
                positions_b, count_b, config.sims_per_move,
                config.max_nodes_per_tree, config.enable_koth, config.c_puct,
                d_weights_b, d_policy_bufs, results_b
            );
        }

        // Merge results back: for each group, find best move and apply
        // Process in reverse so we can swap-remove
        auto process_game = [&](int game_slot, TreeEvalResult& res, int tree_idx) {
            EvalGame& g = games[game_slot];

            // Read children to get visit counts for move selection
            void* d_pool_ptr = nullptr;
            cudaGetSymbolAddress(&d_pool_ptr, g_node_pool);
            int tree_offset = tree_idx * config.max_nodes_per_tree;
            MCTSNode h_root;
            cudaMemcpy(&h_root, (char*)d_pool_ptr + tree_offset * sizeof(MCTSNode),
                       sizeof(MCTSNode), cudaMemcpyDeviceToHost);

            int num_children = h_root.num_children;
            if (num_children > 256) num_children = 256;
            int h_visits[256];
            GPUMove h_moves[256];
            int total_visits = 0;
            for (int c = 0; c < num_children; c++) {
                MCTSNode h_child;
                cudaMemcpy(&h_child, (char*)d_pool_ptr + (h_root.first_child_idx + c) * sizeof(MCTSNode),
                           sizeof(MCTSNode), cudaMemcpyDeviceToHost);
                h_visits[c] = h_child.visit_count;
                h_moves[c] = (GPUMove)h_child.move_from_parent;
                total_visits += h_child.visit_count;
            }

            // Select move (eval uses explore_base for slight randomness)
            GPUMove chosen = 0;
            if (num_children > 0 && total_visits > 0) {
                chosen = sample_move(res, g.board, g.move_number, config.explore_base,
                                     h_visits, h_moves, num_children, g.rng);
            }

            int game_result = 0;  // 0=ongoing

            if (chosen != 0 && num_children > 0) {
                // Apply move
                cudaMemcpy(d_bs, &g.board, sizeof(BoardState), cudaMemcpyHostToDevice);
                kernel_apply_move<<<1, 1>>>(d_bs, chosen);
                cudaMemcpy(&g.board, d_bs, sizeof(BoardState), cudaMemcpyDeviceToHost);
                g.move_number++;

                uint64_t h = compute_hash(g.board);
                if (g.hash_count < SP_MAX_MOVES_PER_GAME)
                    g.hash_history[g.hash_count++] = h;

                // Check game end
                cudaMemcpy(d_bs, &g.board, sizeof(BoardState), cudaMemcpyHostToDevice);
                kernel_gen_legal_moves<<<1, 1>>>(d_bs, d_moves, d_count);
                int legal_count;
                cudaMemcpy(&legal_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

                if (legal_count == 0) {
                    kernel_is_in_check<<<1, 1>>>(d_bs, d_check);
                    int in_check;
                    cudaMemcpy(&in_check, d_check, sizeof(int), cudaMemcpyDeviceToHost);
                    game_result = in_check ? (g.board.w_to_move ? 2 : 1) : 3;
                } else if (g.board.halfmove >= 100) {
                    game_result = 3;
                } else if (is_threefold(g.hash_history, g.hash_count, h)) {
                    game_result = 3;
                } else if (g.move_number > SP_MAX_MOVES_PER_GAME) {
                    game_result = 3;
                }
            } else {
                game_result = 3;  // no legal moves or no visits
            }

            if (game_result != 0) {
                // Classify result from A's perspective
                if (game_result == 3) {
                    result.draws++;
                } else if (game_result == 1) {
                    // White wins
                    if (g.a_is_white) result.wins_a++;
                    else result.wins_b++;
                } else {
                    // Black wins
                    if (!g.a_is_white) result.wins_a++;
                    else result.wins_b++;
                }
                games_completed++;

                // Replace with new game
                if (games_started < config.num_games) {
                    init_eval_game(games[game_slot], games_started++);
                } else {
                    if (game_slot < active_count - 1)
                        games[game_slot] = games[active_count - 1];
                    active_count--;
                }
            }
        };

        // Process group A results (in reverse for safe swap-remove)
        for (int i = count_a - 1; i >= 0; i--)
            process_game(idx_a[i], results_a[i], i);

        // Process group B results
        for (int i = count_b - 1; i >= 0; i--)
            process_game(idx_b[i], results_b[i], i);
    }

    cudaFree(d_policy_bufs);
    cudaFree(d_bs);
    cudaFree(d_moves);
    cudaFree(d_count);
    cudaFree(d_check);
    delete[] games;

    return result;
}

// ============================================================
// Standalone executable
// ============================================================

// Main function moved to selfplay_main.cu
#if 0
int selfplay_main_disabled(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <weights.bin> <num_games> <sims_per_move> <output_dir>\n", argv[0]);
        return 1;
    }

    init_movegen_tables();

    SelfPlayConfig config = {};
    config.num_games = atoi(argv[2]);
    config.sims_per_move = atoi(argv[3]);
    config.max_nodes_per_tree = 4096;
    config.explore_base = 0.80f;
    config.enable_koth = false;
    config.c_puct = 1.414f;
    config.max_concurrent = SP_MAX_CONCURRENT;

    printf("Self-play: %d games, %d sims/move, %d concurrent\n",
           config.num_games, config.sims_per_move, config.max_concurrent);

    clock_t start = clock();
    int samples = run_selfplay(argv[1], config, argv[4]);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    if (samples >= 0)
        printf("Generated %d samples in %.1f seconds (%.1f samples/sec)\n",
               samples, elapsed, samples / elapsed);

    return samples < 0 ? 1 : 0;
}
#endif  // disabled main
