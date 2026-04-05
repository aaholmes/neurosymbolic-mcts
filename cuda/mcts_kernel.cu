#include "mcts_kernel.cuh"
#include "tree_store.cuh"
#include "movegen.cuh"
#include "apply_move.cuh"
#include "quiescence.cuh"
#include "quick_checks.cuh"
#include "nn_forward.cuh"
#include "nn_ops.cuh"

#include <cstdio>
#include <cmath>

// Global simulation counter for multi-warp mode
__device__ int32_t g_sim_counter;

// ============================================================
// Device helpers
// ============================================================

// Get a BoardState pointer from a node (board lives at offset 128 in MCTSNode)
__device__ __forceinline__ BoardState* node_board(MCTSNode* node) {
    return reinterpret_cast<BoardState*>(node->pieces);
}

__device__ __forceinline__ const BoardState* node_board_const(const MCTSNode* node) {
    return reinterpret_cast<const BoardState*>(node->pieces);
}

// Select the child with highest UCB score (PUCT formula).
__device__ int select_child_puct(int parent_idx, float c_puct) {
    MCTSNode* parent = &g_node_pool[parent_idx];
    float sqrt_parent_N = sqrtf((float)parent->visit_count + 1.0f);

    int best_child = parent->first_child_idx;
    float best_ucb = -1e9f;

    for (int i = 0; i < parent->num_children; i++) {
        int child_idx = parent->first_child_idx + i;
        MCTSNode* child = &g_node_pool[child_idx];

        // Q from parent's perspective: negate child's mean value
        float q;
        int n = child->visit_count + child->virtual_loss;
        if (n == 0) {
            q = 0.0f;
        } else {
            q = -(child->total_value / (float)n);
        }

        // U = exploration bonus
        float u = c_puct * child->prior * sqrt_parent_N / (1.0f + n);
        float ucb = q + u;

        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_child = child_idx;
        }
    }
    return best_child;
}

// Expand a leaf node: generate legal moves, allocate children.
__device__ void expand_node(int node_idx) {
    MCTSNode* node = &g_node_pool[node_idx];
    BoardState bs;
    node_to_board(node, &bs);

    // Generate all pseudo-legal moves
    MoveList caps, quiets;
    caps.clear();
    quiets.clear();
    gen_pseudo_legal_moves(&bs, &caps, &quiets);

    // Filter to legal moves
    GPUMove legal_moves[256];
    int num_legal = 0;

    for (int i = 0; i < caps.count; i++) {
        BoardState child_bs = bs;
        apply_move(&child_bs, caps.moves[i]);
        if (is_legal(&child_bs)) {
            legal_moves[num_legal++] = caps.moves[i];
        }
    }
    for (int i = 0; i < quiets.count; i++) {
        BoardState child_bs = bs;
        apply_move(&child_bs, quiets.moves[i]);
        if (is_legal(&child_bs)) {
            legal_moves[num_legal++] = quiets.moves[i];
        }
    }

    if (num_legal == 0) {
        // Checkmate or stalemate
        bool in_check_now = is_in_check(&bs);
        float term_val = in_check_now ? -1.0f : 0.0f; // checkmate = loss for STM, stalemate = draw
        node->terminal_value = term_val;
        node->is_terminal = 1;
        node->num_children = 0;
        return;
    }

    // Allocate children
    int first_child = alloc_children(num_legal);
    if (first_child < 0) {
        // Pool exhausted — treat as leaf with current eval
        node->num_children = 0;
        return;
    }

    node->first_child_idx = first_child;
    node->num_children = (int16_t)num_legal;

    // Initialize children with uniform priors
    float uniform_prior = 1.0f / (float)num_legal;
    for (int i = 0; i < num_legal; i++) {
        MCTSNode* child = &g_node_pool[first_child + i];

        // Zero out the hot data
        child->visit_count = 0;
        child->total_value = 0.0f;
        child->virtual_loss = 0;
        child->expand_lock = UNEXPANDED;
        child->parent_idx = node_idx;
        child->move_from_parent = (int16_t)legal_moves[i];
        child->num_children = 0;
        child->first_child_idx = -1;
        child->prior = uniform_prior;
        child->terminal_value = 0.0f;
        child->is_terminal = 0;

        // Apply move to get child board state
        BoardState child_bs = bs;
        apply_move(&child_bs, legal_moves[i]);
        board_to_node(&child_bs, child);
    }
}

// Backup value from leaf to root.
__device__ void backup_value_to_root(int leaf_idx, float value) {
    int node_idx = leaf_idx;
    float v = value;

    while (node_idx >= 0) {
        backprop_value(&g_node_pool[node_idx], v);
        v = -v; // flip perspective at each level
        node_idx = g_node_pool[node_idx].parent_idx;
    }
}

// Check for insufficient material (K vs K, K+B vs K, K+N vs K)
__device__ bool is_insufficient_material(const BoardState* bs) {
    // Count non-king pieces
    uint64_t all_pieces = bs->pieces_occ[0] | bs->pieces_occ[1];
    int total = popcount(all_pieces);

    if (total == 2) return true;  // K vs K
    if (total == 3) {
        // K+minor vs K
        uint64_t knights = bs->pieces[WHITE * 6 + KNIGHT] | bs->pieces[BLACK * 6 + KNIGHT];
        uint64_t bishops = bs->pieces[WHITE * 6 + BISHOP] | bs->pieces[BLACK * 6 + BISHOP];
        if (knights || bishops) return true;
    }
    return false;
}

// ============================================================
// The persistent MCTS kernel
// ============================================================

__global__ void mcts_kernel(int max_simulations, bool enable_koth, float c_puct) {
    for (int sim = 0; sim < max_simulations; sim++) {

        // === 1. SELECT: walk root→leaf via PUCT ===
        int node_idx = 0; // root
        int path[256];    // track path for backup
        int path_len = 0;
        path[path_len++] = 0;

        while (is_expanded(&g_node_pool[node_idx]) &&
               !g_node_pool[node_idx].is_terminal &&
               g_node_pool[node_idx].num_children > 0) {
            node_idx = select_child_puct(node_idx, c_puct);
            apply_virtual_loss(&g_node_pool[node_idx]);
            path[path_len++] = node_idx;
        }

        // === 2. EXPAND (if not terminal and not already expanded) ===
        MCTSNode* leaf = &g_node_pool[node_idx];
        bool just_expanded = false;
        if (!leaf->is_terminal && try_expand(leaf)) {
            expand_node(node_idx);
            finish_expand(leaf);
            just_expanded = true;
        }

        // === 3. QUICK CHECKS on the just-expanded node (before descending) ===
        float value;
        bool evaluated = false;

        if (just_expanded && !leaf->is_terminal) {
            BoardState bs;
            node_to_board(leaf, &bs);

            if (check_mate_in_1(&bs)) {
                leaf->terminal_value = 1.0f;
                leaf->is_terminal = 1;
                value = 1.0f;
                evaluated = true;
            } else if (enable_koth && check_koth_in_1(&bs)) {
                leaf->terminal_value = 1.0f;
                leaf->is_terminal = 1;
                value = 1.0f;
                evaluated = true;
            }
        }

        // If not resolved by quick checks, descend to first child for evaluation
        if (!evaluated && just_expanded && leaf->num_children > 0 && !leaf->is_terminal) {
            node_idx = leaf->first_child_idx;
            path[path_len++] = node_idx;
        }

        // === Evaluate the current node (if not already resolved) ===
        if (!evaluated) {
            MCTSNode* eval_node = &g_node_pool[node_idx];

            if (eval_node->is_terminal) {
                value = eval_node->terminal_value;
            } else {
                BoardState bs;
                node_to_board(eval_node, &bs);

                // === 4. QUIESCENCE: PE mode ===
                float q_result = gpu_principal_exchange(&bs);

                // === 5. VALUE: Classical fallback ===
                value = tanhf(0.326f * q_result);

                // === 6. TERMINAL: Check draws ===
                if (bs.halfmove >= 100) {
                    value = 0.0f;
                } else if (is_insufficient_material(&bs)) {
                    value = 0.0f;
                }
            }
        }

        // === 7. BACKUP: walk leaf→root ===
        // Value is from the leaf's STM perspective.
        // As we walk up, we negate at each level.
        {
            float v = value;
            for (int i = path_len - 1; i >= 0; i--) {
                backprop_value(&g_node_pool[path[i]], v);
                v = -v;
            }
            // Remove virtual loss from the path (backprop_value already handles visit_count
            // and total_value, but we applied virtual loss during selection that needs undoing)
            // Actually, backprop_value already decrements virtual_loss, so we're good.
        }
    }
}

// ============================================================
// Host-side API
// ============================================================

GPUMctsResult gpu_mcts_search(
    const BoardState& root_position,
    int simulations,
    bool enable_koth,
    float c_puct
) {
    // Initialize
    reset_tree();
    upload_root_position(root_position);

    // Set stack size for recursive q-search + expand_node (need ~32KB per thread)
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    // Launch kernel: 1 block, 1 thread
    mcts_kernel<<<1, 1>>>(simulations, enable_koth, c_puct);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Read back results
    MCTSNode root;
    read_root_node(&root);

    GPUMctsResult result = {};
    result.total_simulations = root.visit_count;
    result.nodes_allocated = get_allocated_count();

    if (root.num_children == 0 || root.is_terminal) {
        result.root_value = root.is_terminal ? root.terminal_value : 0.0f;
        // If terminal with children, find the most-visited child for the best move
        if (root.num_children > 0) {
            int best_visits = -1;
            for (int i = 0; i < root.num_children; i++) {
                MCTSNode child;
                read_node(root.first_child_idx + i, &child);
                if (child.visit_count > best_visits) {
                    best_visits = child.visit_count;
                    GPUMove mv = (GPUMove)child.move_from_parent;
                    result.best_move_from = GPU_MOVE_FROM(mv);
                    result.best_move_to = GPU_MOVE_TO(mv);
                    result.best_move_promo = GPU_MOVE_PROMO(mv);
                }
            }
        }
        return result;
    }

    // Find best child by visit count
    int best_visits = -1;
    float best_q = 0.0f;
    GPUMove best_move = GPU_MOVE_NULL;

    for (int i = 0; i < root.num_children; i++) {
        MCTSNode child;
        read_node(root.first_child_idx + i, &child);

        if (child.visit_count > best_visits) {
            best_visits = child.visit_count;
            best_move = (GPUMove)child.move_from_parent;
            best_q = (child.visit_count > 0)
                ? -(child.total_value / (float)child.visit_count)
                : 0.0f;
        }
    }

    result.best_move_from = GPU_MOVE_FROM(best_move);
    result.best_move_to = GPU_MOVE_TO(best_move);
    result.best_move_promo = GPU_MOVE_PROMO(best_move);
    result.root_value = best_q;

    return result;
}

int read_root_children(
    int* visit_counts,
    float* q_values,
    uint16_t* moves,
    int max_children
) {
    MCTSNode root;
    read_root_node(&root);

    int n = root.num_children;
    if (n > max_children) n = max_children;

    for (int i = 0; i < n; i++) {
        MCTSNode child;
        read_node(root.first_child_idx + i, &child);
        visit_counts[i] = child.visit_count;
        q_values[i] = (child.visit_count > 0)
            ? -(child.total_value / (float)child.visit_count)
            : 0.0f;
        moves[i] = (uint16_t)child.move_from_parent;
    }
    return n;
}

bool get_best_child_board(BoardState* out_board, uint16_t* out_move) {
    MCTSNode root;
    read_root_node(&root);

    if (root.num_children == 0) return false;

    int best_idx = root.first_child_idx;
    int best_visits = -1;

    for (int i = 0; i < root.num_children; i++) {
        MCTSNode child;
        read_node(root.first_child_idx + i, &child);
        if (child.visit_count > best_visits) {
            best_visits = child.visit_count;
            best_idx = root.first_child_idx + i;
        }
    }

    MCTSNode best;
    read_node(best_idx, &best);
    // Copy board from the node's board section
    memcpy(out_board->pieces, best.pieces, sizeof(best.pieces));
    memcpy(out_board->pieces_occ, best.pieces_occ, sizeof(best.pieces_occ));
    out_board->w_to_move = best.w_to_move;
    out_board->en_passant = best.en_passant;
    out_board->castling = best.castling;
    out_board->halfmove = best.halfmove;
    memset(out_board->_pad, 0, sizeof(out_board->_pad));

    *out_move = (uint16_t)best.move_from_parent;
    return true;
}

// ============================================================
// NN-mode kernel: set_child_priors + forward pass
// ============================================================

// Set child priors from the policy log-probability vector.
// Called by thread 0 of the warp after oracle_net_forward().
__device__ void set_child_priors(int parent_idx, const float* log_policy, bool w_to_move) {
    MCTSNode* parent = &g_node_pool[parent_idx];
    int n = parent->num_children;
    if (n == 0) return;

    // Collect raw probabilities for legal moves
    float probs[256]; // one per child
    float total = 0.0f;

    for (int i = 0; i < n; i++) {
        MCTSNode* child = &g_node_pool[parent->first_child_idx + i];
        GPUMove mv = (GPUMove)child->move_from_parent;
        int idx = move_to_policy_index(mv, w_to_move);
        float prob;
        if (idx >= 0 && idx < NN_POLICY_SIZE) {
            prob = expf(log_policy[idx]);
        } else {
            prob = 1e-8f; // tiny default for unmapped moves
        }
        probs[i] = prob;
        total += prob;
    }

    // Normalize to sum to 1
    if (total > 0.0f) {
        for (int i = 0; i < n; i++) {
            g_node_pool[parent->first_child_idx + i].prior = probs[i] / total;
        }
    } else {
        float uniform = 1.0f / (float)n;
        for (int i = 0; i < n; i++) {
            g_node_pool[parent->first_child_idx + i].prior = uniform;
        }
    }
}

// NN-mode MCTS kernel.
// One warp (32 threads) per block. Thread 0 handles MCTS logic,
// all 32 threads cooperate on the NN forward pass.
__global__ void mcts_kernel_nn(
    int max_simulations, bool enable_koth, float c_puct,
    OracleNetWeights* weights, float* nn_scratch, int scratch_stride
) {
    int warp_id = blockIdx.x;
    int lane = threadIdx.x & 31;
    float* my_scratch = nn_scratch + warp_id * scratch_stride;

    // Policy output buffer (within scratch space)
    float* policy_buf = my_scratch + SCRATCH_BUF_SIZE * 2 + SCRATCH_COL_SIZE + SCRATCH_PLANES_SIZE;

    while (true) {
        // Atomic increment to claim a simulation
        int sim;
        if (lane == 0) {
            sim = atomicAdd(&g_sim_counter, 1);
        }
        sim = __shfl_sync(0xFFFFFFFF, sim, 0); // broadcast to all lanes
        if (sim >= max_simulations) break;

        // Only thread 0 does MCTS tree operations
        int node_idx = 0;
        int path[256];
        int path_len = 0;
        float value = 0.0f;
        bool evaluated = false;

        if (lane == 0) {
            path[path_len++] = 0;

            // === SELECT ===
            while (is_expanded(&g_node_pool[node_idx]) &&
                   !g_node_pool[node_idx].is_terminal &&
                   g_node_pool[node_idx].num_children > 0) {
                node_idx = select_child_puct(node_idx, c_puct);
                apply_virtual_loss(&g_node_pool[node_idx]);
                path[path_len++] = node_idx;
            }

            // === EXPAND ===
            MCTSNode* leaf = &g_node_pool[node_idx];
            if (!leaf->is_terminal && try_expand(leaf)) {
                expand_node(node_idx);
                finish_expand(leaf);

                // Quick checks on the expanded node
                if (!leaf->is_terminal && leaf->num_children > 0) {
                    BoardState bs;
                    node_to_board(leaf, &bs);

                    if (check_mate_in_1(&bs)) {
                        leaf->terminal_value = 1.0f;
                        leaf->is_terminal = 1;
                        value = 1.0f;
                        evaluated = true;
                    } else if (enable_koth && check_koth_in_1(&bs)) {
                        leaf->terminal_value = 1.0f;
                        leaf->is_terminal = 1;
                        value = 1.0f;
                        evaluated = true;
                    }
                }

                // If not terminal, descend to first child
                if (!evaluated && leaf->num_children > 0 && !leaf->is_terminal) {
                    node_idx = leaf->first_child_idx;
                    path[path_len++] = node_idx;
                }
            }

            // Handle already-terminal nodes
            if (!evaluated && g_node_pool[node_idx].is_terminal) {
                value = g_node_pool[node_idx].terminal_value;
                evaluated = true;
            }
        }

        // Broadcast state to all lanes
        node_idx = __shfl_sync(0xFFFFFFFF, node_idx, 0);
        int eval_flag = evaluated ? 1 : 0;
        eval_flag = __shfl_sync(0xFFFFFFFF, eval_flag, 0);
        evaluated = (eval_flag != 0);

        // === NN FORWARD PASS (all 32 threads cooperate) ===
        if (!evaluated) {
            BoardState bs;
            if (lane == 0) {
                node_to_board(&g_node_pool[node_idx], &bs);
            }
            // Broadcast board state to all lanes (128 bytes = 16 uint64_t)
            uint64_t* bs_words = (uint64_t*)&bs;
            for (int i = 0; i < 16; i++) {
                bs_words[i] = __shfl_sync(0xFFFFFFFF, bs_words[i], 0);
            }

            // PE q-search (single-thread, thread 0)
            float q_result = 0.0f;
            if (lane == 0) {
                q_result = gpu_principal_exchange(&bs);
            }
            q_result = __shfl_sync(0xFFFFFFFF, q_result, 0);

            // Forward pass (all 32 threads)
            float nn_value, nn_k;
            oracle_net_forward(&bs, q_result, weights, my_scratch,
                              policy_buf, &nn_value, &nn_k);
            __syncwarp();

            // Thread 0: set child priors and extract value
            if (lane == 0) {
                // Find the parent (the node we expanded, which has the children)
                int parent_idx = (path_len >= 2) ? path[path_len - 2] : node_idx;
                MCTSNode* parent = &g_node_pool[parent_idx];
                if (parent->num_children > 0) {
                    // Read w_to_move from parent's board
                    set_child_priors(parent_idx, policy_buf, parent->w_to_move);
                }

                value = nn_value;

                // Terminal checks
                if (bs.halfmove >= 100 || is_insufficient_material(&bs)) {
                    value = 0.0f;
                }
            }
        }

        // === BACKUP (thread 0 only) ===
        if (lane == 0) {
            float v = value;
            for (int i = path_len - 1; i >= 0; i--) {
                backprop_value(&g_node_pool[path[i]], v);
                v = -v;
            }
        }
        __syncwarp();
    }
}

// ============================================================
// Host-side NN-mode search
// ============================================================

static void reset_sim_counter() {
    int32_t zero = 0;
    cudaMemcpyToSymbol(g_sim_counter, &zero, sizeof(int32_t));
}

GPUMctsResult gpu_mcts_search_nn(
    const BoardState& root_position,
    int simulations,
    bool enable_koth,
    float c_puct,
    OracleNetWeights* d_weights,
    float* d_nn_scratch,
    int num_warps
) {
    reset_tree();
    upload_root_position(root_position);
    reset_sim_counter();

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    // Launch: one warp (32 threads) per block
    mcts_kernel_nn<<<num_warps, 32>>>(
        simulations, enable_koth, c_puct,
        d_weights, d_nn_scratch, SCRATCH_TOTAL_FLOATS
    );
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA NN kernel error: %s\n", cudaGetErrorString(err));
    }

    // Read results (same as classical)
    MCTSNode root;
    read_root_node(&root);

    GPUMctsResult result = {};
    result.total_simulations = root.visit_count;
    result.nodes_allocated = get_allocated_count();

    if (root.num_children == 0 || root.is_terminal) {
        result.root_value = root.is_terminal ? root.terminal_value : 0.0f;
        if (root.num_children > 0) {
            int best_visits = -1;
            for (int i = 0; i < root.num_children; i++) {
                MCTSNode child;
                read_node(root.first_child_idx + i, &child);
                if (child.visit_count > best_visits) {
                    best_visits = child.visit_count;
                    GPUMove mv = (GPUMove)child.move_from_parent;
                    result.best_move_from = GPU_MOVE_FROM(mv);
                    result.best_move_to = GPU_MOVE_TO(mv);
                    result.best_move_promo = GPU_MOVE_PROMO(mv);
                }
            }
        }
        return result;
    }

    int best_visits = -1;
    for (int i = 0; i < root.num_children; i++) {
        MCTSNode child;
        read_node(root.first_child_idx + i, &child);
        if (child.visit_count > best_visits) {
            best_visits = child.visit_count;
            GPUMove mv = (GPUMove)child.move_from_parent;
            result.best_move_from = GPU_MOVE_FROM(mv);
            result.best_move_to = GPU_MOVE_TO(mv);
            result.best_move_promo = GPU_MOVE_PROMO(mv);
            result.root_value = (child.visit_count > 0)
                ? -(child.total_value / (float)child.visit_count)
                : 0.0f;
        }
    }

    return result;
}
