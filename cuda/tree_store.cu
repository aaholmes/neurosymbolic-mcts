#define TREE_STORE_IMPL
#include "tree_store.cuh"
#include <cstdio>
#include <cstring>

// ============================================================
// Device-side implementations
// ============================================================

__device__ int32_t alloc_children(int n) {
    int32_t base = atomicAdd(&g_next_node_idx, n);
    if (base + n > MAX_NODES) {
        // Pool exhausted. We don't undo the add (benign: allocator is reset between moves).
        return -1;
    }
    return base;
}

__device__ bool try_expand(MCTSNode* node) {
    return atomicCAS(&node->expand_lock, UNEXPANDED, EXPANDING) == UNEXPANDED;
}

__device__ void finish_expand(MCTSNode* node) {
    // Ensure all child writes are globally visible before we advertise EXPANDED.
    __threadfence();
    atomicExch(&node->expand_lock, EXPANDED);
}

__device__ bool is_expanded(const MCTSNode* node) {
    // Use __ldg for a cached, non-coherent read — fine since expand_lock
    // transitions monotonically (0→1→2) and we only care about seeing 2.
    return __ldg(&node->expand_lock) == EXPANDED;
}

__device__ void apply_virtual_loss(MCTSNode* node) {
    atomicAdd(&node->virtual_loss, 1);
}

__device__ void remove_virtual_loss(MCTSNode* node) {
    atomicSub(&node->virtual_loss, 1);
}

__device__ void backprop_value(MCTSNode* node, float value) {
    atomicAdd(&node->visit_count, 1);
    atomicAdd(&node->total_value, value);
    atomicSub(&node->virtual_loss, 1);
}

// ============================================================
// Host-side implementations
// ============================================================

void reset_tree() {
    // Reset allocator to 1 (node 0 = root)
    int32_t one = 1;
    cudaMemcpyToSymbol(g_next_node_idx, &one, sizeof(int32_t));

    // Zero out the root node
    MCTSNode root;
    memset(&root, 0, sizeof(MCTSNode));
    root.parent_idx = -1;
    root.en_passant = EN_PASSANT_NONE;
    cudaMemcpyToSymbol(g_node_pool, &root, sizeof(MCTSNode));
}

void upload_root_position(const BoardState& board) {
    // Read current root node
    MCTSNode root;
    cudaMemcpyFromSymbol(&root, g_node_pool, sizeof(MCTSNode));

    // Copy board fields into root
    memcpy(root.pieces, board.pieces, sizeof(board.pieces));
    memcpy(root.pieces_occ, board.pieces_occ, sizeof(board.pieces_occ));
    root.w_to_move = board.w_to_move;
    root.en_passant = board.en_passant;
    root.castling = board.castling;
    root.halfmove = board.halfmove;

    // Write back
    cudaMemcpyToSymbol(g_node_pool, &root, sizeof(MCTSNode));
}

void read_root_node(MCTSNode* host_node) {
    cudaMemcpyFromSymbol(host_node, g_node_pool, sizeof(MCTSNode));
}

void read_node(int idx, MCTSNode* host_node) {
    cudaMemcpyFromSymbol(host_node, g_node_pool, sizeof(MCTSNode),
                         idx * sizeof(MCTSNode));
}

int32_t get_allocated_count() {
    int32_t count;
    cudaMemcpyFromSymbol(&count, g_next_node_idx, sizeof(int32_t));
    return count;
}
