#pragma once

#include "common.cuh"

// ============================================================
// Expansion lock states
// ============================================================

enum ExpandState : uint32_t {
    UNEXPANDED = 0,
    EXPANDING  = 1,
    EXPANDED   = 2,
};

// ============================================================
// Device-side tree store API
// ============================================================

#ifdef __CUDACC__

// Node pool and allocator counter (defined in tree_store.cu)
extern __device__ MCTSNode g_node_pool[MAX_NODES];
extern __device__ int32_t  g_next_node_idx;

// Allocate n contiguous node slots. Returns base index, or -1 if pool exhausted.
__device__ int32_t alloc_children(int n);

// --- Expansion lock protocol ---

// Attempt to claim a leaf for expansion. Returns true if this thread won the race.
__device__ bool try_expand(MCTSNode* node);

// Mark expansion complete. MUST call __threadfence() internally before state change
// to ensure child data is globally visible before other threads see EXPANDED.
__device__ void finish_expand(MCTSNode* node);

// Check if a node has been fully expanded (children ready).
__device__ bool is_expanded(const MCTSNode* node);

// --- Backpropagation atomics ---

// Apply virtual loss (increment) before descending during selection.
__device__ void apply_virtual_loss(MCTSNode* node);

// Remove virtual loss (decrement) during backup.
__device__ void remove_virtual_loss(MCTSNode* node);

// Atomic backup: increment visit count, add value, remove virtual loss.
__device__ void backprop_value(MCTSNode* node, float value);

#endif // __CUDACC__

// ============================================================
// Host-side API
// ============================================================

// Reset the tree: set allocator to 1, zero out root node.
void reset_tree();

// Upload a board position into the root node (index 0).
void upload_root_position(const BoardState& board);

// Read back the root node to host memory.
void read_root_node(MCTSNode* host_node);

// Read back an arbitrary node to host memory.
void read_node(int idx, MCTSNode* host_node);

// Read current allocator index (number of nodes allocated).
int32_t get_allocated_count();
