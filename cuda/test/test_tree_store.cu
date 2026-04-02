#include "../tree_store.cuh"
#include "test_helpers.cuh"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

// ============================================================
// Test kernels
// ============================================================

// Test 1: Each thread allocates 1 node, stores its index
__global__ void kernel_alloc_one_each(int32_t* results, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        results[tid] = alloc_children(1);
    }
}

// Test 2: Each thread allocates a batch of nodes
__global__ void kernel_alloc_batch(int32_t* results, int batch_size, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        results[tid] = alloc_children(batch_size);
    }
}

// Test 3: Allocate until exhaustion (single thread)
__global__ void kernel_alloc_until_exhausted(int32_t* results, int* count) {
    int i = 0;
    while (true) {
        int32_t idx = alloc_children(1);
        if (idx < 0) break;
        results[i] = idx;
        i++;
        if (i >= MAX_NODES) break;  // safety
    }
    *count = i;
}

// Test 4: Many threads race to expand the same node
__global__ void kernel_expansion_race(int* winner_count, int32_t* winner_tid) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    MCTSNode* root = &g_node_pool[0];

    if (try_expand(root)) {
        // Winner: write sentinel and finish
        root->first_child_idx = 42;
        root->num_children = 7;
        finish_expand(root);
        atomicAdd(winner_count, 1);
        *winner_tid = tid;
    }

    // All threads wait for expansion to complete
    while (!is_expanded(root)) {
        // spin
    }

    // Verify sentinel is visible
    // (atomicAdd on a separate counter to confirm all threads see the data)
}

// Test 5: Many threads backprop to root
__global__ void kernel_backprop_stress(int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        backprop_value(&g_node_pool[0], 0.5f);
    }
}

// Test 6: Virtual loss symmetry
__global__ void kernel_virtual_loss_apply(int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        apply_virtual_loss(&g_node_pool[0]);
    }
}

__global__ void kernel_virtual_loss_remove(int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        remove_virtual_loss(&g_node_pool[0]);
    }
}

// Test 7: Read back board state (single thread)
__global__ void kernel_read_board(uint64_t* pieces_out, uint64_t* occ_out,
                                  uint8_t* meta_out) {
    MCTSNode* root = &g_node_pool[0];
    for (int i = 0; i < 12; i++) pieces_out[i] = root->pieces[i];
    occ_out[0] = root->pieces_occ[0];
    occ_out[1] = root->pieces_occ[1];
    meta_out[0] = root->w_to_move;
    meta_out[1] = root->en_passant;
    meta_out[2] = root->castling;
    meta_out[3] = root->halfmove;
}

// ============================================================
// Test functions
// ============================================================

void test_allocation_uniqueness(bool& test_failed) {
    reset_tree();

    const int N = 1024;
    int32_t* d_results;
    CHECK_CUDA(cudaMalloc(&d_results, N * sizeof(int32_t)));

    kernel_alloc_one_each<<<(N + 255) / 256, 256>>>(d_results, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int32_t> results(N);
    CHECK_CUDA(cudaMemcpy(results.data(), d_results, N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_results);

    // Sort and verify unique, contiguous 1..1024
    std::sort(results.begin(), results.end());
    for (int i = 0; i < N; i++) {
        ASSERT_EQ(results[i], i + 1);
    }

    // Verify allocator is at N+1
    ASSERT_EQ(get_allocated_count(), N + 1);
}

void test_bulk_allocation(bool& test_failed) {
    reset_tree();

    const int N_THREADS = 64;
    const int BATCH = 16;
    int32_t* d_results;
    CHECK_CUDA(cudaMalloc(&d_results, N_THREADS * sizeof(int32_t)));

    kernel_alloc_batch<<<1, N_THREADS>>>(d_results, BATCH, N_THREADS);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int32_t> results(N_THREADS);
    CHECK_CUDA(cudaMemcpy(results.data(), d_results, N_THREADS * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_results);

    // Sort base indices
    std::sort(results.begin(), results.end());

    // Verify no overlaps: each base should be BATCH apart
    for (int i = 0; i < N_THREADS; i++) {
        ASSERT_EQ(results[i], 1 + i * BATCH);
    }

    // Total consumed = N_THREADS * BATCH + 1 (root)
    ASSERT_EQ(get_allocated_count(), N_THREADS * BATCH + 1);
}

void test_pool_exhaustion(bool& test_failed) {
    // We use the full pool. Allocate MAX_NODES-1 nodes (root is 0, allocator starts at 1).
    reset_tree();

    int32_t* d_results;
    int* d_count;
    CHECK_CUDA(cudaMalloc(&d_results, MAX_NODES * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));

    kernel_alloc_until_exhausted<<<1, 1>>>(d_results, d_count);
    CHECK_CUDA(cudaDeviceSynchronize());

    int count;
    CHECK_CUDA(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Should have allocated exactly MAX_NODES - 1 nodes (indices 1 through MAX_NODES-1)
    ASSERT_EQ(count, MAX_NODES - 1);

    cudaFree(d_results);
    cudaFree(d_count);
}

void test_expansion_lock(bool& test_failed) {
    reset_tree();

    int* d_winner_count;
    int32_t* d_winner_tid;
    CHECK_CUDA(cudaMalloc(&d_winner_count, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_winner_tid, sizeof(int32_t)));
    CHECK_CUDA(cudaMemset(d_winner_count, 0, sizeof(int)));

    // Launch 256 threads all racing to expand root
    kernel_expansion_race<<<1, 256>>>(d_winner_count, d_winner_tid);
    CHECK_CUDA(cudaDeviceSynchronize());

    int winner_count;
    CHECK_CUDA(cudaMemcpy(&winner_count, d_winner_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Exactly one thread should have won
    ASSERT_EQ(winner_count, 1);

    // Root should be EXPANDED with sentinel values
    MCTSNode root;
    read_root_node(&root);
    ASSERT_EQ(root.expand_lock, (uint32_t)EXPANDED);
    ASSERT_EQ(root.first_child_idx, 42);
    ASSERT_EQ(root.num_children, 7);

    cudaFree(d_winner_count);
    cudaFree(d_winner_tid);
}

void test_backprop_atomics(bool& test_failed) {
    reset_tree();

    const int N = 10000;
    kernel_backprop_stress<<<(N + 255) / 256, 256>>>(N);
    CHECK_CUDA(cudaDeviceSynchronize());

    MCTSNode root;
    read_root_node(&root);

    ASSERT_EQ(root.visit_count, N);
    ASSERT_NEAR(root.total_value, 5000.0f, 1.0f);  // N * 0.5
    // Virtual loss: backprop_value does atomicSub(1), so should be -N
    // (started at 0, subtracted N times)
    ASSERT_EQ(root.virtual_loss, -N);
}

void test_virtual_loss_symmetry(bool& test_failed) {
    reset_tree();

    const int N = 5000;
    kernel_virtual_loss_apply<<<(N + 255) / 256, 256>>>(N);
    CHECK_CUDA(cudaDeviceSynchronize());

    MCTSNode mid;
    read_root_node(&mid);
    ASSERT_EQ(mid.virtual_loss, N);

    kernel_virtual_loss_remove<<<(N + 255) / 256, 256>>>(N);
    CHECK_CUDA(cudaDeviceSynchronize());

    MCTSNode after;
    read_root_node(&after);
    ASSERT_EQ(after.virtual_loss, 0);
}

void test_board_round_trip(bool& test_failed) {
    reset_tree();

    BoardState start = make_starting_position();
    upload_root_position(start);

    // Read back via kernel
    uint64_t* d_pieces;
    uint64_t* d_occ;
    uint8_t*  d_meta;
    CHECK_CUDA(cudaMalloc(&d_pieces, 12 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_occ, 2 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_meta, 4 * sizeof(uint8_t)));

    kernel_read_board<<<1, 1>>>(d_pieces, d_occ, d_meta);
    CHECK_CUDA(cudaDeviceSynchronize());

    uint64_t pieces[12], occ[2];
    uint8_t meta[4];
    CHECK_CUDA(cudaMemcpy(pieces, d_pieces, 12 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(occ, d_occ, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(meta, d_meta, 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Verify white pawns on rank 2
    ASSERT_EQ(pieces[WHITE * 6 + PAWN], 0x000000000000FF00ULL);
    // Verify white king on e1
    ASSERT_EQ(pieces[WHITE * 6 + KING], 0x0000000000000010ULL);
    // Verify black pawns on rank 7
    ASSERT_EQ(pieces[BLACK * 6 + PAWN], 0x00FF000000000000ULL);
    // Verify black king on e8
    ASSERT_EQ(pieces[BLACK * 6 + KING], 0x1000000000000000ULL);
    // Verify occupancy
    ASSERT_EQ(occ[WHITE], 0x000000000000FFFFULL);
    ASSERT_EQ(occ[BLACK], 0xFFFF000000000000ULL);
    // Verify metadata
    ASSERT_EQ(meta[0], 1);  // w_to_move
    ASSERT_EQ(meta[1], EN_PASSANT_NONE);
    ASSERT_EQ(meta[2], CASTLE_ALL);
    ASSERT_EQ(meta[3], 0);  // halfmove

    cudaFree(d_pieces);
    cudaFree(d_occ);
    cudaFree(d_meta);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== Phase 1: Tree Store Tests ===\n\n");

    int total = 0, passes = 0, failures = 0;

    RUN_TEST(test_allocation_uniqueness);
    RUN_TEST(test_bulk_allocation);
    RUN_TEST(test_pool_exhaustion);
    RUN_TEST(test_expansion_lock);
    RUN_TEST(test_backprop_atomics);
    RUN_TEST(test_virtual_loss_symmetry);
    RUN_TEST(test_board_round_trip);

    printf("\n%d/%d passed", passes, total);
    if (failures > 0) printf(", %d FAILED", failures);
    printf("\n");

    return failures > 0 ? 1 : 0;
}
