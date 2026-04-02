#pragma once

#include <cstdint>

// ============================================================
// Piece and color constants (match Rust piece_types.rs)
// ============================================================

constexpr int PAWN   = 0;
constexpr int KNIGHT = 1;
constexpr int BISHOP = 2;
constexpr int ROOK   = 3;
constexpr int QUEEN  = 4;
constexpr int KING   = 5;

constexpr int WHITE = 0;
constexpr int BLACK = 1;

// ============================================================
// Castling rights (packed into 1 byte)
// ============================================================

constexpr uint8_t CASTLE_WK = 0x01;  // White kingside
constexpr uint8_t CASTLE_WQ = 0x02;  // White queenside
constexpr uint8_t CASTLE_BK = 0x04;  // Black kingside
constexpr uint8_t CASTLE_BQ = 0x08;  // Black queenside
constexpr uint8_t CASTLE_ALL = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;

constexpr uint8_t EN_PASSANT_NONE = 0xFF;

// ============================================================
// GPUMove: 16-bit move encoding
// ============================================================
//
// bits 0-5:   from square (0-63)
// bits 6-11:  to square (0-63)
// bits 12-14: promotion piece (0=none, 1=KNIGHT, 2=BISHOP, 3=ROOK, 4=QUEEN)
// bit 15:     reserved

typedef uint16_t GPUMove;

#define GPU_MOVE_FROM(m)      ((int)((m) & 0x3F))
#define GPU_MOVE_TO(m)        ((int)(((m) >> 6) & 0x3F))
#define GPU_MOVE_PROMO(m)     ((int)(((m) >> 12) & 0x07))
#define MAKE_GPU_MOVE(f,t,p)  ((GPUMove)((f) | ((t) << 6) | ((p) << 12)))

constexpr GPUMove GPU_MOVE_NULL = 0;

// ============================================================
// BoardState: standalone 128-byte board representation
// Used by apply_move, perft, and as the board portion of MCTSNode.
// ============================================================

struct BoardState {
    uint64_t pieces[12];       // 96 bytes: [color * 6 + piece_type]
    uint64_t pieces_occ[2];    // 16 bytes: [WHITE=0, BLACK=1] occupancy
    uint8_t  w_to_move;        //  1 byte:  1=white, 0=black
    uint8_t  en_passant;       //  1 byte:  target square (0-63), 0xFF=none
    uint8_t  castling;         //  1 byte:  packed bits (CASTLE_WK etc.)
    uint8_t  halfmove;         //  1 byte:  halfmove clock
    uint8_t  _pad[12];         // 12 bytes: pad to 128
};
static_assert(sizeof(BoardState) == 128, "BoardState must be 128 bytes");

// ============================================================
// MCTSNode: 256-byte cache-aligned GPU tree node
//
// Cache line 1 (bytes 0-127):   Selection/backprop hot data
// Cache line 2 (bytes 128-255): Board state
// ============================================================

constexpr int MAX_NODES = 65536;

struct __align__(256) MCTSNode {
    // === Cache line 1: Selection/backprop hot data ===
    int32_t  visit_count;       //  4  atomicAdd during backprop
    float    total_value;       //  4  atomicAdd during backprop (f32 for native atomics)
    int32_t  virtual_loss;      //  4  atomicAdd/Sub for multi-explorer
    uint32_t expand_lock;       //  4  atomicCAS {0=leaf, 1=expanding, 2=expanded}
    int32_t  parent_idx;        //  4  index into node pool (-1 for root)
    int16_t  move_from_parent;  //  2  GPUMove encoding
    int16_t  num_children;      //  2  number of children
    int32_t  first_child_idx;   //  4  children are contiguous in pool
    float    prior;             //  4  P(s,a) from policy head
    float    terminal_value;    //  4  +/-1.0 if terminal, 0.0 otherwise
    uint8_t  is_terminal;       //  1  terminal flag
    uint8_t  _pad1[91];         // 91  pad to 128

    // === Cache line 2: Board state ===
    uint64_t pieces[12];        // 96  [color*6 + piece_type]
    uint64_t pieces_occ[2];     // 16  [WHITE=0, BLACK=1]
    uint8_t  w_to_move;         //  1
    uint8_t  en_passant;        //  1  square index, 0xFF = none
    uint8_t  castling;          //  1  packed castling bits
    uint8_t  halfmove;          //  1
    uint8_t  _pad2[12];         // 12  pad to 128
};
static_assert(sizeof(MCTSNode) == 256, "MCTSNode must be 256 bytes");

// ============================================================
// Board accessor helpers
// ============================================================

#ifdef __CUDACC__

__device__ __forceinline__ uint64_t get_piece_bb(const MCTSNode* n, int color, int piece_type) {
    return n->pieces[color * 6 + piece_type];
}

__device__ __forceinline__ uint64_t get_occupancy(const MCTSNode* n) {
    return n->pieces_occ[0] | n->pieces_occ[1];
}

// Copy board state from MCTSNode to standalone BoardState
__device__ __forceinline__ void node_to_board(const MCTSNode* n, BoardState* bs) {
    // Copy 128 bytes: pieces through _pad2
    const uint64_t* src = n->pieces;
    uint64_t* dst = bs->pieces;
    #pragma unroll
    for (int i = 0; i < 16; i++) {  // 16 x 8 bytes = 128 bytes
        dst[i] = src[i];
    }
}

// Copy board state from BoardState into MCTSNode
__device__ __forceinline__ void board_to_node(const BoardState* bs, MCTSNode* n) {
    const uint64_t* src = bs->pieces;
    uint64_t* dst = n->pieces;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        dst[i] = src[i];
    }
}

// Board accessor helpers for BoardState
__device__ __forceinline__ uint64_t bs_get_piece(const BoardState* bs, int color, int piece_type) {
    return bs->pieces[color * 6 + piece_type];
}

__device__ __forceinline__ uint64_t bs_get_occupancy(const BoardState* bs) {
    return bs->pieces_occ[0] | bs->pieces_occ[1];
}

__device__ __forceinline__ int bs_side_to_move(const BoardState* bs) {
    return bs->w_to_move ? WHITE : BLACK;
}

#endif // __CUDACC__

// ============================================================
// MoveList: fixed-size move array (no heap allocation)
// Max legal moves in any chess position is 218.
// ============================================================

struct MoveList {
    GPUMove moves[256];
    int count;

#ifdef __CUDACC__
    __device__ void clear() { count = 0; }
    __device__ void add(GPUMove m) { moves[count++] = m; }
#endif
};

// ============================================================
// Bit manipulation helpers
// ============================================================

#ifdef __CUDACC__

// Pop least significant bit, return its index (0-63)
__device__ __forceinline__ int pop_lsb(uint64_t& bb) {
    int idx = __ffsll(bb) - 1;  // CUDA intrinsic: find first set bit (1-indexed)
    bb &= bb - 1;               // clear LSB
    return idx;
}

// Count set bits
__device__ __forceinline__ int popcount(uint64_t bb) {
    return __popcll(bb);
}

// Find first set bit without clearing (0-indexed, -1 if empty)
__device__ __forceinline__ int lsb(uint64_t bb) {
    return bb ? __ffsll(bb) - 1 : -1;
}

#endif // __CUDACC__
