# GPU MCTS Architecture and the Tensor Core Staging Problem

## Background: Why GPU MCTS?

Standard AlphaZero runs MCTS on CPU and calls the neural network on GPU for position evaluation. Every evaluation requires a CPU-GPU-CPU round trip. Our engine does everything on the GPU: the entire select-expand-evaluate-backup loop runs inside a single CUDA kernel. This eliminates transfer latency and lets us evaluate positions the instant they're needed.

## The Three MCTS Kernel Generations

We've built three progressively more parallel MCTS kernels, each mapping the computation differently onto GPU hardware:

**1. Classical (no neural network):** One CUDA thread per tree. The thread runs the full MCTS loop using only material-based evaluation (PeSTO + quiescence search). Simple but weak -- no learned policy or value.

**2. Warp-cooperative (OracleNet SE-ResNet):** One warp (32 threads) per tree. Thread 0 handles MCTS logic (select, expand, backup), and all 32 threads cooperate on the neural network forward pass. The network is a small SE-ResNet (squeeze-excitation residual network) with 128 channels and 6 blocks. The 32 threads divide the 128 channels among themselves (4 channels each) and use warp shuffle instructions (`__shfl_sync`) for cross-thread communication. Convolutions use a shifted-copy pattern to avoid shared memory bank conflicts.

**3. Block-cooperative (Transformer):** One thread block (256 threads) per tree. The transformer architecture needs much more parallelism than 32 threads can provide -- matrix multiplies on [64, 128] tensors are the bottleneck. A full thread block (8 warps of 32 threads each) cooperates on the forward pass using shared memory for all intermediate activations. Thread 0 still handles MCTS logic; all 256 threads participate in the neural network evaluation.

The transformer treats the chessboard as 64 tokens (one per square), each with 17 features (12 piece types + EP + 4 castling rights). The architecture is: input projection (17->128), 6 transformer blocks (4-head attention, 128->512->128 FFN), then policy head (128->73 per token, flattened to 4672) and value head (global average pool -> FC -> tanh).

## Shared Memory: The Scarce Resource

GPU shared memory is fast SRAM local to each streaming multiprocessor (SM). On our RTX 4090 (sm_89), each thread block can use up to ~99 KB. This is where ALL intermediate activations live during the forward pass:

```
buf_x:      [T, D]   = [64, 128] = 32 KB   (residual/accumulator)
buf_out:    [T, D]   = [64, 128] = 32 KB   (layer outputs)
workspace:  varies              = 24 KB   (Q, K, V, attention scores, FFN tiles)
reduce:     [256 floats]        =  1 KB   (LayerNorm/softmax scratch)
-----------------------------------------------------
Total:                            89 KB of 99 KB available
```

Everything must fit in this budget. There is no spillover -- if it doesn't fit, the kernel can't launch.

### Memory as a function of hyperparameters

The dominant terms are:

| Buffer | Size (floats) | Size (bytes) | Depends on |
|--------|---------------|-------------|------------|
| buf_x | T * D = 64 * 128 = 8192 | 32 KB | T (tokens), D (hidden dim) |
| buf_out | T * D = 8192 | 32 KB | T, D |
| workspace | max(T*T, T*2*HD) + T*HD = 6144 | 24 KB | T, HD (head dim) |
| reduce | 256 | 1 KB | fixed |

Where T = 64 (one per square, not adjustable for chess), D = 128 (embedding dimension), HD = D/num_heads = 32.

The workspace is bottlenecked by the attention score matrix T*T = 64*64 = 4096 floats, which depends only on the sequence length (fixed at 64 for chess).

## What Are Tensor Cores?

Tensor Cores are specialized hardware units on NVIDIA GPUs (Volta and later) that perform 16x16x16 matrix multiply-accumulate in a single instruction. One warp (32 threads) issues a single `wmma::mma_sync` instruction that computes: D[16,16] += A[16,16] x B[16,16] where A and B are FP16 (half-precision) and D is FP32. This is dramatically faster than having each thread compute individual multiply-adds.

For our transformer, the dominant cost is matrix multiplications:
- Q/K/V projections: [64, 128] x [128, 32] = per-head linear layers
- QK^T: [64, 32] x [32, 64] = attention scores
- Attention x V: [64, 64] x [64, 32] = weighted values
- Output projection: [64, 32] x [32, 128]
- FFN: two [64, 128] x [128, 64] layers (tiled)

With scalar code (one multiply-add per thread per cycle), these are slow. With Tensor Cores, they're ~20x faster. Our measurements showed **56 samples/sec with TC vs 2.3 samples/sec scalar** -- a 24x speedup.

## How the TC Path Works (and Why It Breaks)

To use Tensor Cores via CUDA's `wmma` API, the input matrices must be in shared memory (or global memory) as contiguous FP16 tiles. Our FP32 activations live in shared memory, and the FP16 weights are pre-converted and stored in global memory. The process for each matrix multiply is:

1. **Stage** the FP32 input tile into an FP16 staging buffer (256 half-precision values per warp)
2. **Load** the staged FP16 tile into a wmma fragment register
3. **Compute** the 16x16x16 matrix multiply-accumulate
4. **Store** the FP32 result back to shared memory (with a transpose for `tf_linear`)

Each warp needs its own staging area so warps can work on different tiles simultaneously. With 8 warps, the staging layout is:

```
a_staging: 8 warps x 256 halves = 2048 halves =  4 KB
b_staging: 8 warps x 256 halves = 2048 halves =  4 KB
tile_temp: 8 warps x 256 floats = 2048 floats =  8 KB  (for tf_linear's transpose step)
---------------------------------------------------------
Total staging:                                    16 KB
```

### The Problem: Staging Overlaps Workspace

The staging buffer is passed as `(half*)workspace` -- it lives in the **same memory** as the Q/K/V/attention buffers:

```
workspace (24 KB), as float*:
|-- ws_q / ws_attn  [0:4096 floats]     <-- Q, K, or attention scores
|-- ws_v            [4096:6144 floats]   <-- V vectors

SAME MEMORY, viewed as half*:
|-- a_staging       [0:2048 halves]      <-- FP16 staging for wmma
|-- b_staging       [2048:4096 halves]   <-- FP16 staging for wmma
|-- tile_temp       [4096:8192 halves -> 2048:4096 floats]   <-- transpose scratch
```

When `tf_linear` is called with input or output in workspace (e.g., Q projection: reads `buf_out`, writes `ws_q`), the staging writes corrupt the I/O data:

1. Output (`ws_q`) is zeroed at workspace offset 0
2. Warp 0 stages an input tile: writes FP16 values to `a_staging[0:255]` = workspace bytes 0-511
3. This **overwrites** the first 128 floats of `ws_q` with garbage half-precision bits
4. Later, when the result is scattered to `ws_q`, it lands on partially-corrupted memory
5. Even worse: different warps process different tiles **concurrently**, so warp 0's result store can be overwritten by warp 3's staging write if they happen to target the same bytes

This is a **warp-level race condition**. There's no synchronization between warps within the tile processing loop -- by design, since warp independence is what makes Tensor Cores fast. But it means the staging area and the I/O buffers absolutely cannot share memory.

The same problem affects every TC operation:
- **Q/K projection** -> output at workspace+0, staging at workspace+0
- **V projection** -> output at workspace+4096, staging may overlap
- **QK^T** -> output at workspace+0, staging at workspace+0
- **attn x V** -> inputs at workspace+0 and workspace+4096, staging overlaps inputs
- **FFN tile** -> output at workspace+0, staging at workspace+0

## Why It's Hard to Fix (and How to Fix It)

The fix is conceptually simple: put the staging buffer somewhere that doesn't overlap with any input or output. The challenge is we've used almost all 99 KB:

```
Currently used:  89 KB
Staging needed:  16 KB
Total needed:   105 KB  <-- exceeds 99 KB limit by 6 KB
```

### Approaches that DON'T work

| Approach | Why it fails |
|----------|-------------|
| Add 16 KB staging region | 105 KB > 99 KB limit |
| Add 8 KB and reuse for tile_temp | Warp-level race: warp 0's tile_temp overlaps warp 3's staging |
| Shrink buf_x/buf_out/workspace | Sized by model dims (T=64, D=128) -- can't shrink without changing model |
| Use global memory for staging | ~100 ns/tile latency, negates TC benefit |
| `__syncthreads()` between staging phases | Serializes warps, kills TC parallelism |

### Approaches that WORK

#### Option A: Reduce TC warps from 8 to 4 (easiest, no model change)

Only warps 0-3 participate in TC operations (warps 4-7 idle during wmma). This halves the staging requirement:

```
D=128 H=4 (4 TC warps):  89 KB base + 8 KB staging = 97 KB  <-- FITS (2 KB margin)
```

**Tradeoff:** TC throughput halved (4 warps process tiles vs 8). But TC was 24x faster than scalar, so even at ~12x it's still a massive win.

#### Option B: Reduce embedding dimension

The two buf_x/buf_out buffers are 2 x T x D x 4 bytes = 2 x 64 x D x 4. Reducing D frees significant space:

```
Config                       Base  Stage  Total  Limit  Margin  D%16  HD%16
---------------------------------------------------------------------------
D=128 H=4 (current, 8w)      89K   16K   105K    99K    -6K    OK    OK
D=128 H=4 (4 TC warps)       89K    8K    97K    99K    +2K    OK    OK
D=112 H=4                    80K   16K    96K    99K    +3K    OK    BAD
D=96  H=4 (HD=24)            71K   16K    87K    99K   +12K    OK    BAD
D=96  H=3 (HD=32)            73K   16K    89K    99K   +10K    OK    OK
D=64  H=4 (HD=16)            53K   16K    69K    99K   +30K    OK    OK
D=112 H=4 (4 TC warps)       80K    8K    88K    99K   +11K    OK    BAD
D=96  H=3 (4 TC warps)       73K    8K    81K    99K   +18K    OK    OK
```

Note: "HD%16" matters for wmma efficiency. Tensor Cores operate on 16x16 tiles, so HD (head dimension) should be a multiple of 16. HD=24 or HD=28 wastes 25-44% of each tile on padding zeros.

**Best small-model option:** D=96, H=3 (HD=32) gives 10 KB margin with all 8 TC warps, and HD=32 is perfectly aligned. Model capacity drops ~25% but TC works at full speed.

#### Option C: Per-call staging placement (most complex, no model change, full speed)

For each `tf_linear` call, pick a staging area that doesn't conflict with that specific call's inputs/outputs:

- **Q/K proj** (reads buf_out, writes ws_q): stage in buf_x (zeroed, not yet used for this head)
- **V proj** (reads buf_out, writes ws_v): stage in buf_x
- **Out proj** (reads ws_q, writes buf_x): stage in workspace (output is buf_x, not workspace)
- **FFN1** (reads buf_out, writes ws_tile): stage in buf_x
- **FFN2** (reads ws_tile, writes buf_x): *problematic* -- both workspace and buf_x are in use

This approach requires careful per-call analysis and re-zeroing buf_x between uses. FFN2 has no clean staging location, so it may still need a partial solution (e.g., 4-warp TC for FFN2 only).

#### Option D: Two-phase execution

Split the forward pass into two phases with a `__syncthreads()` boundary:
1. Phase 1: all TC operations (Q/K/V proj, FFN1, FFN2) -- use workspace for staging, buf_x/buf_out for I/O
2. Phase 2: all workspace-dependent operations (QK^T, softmax, attn x V) -- scalar

This avoids staging-workspace overlap for the large matrix multiplies while keeping scalar for the attention operations (which are smaller). Requires restructuring the per-block loop.

## Current Status

- **Scalar path:** Verified correct (matches PyTorch to 4e-6), 58 tests passing. 2.3 samples/sec.
- **TC path:** Disabled. The staging/workspace overlap corrupts all operations silently.
- **Recommended fix:** Option A (4 TC warps) is the lowest-risk path to a ~12x speedup with zero model changes. Can be implemented by adding a `constexpr int TC_WARPS = 4` and allocating staging at `smem + TF_STAGING_OFFSET`.
