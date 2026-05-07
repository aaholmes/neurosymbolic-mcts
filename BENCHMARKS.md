# Throughput Benchmarks

GPU benchmarks measured on RTX 5060 Ti (sm_120, 16 GB VRAM, 36 SMs). Rust benchmarks on a 32-core CPU + LibTorch.

All numbers report **samples/sec** (one completed MCTS search = one training sample) and **sims/sec** (single MCTS simulation = one path through the tree). At 200 sims/move, 1 sample = 200 sims.

## Reproducing

```bash
cd cuda/build && cmake --build . --target bench_throughput -j
./cuda/build/bench_throughput [num_games] [sims_per_move] [max_concurrent]
# Defaults: 100 games, 200 sims/move, 36 concurrent
```

The benchmark uses zero-initialized SE-ResNet weights (no model file required); throughput depends only on architecture, not weight values.

## GPU MCTS v4 — batched conv microbenchmark (B=2)

Measured 2026-05-07 on the `v4-batched-conv` branch. **Primitive-level benchmark only**; no MCTS integration in this phase. The new `block_conv_3x3_shifted_b2` loads each weight tile A once per (n_tile, k_tile) and applies it to both batch elements before advancing — true batched WMMA with weight-load amortization. The new `oracle_net_forward_block_b2` uses it; BN/SE/heads still run sequentially per batch on each batch's slice.

**Microbenchmark** (`bench_forward_b2 1000` — single block, 256 threads, 1000 iterations after 50-iter warmup, CUDA-event timing):

| mode | smem | per-call | per-sim | per-sim ratio |
|------|------|---------:|--------:|--------------:|
| B=1 (existing `oracle_net_forward_block`) | 50,176 B (49 KB) | 1392.7 µs | 1392.7 µs | 1.00× |
| **B=2 (new `oracle_net_forward_block_b2`)** | 99,328 B (97 KB) | 2339.0 µs | **1169.5 µs** | **1.191×** |

Three runs all hit 1.191× to 4 significant figures (run-to-run variation < 0.1%). The 1.19× per-sim speedup confirms the bottleneck includes a substantial weight-bandwidth component — true batched WMMA halves weight memory traffic, and we observe roughly that fraction of the per-sim wall time recovered.

**Tolerance vs single-batch reference** (`test_block_forward_b2_matches_b1` with non-zero pseudo-random weights, two distinct mid-game positions):
- batch 0: val Δ = 0.0001, k Δ = 0.0001, max policy diff = 0.0050, mismatches at tol 0.01 = 0
- batch 1: val Δ = 0.0001, k Δ = 0.0001, max policy diff = 0.0036, mismatches at tol 0.01 = 0

**Note:** these are **single-block** wall times (one game on one of 36 SMs). The high absolute µs (vs the ~35.6 µs implied by 17K sims/sec at 36 concurrent) reflects a microbenchmark on an unsaturated GPU. The **per-sim ratio** is what extrapolates to MCTS throughput. v5 (deferred) will integrate this primitive into multi-explorer or per-warp-tree scheduling and measure the end-to-end samples/sec impact.

**Smem fits 99 KB cap**: 99,328 B with 2 KB headroom for static smem before hitting the per-block hardware cap. Layout in halves: buf1[2,128,64] at 0, buf2[2,128,64] at 16384, reduce[256 floats] at 32768, shifted_b2[2,128,64] at 33280, total 49,664 halves.

The existing single-explorer path is unchanged: `bench_throughput 100 200 36` still reports 86.4 samples/sec on this commit (within 0.1 of the v3 baseline).

## GPU MCTS v3 (FP16 activation storage)

Measured 2026-05-07 on commit `cb796d9` (with v3 conversion applied). Same architecture as v2 (SE-ResNet 6×128, 36 concurrent, pool = max(8192, sims·35)). Inter-layer activation buffers (`buf1`, `buf2`) converted from FP32 to FP16; matmul math was already FP16 + FP32-accumulator WMMA. BN, SE arithmetic, residual add (in registers), value head FC1/FC2, and final tanh remain FP32 for numerical stability.

| sims/move | games | wall time | samples/sec | sims/sec | vs v2 |
|-----------|-------|-----------|-------------|----------|-------|
| 200       | 100   | 133.9 s   | 86.8        | 17,356   | 1.07× |

Three-run median; runs were 133.7s, 133.9s, 134.1s — within 0.3% of each other. Tolerance vs FP32 warp-forward reference (per `test_block_forward_fp16_activations_vs_fp32`): value Δ=0.0000, k Δ=0.0000, max policy Δ=0.0035 — well below the 0.05 tolerance bar.

**Smem footprint**: 82,944 bytes (81 KB) → 50,176 bytes (49 KB), a 39% reduction. The throughput gain is modest because the hot conv path was already FP16-WMMA at v2; the headline savings come from removing the per-conv FP32→FP16 conversion and halving activation-buffer bandwidth. The 32 KB freed unblocks the next planned step — intra-block virtual-loss batched inference, where batch=8 activations at FP32 (~256 KB) exceed the 99 KB smem cap but at FP16 (~128 KB) become approachable with smarter tiling.

## GPU MCTS v2 (post-pool-fix, pre-FP16)

Measured 2026-05-05 on commit `f5278d8`. Architecture: SE-ResNet 6×128 (~2M params), 36 concurrent games, pool = max(8192, sims·35), FP32 inter-layer activations.

| sims/move | games | wall time | samples/sec | sims/sec |
|-----------|-------|-----------|-------------|----------|
| 50        | 100   | 33.2 s    | 284         | 14,200   |
| 200       | 100   | 147.0 s   | 79          | 15,800   |
| 400       | 60    | 188.4 s   | 35          | 13,900   |

Sims/sec is roughly constant (~14K) across configurations — per-sim cost is dominated by the SE-ResNet forward pass (1.5 ms × 36 concurrent trees ≈ 24K theoretical max; ~14K achieved with selection/expand/backup overhead).

## GPU MCTS v1 (deprecated, pool=300)

Measured 2026-04-18 on the pre-fix code path, since deleted in commit `9e2d574`.

| sims/move | samples/sec | sims/sec |
|-----------|-------------|----------|
| 50        | 1,208       | 60,000   |
| 200       | 554         | 111,000  |
| 400       | 330         | 132,000  |

⚠️ **These numbers are inflated** and not directly comparable to v2. The pool=300 configuration was below the production minimum (8192). With insufficient pool space, the kernel ran out of room to allocate children after ~10 expansions, then silently fake-terminalized every subsequent leaf to value=0 — no NN forward, no real search. Roughly 95% of "simulations" were no-ops.

The pool minimum (8192) and the fake-terminalize fix landed in commits `6f6e698` and `f5278d8`. Honest per-sim throughput on v1 was ~5K sims/sec, in the same range as v2.

## Rust selfplay (tch-rs + Rayon, tactical tiers)

Measured 2026-04-18 on a 32-core CPU + LibTorch.

| sims/move | configuration | samples/sec | sims/sec |
|-----------|---------------|-------------|----------|
| 200       | 100 games, batch=16, 31 threads | 12.6 | 2,520 |

Rust uses three-tier tactical search (Tier 1: mate/KOTH; Tier 2: quiescence; Tier 3: NN). Only ~12% of simulations require an NN forward (192K NN calls per 1.6M sims) — tiers resolve the rest classically. Its sims/sec figure is therefore not directly comparable to GPU MCTS, which always evaluates with the network.

## Lc0 (Leela Chess Zero) on the same GPU

Measured 2026-05-06 on the same RTX 5060 Ti, lc0 v0.33.0-dev (master, built from source — see `tools/lc0_install_notes.md`). Network: **Maia-1100, a 6×64 SE-ResNet (~600K params, FP16)** — about 1/3 the parameter count of Caissawary's 6×128. (Architecture verified by parsing `maia-1100.pb.gz` directly: 6 residual blocks × 64 filters × 112 input planes, SE reduction 8.)

**Pure NN inference (`backendbench`, cuda-fp16):**

| batch size | NPS    |
|------------|--------|
| 1          | 4,926  |
| 9          | 51,616 |
| 33         | 109,199 |
| 65         | 163,953 |
| 121        | 215,175 |

Inference scales near-linearly until ~batch=121, where it asymptotes around 215K NPS. At batch=1, Tensor Core utilization is poor; the per-call overhead is the dominant cost.

**End-to-end MCTS selfplay (`selfplay --visits=200 --parallelism=8 --games=10`):**

| metric | value |
|--------|-------|
| games  | 10 |
| moves  | 858 |
| nodes  | 180,497 |
| wall time | 2.33 s |
| **sims/sec** | **77,468** |
| **samples/sec** | **368** |

Lc0's MCTS uses virtual-loss-driven leaf parallelism — at parallelism=8 it accumulates batches of ~30-100 leaves per backend call, hitting the favorable region of the inference scaling curve.

## Apples-to-apples: Caissawary at 6×64 vs Lc0 + Maia

Measured 2026-05-07 on a temporary experiment branch (not merged) that flipped `NN_HIDDEN_DIM` from 128 to 64 in `cuda/nn_weights.cuh` and re-ran `bench_throughput 100 200 36`. Purpose: remove the network-size confound from the Lc0 comparison and check whether the engine is compute-bound or fixed-overhead-bound at smaller networks.

Median of 3 runs (very tight cluster, all within 0.3% of each other):

| comparison | samples/sec | sims/sec | per-sim time |
|------------|-------------|----------|--------------|
| Caissawary 6×64 (this experiment)     | 161.9 | 32,381 | 30.9 µs |
| Caissawary 6×128 v3 (current main)    | 86.8  | 17,356 | 57.6 µs |
| Lc0 + Maia 6×64 (measured 2026-05-06) | 368   | 77,468 | 12.9 µs |

**The apples-to-apples gap with Lc0 is 2.4× sims/sec / 2.27× samples/sec, not the 4.7× headline at unequal architectures.** The 6×128-vs-6×64 architecture mismatch was inflating the previously documented gap by roughly 2×.

### Per-sim time decomposition

Linear-system fit (assuming hidden conv FLOPs scale 4× from 6×64 → 6×128, and other MCTS work is constant):

| component | 6×128 | 6×64 |
|-----------|-------|------|
| NN forward (estimated) | 35.6 µs (62%) | 8.9 µs (29%) |
| MCTS overhead (selection / expand / backup) | 22 µs (38%) | 22 µs (71%) |
| **Total per-sim** | **57.6 µs** | **30.9 µs** |

The MCTS overhead is the unrelievable floor. At smaller networks it dominates.

### Implication for batched inference

If intra-block virtual-loss batching at batch=8 cuts NN forward by ~1.5× (sub-linear Tensor Core utilization scaling from batch=1 → batch=8):

- 6×128: 57.6 → 45.8 µs per sim → **~1.26× speedup** on samples/sec
- 6×64:  30.9 → 25.0 µs per sim → ~1.24× speedup

This revises the earlier 1.5–2× projection downward to ~1.25–1.4×. The MCTS-overhead floor caps the upside. Batching is still the highest-leverage next step but the magnitude of the win is smaller than originally projected.

### Reproduction notes

Not committed to main. To re-run:
1. Edit `cuda/nn_weights.cuh`: change `NN_HIDDEN_DIM = 128` to `NN_HIDDEN_DIM = 64` (and `NN_SE_REDUCTION = 16` to `NN_SE_REDUCTION = 8` to keep `NN_SE_INNER = 8`).
2. Comment out the `static_assert(sizeof(OracleNetWeights) == 7930688, ...)` in `cuda/nn_weights.cu` (export was sized for 6×128).
3. Parameterize the hardcoded `32`s in `block_forward.cu` (residual save/add) and `block_ops.cu` (`block_conv_3x3_smem_w` accumulator) to use `(NN_HIDDEN_DIM * 64 + 255) / 256`.
4. Rebuild and run `bench_throughput 100 200 36`.

## Comparison

At 200 sims/move on a single RTX 5060 Ti, **measured** end-to-end:

| engine | network | precision | samples/sec | sims/sec | vs Caissawary v2 |
|--------|---------|-----------|-------------|----------|------------------|
| Caissawary GPU v3 | SE-ResNet 6×128 (~1.98M, 17 input planes) | FP16 act, FP32 BN/SE/heads | 86.8 | 17,356 | 1.07× |
| Caissawary GPU v2 | SE-ResNet 6×128 (~1.98M, 17 input planes) | FP32 | 79 | 15,800 | 1.0× |
| Caissawary Rust   | SE-ResNet 6×128 | FP32 | 12.6 | 2,520 | 0.16× |
| **Lc0 selfplay**  | SE-ResNet 6×64 (~600K, 112 input planes) | FP16 | **368** | **77,468** | **4.7× / 4.9×** |

**Lc0 is ~5× faster than Caissawary GPU v2, but it is also running a smaller network in lower precision.** The two effects compound, and need to be separated to extract a useful conclusion.

### Compute-throughput decomposition

Approximate per-position FLOPs (hidden + first conv only):

- Maia 6×64 with 112 input planes: ~7M FLOPs/forward
- Caissawary 6×128 with 17 input planes: ~12.5M FLOPs/forward — about 1.8× heavier per call.

GFLOPs/sec actually achieved on the GPU at MCTS time:

- Lc0 + Maia: 77,468 sims/sec × 7M ≈ **540 GFLOPs/sec**
- Caissawary:  15,800 sims/sec × 12.5M ≈ **200 GFLOPs/sec**

**Lc0 extracts roughly 2.7× more useful compute per second on the same hardware.** Combined with running a 1.8× cheaper network, this gives the observed 4.9× sims/sec gap.

### Where the 2.7× compute-efficiency gap comes from

These factors don't multiply cleanly, but they're individually estimable:

1. **FP16 vs FP32 (~1.5–2×).** At batch=1 the GPU is memory-bound on weight reads — FP16 halves the bytes per forward.
2. **Batched inference (~1.5–2×).** Lc0's `parallelism=8` with virtual loss accumulates 30–100 leaves per backend call. We do batch=1 per block. Tensor-Core utilization at batch=64 vs batch=1 is roughly 5–10× per FLOP, but only the matmul portion of the forward benefits, so the end-to-end win is more like 1.5–2×.
3. **Kernel quality (~1.2–1.5×).** Lc0 uses CUTLASS-tuned conv kernels. Our shifted-copy conv is roughly half the per-position throughput of cuDNN at small batch (matches the internal `batched_conv_scaling` data).

1.7 × 1.7 × 1.3 ≈ 3.8 — comfortably explains the measured 2.7× with room to spare.

### Honest extrapolation

The earlier design-doc claim of "Caissawary 3–5× faster than Lc0" (CUDA_DESIGN_DOC_v9.md §5.3, §6.1) was a projection based on coordination-overhead arguments, never measured. The measurement inverts that direction, but the magnitude needs care:

- If Lc0 ran our 6×128 weights in **FP16** with **batched inference**, it would still beat us by ~2–3× (kernel quality + the fact that we'd lose ~1.8× to the heavier network).
- If Lc0 ran our 6×128 in **FP32 at batch=1** (the way our kernel does), the kernel-quality gap alone is probably 1.3–1.5×. We might break even or trail by a small constant factor.

So the bulk of the measured 5× gap is not "Lc0's kernels beat ours" — it's **FP16 + batched inference**. Those are the two big architectural wins we're leaving on the table.

The "no CPU-GPU coordination overhead" advantage Caissawary was designed around does exist, but it's a smaller win than the loss from batch=1 FP32 inference. The crossover would happen for *much* smaller networks where coordination dominates over compute, or at extremely tight latency budgets — neither matches our actual workload.

## Notes

- All GPU benchmarks use the SE-ResNet 6×128 path (`--resnet`/`use_resnet=true`). The transformer 12L D=128 path (`mcts_kernel_eval_transformer`) received the same pool-exhaustion fix; per-sim throughput is similar but not yet re-measured under v2.
- Wall-time measurement uses `clock_gettime(CLOCK_MONOTONIC)`, not `clock()`, so GPU wait time is correctly accounted.
- Bench tool source: `cuda/test/bench_throughput.cu`.
- Lc0 install + reproduction notes: `tools/lc0_install_notes.md`.
