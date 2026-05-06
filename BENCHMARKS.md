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

## GPU MCTS v2 (current main, post-pool-fix)

Measured 2026-05-05 on commit `f5278d8`. Architecture: SE-ResNet 6×128 (~2M params), 36 concurrent games, pool = max(8192, sims·35).

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

Measured 2026-05-06 on the same RTX 5060 Ti, lc0 v0.33.0-dev (master, built from source — see `tools/lc0_install_notes.md`). Network: **Maia-1100, a 12×128 SE-ResNet (~3M params)** — twice as deep as Caissawary's 6×128, the smallest publicly hosted Lc0-format network we could find.

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

## Comparison

At 200 sims/move on a single RTX 5060 Ti, **measured** end-to-end:

| engine | network | samples/sec | sims/sec | vs Caissawary v2 |
|--------|---------|-------------|----------|------------------|
| Caissawary GPU v2 | SE-ResNet 6×128 (~2M) | 79 | 15,800 | 1.0× |
| Caissawary Rust | SE-ResNet 6×128 | 12.6 | 2,520 | 0.16× |
| **Lc0 selfplay** | SE-ResNet 12×128 (~3M, 2× deeper) | **368** | **77,468** | **4.7× / 4.9×** |

**Lc0 is ~5× faster than Caissawary GPU v2 despite running a 2× larger network.** Extrapolating to the same network size (the 6×128 inference is roughly 2× cheaper than 12×128, all else equal), Lc0 would likely be **~9-10× faster** on identical weights.

The earlier design-doc claim of "Caissawary 3-5× faster than Lc0" (CUDA_DESIGN_DOC_v9.md §5.3, §6.1) was a projection based on per-call coordination overhead arguments, never measured. The measurement inverts the claim: **Lc0 wins by ~5× as currently configured**, and probably wins by more on the same network.

**Why Lc0 wins:**
1. **Batched inference at MCTS time.** Lc0's `parallelism=8` with virtual loss accumulates ~30-100 leaves per backend call. Our kernel does batch=1 per block (just 36 batch=1 forwards in parallel via 36 SMs). At Tensor Core utilization, batch=100 is ~30× more arithmetic per memory-bound load than batch=1.
2. **Mature optimized kernels.** Lc0's CUDA backend uses CUTLASS-tuned conv kernels and cuDNN paths. Our shifted-copy conv is decent but ~50% of cuDNN's per-position throughput at small batch (matches our internal `test_batched_conv_scaling` data).
3. **MCTS overlap.** CPU walks the tree while GPU computes the forward pass. Our kernel serializes select → expand → forward → backup within each block.

The "no CPU-GPU coordination overhead" advantage Caissawary was designed around does exist, but it's a smaller win than the loss from batch=1 inference. The crossover would happen for *much* smaller networks (e.g. ~1×64) where coordination dominates over compute, or at extremely tight latency budgets — neither matches our actual workload.

## Notes

- All GPU benchmarks use the SE-ResNet 6×128 path (`--resnet`/`use_resnet=true`). The transformer 12L D=128 path (`mcts_kernel_eval_transformer`) received the same pool-exhaustion fix; per-sim throughput is similar but not yet re-measured under v2.
- Wall-time measurement uses `clock_gettime(CLOCK_MONOTONIC)`, not `clock()`, so GPU wait time is correctly accounted.
- Bench tool source: `cuda/test/bench_throughput.cu`.
- Lc0 install + reproduction notes: `tools/lc0_install_notes.md`.
