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

## Comparison

At 200 sims/move on a single RTX 5060 Ti, **measured** end-to-end:

| engine | network | precision | samples/sec | sims/sec | vs Caissawary v2 |
|--------|---------|-----------|-------------|----------|------------------|
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
