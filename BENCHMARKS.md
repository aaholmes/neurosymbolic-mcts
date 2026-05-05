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

## Comparison

At 200 sims/move:
- **GPU v2 is 6.3× faster than Rust** measured in samples/sec (one completed search per sample) — the honest head-to-head number.
- The original "41× faster" headline compared Rust to v1, which was inflated ~7× by the fake-terminalize bug. The corrected ratio is one-sixth of that.
- Rust's three-tier search reaches comparable search quality with ~10% of the NN calls. Adding GPU-side tactical tiers would further close (or reverse) the gap on equivalent quality-per-second.

## Notes

- All GPU benchmarks use the SE-ResNet 6×128 path (`--resnet`/`use_resnet=true`). The transformer 12L D=128 path (`mcts_kernel_eval_transformer`) received the same pool-exhaustion fix; per-sim throughput is similar but not yet re-measured under v2.
- Wall-time measurement uses `clock_gettime(CLOCK_MONOTONIC)`, not `clock()`, so GPU wait time is correctly accounted.
- Bench tool source: `cuda/test/bench_throughput.cu`.
