# Efficiency Tracking

A running log of measured throughput across approaches, NN architectures, and code commits, on the same hardware (RTX 5060 Ti, sm_120). Reference target: **Lc0 + Maia 6×64 = 368 samples/sec, 77,468 sims/sec**.

To append a new measurement: edit this file and add a row to the table. Source data and reproduction notes are linked from each row.

## End-to-end MCTS throughput (selfplay-equivalent, 200 sims/move, 36 concurrent)

| date       | approach                                       | NN arch       | precision               | samples/sec | sims/sec | vs Lc0 6×64 | commit   | source                                  |
|:-----------|:-----------------------------------------------|:--------------|:------------------------|------------:|---------:|------------:|:---------|:----------------------------------------|
| 2026-04-18 | Rust selfplay (tch-rs + Rayon, tactical tiers) | SE-ResNet 6×128 | FP32 (LibTorch)        | 12.6        | 2,520    | 0.034× | (pre-CUDA) | BENCHMARKS.md "Rust selfplay" |
| 2026-05-05 | GPU MCTS v2 (post-pool-fix)                    | SE-ResNet 6×128 | FP32                   | 79          | 15,800   | 0.21×  | f5278d8    | BENCHMARKS.md "v2"                       |
| 2026-05-07 | GPU MCTS v3 (FP16 activation storage)          | SE-ResNet 6×128 | FP16 act, FP32 BN/SE   | 86.8        | 17,356   | 0.24×  | d2591cd    | BENCHMARKS.md "v3"                       |
| 2026-05-07 | GPU MCTS v3                                    | SE-ResNet 6×64  | FP16 act, FP32 BN/SE   | 161.9       | 32,381   | 0.44×  | (temp branch, not committed) | BENCHMARKS.md "6×64 experiment" |
| 2026-05-07 | GPU MCTS v4 (batched conv primitive only)      | SE-ResNet 6×128 | FP16 act               | 86.4        | 17,267   | 0.23×  | 5a2977f    | BENCHMARKS.md "v4" (microbench separate) |
| 2026-05-07 | **GPU MCTS v5 (2-explorer VL + batched fwd)**  | SE-ResNet 6×128 | FP16 act               | **101.4**   | **20,283** | **0.28×** | 227a94b | BENCHMARKS.md "v5"                       |
| 2026-05-07 | GPU MCTS v5                                    | SE-ResNet 6×64  | FP16 act               | 188.6       | 37,729   | 0.51×  | (temp branch, not committed) | BENCHMARKS.md "v5 6×64"                  |
| 2026-05-06 | **Lc0 + Maia (reference)**                     | SE-ResNet 6×64  | FP16                   | **368**     | **77,468** | **1.00×** | lc0 v0.33.0-dev | BENCHMARKS.md "Lc0"                      |

## Forward-only NN throughput (microbenchmark, single block, 1000 iters)

| date       | kernel path                          | NN arch      | per-call (µs) | per-sim (µs) | speedup | commit   | source                |
|:-----------|:-------------------------------------|:-------------|--------------:|-------------:|--------:|:---------|:----------------------|
| 2026-05-07 | `oracle_net_forward_block` (B=1, v3) | SE-ResNet 6×128 | 1393 | 1393 | 1.00× | 5a2977f | BENCHMARKS.md "v4 microbench" |
| 2026-05-07 | `oracle_net_forward_block_b2` (B=2)  | SE-ResNet 6×128 | 2339 | 1170 | 1.19× | 5a2977f | BENCHMARKS.md "v4 microbench" |

## Calibration tests (for hybrid CPU-NNUE + GPU-bignet evaluation)

| date       | test                                                    | overall RMSE | drawish RMSE | midrange RMSE | by-ply spread | verdict | output                |
|:-----------|:--------------------------------------------------------|-------------:|-------------:|--------------:|--------------:|:--------|:----------------------|
| 2026-05-07 | Stockfish d2 → Stockfish d12 (same eval, fast→slow)    | 0.195        | 0.121        | 0.202         | 0.054         | Calibration is tight at the extremes (drawish, clearly winning) but noisy in the tactical mid-range. Off-the-shelf NNUE used as a speculative provisional value would mislead exploration there. | `/tmp/calib/results.json` |
| 2026-05-07 | Stockfish d12 → Caissawary v3 (cross-model)             | 0.350        | —            | —             | —             | Uninformative — dominated by ~1000 Elo strength gap, not by the calibration architecture. | `/tmp/calib/results.json` |

A later reframing made these tests mostly moot. The shared-formula approach below uses the existing `V_final = tanh(V_logit + k · ΔM)` — there's no sigmoid mapping to fit, the slow value is literally the fast value plus an additive residual.

## Planned approaches (not yet measured)

Two roughly independent directions. The NNUE one looks more promising; the GPU-engine one is a fallback for incremental wins.

**Replace PeSTO with NNUE inside the existing Q-search.** The fast eval becomes `tanh(k · ΔM_NNUE)`; the slow eval (after GPU returns) becomes `tanh(V_logit + k · ΔM_NNUE)`. NNUE is dramatically stronger than PeSTO, so this likely gives ~+200 Elo at the same MCTS budget without retraining. ~1 week. Touches `src/search/quiescence.rs` and `src/eval.rs`. May need to re-tune `k` for the new ΔM scale.

**Retrain against NNUE-based ΔM.** Once the Q-search uses NNUE, V_logit learns a residual on top of a stronger baseline. Probably another +50–100 Elo. One training cycle.

**Async pipelined CPU MCTS + GPU big-net.** The CPU descends the tree, runs Q-search-with-NNUE at every leaf, backs up the provisional value immediately, and queues the position for batched GPU inference. The GPU returns V_logit; CPU updates the node value to the slow form. The tree never stalls on GPU latency. ~3 months. Plausibly Lc0-class throughput, with the architectural advantage that every node always has *some* value.

**FP8 activations on the GPU MCTS path.** Lets B=4 fit in smem. Probably ~+15–25% throughput at 6×128. ~3 weeks. Independent of the NNUE work.

**Cross-block batching (Lc0-style rewrite of the GPU path).** Match Lc0 throughput by abandoning persistent-kernel for a request-queue + batched-inference design. ~3 months. Same end-state as the async-pipelined approach above, just reached via the GPU side rather than the CPU side.

The NNUE-Q-search change is the highest-ROI next step. It's small, delivers strength gains independent of any larger architectural decision, and validates the "NNUE as fast eval" hypothesis before committing to the bigger work.

## Notes for future entries

**Hardware**: All numbers are on the RTX 5060 Ti (sm_120, 16 GB, 36 SMs) unless stated otherwise. Mark divergent hardware explicitly.

**`vs Lc0 6×64`** is computed as `samples_per_sec / 368`. It's the apples-to-apples ratio at Lc0+Maia's network architecture; Caissawary at 6×128 also reports this ratio for trajectory continuity even though the architectures differ.

**Calibration RMSE** is in win-probability units (range [0, 1]). For MCTS hybrid use, RMSE < 0.10 in the tactical mid-range is roughly the bar for "won't materially mislead exploration" given typical visit-count averaging.

**Source convention**: prefer pointing to a section anchor in `BENCHMARKS.md` rather than re-pasting the full numbers here. If the measurement was on a temporary branch (not committed), say so explicitly.

**When a new optimization or architecture lands**: append a row, don't replace existing rows. The trajectory matters as much as the latest number.
