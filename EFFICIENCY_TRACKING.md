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

The lead architecture is **CPU-resident MCTS with batched GPU inference**, not an extension of the persistent-kernel path. The persistent-kernel code remains usable as a self-contained GPU-only baseline, but it isn't the destination.

The pipeline below is layered — each step delivers strength gains on its own and unblocks the next.

**1. Replace PeSTO with NNUE inside the existing Q-search.** The fast eval becomes `tanh(k · ΔM_NNUE)`; the slow eval (after GPU returns) becomes `tanh(V_logit + k · ΔM_NNUE)`. NNUE is dramatically stronger than PeSTO, so this likely gives ~+200 Elo at the same MCTS budget without retraining. ~1-2 weeks. Touches `src/search/quiescence.rs` and `src/eval.rs`. May need to re-tune `k` for the new ΔM scale.

NNUE source is **Akimbo** (https://github.com/jw1912/akimbo, MIT). Architecture `(768×4 king-input buckets → 1024)_per_perspective → concat → 1`, SCReLU, single output bucket — maps cleanly to noru's `NnueConfig::new_static(3072, 1024, &[], Activation::SCReLU)`. File format is a raw `i16` struct dump (no header, no compression) so the loader is roughly a memcpy. Akimbo at ~3470 CCRL Blitz gives ~+1000 Elo over PeSTO. Originally planned to use Stockfish .nnue, but a Phase 3 spike (2026-05-09) confirmed noru cannot load Stockfish format and SF's modern 8-bucket layer-stack is architecturally incompatible — Akimbo is dramatically simpler. Bonus: Akimbo's author wrote `bullet` (the trainer most modern NNUE engines use), so adopting this file format means future custom-trained nets drop in without loader changes.

**2. Retrain against NNUE-based ΔM.** Once the Q-search uses NNUE, V_logit learns a residual on top of a stronger baseline. Probably another +50–100 Elo. One training cycle.

**3. Network-adapter layer (input/output decoupling).** Generalize the engine to consume any (input-plane spec, policy-encoding spec) pair so different networks can be dropped in without rewrites. Add 8-ply history tracking to `BoardStack`, make the input-plane builder configurable, and add a static `Move ↔ policy index` map keyed by the chosen encoding. Worth doing on its own merits — unblocks any future pretrained-network experiment, not just Lc0. ~1 week.

**4. Async pipelined CPU MCTS + batched GPU inference (lead architecture).** CPU descends the tree, runs NNUE-Q at every leaf, backs up the provisional value immediately, and queues the position for batched GPU inference. The GPU runs continuously at high batch sizes using standard inference libraries (cuBLAS/cuDNN/TensorRT), returns V_logit; CPU folds it into the slow value when it arrives. CPU never stalls on GPU latency; GPU never stalls on small batches. ~2–3 months. Plausibly Lc0-class throughput, with the architectural property that every node always has *some* backed-up value.

**5. Iterative-deepening Q-search at high-uncertainty leaves.** Once the CPU is the home of leaf evaluation, spend more CPU time on tactically interesting positions — extend the NNUE Q-search adaptively when the static eval is volatile or the position has many forcing moves. CPU-only refinement, naturally fits the async pipeline (the GPU is busy with other batches anyway).

**6. Lc0 trunk import (with v_logit adapter).** Take a small Maia-class Lc0 network (6×64 or 10×128 T-net) and reuse its trunk + policy head directly; replace its value head with a small adapter trained to predict the residual on top of NNUE-Q. Inherits Lc0's training data implicitly via the pretrained weights. The adapter learns to improve the evaluation in ways that aren't easy to capture by a Q-search over a shallow network — long-horizon and global-pattern judgments that benefit from a deeper feature stack and explicit policy guidance. Real constraint: network size — going past 10×128 means the GPU inference path needs to handle bigger nets (which the standard-library batched-inference design does naturally; it was the persistent kernel that capped at small nets). Fine-tuning: freeze trunk + policy head, add a small FC adapter on top of the value-head input, keep `k_logit` as a tunable scalar, train MSE against Stockfish-labeled positions. ~3–4 weeks (after the network-adapter layer is in place).

The NNUE-Q-search change is the highest-ROI next step. It's small, delivers strength gains independent of any larger architectural decision, and validates the "NNUE as fast eval" hypothesis before committing to the bigger work. Lc0 trunk import is the natural follow-up — it depends on NNUE-Q being in place (the adapter is trained against `NNUE_q`, not against PeSTO's ΔM) and on the network-adapter layer being in place. The async pipeline can be staged in either before or after the trunk import; doing it before makes the trunk import simpler because there's already a queue to feed.

(Previously listed: FP8 activations for B=4, cross-block-batching rewrite of the persistent kernel. Both were persistent-kernel optimizations; both are subsumed by the async-pipeline pivot above. Dropped.)

## Notes for future entries

**Hardware**: All numbers are on the RTX 5060 Ti (sm_120, 16 GB, 36 SMs) unless stated otherwise. Mark divergent hardware explicitly.

**`vs Lc0 6×64`** is computed as `samples_per_sec / 368`. It's the apples-to-apples ratio at Lc0+Maia's network architecture; Caissawary at 6×128 also reports this ratio for trajectory continuity even though the architectures differ.

**Calibration RMSE** is in win-probability units (range [0, 1]). For MCTS hybrid use, RMSE < 0.10 in the tactical mid-range is roughly the bar for "won't materially mislead exploration" given typical visit-count averaging.

**Source convention**: prefer pointing to a section anchor in `BENCHMARKS.md` rather than re-pasting the full numbers here. If the measurement was on a temporary branch (not committed), say so explicitly.

**When a new optimization or architecture lands**: append a row, don't replace existing rows. The trajectory matters as much as the latest number.
