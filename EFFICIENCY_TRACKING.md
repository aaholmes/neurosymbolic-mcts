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
| 2026-05-07 | Stockfish d2 → Stockfish d12 (same eval, fast→slow)    | 0.195        | 0.121        | 0.202         | 0.054         | YELLOW — calibration is tight at extremes but noisy in tactical mid-range where guidance matters most. Distilled-net variant likely required for useful strength. | `/tmp/calib/results.json` |
| 2026-05-07 | Stockfish d12 → Caissawary v3 (cross-model)             | 0.350        | —            | —             | —             | RED but uninformative — dominated by ~1000 Elo strength gap, not architectural calibration | `/tmp/calib/results.json` |

## Notes for future entries

**Hardware**: All numbers are on the RTX 5060 Ti (sm_120, 16 GB, 36 SMs) unless stated otherwise. Mark divergent hardware explicitly.

**`vs Lc0 6×64`** is computed as `samples_per_sec / 368`. It's the apples-to-apples ratio at Lc0+Maia's network architecture; Caissawary at 6×128 also reports this ratio for trajectory continuity even though the architectures differ.

**Calibration RMSE** is in win-probability units (range [0, 1]). For MCTS hybrid use, RMSE < 0.10 in the tactical mid-range is roughly the bar for "won't materially mislead exploration" given typical visit-count averaging.

**Source convention**: prefer pointing to a section anchor in `BENCHMARKS.md` rather than re-pasting the full numbers here. If the measurement was on a temporary branch (not committed), say so explicitly.

**When a new optimization or architecture lands**: append a row, don't replace existing rows. The trajectory matters as much as the latest number.
