# Self-play throughput

Self-play speed decides how much training data each hour of compute buys. It is measured in
standard chess with a randomly initialized network and the classical tiers off, so that every run
does the same work per move and only speed differs. All runs use 200 simulations per move, an RTX
5060 Ti and a 16-core Ryzen 9 7950X, with one game per search thread.

The network alone makes **730 moves per second** (729.8 ± 0.7) with the defaults of `self_play`
and `orchestrate.py`: f16, CUDA graphs, four batching threads serving 1,920 search threads, batch
cap 128, built with `--features cudnn`. With the default mate search and endgame tables it makes
**765** (764.5 ± 0.7):

```bash
./target/release/self_play 1920 200 out/ model.pt false false false none --syzygy /data/syzygy
```

## From 65 to 765 moves per second

Changes in the order they were made. Each gain comes from before-and-after runs of that change
alone. The first five were measured with fewer threads, so moves per second is given only from
the fourth batching thread on, when the final setting (1,920 search threads, cap 128) was reached.
After that, ± is half the difference between two runs, and rows without it are single runs.
Rows marked ¹ change the search's work per move, so their gain is not a pure speed-up.

| Change | Gain | Moves/s |
|---|---|---|
| LibTorch, the starting point | | about 65 |
| **Candle with cuDNN** (described above) | network call 39% faster | |
| **Fused kernels** for bias, ReLU and squeeze-and-excitation | +19% | |
| **Priors for legal moves only**, gathered on the GPU | +26% | |
| **f16 on tensor cores** instead of f32 | +58% | |
| **mimalloc** instead of glibc's allocator (measured in f32) | +35% | |
| **Four batching threads** and 1,920 search threads, from one and 480 | +52% | 453 (mean of 5) |
| **One cuDNN call for convolution, bias and ReLU** | +1% | 442 |
| **Channels-last activations** throughout | +7% | 474 |
| **CUDA graphs**: each batch replays a recorded forward pass | +13% | 534.3 ± 0.3 |
| **Two-byte moves** | +4% | 560.1 ± 1.3 |
| **Reused subtree counts toward the budget**¹ | +10% | 615.9 ± 1.6 |
| **No root quiescence search in vanilla self-play** | +18% | 730.2 ± 0.3 |
| **f16 and CUDA graphs as defaults**, rather than set by environment variables | none | 729.8 ± 0.7 |
| **Mate in one at leaves**¹ | +13% | 824.1 ± 0.6 |
| **Endgame tables at leaves**, ending games at a covered position | none | 823.4 ± 0.1 |
| **Games end at a mate in one**¹, as at a covered position | −10% | 742.2 ± 3.7 |
| **Mates in up to 5 by checking moves**¹ (the default) | +3% | 763.4 ± 0.5 |
| **Faster mate search** ([MATE_SEARCH.md](MATE_SEARCH.md)) | none | 764.5 ± 0.7 |

- **The 65 moves per second was not recorded** in this repository, and its settings (game, threads,
  tiers) are not confirmed to match the later rows.
- **Moves per second sometimes falls between rows** (453 → 442) because rebuilding the earlier
  commit measured 437; see "Unexplained" below.
- **Candle and fused kernels.** Candle's generic broadcast kernels for bias, ReLU and
  squeeze-and-excitation took about 40% of GPU time, 18–30 µs each on tensors that need about 2 µs.
  Three custom CUDA kernels, compiled when the model loads, do each in one pass and match the unfused
  operations to 1e-5.
- **Legal-move priors.** The policy head scores all 4,672 possible moves; copying only the legal
  moves' logits back cuts the transfer from 0.36 ms to 0.04 ms per batch.
- **f16.** Candle's own half-precision convolution rebuilt its cuDNN setup on every call and
  accumulated in 16 bits, which cuDNN serves with slow kernels; a small wrapper
  (`src/nn/cudnn_conv.rs`) keeps the setup for each batch size and accumulates in 32 bits.
- **mimalloc.** With hundreds of search threads, glibc's allocator spent most of its time on shared
  locks; mimalloc keeps a separate heap per thread.
- **Several batching threads.** A batch of about 120 positions launches too few thread blocks to
  fill the GPU. Each batching thread has its own copy of the network on its own CUDA stream, so
  kernels from different batches overlap. With CUDA graphs on, 2 batching threads give 457 and 3
  give 520, compared with 534 for 4; 6 threads and 2,880 search threads are no faster.
- **Channels-last and fused convolution.** cuDNN's tensor-core kernels work in channels-last
  (NHWC) layout, so with the network in channels-first layout cuDNN converted every convolution's
  input and output, 15% of kernel time. Fusing bias and ReLU into the convolution alone gained
  little because it added layout conversions; keeping activations channels-last removed them.
- **CUDA graphs.** An eager forward pass launches about 120 kernels and makes about 160 GPU
  allocations, each launch taking ~9 µs in the driver with four batching threads. A recorded graph
  replays the pass with one launch; batches are padded to a few sizes, each with its own graph. The
  batching thread's launch time per batch fell from 2,715 µs to 438 µs.
- **Two-byte moves.** A move was a 32-byte struct; packed into 16 bits (6 bits per square, 4 for
  the promotion piece, as Stockfish does) it shrinks move lists and search nodes and cut search CPU
  time 8.8%. With tiers on, the gain was +9.6%.
- **Reuse budget.** Previously each move ran 200 new simulations on top of the reused subtree. This
  changes the search's work per move by design; with tiers on, network evaluations per move fell 36%
  and throughput rose from 68.9 to 109.7.
- **Root quiescence search.** Self-play ran a full quiescence search at the root of every move to
  record the network's material input, even in vanilla mode, where training sets that input to zero.
  On positions from games between random networks it took more than half of the search threads' CPU
  time, sometimes seconds for one position. Skipping it cut CPU time per move from 34.3 ms to
  15.6 ms.
- **Defaults.** Making f16 and graphs the defaults played identical games (234,831 moves) and
  changed nothing in speed. It exposed a crash: creating a CUDA stream or cuDNN handle on one
  thread invalidated a graph being recorded on another, so these steps now share one lock.
- **Mate in one.** A leaf whose side to move can mate in one gets the value +1 without a network
  call. Network evaluations per move fell from 46.7 to 41.1, and the mean game from 134 to 122
  moves. The check costs about one move generation per leaf on the CPU, which has spare time
  while the GPU is the bottleneck.
- **Endgame tables.** Probes are needed only at 5 or fewer pieces, so they add no measurable cost;
  22% of games (430 of 1,920) ended at a covered position.
- **Ending games at a mate in one.** The move that mates, and the replies before it, cost almost
  no network calls, so removing them from the count lowers moves per second: network evaluations
  per move rose from 40.7 to 44.6. The rate falls because cheap moves are no longer played, not
  because any move got slower.
- **Mates in up to 5.** A deeper search ends more games early (82% of games end at a forced mate,
  compared with 72% for mate in one) and saves network calls (41.9 per move, compared with 44.3),
  for 3 ms more CPU time per move on cores that were idle. See the sweep below.
- **Faster mate search.** At mate in 5 the GPU is the bottleneck, so the faster search changed
  nothing (764.9 ± 2.5 before, in the same session); it matters at deeper settings.
- **Unexplained.** Rebuilding the move-generation change that preceded the fused convolution
  measured 437 instead of 453 moves per second (two runs each); the cause was not found.

## Running the network: candle and precision

**Why candle instead of LibTorch.** The engine first ran the network through LibTorch, PyTorch's C++
library, via the `tch` bindings. Candle, a machine learning library written in Rust, removed that
dependency and its run-time CUDA library setup, lets the network tests run in continuous integration
on CPU, and allows custom CUDA kernels and cuDNN calls next to the network code. With batch
normalization folded into the convolutions at load time and the cuDNN algorithm fixed per batch
size, a batch of 128 took 4.40 ms in candle and 7.19 ms in LibTorch (f32, before any of the changes
below).

**Precision.** In builds with cuDNN, the network runs on the GPU in 16-bit floating point (f16) on
the tensor cores, accumulating in 32 bits; `KF_PRECISION=f32` (or `bf16`) overrides this, and the CPU
always uses f32. On the generation-34 model, f16 changes move priors by
at most 0.004 and the value logit by at most 0.023, and picks the top move of f32 on 64 of 64 test
positions. Time per call through `predict_batch` (`nn_bench`, median of 500 calls; interquartile
ranges within 1.5% of the median):

| Batch size | f32 | f16 |
|---|---|---|
| 32 | 1.74 ms | 0.93 ms |
| 64 | 2.30 ms | 1.07 ms |
| 128 | 3.25 ms | 1.72 ms |

## What limits it now

With the network alone, the GPU is busy 92–93% of the time. Profiled before exact endings were added,
each batch spent 1.4 ms waiting for the GPU and about 1 ms in other driver calls: creating and
destroying an event (0.3 ms), four uploads from pageable host memory (0.29 ms), two downloads
(0.2 ms) and six allocations (0.15 ms). Reserving room for 256 moves in each move list
(`KF_RESERVE_MOVES=1`) gave a further +4.5% with two-byte moves, but has not been measured on the
current code.

With the tiers on, self-play is limited by the CPU: 109.0–110.3 moves per second with 480 search
threads and one batching thread, the GPU busy 14% of the time. Nearly all of that CPU time goes to
the classical search's move handling.

## How deep to search for mates

Each setting ran 1,920 games with endgame tables on, two runs alternated (± is half their
difference), except the last six rows, which are single runs. The settings play different games,
so moves per second compares the cost per training position rather than equal work. The untrained
network blunders into mates far more often than a trained one would, so the savings in training
are likely smaller. All rows predate the mate search speed-ups below.

| Mate search | Moves/s | Network calls per move | CPU ms per move | Games ending at a mate |
|---|---|---|---|---|
| Mate in 1 | 742.2 ± 3.7 | 44.3 | 15.7 | 72% |
| Every move, mate in 2 | 732.7 ± 1.4 | 43.2 | 25.2 | 80% |
| Every move, mate in 3 | 124.2 ± 0.2 | 42.7 | 241 | 84% |
| Checking moves, mate in 2 | 748.3 ± 0.2 | 43.1 | 16.4 | 77% |
| Checking moves, mate in 3 | 754.0 ± 0.6 | 42.5 | 16.6 | 80% |
| Checking moves, mate in 4 | 760.1 ± 0.7 | 42.1 | 17.3 | 82% |
| **Checking moves, mate in 5** | **763.4 ± 0.5** | 41.9 | 18.5 | 82% |
| Checking moves, mate in 6 | 765.6 | 41.5 | 22.1 | 83% |
| Checking moves, mate in 7 | 640.3 | 41.3 | 35.6 | 85% |
| Checking moves, mate in 8 | 305.2 | 41.4 | 87.3 | 84% |
| Checking moves, mate in 3, one quiet move per line | 672.7 | 42.5 | 33.6 | 83% |
| Checking moves, mate in 4, one quiet move per line | 478.5 | 42.0 | 54.9 | 85% |
| Checking moves, mate in 5, one quiet move per line | 250.2 | 41.9 | 115.9 | 87% |

Once the mate search uses more CPU time than the cores have spare, the network waits on it and
throughput collapses: searching every move to mate in 3 leaves the GPU busy 19% of the time.
Mate in 6 is within noise of mate in 5 (the gap between two identical settings was 0.4%) at 19%
more CPU time, so mate in 5 is the default. The sweep was stopped to speed up the search first,
and will be rerun with the faster code.

## Changes measured and not adopted

- **Several leaves per game (optional, off by default).** Each game selects K leaves before waiting
  for the network, with a temporary "virtual loss" steering later selections elsewhere. In King of
  the Hill, K = 4 raised throughput 58% at 60 threads, but at equal simulations it plays weaker: in
  200 games of the generation-34 model playing itself, K = 4 scored 0.435, or −45 Elo [−92, +2]
  relative to K = 1 (`--leaves-per-game`). More threads raise batch sizes without this cost.
- **Collecting the next batch while the GPU runs the current one** (on by default, `KF_PIPELINE=0`
  turns it off): 298.0 compared with 310.0 with one batching thread, and 453.1 compared with 456.9
  with four.
- **Reserving CPU cores for the batching thread** (`KF_INFER_CPUS`). With the tiers off, the thread
  waited for a free core only 1–5% of the time, which bounds what reserving cores can gain.
- **A shorter wait for a batch to fill.** An idle batching thread waits up to 500 µs for more
  requests; 0 µs gave 244 moves per second and 2,000 µs gave 287, compared with 298.
- **Larger batches from one batching thread:** 326 moves per second at cap 256 with 960 threads,
  and 334 and 329 at cap 512 with 1,920, compared with 453 for four threads at cap 128.
