# Neurosymbolic Monte Carlo tree search

**Exact solutions for tractable positions as terminal search nodes, for sample-efficient reinforcement learning.**

Self-play engines like AlphaZero learn everything from scratch, including positions a classical
solver can evaluate exactly in microseconds. This chess engine searches classically first. Any
position an exact method can evaluate, such as a forced mate in N moves or an endgame with five
pieces or fewer, becomes a terminal node of the Monte Carlo tree search (MCTS) with its exact value,
just as checkmate does. The engine is then rewarded for reaching such a position, not only for
checkmate, so the training signal is denser, and unlike a learned reward model it cannot be gamed.
Only positions no exact method can evaluate go to the neural network.

In a matched test, this engine trained for one hour rated 131 Elo [116, 146] above an
AlphaZero-style engine that relies on the network alone, given the same settings and training time.

Where no exact answer exists, classical search still informs the network. A quiescence search plays
out the forcing moves in a position, mainly captures, and scores the quiet position that results.
That score is added to the network's output as a prior and is also one of its inputs. Forced
sequences are easy for classical engines but hard for a network that evaluates a position in one
pass: AlphaGo fed the result of a ladder search (a forced capturing sequence in Go) to its networks
([Silver et al., *Nature* 2016](https://doi.org/10.1038/nature16961)), and AlphaGo Zero, trained
without it, understood ladders "only … much later in training"
([Silver et al., *Nature* 2017](https://doi.org/10.1038/nature24270)).

The idea, exact solutions for tractable subproblems used as terminal search nodes, applies to any
domain with classically solvable subproblems, such as mathematical reasoning with an automated
theorem prover.

## Results so far

All numbers are for standard chess on one RTX 5060 Ti GPU and a 16-core Ryzen 9 7950X CPU.

- **Matched comparison: the full engine is 131 Elo stronger than the network alone** after the
  same training. Each of the three engines (below) trained from scratch for one hour with the same
  settings, two seeds each, then all six played a round robin (100 games per pair, 100 simulations
  per move, each searching as it was trained). Differences of arm means, with 95% intervals from
  resampling games:

  | Comparison | Elo difference |
  |---|---|
  | Full engine − network only | +131 [+116, +146] |
  | Policy-only engine − network only | −18 [−29, −6] |
  | Full engine − policy-only engine | +149 [+135, +163] |

  The two seeds of each engine differed by 12–25 Elo, so the full engine's lead is far larger than
  the run-to-run spread; the policy-only engine's deficit is not. The policy-only engine generated
  three times as many positions in the hour (21–22M compared with 7.5M), and still did not catch up.
  Limits: the intervals cover game noise, not seed variance (two seeds per arm); the matches fixed
  simulations per move, not time, so the full engine spent more CPU per move; most games were drawn
  by repetition, which compresses rating gaps; and part of the lead may be a head start, since the
  classical score lets the untrained full engine play reasonably. Settings were tuned for the
  network-only engine, which favours it.
- **Training settings, tuned in 1-hour runs** of the network-only engine. The best of 43 runs
  rates 233 ± 8 Elo above the default settings. They train on each position 0.65–1.2 times rather
  than 4, and search 100 positions per move rather than 200, so more of the GPU goes to self-play:
  up to 3.3 times as many positions in the hour. Ratings follow the number of positions generated
  (about +106 Elo per doubling), not the number of training steps. Two reruns of the best settings
  with new seeds averaged 4 Elo above it [−14, +21].
- **Self-play speed**: 765 moves per second with the network alone (200 simulations per move), up
  from about 65, through custom GPU kernels, half precision, CUDA graphs and many search threads
  sharing a few batching threads ([docs/THROUGHPUT.md](docs/THROUGHPUT.md)).
- **Cheaper classical search.** The mate search is 3.7× faster at its default depth
  ([docs/MATE_SEARCH.md](docs/MATE_SEARCH.md)). The quiescence search is 3.4× faster with
  identical scores, by generating only checking and forking quiet moves, testing legality from
  pinned pieces, and keeping the piece on each square and the piece-square score up to date on the
  board; standard pruning makes it about 24× faster but changes 4% of values by more than 0.1, so
  it is optional ([docs/QUIESCENCE.md](docs/QUIESCENCE.md)). The policy-only engine needs no
  network value, so it makes 4.5 network calls per move instead of 29.7 and is CPU-bound: 2,266
  moves per second with 6-piece endgame tables, 26% faster than before the exact speed-ups.

## How the search works

Each position is evaluated by the cheapest method that can evaluate it. Costs are means per
position in standard chess, on one CPU thread for Tiers 1 and 2:

| Tier | Method | Result | Cost |
|---|---|---|---|
| 1 | Exact search for a forced mate in up to 5 moves in which every attacking move gives check, and endgame tables (Syzygy) for positions with 5 or fewer pieces | Exact win, loss or draw | 12 µs (mate search) |
| 2 | Quiescence search: captures, promotions, replies to check and one quiet check or fork per side in each line are played out, threats are tested by letting the side to move pass, and the quiet position is scored with a standard hand-tuned evaluation (PeSTO); a line that reaches a position the endgame tables cover ends with their result | Heuristic score ΔM | 88 µs (median 14 µs) |
| 3 | Neural network (a 6-block residual network, about 2M parameters) | Learned value and move priors | 13 µs of GPU time in a batch of 128 |

A position solved in Tier 1 is terminal, so its exact value propagates up the tree without being
averaged with approximate evaluations of its children. Tier 1 also stops the search for a move once
a root move averages a value above 0.99, which a large material lead alone can reach: in self-play
with an untrained network, 4.9% of searches stopped early and the root averaged 194 visits instead
of 200, so engines with and without Tier 1 search nearly equally hard. Otherwise the value is

$$V = \tanh(V_{\text{net}} + k \cdot \Delta M)$$

where $V_{\text{net}}$ is the network's unbounded output and $k$ is a learned weight on the
classical score. With no network, $V = \tanh(0.326 \cdot \Delta M)$, which is why the untrained
engine already plays reasonably.

Three engines are compared:

- **Network only** (the baseline, as in AlphaZero): $V = \tanh(V_{\text{net}})$, with forced mates
  and endgame tables still ending games with their exact result.
- **Full engine**: all three tiers, with the value above.
- **Policy-only engine** (`--policy-only revisit`): Tiers 1 and 2 give the value,
  $V = \tanh(0.326 \cdot \Delta M)$, and the network supplies only which moves to search. This
  is the long-term goal described under Next steps.

Moves are selected with AlphaZero's rule (PUCT, which adds to each move's average value a bonus
that grows with the network's prior and shrinks with visits). Two additions keep the search from
ignoring moves the network rates low: every capture, check and fork gets one visit before the rule
applies, and so does every move at the root. Details: [docs/SEARCH.md](docs/SEARCH.md).

## How self-play generates training data

Each position in a self-play game gives the network two targets: a policy target from the search,
which depends only on the position and the current network, and a value target from the game's
result, which depends on everything played afterwards.

- **Policy target: visits minus one.** Every root move gets one visit, which is the price of
  evaluating it, so one visit is subtracted from each move's count. A move whose only visit showed
  it was bad gets no weight.
- **Move played: sampled from the same counts**, in proportion to (visits − 1)^(1/τ). The
  temperature τ starts at 1 and halves every few moves, toward always playing the most-visited
  move, so openings vary and later play follows the search.
- **Exploration that did not help.** AlphaZero's random (Dirichlet) noise on the root priors lost
  56 ± 6 Elo. So did searching side positions, one or two random moves off the main line, to teach
  the network to play on after a blunder: at half of all moves they halved the games played in an
  hour and lost 112 ± 8 Elo. Both are off by default.
- **Board symmetries.** A position without castling rights equals its mirror image, and a pawnless
  one equals all eight rotations and reflections. Each position is trained on one image drawn at
  random, so every position counts once whatever its symmetry.

## Tech stack and how the parts fit together

| Component | Language | Role |
|---|---|---|
| Search engine (`src/`) | Rust | Board representation, move generation, the three tiers, MCTS, self-play and evaluation |
| Network inference (`src/nn/`) | Rust, [candle](https://github.com/huggingface/candle), cuDNN, custom CUDA kernels | Runs the network on the GPU in batches collected from all search threads |
| Training and tuning (`python/`) | Python, PyTorch, Optuna | Defines and trains the network, runs the training loop and the settings search |
| GPU-only search (`cuda/`) | CUDA C++ | An experimental version that runs the whole search in one GPU kernel ([BENCHMARKS.md](BENCHMARKS.md)) |

Python runs the loop: it starts the Rust self-play binary, which writes training positions to disk,
and trains the network on them in PyTorch. Rust loads the PyTorch weights file directly and runs
its own implementation of the network, which a test keeps within 1e-4 of PyTorch's outputs.

Why this split:

- **The search is in Rust** because the classical tiers visit hundreds to thousands of positions
  per call, which needs bit-level board operations with no interpreter overhead.
- **The classical search runs on the CPU, beside the GPU.** Each search thread runs the mate and
  quiescence searches while the GPU evaluates other threads' batches, so classical search that is
  cheap enough costs no GPU time; it only needs enough CPU threads.
- **Each game has its own search thread** (1,920 by default), and a few batching threads collect
  their network requests into batches of up to 128. Large batches use the GPU far better: 13 µs per
  position at 128 compared with 29 µs at 32.
- **Inference uses candle, not LibTorch.** Candle is a machine learning library written in Rust. It
  removed a large C++ dependency, lets the network tests run on CPU in continuous integration, and
  allows custom CUDA kernels and cuDNN calls next to the network code. A batch of 128 took 4.40 ms
  in candle and 7.19 ms in LibTorch (both f32), and 1.72 ms after moving to half precision.
- **Training stays in PyTorch**, for its mature automatic differentiation and optimizers. Its cost
  is GPU time, not Python: on one GPU, training each position 4 times on average slows self-play
  by 15% with the trainer in bf16 and channels-last layout, compared with 24% in f32.

## Training

Two loops are available ([docs/TRAINING.md](docs/TRAINING.md)):

- **In generations** (`python/orchestrate.py`), as in AlphaGo Zero: train a candidate on recent
  games and keep it only if it beats the current model in a sequential probability ratio test. In a
  30-generation pilot, candidates kept beating the current model but the test accepted only 4,
  which motivated the continuous loop.
- **Continuously** (`python/continuous.py`), as in AlphaZero and KataGo: self-play never stops,
  each new game uses the newest weights, and the trainer trains on a sliding window of recent games
  so that each position is used a fixed number of times on average (the replay ratio).

**Tuning** (`python/hpo.py`) searches six settings: learning rate, replay ratio, simulations per
move, how fast the temperature halves, the exploration constant in PUCT, and the window size. Each
run trains from scratch for one hour, then plays 800 games with up to five earlier runs, and all
runs are rated jointly by Elo (a Bradley–Terry fit with bootstrap standard errors). A
Gaussian-process optimizer (Optuna) proposes each new run from the ratings so far; after it kept
proposing settings at the range limits, where its model is least certain, it proposed only inside
narrower ranges around the best runs, with its model still fitted on every run. Positions
generated per hour fit 1/(a·sims + b·ratio + c) within 7%: one unit of replay ratio costs as much
GPU time as 25 simulations per move.

## Next steps

In order of priority:

1. **Separate the head start from the training effect**: rate the untrained full engine using the
   trained network-only engines. If it is already close to +131, the classical score helps play
   more than it helps learning.
2. **Matched by time, not simulations**: replay the round robin with each side given the same
   wall-clock time per move, since the full engine's classical search costs CPU per simulation.
3. **Longer matched runs (4 hours, three seeds per engine)**, to see whether the full engine's
   lead grows or shrinks as the network improves, with intervals over seeds as well as games.
4. **Tune the full engine's own settings**, since the current ones were tuned for the network alone.
5. **Whether the pruned quiescence search changes playing strength**: the full engine with exact
   and pruned quiescence, same network and seeds.

Further experiments (full searches on only some moves, network size, Gumbel AlphaZero, a mixed
value target) are listed in [docs/TRAINING.md](docs/TRAINING.md#next-experiments).

**Long-term goal: a strong engine that learns only its policy.** The value of a position would come
from classical methods alone: exact search for forced mates and endgame tables, then the quiescence
score, $V = \tanh(k \cdot \Delta M)$, with the network's value fixed at 0. The network learns only
which moves to search. In tuning, the value loss fell 1.7 times its noise over the last 40 minutes
of an hour while the policy loss fell 14 times, because a game's result is one noisy label shared
by all its positions; training only the policy removes that target. The cost is a ceiling: the
engine can at best match a shallow search on a hand-tuned evaluation (PeSTO), with the best possible
choice of moves to search, and averaging values in MCTS makes a weak evaluation hurt more than in
minimax search. How high that ceiling is has not been measured. A learned evaluation such as
Stockfish's NNUE would raise it, but would no longer be learning from scratch on limited compute
([docs/TRAINING.md](docs/TRAINING.md#long-term-goal-training-only-the-policy)).

The policy-only engine is implemented (`self_play --policy-only revisit`). In the matched
one-hour comparison it trailed the network-only engine by 18 Elo despite three times as much data,
so its ceiling is the open question; a learned evaluation, or a larger policy network using the
idle GPU, would be the next things to try.

Later directions: network layers that connect squares a knight or queen move apart, and applying
the tiered search to mathematical reasoning, with an automated theorem prover as the exact method.

## Quick start

```bash
cargo build --release --features cudnn        # needs CUDA (nvcc >= 12.8 for RTX 50-series) and cuDNN
./target/release/self_play 1920 200 out/ model.pt false false false none --syzygy /data/syzygy
# --syzygy takes several directories separated by colons, e.g. /data/syzygy:/data/syzygy6
# tables are read through memory maps (--syzygy-reads pread reads them by system call instead);
# --no-qsearch-tables keeps the quiescence search from probing them; --syzygy-set none, 5, 5+, 6 or 6+
# limits the tables used (N+ adds the N+1-piece rook and pawn endings with equal rooks, at most two pawns apart)
python python/continuous.py --work-dir runs/c1 --replay-ratio 4 --window 200000
cargo test                                    # fast Rust tests (~50 s)
cd python && python -m pytest                 # training pipeline
```

About 1,400 tests in Rust, CUDA and Python. All build options, settings, binaries and tests:
[docs/USAGE.md](docs/USAGE.md) and [TESTING.md](TESTING.md).

## Further reading

- [docs/SEARCH.md](docs/SEARCH.md): the tiers, the value function and move selection in full
- [docs/THROUGHPUT.md](docs/THROUGHPUT.md): how self-play went from 65 to 765 moves per second with the network alone
- [docs/MATE_SEARCH.md](docs/MATE_SEARCH.md): the mate search speed-ups and move orders tried
- [docs/QUIESCENCE.md](docs/QUIESCENCE.md): the quiescence search speed-ups and pruning
- [docs/TRAINING.md](docs/TRAINING.md): both training loops, the tuning results and the long-term goal
- [docs/VARIANT_RESULT.md](docs/VARIANT_RESULT.md): an early, unmatched result in King of the Hill, a chess variant
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md): what was tried, what failed, and why

## License

MIT. See [LICENSE](LICENSE).
