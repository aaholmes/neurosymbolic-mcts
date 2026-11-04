# Neurosymbolic Monte Carlo tree search

**Exact solutions for tractable positions as terminal search nodes, for sample-efficient reinforcement learning.**

Self-play engines like AlphaZero learn everything from scratch, including positions a classical
solver can evaluate exactly in microseconds. This chess engine searches classically first. Any
position an exact method can evaluate, such as a forced mate in N moves, becomes a terminal node of
the Monte Carlo tree search (MCTS) with its exact value, just as checkmate does. The engine is then
rewarded for reaching such a position, not only for checkmate, so the training signal is denser,
and unlike a learned reward model it cannot be gamed. Exact evaluation is also cheaper: at most
about 140 µs per position, compared with about 1,350 µs for a network evaluation as the search
sees it. Only positions no exact method can evaluate go to the neural network.

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

- **A first result, in a chess variant where exact solutions are common**: after 18 generations
  of self-play the engine rates about 625 Elo above a network-only baseline trained for 28. It is
  preliminary: compute was not matched (the classical search is extra), the two runs are linked by
  a single 100-game match, and the untrained engine already plays about as well as the trained
  baseline, so part of the gap is a head start. Details: [docs/VARIANT_RESULT.md](docs/VARIANT_RESULT.md).
- **Training settings for standard chess**, the intended test, tuned on the network-only engine
  with forced mates and endgame tables as exact endings. Each of 29 runs so far trained from
  scratch for one hour on one GPU. The best of 43 rates 233 ± 8 Elo above the default settings,
  and four others are within 41 Elo of it. They all train each position 0.65–1.2 times rather than
  4, and most use 100 simulations per move rather than 200, so more of the GPU goes to self-play:
  up to 3.3 times as many positions in the hour. Ratings follow the positions generated, at about
  +106 Elo per doubling, not the number of training steps. Runs with similar settings differed by
  50–120 Elo, so the top settings are next repeated with new seeds, then rerun for longer.
- **Self-play speed** (200 simulations per move, one RTX 5060 Ti): **765 moves per second**, up
  from about 65 at the start, through custom GPU kernels, half precision, CUDA graphs and many
  search threads sharing a few batching threads ([docs/THROUGHPUT.md](docs/THROUGHPUT.md)). With
  the full classical search on, it is limited by the CPU, at about 110.
- **Mate search** 3.7× faster at the default depth, by generating only checking moves and replies
  to check, trying first the checks that leave the enemy king fewest escape squares, and
  remembering positions already searched ([docs/MATE_SEARCH.md](docs/MATE_SEARCH.md)).

## How the search works

Each position is evaluated by the cheapest method that can evaluate it:

| Tier | Method | Result | Mean cost |
|---|---|---|---|
| 1 | Exact search for a forced mate in up to 5 moves in which every attacking move gives check, and endgame tables (Syzygy) for positions with 5 or fewer pieces | Proven win, loss or draw | ≤ 140 µs |
| 2 | Quiescence search: captures, promotions and checks are played out, and the quiet position is scored with a standard hand-tuned evaluation (PeSTO) | Heuristic score ΔM | 8 µs |
| 3 | Neural network (a 6-block residual network, ~2M parameters) | Learned value and move priors | 1,346 µs |

A position solved in Tier 1 is terminal, so its proven value propagates up the tree without being
averaged with approximate evaluations of its children. Otherwise the value is

$$V = \tanh(V_{\text{net}} + k \cdot \Delta M)$$

where $V_{\text{net}}$ is the network's unbounded output and $k$ is a learned weight on the
classical score. With no network, $V = \tanh(0.326 \cdot \Delta M)$, which is why the untrained
engine already plays reasonably. The baseline, as in AlphaZero, uses the network alone.

Moves are selected with AlphaZero's rule (PUCT, which adds to each move's average value a bonus
that grows with the network's prior and shrinks with visits). Two additions keep the search from
ignoring moves the network rates low: every capture, check and fork gets one visit before the rule
applies, and so does every move at the root. Details: [docs/SEARCH.md](docs/SEARCH.md).

## How self-play generates training data

Each position in a self-play game gives the network two targets: a policy target from the search
and a value target from the game's result.

- **Policy target: visits minus one.** Every root move gets one visit, which is the price of
  evaluating it, so one visit is subtracted from each move's count. A move whose only visit showed
  it was bad gets no weight.
- **Move played: sampled from the same counts.** The move is drawn in proportion to
  (visits − 1)^(1/τ). The temperature τ starts at 1 (in proportion to the counts) and halves every
  few moves, toward always playing the most-visited move, so openings vary and later play follows
  the search. Five of the six best tuning runs halve it every 5–13 moves; the one run at 3 rated
  far lower.
- **Exploration that did not help.** AlphaZero's random (Dirichlet) noise on the root priors lost
  56 ± 6 Elo. So did searching side positions, one or two random moves off the main line, to
  teach the network to play on after a blunder: at half of all moves they halved the games played
  in an hour and lost 112 ± 8 Elo. Both are off by default.
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

- **The search is in Rust** because one exact search visits about 1,000 positions, which needs
  bit-level board operations with no interpreter overhead.
- **Each game has its own search thread** (1,920 by default), and a few batching threads collect
  their network requests into batches of up to 128. Large batches use the GPU far better: 13 µs per
  position at 128 compared with 29 µs at 32.
- **Inference uses candle, not LibTorch.** Candle is a machine learning library written in Rust. It
  removed a large C++ dependency, lets the network tests run on CPU in continuous integration, and
  allows custom CUDA kernels and cuDNN calls next to the network code; a batch ran 39% faster.
- **Training stays in PyTorch**, for its mature automatic differentiation and optimizers. Its cost
  is GPU time, not Python: on one GPU, training each position 4 times on average slows self-play
  by 15% with the trainer in bf16 and channels-last layout, compared with 24% in f32.

## Training

Two loops are available ([docs/TRAINING.md](docs/TRAINING.md)):

- **In generations** (`python/orchestrate.py`), as in AlphaGo Zero: train a candidate on recent
  games and keep it only if it beats the current model in a sequential probability ratio test. In
  a 30-generation pilot in standard chess, candidates kept beating the current model but the test
  accepted only 4, which motivated the continuous loop.
- **Continuously** (`python/continuous.py`), as in AlphaZero and KataGo: self-play never stops,
  each new game uses the newest weights, and the trainer trains on a sliding window of recent
  games so that each position is used a fixed number of times on average (the replay ratio). The
  trainer runs in 16-bit floating point (bf16) with the channels-last memory layout, which cuts its
  GPU time per step by 43%, with training loss within 0.1% of f32 over 320 steps. A run can stop
  after a set time, with the learning rate decaying to 0 over it.

**Tuning** (`python/hpo.py`) searches six settings: learning rate, replay ratio, simulations per
move, how fast the temperature halves, the exploration constant in PUCT, and the window size. Each
run trains from scratch for one hour, then plays 800 games with up to five earlier runs, and all
runs are rated jointly by Elo (a Bradley–Terry fit with bootstrap standard errors). A
Gaussian-process optimizer (Optuna) proposes each new run from the ratings so far. Two runs at the
default settings with different seeds rated 22 ± 6 Elo apart, but runs with similar settings near
the best differed by 50–120 Elo, so single runs cannot rank the top settings. The optimizer kept
proposing settings at the range limits, where its model is least certain, so after 13 trials it
proposed only inside narrower ranges around the best runs, with its model still fitted on every
run. Positions generated per hour fit 1/(a·sims + b·ratio + c) within 7%; one unit of replay
ratio costs as much GPU time as 25 simulations per move ([docs/TRAINING.md](docs/TRAINING.md)).

Each new weights version starts a fresh inference server inside the running self-play process.
Two bugs made long runs fail: a server loading or shutting down while another recorded a CUDA
graph could break that recording and freeze self-play, and each retired server kept about 75 MiB
of GPU memory. Loading and release now wait for any recording to finish, each retired worker frees
its cuDNN state and graph memory, and a run stops if no game finishes for five minutes.

## Next steps

In order of priority:

1. **Repeat the top 1-hour runs with new seeds** (under way), and check how much of each rating
   comes from the exploration constant used in rating matches rather than in training.
2. **A 12-hour continuous run of the network-only engine** with the best 1-hour settings, its
   checkpoints rated hourly with a fixed opponent, to see whether the 1-hour settings hold.
3. **Make the quiescence search cheaper**, so the full classical search fits in standard chess.
   It averages 9.4 ms per call, mostly from a few positions with exposed kings; standard pruning
   (static exchange evaluation, delta pruning, a size cap) should shrink it at least 10×.
4. **Three matched 4-hour runs from scratch, two seeds each**: the network alone, the full tiered
   engine, and the tiered engine training only the policy (below), rated with a fixed opponent and
   with bootstrap intervals on the differences. This measures whether exact endings and the
   classical score help training, and whether a learned value is worth its noise.
5. **Only if the policy-only engine is close or ahead**: tune its settings, whose best values
   likely differ because its self-play is limited by the CPU, then a 12-hour run.

Further experiments (full searches on only some moves, network size, Gumbel AlphaZero, a mixed
value target) are listed in [docs/TRAINING.md](docs/TRAINING.md#next-experiments).

**Long-term goal: a strong engine that learns only its policy.** The value of a position would
come from classical methods alone: exact search for forced mates and endgame tables, then the
quiescence score, $V = \tanh(k \cdot \Delta M)$, with the network's value fixed at 0. The network
learns only which moves to search. In tuning, the value loss fell 1.7 times its noise over the
last 40 minutes of an hour while the policy loss fell 14 times, because a game's result is one
noisy label shared by all its positions; training only the policy removes that target. The cost
is a ceiling: the engine can at best match a shallow search on a hand-tuned evaluation (PeSTO),
with the best possible choice of moves to search, and averaging values in MCTS makes a weak
evaluation hurt more than in minimax search. How high that ceiling is has not been measured.
Using a learned evaluation such as Stockfish's NNUE would raise it, but would no longer be
learning from scratch on limited compute. Details: [docs/TRAINING.md](docs/TRAINING.md#long-term-goal-training-only-the-policy).

Later directions: a small learned evaluation inside quiescence search, as in Stockfish's
efficiently updatable neural network (NNUE); network layers that connect squares a knight or queen
move apart; and applying the tiered search to mathematical reasoning, with an automated theorem
prover as the exact method.

## Quick start

```bash
cargo build --release --features cudnn        # needs CUDA (nvcc >= 12.8 for RTX 50-series) and cuDNN
./target/release/self_play 1920 200 out/ model.pt false false false none --syzygy /data/syzygy
python python/continuous.py --work-dir runs/c1 --replay-ratio 4 --window 200000
cargo test                                    # fast Rust tests (~50 s)
cd python && python -m pytest                 # training pipeline
```

About 1,400 tests in Rust, CUDA and Python. All build options, settings, binaries and tests:
[docs/USAGE.md](docs/USAGE.md) and [TESTING.md](TESTING.md).

## Further reading

- [docs/SEARCH.md](docs/SEARCH.md): the tiers, the value function and move selection in full
- [docs/THROUGHPUT.md](docs/THROUGHPUT.md): how self-play went from 65 to 765 moves per second
- [docs/MATE_SEARCH.md](docs/MATE_SEARCH.md): the mate search speed-ups and move orders tried
- [docs/TRAINING.md](docs/TRAINING.md): both training loops, their settings and the tuning results
- [docs/VARIANT_RESULT.md](docs/VARIANT_RESULT.md): the first result, with all ratings
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md): what was tried, what failed, and why

## License

MIT. See [LICENSE](LICENSE).
