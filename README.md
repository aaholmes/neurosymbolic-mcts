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
- **Standard chess**, the intended test: a 30-generation pilot of the network-only engine, with
  forced mates and endgame tables as exact endings, ran cleanly. Candidates kept beating the current
  model, but the acceptance test accepted only 4 of 30, which motivated continuous training (below).
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
engine already plays reasonably. The baseline, as in AlphaZero, uses the network alone. Moves are
selected with AlphaZero's PUCT rule, with one change so that the network's top move is tried first
on a node's first visit. Details: [docs/SEARCH.md](docs/SEARCH.md).

## Tech stack and how the parts fit together

| Component | Language | Role |
|---|---|---|
| Search engine (`src/`) | Rust | Board representation, move generation, the three tiers, MCTS, self-play and evaluation |
| Network inference (`src/nn/`) | Rust, [candle](https://github.com/huggingface/candle), cuDNN, custom CUDA kernels | Runs the network on the GPU in batches collected from all search threads |
| Training (`python/`) | Python, PyTorch | Defines and trains the network, and runs the training loop |
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
- **Training stays in PyTorch**, for its mature automatic differentiation and optimizers. On its
  own the trainer runs 2.6 times as fast as continuous training needs; how much self-play slows
  when both share the GPU is being measured.

## Training

Two loops are available ([docs/TRAINING.md](docs/TRAINING.md)):

- **In generations** (`python/orchestrate.py`), as in AlphaGo Zero: train a candidate on recent
  games and keep it only if it beats the current model in a sequential probability ratio test.
- **Continuously** (`python/continuous.py`), as in AlphaZero and KataGo: self-play never stops,
  each new game uses the newest weights, and the trainer trains on a sliding window of recent
  games so that each position is used a fixed number of times on average. This runs end to end
  but has not yet trained a model.

Positions with board symmetries (mirror images without castling rights, all eight rotations and
reflections without pawns) are trained on one randomly drawn equivalent image, so every position
counts once whatever its symmetry.

## Next steps

1. **Train continuously in standard chess**, compared with the generation loop at equal compute,
   with a periodic match with a fixed reference to catch a collapse.
2. **Network alone, with and without exact endings**: matched runs from scratch, with a bootstrap
   interval on the Elo difference. Whether exact endings help training is not yet measured.
3. **Make the quiescence search cheaper**, so the full classical search fits in standard chess.
   It averages 9.4 ms per call, mostly from a few positions with exposed kings; standard pruning
   (static exchange evaluation, delta pruning, a size cap) should shrink it at least 10×.
4. **The full engine versus the network alone in standard chess at equal wall-clock time**,
   including the untrained engine, to separate a head start from better learning.
5. **Full searches on only some moves** (playout cap randomization,
   [Wu, 2019](https://arxiv.org/abs/1902.10565)), judged by a training comparison.

Later directions: a small learned evaluation inside quiescence search, as in Stockfish's NNUE;
network layers that connect squares a knight or queen move apart; tuning the training settings
cheaply on saved data before full runs; and applying the tiered search to mathematical reasoning,
with an automated theorem prover as the exact method.

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
- [docs/TRAINING.md](docs/TRAINING.md): both training loops and their settings
- [docs/VARIANT_RESULT.md](docs/VARIANT_RESULT.md): the first result, with all ratings
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md): what was tried, what failed, and why

## License

MIT. See [LICENSE](LICENSE).
