# Neurosymbolic Monte Carlo tree search

**Exact solutions for tractable positions as terminal search nodes, for sample-efficient reinforcement learning.**

Self-play engines like AlphaZero learn everything from scratch, including positions a classical
solver can evaluate exactly in microseconds. This engine searches classically first. Any position
an exact method can evaluate, such as a forced mate in N moves, becomes a terminal node of the Monte
Carlo tree search (MCTS) with its exact value, just as checkmate does, so the engine is rewarded for
reaching it rather than only for checkmate. The training signal is denser, and unlike a learned
reward model it cannot be gamed. The exact evaluation is also cheaper: an exact search costs about
140 µs per position, compared with about 1,500 µs for a network evaluation. Only positions no exact
method can evaluate are sent to the neural network.

Where no exact answer exists, the classical search still informs the network. A quiescence search
plays out the forcing moves in a position, mainly captures, and scores the quiet position that
results. That score enters the value function twice: as a prior term added to the network's
output, and as an input to the network. Forced sequences are easy for human players and classical
engines but hard for a network that evaluates a position in a single pass. AlphaGo fed the result of
a ladder search, a forced capturing sequence in Go, to its networks as an input feature
([Silver et al., *Nature* 2016](https://doi.org/10.1038/nature16961)), and AlphaGo Zero, trained
without such features, understood ladders "only … much later in training"
([Silver et al., *Nature* 2017](https://doi.org/10.1038/nature24270)).

The pattern, exact solutions for tractable subproblems injected as terminal search nodes, applies to
any domain with classically solvable subproblems, such as mathematical reasoning with an automated
theorem prover.

## Result

Both engines below were trained by self-play in King of the Hill, a chess variant in which moving
your king to one of the four central squares also wins. The variant makes positions with exact
solutions common, which gives the classical layer room to matter
([DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)). Both use the same network (6 residual blocks, 128
channels, ~2M parameters) and 200 search simulations per move; the baseline uses the neural network
alone, as AlphaZero does.

After 18 generations the engine rates **2663 Elo [2565, 2763]**, about 625 Elo above the baseline
after 28 generations, **2038 [1957, 2111]**.

![Elo by training generation for both runs](tournament_results_800eval_elo_plot.png)

Ratings come from 6,579 games over 17 pairings, fitted with the Bradley–Terry model, with 95%
bootstrap intervals (1,000 resamples) relative to the baseline's first generation. Each training run
was rated as a ladder of adjacent generations, and the two ladders are joined by a single 100-game
match: this engine before any training, with its network initialized to zero so that it evaluates
positions classically, versus the baseline after 28 generations. That match gives +30 Elo
[−20, +74], so the untrained engine is about level with the fully trained baseline. The 625-Elo gap
between the final models is far larger than that uncertainty; a bootstrap interval on the gap itself
has not yet been computed.

<details>
<summary>All 18 ratings</summary>

| Model | Elo | 95% interval |
|---|---|---|
| tiered_gen18 | 2663 | [2565, 2763] |
| tiered_gen12 | 2595 | [2502, 2692] |
| tiered_gen8 | 2542 | [2448, 2633] |
| tiered_gen6 | 2478 | [2391, 2567] |
| tiered_gen5 | 2347 | [2260, 2428] |
| tiered_gen4 | 2312 | [2223, 2395] |
| tiered_gen3 | 2228 | [2144, 2306] |
| tiered_gen1 | 2167 | [2082, 2242] |
| tiered_gen0 | 2068 | [1984, 2148] |
| vanilla_gen28 | 2038 | [1957, 2111] |
| vanilla_gen23 | 1996 | [1920, 2063] |
| vanilla_gen21 | 1866 | [1801, 1933] |
| vanilla_gen13 | 1833 | [1772, 1902] |
| vanilla_gen10 | 1794 | [1734, 1859] |
| vanilla_gen7 | 1726 | [1672, 1781] |
| vanilla_gen4 | 1639 | [1598, 1684] |
| vanilla_gen1 | 1600 | [1566, 1636] |
| vanilla_gen0 | 1500 | (anchor) |

`tiered` is this engine and `vanilla` the baseline.

</details>

## How the search works

Each position is handled by the cheapest method that can evaluate it:

| Tier | Method | Result | Mean cost |
|---|---|---|---|
| 1 | Exact search for a forced mate in up to 5 moves (checking moves only), or a forced king march to the center in up to 3 | Proven value | 132–149 µs |
| 2 | Quiescence search: captures and promotions are played out until the position is quiet, with all replies to check, one quiet check or fork per side, and a test for threats made by letting the side to move pass; the quiet position is scored with PeSTO, a standard hand-tuned evaluation of material and piece placement | Heuristic value | 8 µs |
| 3 | Neural network | Learned value | 1,487 µs |

The Tier 3 cost is the mean time per network call as the search sees it, measured with
`profile_engine` (4 games played one after another, 200 simulations per move). It includes up to
0.5 ms that the inference server waits for other requests to fill a batch.

A position evaluated exactly in Tier 1 is terminal, like checkmate or stalemate, so its proven value
propagates up the tree without being averaged with approximate evaluations of its children.

The final value combines the network with the classical score:

$$V = \tanh(V_{\text{net}} + k \cdot \Delta M)$$

where $\Delta M$ is the Tier 2 score, $V_{\text{net}}$ is the network's unbounded output (which also
receives $\Delta M$ as an input), and $k$ is a learned scalar for how much to trust material,
initialized to 0.326. With no network, $V = \tanh(0.326 \cdot \Delta M)$; this is the untrained
engine in the result above.

Moves are selected with AlphaZero's rule (PUCT, a variant of the upper confidence bound for trees),
with one change:

$$U(s,a) = c \cdot P(s,a) \cdot \frac{\sqrt{\max(N(s), 1)}}{1 + n(s,a)}$$

AlphaZero uses $N(s)$, the visit count of the parent, in place of $\max(N(s), 1)$. On a node's first
visit $N = 0$, which makes the exploration bonus zero for every move, so the first move tried is
whichever the move generator lists first. With $\max(N, 1)$ the bonus is proportional to the
network's prior $P(s,a)$ from the first visit, so the network's top choice is tried first. For
$N \geq 1$ the two rules are identical.

## How the components fit together

| Component | Language | Role |
|---|---|---|
| Search engine (`src/`) | Rust | Board representation, move generation, the three tiers, MCTS, self-play and evaluation binaries |
| Network inference (`src/nn/`) | Rust, [candle](https://github.com/huggingface/candle) | Runs the network on CPU or GPU, called from a server thread that batches requests from all search threads |
| Training (`python/`) | Python, PyTorch | Defines the network, trains it on self-play data, and runs the whole loop (`orchestrate.py`) |
| GPU search (`cuda/`) | CUDA C++ | An alternative in which the entire search runs on the GPU (see below) |

The network is trained in PyTorch and run in Rust. The orchestrator calls the Rust binaries for
self-play and evaluation, and they write training positions to disk for the Python trainer. The
Rust side re-implements the network layer by layer and loads the PyTorch weights file (`.pth` or
`.safetensors`) directly, so a trained model needs no export step. A committed set of PyTorch
reference outputs, from a small network with every weight randomized, checks that the two
implementations agree to within 1e-4.

The search runs in Rust because a Tier 1 search visits about 1,000 positions per call, which
needs bit-level board operations with no interpreter overhead. Network calls from all games
are collected into batches, because a GPU call on 16 positions takes 1.38 ms, compared with 0.95 ms
for a single position.

### Inference with candle instead of LibTorch

The engine originally ran the network through LibTorch, PyTorch's C++ library, using the `tch`
Rust bindings. It now uses candle, a machine learning library written in Rust, which removed the
LibTorch dependency and the work needed to make LibTorch find its CUDA libraries at run time. The
network tests now run in continuous integration on CPU. On the GPU, convolutions go through cuDNN,
each batch normalization is folded into the preceding convolution when the model loads, and the
cuDNN algorithm is fixed by batch size rather than searched for on every call.

Measured on one RTX 5060 Ti, with one trained model (6×128) and identical positions for both
backends, the time for one batched network call (median of 200–500 calls; interquartile ranges
are within 2% of the median):

| Batch size | LibTorch | candle | Speedup |
|---|---|---|---|
| 1 | 0.97 ms | 0.95 ms | 1.02× |
| 16 | 1.51 ms | 1.38 ms | 1.09× |
| 128 | 7.19 ms | 4.40 ms | 1.63× |
| 512 | 31.3 ms | 17.4 ms | 1.80× |
| 2048 | 140 ms | 92.6 ms | 1.51× |

The gain grows with batch size up to about 1,000 positions. Self-play uses batches of at most 16, where it is small, so
end-to-end self-play (30 games, 200 simulations per move, 15 threads, 2 runs per backend) went from
27.8 and 26.9 moves per second with LibTorch to 28.8 and 30.7 with candle, about 9% faster. Running
more games in parallel, so that batches fill up, is where the larger speedups would show.

The outputs agree with plain 32-bit floating-point arithmetic to within 5e-6 when candle uses its
own GPU kernels. With cuDNN, and with LibTorch, they differ by up to 3e-3 and 4e-3 respectively in
the value logit, consistent with cuDNN computing 32-bit convolutions in the lower-precision TF32
format. On 512 positions all three agree on the top move.

## Training

Self-play, a replay buffer, training, and evaluation of each new generation in a loop:

- **Optimizer:** Muon (learning rate 0.02), with adaptive epochs and early stopping.
- **Accepting a new generation:** a sequential probability ratio test (SPRT) over up to 800
  evaluation games, which usually stops after about 30 when the result is clear.
- **Data:** after generation 1 the evaluation games also supply training data, so separate self-play
  is optional. Positions are weighted by the strength of the model that played them (a 200-Elo gap
  gives about 32% inclusion). Mirror images are added for positions without castling rights (2×), and
  all eight board symmetries for pawnless endgames (8×).

```bash
python python/orchestrate.py --enable-koth \
  --sims-schedule "0:100,5:200,10:400,20:800"    # full training loop, King of the Hill
python python/orchestrate.py --disable-tier1       # ablation: no exact search
python python/orchestrate.py --disable-material    # ablation: no classical material score
python python/orchestrate.py --skip-self-play      # train on evaluation games only
python python/orchestrate.py --arch transformer --enable-koth
```

## Search on the GPU

The `cuda/` folder holds a version in which the whole search runs inside one persistent CUDA kernel:
tree traversal, node expansion, the tactical checks, quiescence search and the network itself, with
no round trips to the CPU and no cuBLAS or cuDNN. On an RTX 5060 Ti with 36 concurrent games at 200
simulations per move it produces 101.4 training samples per second (20,300 simulations per second).
It is limited by the GPU's ~99 KB of shared memory per multiprocessor, which caps the network size,
so current work runs the search on CPU threads and batches network evaluations on the GPU. Design
details and measurements: [BENCHMARKS.md](BENCHMARKS.md) and [CUDA_TC_ANALYSIS.md](CUDA_TC_ANALYSIS.md).

## Current state and next steps

The engine, training loop and both search back ends work, and the result above is complete apart
from a confidence interval on the Elo gap between the final models. Open directions:

- **The Elo gap:** compute a bootstrap interval on the Elo difference between the final models.
- **Larger batches:** run more games in parallel so that network calls reach the batch sizes where
  candle is 1.6–1.8× faster than LibTorch.
- **Networks built around piece moves:** layers that pass information between squares a knight or
  queen move apart, rather than only between neighbours as a 3×3 convolution does. Each new
  architecture must be written in both PyTorch and Rust; the reference-output tests make that a
  repeatable check.
- **Other domains:** this tiered search for problems with classically solvable subproblems, such
  as theorem proving.

## Building and running

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install torch numpy python-chess

cargo build --release                      # search without the network
cargo build --release --features neural    # with the network, on CPU (candle)
cargo build --release --features cuda      # with the network on the GPU (needs nvcc >= 12.8 for RTX 50-series)
cargo build --release --features cudnn     # as cuda, with cuDNN convolutions (fastest; libcudnn on LIBRARY_PATH)
./target/release/caissawary                # the engine, speaking the Universal Chess Interface (UCI)

cd cuda && mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 && make
```

## Testing

About 1,350 tests: 829 in Rust, 314 in CUDA and 203 in Python. The fast `cargo test` suite runs about
750 of the Rust tests in about 50 s. See [TESTING.md](TESTING.md).

```bash
cargo test                                        # fast Rust tests (~50 s)
cargo test --features neural                      # plus the network, compared with PyTorch reference outputs
cargo test --features slow-tests                  # full suite, including move-generation counts (~200 s)
cd python && python -m pytest test_*.py -v        # training pipeline
cuda/build/test_mcts_kernel                       # GPU search
cuda/build/test_transformer
cuda/build/test_block_ops
cuda/build/test_budget_reuse
cuda/build/test_nn_ops
cuda/build/bench_throughput 100 200 36 [p2]       # end-to-end self-play throughput
cuda/build/bench_forward_b2 1000                  # batched convolution microbenchmark
cargo test --release --features cudnn --test nn_bench -- --ignored --nocapture   # network latency by batch size
```

<details>
<summary>Binary targets</summary>

| Binary | Description |
|--------|-------------|
| `caissawary` | The engine (UCI) |
| `self_play` | Self-play data generation with game logs |
| `evaluate_models` | Head-to-head model evaluation with SPRT |
| `round_robin` | Tournament with Elo estimation |
| `mcts_inspector` | Search tree visualization (Graphviz DOT) |
| `verbose_search` | Real-time search narration |
| `verbose_game` | Full game between two classical engines |
| `benchmark` | Performance testing |
| `strength_test` | Engine strength assessment |
| `run_experiments` | Ablation studies |
| `elo_tournament` | Elo estimation for classical engines |
| `texel_tune` | Evaluation weight tuning |
| `generate_training_data` | Standalone training data generation |
| `play_engine` | Play the engine yourself |
| `profile_engine` | Per-operation timing profiler |
| `qsearch_gui` | Interactive quiescence-search tree viewer (web UI at localhost:8088) |

</details>

## Further reading

- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md): what was tried, what failed, and why
- [BENCHMARKS.md](BENCHMARKS.md): throughput measurements and the GPU design
- [TESTING.md](TESTING.md): test suite documentation and coverage
- [SPRT_GUIDE.md](SPRT_GUIDE.md): how new generations are evaluated
- [CUDA_TC_ANALYSIS.md](CUDA_TC_ANALYSIS.md): Tensor Core memory analysis

## License

MIT. See [LICENSE](LICENSE).
