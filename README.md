# Caissawary

**Neurosymbolic MCTS with exact subgame resolution for sample-efficient reinforcement learning.**

A chess engine whose core innovation is a three-tier tactical MCTS that triages positions by complexity. Classically solved positions become **terminal nodes** -- they are never expanded further, so their exact values cannot be diluted by approximate child evaluations. Only genuinely uncertain positions invoke the neural network.

The architecture pattern -- injecting exact solutions for tractable subproblems as terminal MCTS nodes -- generalizes beyond chess to any domain with classically solvable subproblems (e.g., mathematical reasoning with automated theorem provers).

The name is a hybrid, like the engine: **Caissa** (the mythical goddess of chess) + **Cassowary** (a large, formidable, and famously aggressive bird).

![Caissawary Logo](Caissawary.png)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange)](https://rustup.rs/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Three-Tier MCTS

| Tier | Mechanism | Property | Mean cost |
|------|-----------|----------|-----------|
| **Tier 1** | Safety Gates (mate-in-5, KOTH-in-3) | Provably correct, exact values | 132-149 us |
| **Tier 2** | Iterative-widening quiescence search | PeSTO eval after captures, checks, forks | 8 us |
| **Tier 3** | Neural network (OracleNet) | Learned positional evaluation | 1,765 us |

Gate-resolved nodes are **terminal** -- identical to checkmate or stalemate -- so proven values propagate through the tree without dilution.

### PUCT Selection

$$U(s,a) = c \cdot P(s,a) \cdot \frac{\sqrt{\max(N(s),\,1)}}{1 + n(s,a)}$$

The denominator uses $\max(N,1)$ instead of vanilla AlphaZero's $N$. When a node is first visited ($N=0$), the standard formula collapses $U$ to zero for every child, so the first simulation is selected by move-generator order — a systematic bias toward structurally poor moves. With $\max(N,1)$, the exploration bonus is proportional to the policy prior $P(s,a)$ from the first visit, so the network's top recommendation is explored first. For $N \geq 1$ the formulas are identical; no retuning of $c$ is needed.

### Value Function

$$V_{final} = \tanh(V_{logit} + k \cdot \Delta M)$$

- $\Delta M$: PeSTO evaluation from Tier 2 Q-search (material + piece placement after tactical resolution)
- $V_{logit}$: NN positional assessment (unbounded). $\Delta M$ also feeds as a direct input feature.
- $k$: learned scalar for global material trust (initialized to 0.326)
- Classical fallback (no NN): $V_{logit}=0$, $k=0.326$, so $V_{final} = \tanh(0.326 \cdot \Delta M)$

## Tournament Results: Caissawary vs Vanilla MCTS

An 18-model round-robin tournament compared two training runs (both 6 blocks, 128 channels, ~2M params, 200 MCTS sims/move):
- **Caissawary** (9 models, 20 generations): all three tiers
- **Vanilla** (9 models, 29 generations): KOTH only, no tier 1, no material (pure AlphaZero-style)

![Elo vs Generation](tournament_results_800eval_elo_plot.png)

| Rank | Model | Elo | 95% CI | Type |
|------|-------|-----|--------|------|
| 1 | tiered_gen18 | 2663 | [2565, 2763] | Caissawary |
| 2 | tiered_gen12 | 2595 | [2502, 2692] | Caissawary |
| 3 | tiered_gen8 | 2542 | [2448, 2633] | Caissawary |
| 4 | tiered_gen6 | 2478 | [2391, 2567] | Caissawary |
| 5 | tiered_gen5 | 2347 | [2260, 2428] | Caissawary |
| 6 | tiered_gen4 | 2312 | [2223, 2395] | Caissawary |
| 7 | tiered_gen3 | 2228 | [2144, 2306] | Caissawary |
| 8 | tiered_gen1 | 2167 | [2082, 2242] | Caissawary |
| 9 | tiered_gen0 | 2068 | [1984, 2148] | Caissawary |
| 10 | vanilla_gen28 | 2038 | [1957, 2111] | Vanilla |
| 11 | vanilla_gen23 | 1996 | [1920, 2063] | Vanilla |
| 12 | vanilla_gen21 | 1866 | [1801, 1933] | Vanilla |
| 13 | vanilla_gen13 | 1833 | [1772, 1902] | Vanilla |
| 14 | vanilla_gen10 | 1794 | [1734, 1859] | Vanilla |
| 15 | vanilla_gen7 | 1726 | [1672, 1781] | Vanilla |
| 16 | vanilla_gen4 | 1639 | [1598, 1684] | Vanilla |
| 17 | vanilla_gen1 | 1600 | [1566, 1636] | Vanilla |
| 18 | vanilla_gen0 | 1500 | (anchor) | Vanilla |

**Key findings:**
- All Caissawary models outrank all vanilla models. Even tiered_gen0 (zero-initialized NN) exceeds vanilla_gen28, the best vanilla model after 29 generations of training.
- Caissawary gains +595 Elo over 18 generations; vanilla gains +538 Elo over 28 generations -- similar NN learning rates, but tiered starts ~570 Elo higher.
- The classical fallback alone ($V_{final} = \tanh(0.326 \cdot \Delta M)$) provides stronger play than 29 generations of pure NN training.

Elo via MLE on the Bradley-Terry model with 95% bootstrap CIs (1000 resamples). Full results: [`runs/tournaments/round_robin_800eval/round_robin_results.json`](runs/tournaments/round_robin_800eval/round_robin_results.json).

## GPU MCTS (CUDA)

A fully GPU-resident MCTS implementation in `cuda/`. The entire search loop -- tree traversal, node expansion, quick checks, quiescence search, and neural network inference -- runs inside a persistent CUDA kernel with no CPU interaction during search. No cuBLAS, no cuDNN, no host round-trips.

### Two Inference Architectures

Which is faster depends on model size:

| Architecture | Mechanism | Best for | Throughput |
|---|---|---|---|
| **GPU-resident MCTS** (CUDA) | Full MCTS loop on GPU, NN inline per-SM | Small models (<=12L D=128) | ~8,200 pos/sec (12L), ~130 samp/sec |
| **CPU MCTS + GPU batched** (Rust+LibTorch) | Rayon threads + InferenceServer batching | Large models (>=24L) | ~38 samp/sec (6L) |

**GPU-resident wins for small models:** Each SM runs select-expand-evaluate-backup with zero CPU-GPU transfers. At 12L D=128 (2.4M params), transfer overhead of batched inference exceeds the cost of computing inline on each SM.

**Batched wins for large models:** At 24+ layers (4.8M+ params), GPU forward pass exceeds 3.6ms per batch -- CPU tree operations can overlap. Larger models also exceed the ~99 KB shared memory limit for GPU-resident (D>=192 is impossible). See [CUDA_TC_ANALYSIS.md](CUDA_TC_ANALYSIS.md).

### Current Model: 12L Transformer, D=128, 2.4M params

- **Throughput:** ~8,200 positions/sec on RTX 5060 Ti (36 SMs, 200 sims/move)
- **Self-play:** ~130 samples/sec (36 concurrent games, zero-init shorter games)
- **Architecture:** Pre-LayerNorm encoder, 4 attention heads (head_dim=32), FFN 4x expansion
- **Tensor Cores:** FP16 `buf_out` frees 16 KB for TC staging; Q/K/V projections and FFN use wmma (FP16 in, FP32 accumulate); QK^T, softmax, attn*V stay scalar (workspace aliasing)

### SE-ResNet Also Supported

SE-ResNet 128x6 (~2M params) achieves 1.51 ms forward pass with shifted-copy TC (conv3x3 decomposed into 9 dense GEMMs). Better TC tiling than transformer due to regular structure. Per-move: ~19 ms at 36 blocks, 200 sims.

### GPU Self-Play and Evaluation

```bash
# Self-play: 36 concurrent games, 200 sims/move
cuda/build/selfplay /tmp/transformer_weights.bin 36 200 /tmp/selfplay_data/

# Two-network SPRT evaluation (colors alternated for fairness)
cuda/build/evaluate weights_a.bin weights_b.bin 36 200 /tmp/eval_data/
```

Both binaries produce training data compatible with `train.py`. Two-network eval partitions games by side-to-move, making two kernel calls per round -- both weight sets (~4.7 MB each) reside in GPU global memory simultaneously.

## Training Pipeline

AlphaZero-style loop: self-play -> replay buffer -> train -> export -> evaluate -> SPRT gate.

- **Optimizer:** Muon (lr=0.02) with adaptive epochs and early stopping
- **SPRT gating:** Up to 800 eval games with early stopping (~30 games for clear results)
- **Skip-self-play:** After gen 1, eval games produce training data, so self-play is optional
- **Elo-weighted buffer:** Positions weighted by model strength (200 Elo gap -> ~32% inclusion)
- **Data augmentation:** Horizontal flip (2x) for positions without castling; D4 dihedral (8x) for pawnless endgames

```bash
# Full training loop
python python/orchestrate.py --enable-koth \
  --sims-schedule "0:100,5:200,10:400,20:800"

# Ablations
python python/orchestrate.py --disable-tier1        # No safety gates
python python/orchestrate.py --disable-material      # Pure AlphaZero

# Skip self-play (eval games produce training data)
python python/orchestrate.py --skip-self-play

# Transformer architecture
python python/orchestrate.py --arch transformer --enable-koth
```

## Building and Running

```bash
# Prerequisites
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install torch numpy python-chess

# Build
cargo build --release                    # Tactical MCTS only
cargo build --release --features neural  # With LibTorch neural network support

# Run UCI engine
./target/release/caissawary

# Build CUDA kernel
cd cuda && mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 && make
```

## Testing

~790 tests (609 Rust + 157 CUDA + 24 Python). See [TESTING.md](TESTING.md).

```bash
cargo test                                        # Fast Rust tests (~50s)
cargo test --features slow-tests                  # Full suite including perft (~200s)
cd python && python -m pytest test_*.py -v        # Python pipeline tests
cuda/build/test_mcts_kernel                       # 13/13 CUDA MCTS tests
cuda/build/test_transformer                       # 11/11 transformer tests
cuda/build/test_block_ops                         # 15/15 SE-ResNet tests
cuda/build/test_nn_ops                            # 21/21 NN ops tests
```

<details>
<summary>Binary targets</summary>

| Binary | Description |
|--------|-------------|
| `caissawary` | Main UCI chess engine |
| `self_play` | Self-play data generation with SAN game logs |
| `evaluate_models` | Head-to-head model evaluation with SPRT |
| `round_robin` | Round-robin tournament with Elo estimation |
| `mcts_inspector` | MCTS search tree visualization (Graphviz DOT) |
| `verbose_search` | Real-time search narration |
| `verbose_game` | Full game between two classical MCTS agents |
| `benchmark` | Performance testing and NPS measurement |
| `strength_test` | Engine strength assessment |
| `run_experiments` | Ablation studies framework |
| `elo_tournament` | Elo rating estimation (classical engines) |
| `texel_tune` | Evaluation weight optimization |
| `generate_training_data` | Standalone training data generation |
| `play_engine` | Interactive human-vs-engine play |
| `profile_engine` | Per-operation MCTS timing profiler |
| `qsearch_gui` | Interactive Q-search tree visualizer (web UI at localhost:8088) |

</details>

## Further Reading

- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) -- Scientific journey: what was tried, what failed, and why
- [TESTING.md](TESTING.md) -- Test suite documentation and coverage
- [SPRT_GUIDE.md](SPRT_GUIDE.md) -- SPRT evaluation methodology
- [CUDA_TC_ANALYSIS.md](CUDA_TC_ANALYSIS.md) -- Tensor Core memory analysis

## License

MIT License. See [LICENSE](LICENSE) for details.
