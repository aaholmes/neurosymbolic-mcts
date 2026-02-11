# Caissawary

**Neurosymbolic MCTS with formal verification injection for sample-efficient reinforcement learning.**

Caissawary decomposes MCTS positions into tractable subgames solved exactly by classical methods and uncertain residuals evaluated by a neural network. When a subproblem is tractable, the engine *proves* the answer rather than learning it — injecting ground-truth values directly into the search tree. This reduces the sample complexity of self-play RL by reserving neural network queries for genuinely uncertain positions. The approach generalizes to any MCTS domain with tractable subproblems (theorem proving, program synthesis, robotics planning). Demonstrated here on chess and King of the Hill (KOTH).

The name is a hybrid, like the engine: **Caissa** (the mythical goddess of chess) + **Cassowary** (a large, formidable, and famously aggressive bird).

![Caissawary Logo](Caissawary.png)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange)](https://rustup.rs/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## The Approach: Subgame Decomposition for MCTS

| Tier | Mechanism | Property | Cost |
|------|-----------|----------|------|
| **Tier 1** | Safety Gates (mate search, KOTH geometry) | Provably correct, exact values | ~microseconds |
| **Tier 2** | MVV-LVA tactical move ordering | Prioritizes good captures on first visit | Zero (integer scores) |
| **Tier 3** | Neural network (OracleNet) | Learned evaluation for uncertain positions | ~milliseconds |

The key insight: gate-resolved nodes are **terminal** — they are never expanded, so proven values cannot be diluted by approximate child evaluations. This is the critical difference from using exact values as priors.

## How It Works

**Tier 1** runs ultra-fast safety gates before expansion: a checks-only mate search and KOTH geometric pruning. When a gate fires, the node receives an exact cached value and becomes terminal — identical to checkmate/stalemate.

**Tier 2** orders capture/promotion children by MVV-LVA scores on their first visit (e.g., PxQ = 39 is visited before QxP = 5). This is purely visit ordering — no Q-values are initialized. After the first visit, normal UCB selection takes over.

**Tier 3** evaluates leaf nodes with a neurosymbolic value function:

$$V_{final} = \tanh(V_{logit} + k \cdot \Delta M)$$

Where $V_{logit}$ is the NN's positional assessment (unbounded), $k$ is a learned material confidence scalar, and $\Delta M$ is the material balance after forced captures computed by `forced_material_balance()`. The NN also provides a policy prior over moves, cached on each node after evaluation and used for PUCT selection on subsequent visits. Without a neural network, the classical fallback uses $V_{logit}=0$, $k=0.5$: $V_{final} = \tanh(0.5 \cdot \Delta M)$ with uniform policy priors.

### OracleNet Architecture

OracleNet is a ~2M parameter SE-ResNet (6 blocks, 128 channels) with three heads:

- **Policy head:** 4672 logits (AlphaZero encoding)
- **Value head ($V_{logit}$):** Unbounded positional assessment
- **Confidence head ($k$):** Handcrafted features + 5x5 king patches

The $k$ head uses domain knowledge rather than learned convolutions: 8 scalar features (pawn counts, piece counts, queen presence, castling rights, king rank) plus two 5x5 spatial patches centered on each king, combined via small FC layers (~21.6k parameters). This lets $k$ reason about king safety and material distribution without needing to learn these patterns from scratch.

## Example: Material-Aware Evaluation at Initialization

After White plays 1.b4 (100 MCTS iterations, zero-initialized OracleNet with all logits at 0), Black's root children:

```
  +-----------------+
8 | r n b q k b n r |
7 | p p p p p p p p |
6 | . . . . . . . . |
5 | . . . . . . . . |
4 | . P . . . . . . |
3 | . . . . . . . . |
2 | P . P P P P P P |
1 | R N B Q K B N R |
  +-----------------+
    a b c d e f g h

Move       Visits   Q-value
e7e5           29    +0.239
e7e6           27    +0.274
b8a6            8    +0.289
b8c6            5    +0.185
...            ...      0.000
c7c5            1    -0.462
```

With zero training, the engine already plays intelligently. All four top moves (e5, e6, Na6, Nc6) attack White's hanging b4 pawn — preferring pawn advances over knight moves, since advancing opens a bishop line to b4 while knights on a6/c6 can be chased by b5. Meanwhile 1...c5 is correctly avoided (Q = -0.462) because bxc5 wins a pawn outright. This emerges from the zero-initialized network: with $V_{logit} = 0$ and $k = 0.5$, the value function reduces to $\tanh(0.5 \cdot \Delta M)$ — pure material-aware evaluation via quiescence search, with no training required.

## Training Pipeline

AlphaZero-style loop: self-play → replay buffer → train → export → evaluate → gate (SPRT).

Each generation trains three model variants in parallel and evaluates each via SPRT:
- **Policy-only:** freeze value + k heads, train only the policy head
- **Value-only:** freeze policy head, train only value + k heads
- **All-heads:** standard joint training of all parameters

The best passing variant (if any) is promoted. This isolates whether gains come from better move selection or better position evaluation, and avoids catastrophic forgetting where improving one head degrades another.

Evaluation games use greedy move selection (most-visited child) after the first 10 plies, with proportional sampling only in the opening for diversity. All three variants are evaluated with the same random seeds so game differences are attributable to the model, not randomness.

```bash
# Full training loop with KOTH, ramping sims from 100→800 over generations
python python/orchestrate.py --enable-koth \
  --sims-schedule "0:100,5:200,10:400,20:800"

# Ablation: disable Tier 1 safety gates
python python/orchestrate.py --disable-tier1

# Ablation: disable material-aware evaluation (pure AlphaZero)
python python/orchestrate.py --disable-material

# Quick smoke test
python python/orchestrate.py \
  --games-per-generation 2 --simulations-per-move 50 \
  --minibatches-per-gen 10 --eval-max-games 4 --buffer-capacity 1000
```

The orchestrator supports adaptive minibatch scaling (~1.5 epochs per generation), recency-weighted sampling from the replay buffer, and Muon optimizer by default.

Evaluation uses SPRT (Sequential Probability Ratio Test) with early stopping — clear winners/losers decided in ~30 games, marginal cases use up to the configured maximum (default 400).

## Building and Running

### Prerequisites

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust
pip install torch numpy python-chess                                # NN (optional)
```

### Build

```bash
cargo build --release                    # Tactical MCTS only
cargo build --release --features neural  # With neural network support
```

### Usage

```bash
./target/release/caissawary  # UCI engine — use with any chess GUI
```

## Testing

~800 tests (650 Rust + 150 Python). See [TESTING.md](TESTING.md) for details.

```bash
cargo test                                        # Fast Rust tests (~50s)
cargo test --features slow-tests                  # Full suite including perft (~200s)
cd python && python -m pytest test_*.py -v        # Python pipeline tests
```

## Binary Targets

| Binary | Description |
|--------|-------------|
| `caissawary` | Main UCI chess engine |
| `self_play` | Self-play data generation with SAN game logs |
| `evaluate_models` | Head-to-head model evaluation with SPRT |
| `mcts_inspector` | MCTS search tree visualization (Graphviz DOT) |
| `verbose_search` | Real-time search narration |
| `verbose_game` | Full game between two classical MCTS agents |
| `benchmark` | Performance testing and NPS measurement |
| `strength_test` | Engine strength assessment |
| `run_experiments` | Ablation studies framework |
| `elo_tournament` | Elo rating estimation |
| `texel_tune` | Evaluation weight optimization |
| `generate_training_data` | Standalone training data generation |

## Further Reading

- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) — Scientific journey: what was tried, what failed, and why
- [TESTING.md](TESTING.md) — Test suite documentation and coverage
- [SPRT_GUIDE.md](SPRT_GUIDE.md) — SPRT evaluation methodology

## References

- Silver, D. et al. (2017). ["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"](https://arxiv.org/abs/1712.01815)
- Tian, Y. et al. (2019). ["ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero"](https://arxiv.org/abs/1902.04522)
- Nasu, Y. (2018). ["Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi"](https://www.apply.computer-shogi.org/wcsc28/appeal/the_end_of_genesis_T.N.K.evolution_turbo_type_D/nnue.pdf)
- [Leela Chess Zero](https://lczero.org/) — Open-source neural network chess engine
- [Stockfish](https://stockfishchess.org/) — State-of-the-art classical + NNUE engine
- [Berserk](https://github.com/jhonnold/berserk) — Strong open-source engine with NNUE
- [Chess Programming Wiki](https://www.chessprogramming.org/) — Encyclopedic reference for chess engine techniques

## License

MIT License. See [LICENSE](LICENSE) for details.
