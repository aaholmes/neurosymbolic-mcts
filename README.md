# Caissawary

**Neurosymbolic MCTS with formal verification injection for sample-efficient reinforcement learning.**

Caissawary decomposes MCTS positions into tractable subgames solved exactly by classical methods and uncertain residuals evaluated by a neural network. When a subproblem is tractable, the engine *proves* the answer rather than learning it — injecting ground-truth values directly into the search tree. This reduces the sample complexity of self-play RL by reserving neural network queries for genuinely uncertain positions. The approach generalizes to any MCTS domain with tractable subproblems (theorem proving, program synthesis, robotics planning). Demonstrated here on chess and King of the Hill (KOTH).

![Caissawary Logo](Caissawary.png)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange)](https://rustup.rs/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## The Approach: Subgame Decomposition for MCTS

| Tier | Mechanism | Property | Cost |
|------|-----------|----------|------|
| **Tier 1** | Safety Gates (mate search, KOTH geometry) | Provably correct, exact values | ~microseconds |
| **Tier 2** | MVV-LVA Q-init for captures | Heuristic ordering via UCB | Zero (integer scores) |
| **Tier 3** | Neural network (OracleNet) | Learned evaluation for uncertain positions | ~milliseconds |

The key insight: gate-resolved nodes are **terminal** — they are never expanded, so proven values cannot be diluted by approximate child evaluations. This is the critical difference from using exact values as priors.

## How It Works

**Tier 1** runs ultra-fast safety gates before expansion: a checks-only mate search and KOTH geometric pruning. When a gate fires, the node receives an exact cached value and becomes terminal — identical to checkmate/stalemate.

**Tier 2** assigns MVV-LVA scores to capture/promotion children at expansion time (e.g., PxQ = 39, QxP = 5, normalized to [-1, 1]). These serve as Q-init values for UCB selection, prioritizing good captures with zero computational overhead.

**Tier 3** evaluates leaf nodes with a neurosymbolic value function:

$$V_{final} = \tanh(V_{logit} + k \cdot \Delta M)$$

Where $V_{logit}$ is the NN's positional assessment (unbounded), $k$ is a learned material confidence scalar, and $\Delta M$ is the material balance after forced captures computed by `forced_material_balance()`. Without a neural network, the classical fallback uses $V_{logit}=0$, $k=0.5$: $V_{final} = \tanh(0.5 \cdot \Delta M)$.

## Example: Material-Aware Evaluation (No Neural Network)

After White plays 1.b4 (100 MCTS iterations, classical fallback only), Black's root children:

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

With zero training, the engine already plays intelligently. All four top moves (e5, e6, Na6, Nc6) attack White's hanging b4 pawn — preferring pawn advances over knight moves, since advancing opens a bishop line to b4 while knights on a6/c6 can be chased by b5. Meanwhile 1...c5 is correctly avoided (Q = -0.462) because bxc5 wins a pawn outright. This emerges entirely from the material-aware quiescence search in the value function — $\tanh(0.5 \cdot \Delta M)$ with no neural network.

## Training Pipeline

AlphaZero-style loop: self-play → replay buffer → train → export → evaluate → gate (SPRT).

```bash
# Full training loop (default settings)
python python/orchestrate.py --enable-koth

# Ablation: disable Tier 1 safety gates
python python/orchestrate.py --disable-tier1

# Ablation: disable material-aware evaluation (pure AlphaZero)
python python/orchestrate.py --disable-material

# Quick smoke test
python python/orchestrate.py \
  --games-per-generation 2 --simulations-per-move 50 \
  --minibatches-per-gen 10 --eval-max-games 4 --buffer-capacity 1000
```

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
| `self_play` | Self-play data generation for training |
| `evaluate_models` | Head-to-head model evaluation with SPRT |
| `mcts_inspector` | MCTS search tree visualization (Graphviz DOT) |
| `verbose_search` | Real-time search narration |
| `verbose_game` | Full game between two classical MCTS agents |
| `benchmark` | Performance testing and NPS measurement |
| `run_experiments` | Ablation studies framework |
| `elo_tournament` | Elo rating estimation |
| `texel_tune` | Evaluation weight optimization |

## Further Reading

- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) — Scientific journey: what was tried, what failed, and why
- [TESTING.md](TESTING.md) — Test suite documentation and coverage
- [SPRT_GUIDE.md](SPRT_GUIDE.md) — SPRT evaluation methodology

## References

- Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- Campbell, M. et al. (2002). "Deep Blue"
- The Stockfish Engine and the NNUE architecture

## License

MIT License. See [LICENSE](LICENSE) for details.
