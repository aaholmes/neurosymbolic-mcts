# Architecture: Current Design

Concise summary of what the system does and why. For the historical evolution, see `DESIGN_DECISIONS.md`. For parameter values, see `HYPERPARAMETERS.md`.

## Core Idea

Decompose MCTS evaluation into three tiers: **compute what you can exactly, order what you understand heuristically, learn what remains.** Only genuinely uncertain positions use the neural network.

## Three-Tier MCTS

**Tier 1 — Safety Gates (exact).** Mate search (checks-only for mate-in-1/2/3) and KOTH geometric pruning (can king reach center in 3?). Resolved nodes become **terminal** — never expanded, value is exact. Terminal semantics prevent proven values from being diluted by approximate child evaluations. An `exhaustive_mate_depth` parameter (default 0) can enable exhaustive search at shallower depths to catch quiet-first forced mates, but checks-only is sufficient in practice.

**Tier 2 — Quiescence Search (classical tree search).** Every MCTS leaf evaluation runs `forced_material_balance()`: a material-only alpha-beta Q-search (depth 20) that resolves all forced captures and promotions. This produces deltaM — the true material balance after tactical dust settles. A position may look equal (P=P) but after forced exchanges be +3. No neural network can easily replicate this because it requires a classical tree search over variable-depth exchange sequences. deltaM feeds directly into both the value function additive path (`k * deltaM`) and as an input feature to the value head's FC layer, giving the NN position-dependent control over material trust. Q-search runs before NN inference so the result is available as a model input. Additionally, captures are visited in MVV-LVA order on their first visit (PxQ before QxP), though this is a minor optimization.

**Tier 3 — Neural Network (learned).** OracleNet provides policy (move probabilities) and V_logit (positional assessment). The NN only needs to learn what the Q-search can't compute: piece activity, king safety, pawn structure, and how much to trust material (k).

## Value Function

The value function is where Tiers 2 and 3 combine:

```
V_final = tanh(V_logit + k * deltaM)
```

- **deltaM** (Tier 2, computed): material balance after forced exchanges — the output of the Q-search tree search, run at every leaf
- **V_logit** (Tier 3, NN, unbounded): positional assessment — everything the Q-search doesn't capture
- **k** (Tier 3, NN, positive): learned scalar that sets a global baseline trust level in material

The Q-search result is the foundation: it provides information that requires a classical tree search to compute (hanging pieces, exchange sequences, forced promotions). The NN adds a positional residual through V_logit; position-dependent modulation of material trust is learned through the value head FC, which receives deltaM as a direct input feature. The scalar k provides a global operating point for material trust. This factorization avoids the NN having to rediscover piece values and tactical exchange outcomes from scratch. The classical fallback (no NN) uses V_logit=0, k=0.5: pure material evaluation from the Q-search.

## OracleNet

SE-ResNet backbone (default: 6 blocks, 128 channels, ~2M params). 17x8x8 input in side-to-move perspective.

Three outputs:
- **Policy**: 73-plane AlphaZero encoding (4672 logits). Zero-initialized → uniform prior at init.
- **Value**: 1x1 conv → flatten(64) → concat(q_result) → FC(65→256→1) producing unbounded V_logit. The Q-search result (deltaM) is fed as a direct input feature, giving the value head position-dependent control over material trust. Zero-initialized → V_logit=0 at init.
- **k**: Single learned scalar (`nn.Parameter`). Output: softplus(k_logit)/(2 ln 2), so k=0.5 at init. Sets a global baseline trust level in material; position-dependent modulation is absorbed by the value head FC.

A freshly initialized network produces meaningful behavior: uniform exploration with material-aware evaluation.

## Training Pipeline

**Loop**: self-play → buffer → train → export → evaluate (SPRT) → accept/reject.

**Self-play**: Games against self using current best model. Proportional move selection from MCTS visit counts. Training targets: visit-count policy distribution + game outcome.

**Replay buffer**: Sliding window (100K positions, FIFO eviction). Elo-based strength weighting: data from stronger models sampled more. Buffer accumulates across acceptances — Elo weighting and early stopping handle data staleness without clearing. Eval game data from both sides ingested after each evaluation.

**Training**: Muon optimizer (lr=0.02). Up to 10 epochs with validation-based early stopping (patience=1, 90/10 split). Policy loss: KL divergence. Value loss: MSE. Equal weight. Always trains from the latest candidate (accepted or rejected), so rejected candidates' incremental learning is preserved.

**Evaluation**: SPRT with elo0=0, elo1=10, alpha=beta=0.05. Up to 800 games. Proportional-or-greedy move selection: with probability p=0.90^(move-1) sample proportionally from visit counts, otherwise play greedy (most-visited). This decays from full proportional sampling on move 1 to ~88% greedy by move 20. Forced wins always played deterministically.

**Eval-only mode** (`--skip-self-play`): After gen 1, eval games (up to 800/gen) provide training data at zero marginal cost, replacing self-play. Cuts wall time ~50%.

**Data augmentation**: Horizontal flip when no castling rights (2x). Full D4 dihedral for pawnless+no-castling positions (8x). Both board and policy vector transformed consistently.

## KOTH Variant

King of the Hill: moving your king to {d4, e4, d5, e5} also wins. This stresses Tier 1 gates far more than standard chess — forced king-march threats and checkmate threats coexist in every midgame position.

## Key Implementation Details

- **Incremental Zobrist hashing**: O(1) per move via XOR updates
- **Clone-free legality**: `is_legal_after_move()` applies/undoes in-place
- **Pseudo-legal validation**: `is_pseudo_legal()` validates TT moves without full move generation
- **Lightweight SEE**: `SeeBoard` struct avoids full Board clone for static exchange evaluation
- **LTO enabled**: thin LTO + codegen-units=1 for 5-15% release speedup

## Domain Generality

The three-tier pattern applies wherever MCTS encounters tractable subproblems: mathematical reasoning (theorem provers as Tier 1), program synthesis (type checkers/SMT solvers as Tier 1), any game with solved subgames. The key requirement: proven nodes must be **terminal** in the MCTS tree. The value factorization (V_learned + k * V_computed) transfers to any domain where part of the evaluation is computable.
