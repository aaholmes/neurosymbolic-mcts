# Hyperparameters Reference

All tunable parameters for the Caissawary training pipeline, organized by stage.

## Model Architecture

| Parameter | Value | CLI Flag | Notes |
|-----------|-------|----------|-------|
| Residual blocks | 6 | `--num-blocks` | SE-ResNet blocks |
| Hidden dim | 128 | `--hidden-dim` | ~2M params total |
| Input channels | 17 | -- | 12 piece planes + 4 castling + 1 side-to-move |
| Policy planes | 73 | -- | AlphaZero encoding (4672 = 73x8x8) |
| SE reduction | 16 | -- | Squeeze-and-excitation ratio |
| Value head FC | 64 -> 256 -> 1 | -- | 1x1 conv then 2-layer MLP |
| K head scalars | 12 | -- | Handcrafted features (pawns, pieces, castling, etc.) |
| K head patches | 5x5 | -- | Window around each king, FC(300->32) |
| K head combine | 76 -> 32 -> 1 | -- | [12 scalars \| 32 STM patch \| 32 opp patch] |
| K scale | softplus(x) / (2 ln 2) | -- | Ensures k(init) = 0.5 |

## MCTS Search

| Parameter | Value | CLI Flag | Notes |
|-----------|-------|----------|-------|
| Simulations | 800 | `--simulations-per-move` | Per-move budget (self-play and eval) |
| Exploration (c) | 1.414 | -- | PUCT constant (sqrt(2)) |
| Mate search depth | 5 | -- | Checks-only; disabled if `--disable-tier1` |
| Mate search nodes | 100,000 | -- | Node budget per search |
| Q-search depth | 8 | -- | For `forced_material_balance()` |
| Time limit | 120s | -- | Per-move hard cutoff (eval games) |
| Dirichlet alpha | 0.0 | -- | Disabled; proportional sampling suffices |
| Dirichlet epsilon | 0.0 | -- | Disabled |
| UCB formula | Q + c * P * sqrt(N_parent) / (1 + N_child) | -- | PUCT with NN prior P |
| Unvisited Q | 0.0 | -- | Default Q for unexplored children |

## Move Selection (Evaluation Games)

| Parameter | Value | CLI Flag | Notes |
|-----------|-------|----------|-------|
| Top-p base | 0.95 | `--eval-top-p-base` / `--top-p-base` | p = base^(move_number - 1) |
| Move 1 p | 1.0 | -- | Full sampling |
| Move 11 p | ~0.60 | -- | Moderate nucleus |
| Move 31 p | ~0.21 | -- | Nearly greedy |
| Distribution | visits - 1 | -- | Proportional to adjusted visit counts |
| Forced wins | Deterministic | -- | Terminal/mate children picked greedily |

## Self-Play

| Parameter | Value | CLI Flag | Notes |
|-----------|-------|----------|-------|
| Games per generation | 100 | `--games-per-generation` | |
| Sims schedule | "" (none) | `--sims-schedule` | e.g. `"0:200,10:400,20:800"` |
| Move randomization | On | -- | Children shuffled after expansion |
| Game logging | "first" | `--log-games` | `all`, `first`, or `none` |
| Seed offset | gen * games | -- | Deterministic per-generation seeds |
| Max game length | 200 moves | -- | Draw if exceeded |
| Draw rules | Repetition, 50-move | -- | Standard chess draws |

## Training

| Parameter | Value | CLI Flag | Notes |
|-----------|-------|----------|-------|
| Optimizer | Muon | `--optimizer` | `adam`, `adamw`, `muon` |
| Learning rate | 0.02 | `--initial-lr` | Muon default; Adam default is 0.001 |
| Muon backend LR | lr * 0.1 | -- | For non-Muon parameters |
| AdamW weight decay | 1e-4 | -- | |
| Max epochs | 10 | `--max-epochs` | Early stopping with patience=1 |
| Validation split | 10% | -- | 90/10 train/val |
| Batch size | 64 | `--batch-size` | |
| LR schedule | "" (none) | `--lr-schedule` | e.g. `"500:0.01,1000:0.001"` |
| Policy loss | KL divergence | -- | `F.kl_div(log_softmax, target)` |
| Value loss | MSE | -- | `F.mse_loss(v, target)` |
| Loss weights | 1:1 | -- | Policy + Value equally weighted |
| Augmentation | On | `--no-augment` | Board symmetry (horizontal flip) |
| Train heads | "all" | `--train-heads` | `all`, `policy`, `value` |

## Replay Buffer

| Parameter | Value | CLI Flag | Notes |
|-----------|-------|----------|-------|
| Capacity | 100,000 | `--buffer-capacity` | Positions; oldest evicted FIFO |
| Sampling weight | count * 2 * E(score) | -- | Elo-based: E = 1/(1+10^((max_elo-elo)/400)) |
| Clear on accept | Yes | -- | Buffer cleared when model accepted |
| Eval data ingestion | Both sides | -- | Winner at current Elo, loser at lower Elo |

## Evaluation (SPRT)

| Parameter | Value | CLI Flag | Notes |
|-----------|-------|----------|-------|
| Max games | 400 | `--eval-max-games` | Per evaluation round |
| Eval simulations | 800 | `--eval-simulations` | Can follow sims schedule |
| SPRT elo0 | 0.0 | `--sprt-elo0` | Null hypothesis |
| SPRT elo1 | 10.0 | `--sprt-elo1` | Alternative hypothesis |
| SPRT alpha | 0.05 | `--sprt-alpha` | Type I error rate |
| SPRT beta | 0.05 | `--sprt-beta` | Type II error rate |
| Training data | Saved | `--no-save-training-data` | Both sides' data saved for buffer |

## Parallelism

| Parameter | Value | CLI Flag | Notes |
|-----------|-------|----------|-------|
| Inference batch size | 16 | `--inference-batch-size` | GPU batching for NN eval |
| Game threads | Auto | `--game-threads` | 0 = RAYON_NUM_THREADS default |
| RAYON_NUM_THREADS | cpu_count/2 - 1 | -- | Set by orchestrator |

## Feature Toggles

| Flag | Default | CLI Flag | Effect |
|------|---------|----------|--------|
| KOTH mode | Off | `--enable-koth` | King of the Hill win detection |
| Tier 1 gates | On | `--disable-tier1` | Mate search + KOTH geometric pruning |
| Material value | On | `--disable-material` | V = tanh(v_logit + k*dM) vs V = tanh(v_logit) |
| Skip self-play | Off | `--skip-self-play` | Skip after gen 1; eval games provide data |
| Multi-variant | Off | `--multi-variant` | Train policy/value/all vs just all |
| LTO | On | -- | `lto = "thin"`, `codegen-units = 1` |
