# Design Decisions: A Scientific Journey

This document traces the evolution of Caissawary's architecture — what was tried, what failed, and why the current design emerged. The core pattern — exact subgame resolution injected as terminal MCTS nodes — applies to any domain with tractable subproblems.

### Why King of the Hill?

Standard chess is a surprisingly poor testbed for Tier 1 safety gates. Forced checkmates are rare in typical play — most games are decided by slow accumulation of positional and material advantages. In King of the Hill (KOTH), where moving your king to a central square also wins, the threat of forced wins is *always in the air*. Every midgame position has the dual tension of checkmate threats and king-march threats, both of which are tractable subgames solvable by Tier 1 gates. This makes KOTH an ideal stress test: the three-tier system gets far more opportunities to prove its value compared to standard chess, where Tier 1 would rarely fire.

## 1. Three-Tier MCTS: Why Not Pure AlphaZero?

**The problem.** AlphaZero treats every position uniformly: expand, evaluate with the neural network, backpropagate. But many positions have known answers. A mate-in-2 doesn't need a neural network — it needs a proof. A king that can trivially walk to the center in KOTH doesn't need 800 simulations — it needs geometry. Pure AlphaZero wastes its most expensive resource (NN calls) on positions where a microsecond of classical analysis gives an exact answer.

**The hypothesis.** Decompose positions into three tiers: (1) tractable subgames with exact solutions, (2) positions with useful heuristic structure, and (3) genuinely uncertain positions requiring learned evaluation. Only Tier 3 needs the neural network.

**Domain-general framing.** This decomposition applies wherever MCTS encounters tractable subproblems. The most natural next domain is mathematical reasoning: automated theorem provers (Lean's `decide`, `omega`, `norm_num`) can resolve certain subgoals exactly, while a neural policy guides the high-level proof search through uncertain creative steps. The confidence scalar $k$ maps directly: high when the subgoal is prover-friendly, low when it requires creative insight.

The pattern is always the same: compute what you can exactly, heuristically order what you understand, learn what remains.

## 2. Tier 1: Safety Gates as Terminal Nodes

### First attempt: gate values as priors

The initial design ran mate search at expansion, then used the result as a value prior for the node. The node was still expanded normally — children were created and evaluated by the NN or classical eval.

**What went wrong.** MCTS diluted the proven values. If a node was proven to be mate-in-2 (value = +1.0), but it got expanded, subsequent visits would descend into children evaluated by approximate methods. Those children might return values like +0.3 or -0.1 (the NN hadn't learned this pattern yet). The exact +1.0 from the mate search would get averaged with dozens of approximate values, converging toward something much weaker.

### Solution: terminal semantics

Gate-resolved nodes became **terminal** — treated identically to checkmate and stalemate. No children are ever created. Every future visit simply re-uses the cached exact value. This preserves the proof throughout the entire search.

**Key insight:** Proven values must be *cached*, not *mixed with approximations*. The terminal semantics are what make the proof sticky — once an MCTS node is proven, it stays proven forever.

### Checks-only mate search

Full-width mate search is too expensive to run at every MCTS expansion. The compromise: search only checking moves. This catches the vast majority of short forced mates (which are always delivered through checks) while keeping the cost to microseconds. The depth limit is configurable (default: 3 plies of checks).

### KOTH geometric pruning

In King of the Hill, a king that can reach {d4, e4, d5, e5} wins. The gate computes: can the side-to-move's king reach any center square in at most 3 moves, considering blocking pieces and opponent interception? This is pure geometry — no search needed.

## 3. Tier 2: From Grafting to MVV-LVA Visit Ordering

### v1: Q-search grafting at expansion

The first approach was ambitious: at every MCTS expansion, run a quiescence search (Q-search) rooted at the expanded position. The Q-search tree was then "grafted" onto the MCTS tree — Q-search leaf values became initial values for MCTS children.

**What went wrong.** Q-search at every expansion was expensive (~10x slower expansions), and the grafted values were noisy — Q-search with alpha-beta in a random MCTS leaf often had poor alpha/beta bounds. The complexity was high (converting Q-search nodes to MCTS nodes, handling transpositions between the two trees) and the benefit was marginal over simpler approaches.

### v2: MVV-LVA visit ordering

The simple alternative: at expansion, score each capture/promotion child with `10*victim - attacker` (Most Valuable Victim - Least Valuable Attacker). On their first visit, tactical children are visited in MVV-LVA order — PxQ (score = 39) before QxP (score = 5). No Q-values are initialized; after the first visit, normal UCB selection takes over based on the actual backpropagated values.

**The lesson:** The material dynamics are fully handled by the Tier 3 leaf value function ($\tanh(V_{logit} + k \cdot \Delta M)$), which runs `forced_material_balance()` — a material-only Q-search at evaluation time. Tier 2 only needs to control the *order* of first visits to captures. A simple heuristic suffices.

## 4. The Value Function: V = tanh(V\_logit + k * DeltaM)

### The insight: separate what's learnable from what's computable

Standard AlphaZero learns everything end-to-end: the NN must discover that material matters, learn piece values, and learn to evaluate positions — all from game outcomes. This works, but it's sample-inefficient. The NN spends millions of games rediscovering that a queen is worth more than a pawn.

**The key idea:** Factor the evaluation into a *computable* component (material after forced exchanges) and a *learnable* component (positional factors the NN must discover).

$$V_{final} = \tanh(V_{logit} + k \cdot \Delta M)$$

- $V_{logit}$ (NN output, unbounded): positional assessment only — piece activity, king safety, pawn structure
- $\Delta M$ (computed exactly): material balance after forced captures/promotions via `forced_material_balance()`
- $k$ (NN output, positive): how much material should matter in this position

### Why forced\_material\_balance(), not simple piece counting

A position might have equal material (P=P) but after forced exchanges end up +3 (winning a piece). Simple piece counting misses hanging pieces, discovered attacks, and forced exchange sequences. `forced_material_balance()` runs a material-only quiescence search — no positional terms, just piece values — to resolve all forced captures. This gives $\Delta M$ the "true" material balance.

### Why the NN returns V\_logit (unbounded), not tanh(V\_logit)

If the NN returned a bounded value in [-1, 1], adding $k \cdot \Delta M$ would saturate quickly. With unbounded $V_{logit}$, the NN can learn arbitrary positional offsets that interact smoothly with the material term. The final $\tanh$ constrains the output at the end.

### Dynamic k: learned confidence in material

Not all positions are equally material-sensitive. In a closed Sicilian with locked pawns, material imbalances matter less than in an open position. The network learns $k$ per position: high $k$ in open, tactical positions; low $k$ in closed, strategic ones.

$k$ is computed as $\text{Softplus}(K_{net}) / (2\ln 2)$ where $K_{net}$ is a raw network output. At initialization ($K_{net} = 0$): $k = \ln(2) / (2\ln 2) = 0.5$, giving a reasonable material weight from the start.

### k architecture: handcrafted features, not learned convolutions

The first implementation computed $k$ from the backbone's penultimate features (shared with the value head). This caused $k$ to overfit to specific positions — the deep features encoded too much position-specific detail.

The second design used a **separate shallow input network** for $k$: a single 3x3 conv → BN → global average pool → two linear layers. This was too shallow — one 3x3 conv + global average pool is a "bag of local patterns" that knows what local patterns exist but not where they are. It cannot reliably detect king safety (diluted to 1/64 of signal), pawn structure (multi-hop), or open files (long-range).

The current design uses **handcrafted scalar features + king patches**, consistent with the project philosophy of "compute what you can, learn what you must":

**Scalar features (12 values):** total pawns (open vs closed), STM/opponent non-pawn piece counts (endgame detection), STM/opponent queen presence (tactical complexity), pawn contacts (direct closedness measure), castling rights count (game phase), STM king rank (exposed vs castled), plus 4 bishop square-color features (light-squared and dark-squared bishop presence for each side). The bishop features enable $k$ to detect opposite-colored bishop endgames (lower $k$ — material harder to convert) and bishop pair advantage (higher $k$). These are extracted directly from the input tensor with simple sums and a precomputed checkerboard mask — no learned parameters.

**King patches (2 × 32 features):** 5×5 windows of all 12 piece planes centered on each king, padded for edge kings. Each 300-dim patch is compressed by a separate FC(300→32) + ReLU — separate weights for STM and opponent kings since they have different semantics (king safety vs. attack potential). This captures local piece configurations around each king without global average pooling's dilution.

**Combination:** `[12 scalars | 32 STM patch features | 32 opp patch features]` → FC(76→32) → ReLU → FC(32→1) → k_logit. Only the final FC(32→1) is zero-initialized; patch FCs and combine layer use standard He init. Total: ~21.8k parameters (tiny vs ~2M model total).

### Classical fallback: V\_logit=0, k=0.5

With no neural network, the engine uses $V_{logit} = 0$ (no positional knowledge) and $k = 0.5$ (moderate material weight), giving $V_{final} = \tanh(0.5 \cdot \Delta M)$. This matches the NN's initialization — a freshly initialized network produces identical values to the classical fallback, ensuring smooth bootstrapping.

## 5. OracleNet Architecture

### SE-ResNet backbone

Configurable depth and width (default: 6 residual blocks, 128 channels, ~2M params; `--num-blocks 2 --hidden-dim 64` gives ~240K params for faster iteration). Squeeze-and-Excitation (SE) attention in each block lets the network learn which feature channels are important for each position — essentially a per-position "volume knob" for different types of features (material patterns, king safety features, pawn structure features).

### Board encoding: STM perspective

17x8x8 input tensor in side-to-move (STM) perspective:
- Planes 0-5: STM pieces (P, N, B, R, Q, K)
- Planes 6-11: opponent pieces
- Plane 12: en passant
- Planes 13-16: castling rights (STM-relative)

When Black is to move, ranks are flipped so STM pieces always appear at the "bottom." This halves the effective state space — the network only needs to learn from one perspective.

### Policy head: 4672-move AlphaZero encoding

73 planes x 64 squares = 4672 logits. Planes encode direction (8 queen directions x 7 distances = 56) + 8 knight moves + 9 underpromotions.

### Value head: symbolic residual design

1x1 conv → BN → flatten (64 features) → FC(64→256) → FC(256→1). The output is $V_{logit}$ (unbounded). This feeds into the symbolic residual formula with the independently-computed $k$ and $\Delta M$.

### k head: handcrafted features + king patches

12 scalar features (pawn count, piece counts, queen presence, pawn contacts, castling rights, king rank, bishop square-color presence) + two 5×5 king-centered patches compressed via FC(300→32). Scalars + compressed patches → FC(76→32) → FC(32→1). Operates on raw input, independent of the backbone. See Section 4 for the rationale.

### Zero initialization

All three output layers ($V_{out}$, $K_{out}$, policy head) are initialized to zero:
- **Policy:** $\text{softmax}(0, 0, ...) = $ uniform distribution (explore everything equally)
- **Value:** $V_{logit} = 0$, so $V_{final} = \tanh(k \cdot \Delta M)$ (pure material evaluation)
- **k:** $K_{net} = 0 \Rightarrow k = 0.5$ (moderate material weight)

This means a freshly initialized network immediately produces meaningful behavior: uniform exploration with material-aware evaluation.

## 6. Training Pipeline Evolution

### Gating: from fixed threshold to SPRT

**v1: Fixed threshold (55% winrate).** Simple but noisy — a 55% result in 100 games is barely significant. Promising but weak improvements were rejected, while lucky noise was occasionally accepted.

**v2: One-sided binomial test (p < 0.05).** Better statistical rigor but still problematic with small sample sizes. A candidate winning 6/10 games would be rejected, while one winning 30/50 might be accepted despite similar effect sizes.

**v3: SPRT early stopping (current).** Sequential Probability Ratio Test with trinomial GSPRT matching fishtest's implementation. Tests "candidate is ≥ elo1 Elo stronger" vs. "candidate is ≤ elo0 Elo stronger." Clear improvements terminate in ~30 games; marginal cases use up to 400. This saves significant wall-clock time while maintaining statistical rigor.

**Why 400 eval games.** With 100 eval games and a high draw rate (~84% at 100 sims), even a WR of 0.575 yields p ≈ 0.067 — not enough to reject the null hypothesis. A one-sided binomial test needs ~275 games for 80% power to detect a 53% true WR. The default of 400 games provides comfortable statistical power while SPRT early stopping keeps the average cost much lower for clear-cut results.

### Replay buffer: from clear-on-accept to sliding window

**v1: Clear buffer on model acceptance.** Every time a new model was accepted, the entire replay buffer was cleared. Rationale: old data was generated by a weaker model. Problem: the first training iteration after acceptance had very little data, causing severe overfitting.

**v2: Sliding window.** FIFO eviction — oldest games are removed when capacity is exceeded, regardless of model acceptance. The buffer always contains a mix of recent and older data.

**v3: Recency-weighted sampling.** Sliding window with exponential half-life decay across model generations. Recent games sampled more frequently, old games still contribute. Problem: the half-life parameter was arbitrary and disconnected from actual model strength — a 55% winrate gap (barely detectable) produced the same 7:1 sampling ratio as a massive strength improvement if the same number of positions separated them.

**v4: Elo-based strength weighting (current).** Each model acceptance chains the SPRT winrate into a cumulative Elo rating: $\Delta_{Elo} = -400 \cdot \log_{10}(1/WR - 1)$. Training data is weighted by expected score against the current best model:

$$w_i = n_i \cdot 2 \cdot \frac{1}{1 + 10^{(Elo_{max} - Elo_i) / 400}}$$

This produces sampling ratios proportional to actual measured strength differences. A 55% winrate gap (~35 Elo) yields a ~1.2:1 ratio (not 7:1). A 200 Elo gap yields ~0.48x weight. No data is ever fully discarded — even data from much weaker models contributes at a reduced rate. The weighting adapts automatically: rapid improvement produces steeper gradients, while plateaus produce near-uniform sampling.

### Training data scheduling: from fixed minibatches to epoch-based inclusion

**v1: Fixed minibatch count.** Every generation trained for the same number of minibatches regardless of buffer size. This caused overfitting in early generations (small buffer, many passes over same data) and underfitting in late generations (large buffer, most positions never seen).

**v2: Adaptive minibatch count.** Scaled minibatches to target ~1.5 epochs over the buffer, capped at a configurable maximum. Better, but still used random sampling with replacement — positions could be sampled multiple times while others were never seen. The sampling weights (Elo-based expected score) also didn't prevent high-Elo positions from being trained on repeatedly.

**v3: Epoch-based training with Elo-weighted inclusion (current).** Each epoch iterates over the buffer exactly once, but each position's *inclusion probability* is determined by its model's Elo:

$$p_{include} = \min\left(1, \frac{E_i}{1 - E_i}\right) \quad \text{where} \quad E_i = \frac{1}{1 + 10^{(\text{Elo}_{max} - \text{Elo}_i) / 400}}$$

This is the odds ratio of the expected score — the probability of winning divided by the probability of losing. Max-Elo data has 100% inclusion. At 100 Elo weaker: 56%. At 200 Elo: 32%. At 400 Elo: 10%. The number of epochs per generation is configurable via `--max-epochs` (default: 10), with validation-based early stopping selecting the optimal count automatically.

**Why odds ratio, not expected score.** Expected score ranges from 0.5 (equal) to ~0 (much weaker), which would include only half of max-Elo data. The odds ratio maps equal strength to 1.0 (full inclusion) and decays smoothly to 0 for very weak data, giving the desired semantics: "include this position with probability proportional to how much we trust the model that generated it."

**Why this is better than weighted sampling.** With weighted random sampling, a position from the strongest model might be sampled 3 times while a position from a slightly weaker model is never seen — pure randomness. Epoch-based inclusion guarantees that every max-Elo position is trained on exactly once per epoch, while older data is deterministically downsampled. This eliminates both wasted data (positions evicted before being trained on) and redundant training (positions sampled multiple times in one pass).

**How many epochs?** The optimal number depends on the ratio of model capacity to buffer size. AlphaZero and ELF OpenGo train on thousands of GPUs generating data far faster than they consume it (ELF reports a 13:1 self-play-to-training ratio), so each position is rarely seen twice. Running on a single GPU, we need to extract maximum signal from every position — multiple epochs are the natural lever.

Early experiments with the 2M parameter model at 1 epoch showed rapid initial gains (+67 Elo in 2 generations) followed by 4 consecutive rejections — the model had capacity to learn more but wasn't getting enough gradient updates per generation. Switching to 2 epochs broke through the plateau: the 2-epoch run reached +88 Elo in 3 generations where the 1-epoch run stalled at +67. The risk of overfitting increases with more epochs, but with 100K+ samples and a 2M parameter model that's clearly underfitting, 2 epochs is well within the safe regime.

### Adaptive epochs: validation-based early stopping (current)

Rather than fixing the epoch count as a hyperparameter, the current approach uses a 90/10 train/validation split with patience-1 early stopping. Each generation trains up to `--max-epochs` (default 10), evaluating on the held-out 10% after each epoch. If validation loss fails to improve for 1 epoch, training stops and the best-epoch checkpoint is restored. When `--max-epochs 1` is specified, the validation split is skipped entirely (single epoch trains on 100% of data).

**Why this is better than a fixed count.** The optimal epoch count varies across training: early generations have small buffers and benefit from multiple passes (the model typically selects 2-3 epochs), while later generations with large buffers may need only 1 epoch before validation loss plateaus. A fixed count either underfits early or overfits late. Adaptive selection automatically adjusts.

**Empirical results.** In the 2M parameter scale-up experiment, the adaptive approach consistently selected 2 epochs in early generations (gens 1-6, small buffer) and 1 epoch in later generations (gens 7+, buffer >100K positions). This matched the manually-tuned finding that "2 epochs is about right" while automatically transitioning as the buffer grew.

### Train-from-latest: tried and reverted

**The hypothesis.** When a candidate is rejected by SPRT, it's probably slightly better than the current best — just not statistically provably so. Discarding it and training the next generation from the last *accepted* checkpoint wastes this incremental progress. Training from the most recent candidate (accepted or rejected) should produce a continuous improvement trajectory.

**What happened.** The "train-from-latest" run (adaptive epochs + train from rejected candidates) significantly underperformed the "train-from-best" run (fixed 2 epochs + train from last accepted): 152 Elo at gen 15 vs 340 Elo at gen 27. The adaptive run showed a distinctive pattern: after a string of rejections, it consistently selected only 1 training epoch, suggesting the validation loss was already near-optimal after minimal training.

**The diagnosis.** A rejected candidate has already been trained on most of the buffer. Training from it again on the same (or similar) buffer is effectively adding more epochs on the same data — cumulative overfitting across generations. Each rejection compounds: gen N trains 2 epochs from best, gen N+1 trains from that rejected candidate (effectively epoch 3-4 on the same data), gen N+2 trains from *that* rejected candidate (effectively epoch 5-6), and so on. The adaptive early stopping correctly detected this by selecting 1 epoch — but even 1 additional epoch on an already-overfit checkpoint wasn't enough to overcome the accumulated staleness.

**The lesson.** "Always train from the last accepted checkpoint" is not just a safety measure — it's actively beneficial. Each new training run starts from a clean baseline, avoiding cumulative overfitting. The "wasted" progress from rejected candidates is genuinely wasted: the information that made them slightly better is already encoded in the buffer data, and a fresh training run from the accepted checkpoint will rediscover it (along with newer data).

### Data augmentation: exploiting board symmetries

Chess has a horizontal flip symmetry (a-file ↔ h-file) that holds whenever castling rights are absent. Pawnless, castling-free positions additionally have the full D4 dihedral symmetry (4 rotations x 2 reflections = 8 transforms). The augmentation system classifies each training sample into one of three symmetry groups:

- **D4** (no pawns, no castling, no en passant): expand into all 8 dihedral transforms
- **Horizontal flip** (no castling): expand into original + h-flip (2 samples)
- **None** (castling rights present): no augmentation

Both the board tensor and the 4672-element policy vector must be transformed consistently — each spatial transform induces a permutation on the policy indices (queen slide directions rotate, knight move indices permute, underpromotion capture directions swap). The permutation tables are precomputed at module load time.

**Full expansion, not random sampling.** The earlier approach randomly selected one of $N$ equivalent transforms and set weight $= 1/N$, which systematically underweighted symmetric positions (expected contribution $1/N$ vs. $1.0$ for non-symmetric positions). The current approach expands each sample into *all* equivalent transforms during chunk loading, with uniform weight. This means positions without castling rights receive 2x the training signal, and rare pawnless endgames receive 8x. This overweighting is a feature: endgames are underrepresented in self-play data (most samples come from openings and middlegames where castling rights exist) and are precisely the positions where accurate evaluation matters most for converting advantages.

**Especially sensible for KOTH.** In King of the Hill, many critical positions involve voluntary loss of castling rights — the king *wants* to be within striking distance of the center squares. These positions are strategically important and benefit from the 2x augmentation. Meanwhile, D4-eligible positions (pawnless, no castling) are even rarer than in standard chess, since many KOTH games end with a king-march victory before reaching a pawnless endgame. So the 8x overweighting applies to a vanishingly small fraction of training data.

### Optimizer: Adam → Muon

Adam was the initial default. Muon (Momentum + Newton-Schulz orthogonalization) converges faster on ResNet-style architectures — it normalizes gradient updates using Newton-Schulz iterations, which helps with the ill-conditioning typical of deep convnets. Falls back to AdamW for 1D parameters (biases, batch norm).

### Multi-variant training: tried and retired

**The problem.** Standard joint training updates all three heads (policy, value, k) simultaneously. When a candidate fails SPRT, it's unclear *why* — did the policy get worse? Did the value head overfit? Did improving one head degrade another?

**The experiment.** Each generation trained three variants from the same checkpoint:

- **Policy-only:** freeze value + k heads, train only the policy head
- **Value-only:** freeze policy head, train only value + k heads
- **All-heads:** standard joint training

Each variant was evaluated independently via SPRT. The best passing variant was promoted.

**What the data showed.** Over 23 generations of local testing, policy-only was consistently the weakest variant (average WR 0.474 — *worse* than the previous generation). Value-only and all-heads tied for best. Policy-only never outperformed all-heads in a single generation. The diagnostic value was real but the compute cost was not justified: 3 variants × 400 eval games = 1,200 evaluation games per generation, tripling the eval overhead.

**Current default: single-variant (all-heads only).** Joint training is the default (`--multi-variant` flag available to opt back in). This cuts per-generation wall time by ~60% while losing no measurable strength. The lesson: with a well-designed value function ($V_{logit} + k \cdot \Delta M$), joint training is stable — the feared interference between heads didn't materialize.

### Evaluation: greedy selection after opening diversity

**The problem.** AlphaZero-style proportional move selection (sampling from visit counts) adds noise that obscures model differences. Two models might produce very different search trees but — by random sampling — end up playing similar moves, making SPRT evaluation less sensitive.

**The solution.** Use proportional sampling only for the first 10 plies (opening diversity), then switch to greedy selection (most-visited child) for the rest of the game. Forced wins are always played deterministically regardless of move number. This gives evaluation games a diverse opening book while measuring pure strength in the middlegame and endgame.

### Evaluation game data reuse

**The problem.** Each generation runs 50-400 MCTS evaluation games (candidate vs current model) purely for gating. These games run full MCTS searches with policy outputs at every move, but all position data is discarded — only W/L/D is kept. This is wasted training data. Worse, eval games actually produce *higher quality* samples than self-play: each move starts a fresh MCTS search (no subtree reuse between moves), making samples more independent.

**The solution.** Evaluation games now collect training samples by default. At each move, the MCTS root's visit-count distribution becomes the policy target and `forced_material_balance()` provides the material scalar — the same extraction used by self-play. Samples are partitioned by which model was side-to-move: candidate's moves go to one vector, current model's moves to another.

**Selective ingestion based on gating outcome.** If the candidate wins SPRT: both sides' data is added to the replay buffer (the candidate is stronger, and the current model's data is still valid). If the candidate loses: only the current model's data is kept — the rejected candidate may be overfit or degenerate, so its policy targets could be harmful. The current model's data is always useful regardless of outcome, since by definition it represents the best known model.

**Elo tagging.** Candidate samples are tagged with the newly accepted model's cumulative Elo, and current samples with the previous model's Elo. This integrates naturally with the Elo-based strength weighting — eval data from a newly accepted model receives proportionally higher sampling weight, while data from the previous model is downweighted by exactly the measured strength gap.

**Cost-benefit.** A typical SPRT evaluation (mean ~75 games, ~100 positions per game) yields ~7,500 training samples — roughly equivalent to 75 self-play games but at zero marginal compute cost (the MCTS searches were already being run for evaluation). With 100 self-play games per generation, this represents a ~75% increase in training data per generation for free.

### Eval-only mode: skipping self-play after gen 1

Since eval games produce training data at zero marginal cost, self-play is redundant after the initial buffer seeding. The MCTS training signal (visit-count policy targets, game-outcome value targets) comes from the engine's own search regardless of opponent — the opponent only determines which positions arise. Eval games actually produce *more* data (up to 400 games vs 100 self-play) at higher quality (greedy play after ply 10, fresh MCTS search per move).

With `--skip-self-play`, the loop after gen 1 becomes: train on buffer → eval (400 games, producing training data) → gate → ingest eval data. Gen 1 always runs self-play to seed the buffer. This cuts per-generation wall time roughly in half by eliminating the self-play phase.

## 7. Performance Optimizations

### Incremental Zobrist hashing

Each move XOR-updates the hash rather than recomputing from scratch. This makes hash updates O(1) instead of O(number of pieces). Critical for MCTS where millions of positions are hashed per second.

### Clone-free legality checking

`is_legal_after_move()` checks if a move leaves the king in check without cloning the board. It applies the move, checks legality, then undoes it in-place. This avoids the allocation overhead of `Board::clone()` which was a hot path in move generation.

### Pseudo-legal move validation

`is_pseudo_legal()` validates transposition table moves without generating all legal moves. Given a move from the TT, it checks basic consistency (right color piece on source square, target square not occupied by friendly piece, etc.) in O(1). If valid, the move can be used directly.

### SeeBoard for SEE

Static Exchange Evaluation (SEE) uses a lightweight `SeeBoard` struct instead of cloning the full `Board`. `SeeBoard` contains only the minimum state needed for exchange evaluation — occupancy bitboards and piece types.

## 8. Applicability Beyond Chess

The three-tier decomposition is not chess-specific. The pattern — injecting exact solutions for tractable subproblems as terminal MCTS nodes — applies wherever a domain has classical solvers for subproblems:

| Domain | Tier 1 (Exact) | Tier 2 (Heuristic) | Tier 3 (Learned) |
|--------|---------------|-------------------|-----------------|
| **Chess** | Mate search, KOTH geometry | MVV-LVA tactical visit ordering | Neural position evaluation |
| **Mathematical reasoning** | Automated theorem provers (e.g., Lean's `decide`, `omega`) resolving subgoals | Lemma relevance ranking, proof-term similarity | Neural proof step prediction |
| **Program synthesis** | Type checking, partial evaluation, SMT solvers | API frequency heuristics | Code generation model |
| **Game playing** | Endgame tablebases, solved subgames | Domain heuristics | Value/policy networks |

The key requirement: exactly resolved nodes must be **terminal** in the MCTS tree. If a proven node can be expanded and diluted by approximate child evaluations, the proof is wasted. This terminal semantics insight is the most transferable contribution.

The value function factorization ($V = \tanh(V_{learned} + k \cdot V_{computed})$) also transfers: any domain where part of the evaluation can be computed exactly benefits from separating the learnable residual from the computable component. The learned confidence scalar $k$ modulates trust in the exact component based on context — in chess, $k$ adapts to how convertible a material advantage is; in mathematical reasoning, an analogous scalar could modulate trust in a theorem prover's assessment based on how prover-friendly the current subgoal is.

## Related Work

Caissawary draws on and extends several lines of prior work:

- **AlphaZero** (Silver et al., 2017) established pure neural MCTS for chess, learning everything from self-play. Caissawary uses the same MCTS + self-play framework but decomposes evaluation into learnable and computable components rather than learning end-to-end.

- **MCTS-Solver** (Winands et al., 2008) propagates proven game-theoretic values (wins/losses) through MCTS trees, avoiding the dilution of exact values by approximate backups. Caissawary extends this idea by treating *any* provably resolved node as terminal — not just endgame wins/losses, but also short forced mates and KOTH geometric wins detected mid-search.

- **KataGo** (Wu, 2019) incorporates handcrafted features alongside neural evaluation, including ownership predictions and score estimation. Caissawary's $k$-head similarly uses handcrafted features (pawn structure, piece counts, king patches) but in a different role: as a learned confidence scalar that modulates the influence of a separately computed material balance, rather than as auxiliary prediction targets.

- **MT-MCTS** (Mannen & Wiering, 2012) decomposes games into subgames solved by specialized agents. Caissawary's three-tier structure is a specific instance of this: Tier 1 handles tractable subgames exactly, Tier 2 applies domain heuristics, and Tier 3 handles the uncertain residual with learned evaluation.

The primary novelty is the *combination*: terminal semantics for proven nodes (preventing value dilution), factored value function with a learned confidence scalar, and systematic subgame decomposition — integrated into a single MCTS framework.
