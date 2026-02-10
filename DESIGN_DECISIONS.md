# Design Decisions: A Scientific Journey

This document traces the evolution of Caissawary's architecture — what was tried, what failed, and why the current design emerged. The core ideas generalize beyond chess: any MCTS domain with tractable subproblems can benefit from this approach.

### Why King of the Hill?

Standard chess is a surprisingly poor testbed for Tier 1 safety gates. Forced checkmates are rare in typical play — most games are decided by slow accumulation of positional and material advantages. In King of the Hill (KOTH), where moving your king to a central square also wins, the threat of forced wins is *always in the air*. Every midgame position has the dual tension of checkmate threats and king-march threats, both of which are tractable subgames solvable by Tier 1 gates. This makes KOTH an ideal stress test: the three-tier system gets far more opportunities to prove its value compared to standard chess, where Tier 1 would rarely fire.

## 1. Three-Tier MCTS: Why Not Pure AlphaZero?

**The problem.** AlphaZero treats every position uniformly: expand, evaluate with the neural network, backpropagate. But many positions have known answers. A mate-in-2 doesn't need a neural network — it needs a proof. A king that can trivially walk to the center in KOTH doesn't need 800 simulations — it needs geometry. Pure AlphaZero wastes its most expensive resource (NN calls) on positions where a microsecond of classical analysis gives an exact answer.

**The hypothesis.** Decompose positions into three tiers: (1) tractable subgames with exact solutions, (2) positions with useful heuristic structure, and (3) genuinely uncertain positions requiring learned evaluation. Only Tier 3 needs the neural network.

**Domain-general framing.** This decomposition applies wherever MCTS encounters tractable subproblems:
- **Theorem proving:** Lemmas provable by simple rewriting rules (Tier 1) vs. requiring creative proof search (Tier 3)
- **Program synthesis:** Type-checkable partial programs (Tier 1) vs. open synthesis choices (Tier 3)
- **Robotics planning:** Collision-free paths provable by geometry (Tier 1) vs. uncertain contact dynamics (Tier 3)

The pattern is always the same: prove what you can, heuristically order what you understand, learn what remains.

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

**Scalar features (8 values):** total pawns (open vs closed), STM/opponent non-pawn piece counts (endgame detection), STM/opponent queen presence (tactical complexity), pawn contacts (direct closedness measure), castling rights count (game phase), STM king rank (exposed vs castled). These are extracted directly from the input tensor with simple sums — no learned parameters.

**King patches (2 × 32 features):** 5×5 windows of all 12 piece planes centered on each king, padded for edge kings. Each 300-dim patch is compressed by a separate FC(300→32) + ReLU — separate weights for STM and opponent kings since they have different semantics (king safety vs. attack potential). This captures local piece configurations around each king without global average pooling's dilution.

**Combination:** `[8 scalars | 32 STM patch features | 32 opp patch features]` → FC(72→32) → ReLU → FC(32→1) → k_logit. Only the final FC(32→1) is zero-initialized; patch FCs and combine layer use standard He init. Total: ~21.6k parameters (tiny vs ~1.98M model total).

### Classical fallback: V\_logit=0, k=0.5

With no neural network, the engine uses $V_{logit} = 0$ (no positional knowledge) and $k = 0.5$ (moderate material weight), giving $V_{final} = \tanh(0.5 \cdot \Delta M)$. This matches the NN's initialization — a freshly initialized network produces identical values to the classical fallback, ensuring smooth bootstrapping.

## 5. OracleNet Architecture

### SE-ResNet backbone

6 residual blocks with Squeeze-and-Excitation (SE) attention. SE blocks let the network learn which feature channels are important for each position — essentially a per-position "volume knob" for different types of features (material patterns, king safety features, pawn structure features).

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

8 scalar features (pawn count, piece counts, queen presence, pawn contacts, castling rights, king rank) + two 5×5 king-centered patches compressed via FC(300→32). Scalars + compressed patches → FC(72→32) → FC(32→1). Operates on raw input, independent of the backbone. See Section 4 for the rationale.

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

### Replay buffer: from clear-on-accept to sliding window

**v1: Clear buffer on model acceptance.** Every time a new model was accepted, the entire replay buffer was cleared. Rationale: old data was generated by a weaker model. Problem: the first training iteration after acceptance had very little data, causing severe overfitting.

**v2: Sliding window.** FIFO eviction — oldest games are removed when capacity is exceeded, regardless of model acceptance. The buffer always contains a mix of recent and older data.

**v3: Recency-weighted sampling (current).** Sliding window with exponential recency weighting (configurable half-life, default 20k positions). Recent games are sampled more frequently, but old games still contribute. This smooths the distribution shift between model generations.

### Adaptive minibatches

Fixed minibatch counts caused overfitting in early generations (small buffer, many passes) and underfitting in late generations (large buffer, few passes). The current approach targets ~1.5 epochs over the buffer, capped at a configurable maximum. This scales training proportionally to data availability.

### Data augmentation: exploiting board symmetries

Chess has a horizontal flip symmetry (a-file ↔ h-file) that holds whenever castling rights are absent. Pawnless, castling-free positions additionally have the full D4 dihedral symmetry (4 rotations x 2 reflections = 8 transforms). The augmentation system classifies each training sample into one of three symmetry groups:

- **D4** (no pawns, no castling, no en passant): randomly apply one of 8 transforms
- **Horizontal flip** (no castling): randomly apply identity or h-flip
- **None** (castling rights present): no augmentation

Both the board tensor and the 4672-element policy vector must be transformed consistently — each spatial transform induces a permutation on the policy indices (queen slide directions rotate, knight move indices permute, underpromotion capture directions swap). The permutation tables are precomputed at module load time.

Samples are weighted by $1/N$ where $N$ is the symmetry group size (8, 2, or 1), so that each original position contributes equally to the loss regardless of how many equivalent views exist.

### Optimizer: Adam → Muon

Adam was the initial default. Muon (Momentum + Newton-Schulz orthogonalization) converges faster on ResNet-style architectures — it normalizes gradient updates using Newton-Schulz iterations, which helps with the ill-conditioning typical of deep convnets. Falls back to AdamW for 1D parameters (biases, batch norm).

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

The three-tier decomposition is not chess-specific. The pattern generalizes:

| Domain | Tier 1 (Exact) | Tier 2 (Heuristic) | Tier 3 (Learned) |
|--------|---------------|-------------------|-----------------|
| **Chess** | Mate search, KOTH geometry | MVV-LVA tactical visit ordering | Neural position evaluation |
| **Theorem proving** | Decidable fragments, rewriting rules | Lemma relevance ranking | Proof step prediction |
| **Program synthesis** | Type checking, partial evaluation | API frequency heuristics | Code generation model |
| **Robotics** | Collision geometry, kinematic limits | Distance-to-goal heuristics | Contact dynamics prediction |
| **Game playing** | Endgame tablebases, solved subgames | Domain heuristics | Value/policy networks |

The key requirement: Tier 1 solutions must be **terminal** in the MCTS tree. If a proven node can be expanded and diluted by approximate children, the proof is wasted. This terminal semantics insight is the most transferable contribution.

The value function factorization ($V = \tanh(V_{learned} + k \cdot V_{computed})$) also generalizes: any domain where part of the evaluation can be computed exactly benefits from separating the learnable residual from the computable component. The learned $k$ allows the network to adaptively weight the two components based on context.
