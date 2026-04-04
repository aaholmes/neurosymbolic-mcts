# Design Decisions: A Scientific Journey

This document traces the evolution of Caissawary's architecture — what was tried, what failed, and why the current design emerged. The core pattern — exact subgame resolution injected as terminal MCTS nodes — applies to any domain with tractable subproblems.

### Why King of the Hill?

Standard chess is a surprisingly poor testbed for Tier 1 safety gates. Forced checkmates are rare in typical play — most games are decided by slow accumulation of positional and material advantages. In King of the Hill (KOTH), where moving your king to a central square also wins, the threat of forced wins is *always in the air*. Every midgame position has the dual tension of checkmate threats and king-march threats, both of which are tractable subgames solvable by Tier 1 gates. This makes KOTH an ideal stress test: the three-tier system gets far more opportunities to prove its value compared to standard chess, where Tier 1 would rarely fire.

## 1. Three-Tier MCTS: Why Not Pure AlphaZero?

**The problem.** AlphaZero treats every position uniformly: expand, evaluate with the neural network, backpropagate. But many positions have known answers. A mate-in-2 doesn't need a neural network — it needs a proof. A king that can trivially walk to the center in KOTH doesn't need 800 simulations — it needs geometry. Pure AlphaZero wastes its most expensive resource (NN calls) on positions where a microsecond of classical analysis gives an exact answer.

**The hypothesis.** Decompose positions into three tiers: (1) tractable subgames with exact solutions, (2) positions with useful heuristic structure, and (3) genuinely uncertain positions requiring learned evaluation. Only Tier 3 needs the neural network.

**Domain-general framing.** This decomposition applies wherever MCTS encounters tractable subproblems. The most natural next domain is mathematical reasoning: automated theorem provers (Lean's `decide`, `omega`, `norm_num`) can resolve certain subgoals exactly, while a neural policy guides the high-level proof search through uncertain creative steps. The confidence scalar $k$ maps directly: a global trust level in the computable component, modulated per-position by the value head.

The pattern is always the same: compute what you can exactly, heuristically order what you understand, learn what remains.

## 2. Tier 1: Safety Gates as Terminal Nodes

### First attempt: gate values as priors

The initial design ran mate search at expansion, then used the result as a value prior for the node. The node was still expanded normally — children were created and evaluated by the NN or classical eval.

**What went wrong.** MCTS diluted the proven values. If a node was proven to be mate-in-2 (value = +1.0), but it got expanded, subsequent visits would descend into children evaluated by approximate methods. Those children might return values like +0.3 or -0.1 (the NN hadn't learned this pattern yet). The exact +1.0 from the mate search would get averaged with dozens of approximate values, converging toward something much weaker.

### Solution: terminal semantics

Gate-resolved nodes became **terminal** — treated identically to checkmate and stalemate. No children are ever created. Every future visit simply re-uses the cached exact value. This preserves the proof throughout the entire search.

**Key insight:** Proven values must be *cached*, not *mixed with approximations*. The terminal semantics are what make the proof sticky — once an MCTS node is proven, it stays proven forever.

### Exhaustive mate-in-2 + checks-only mate-in-3

Full-width mate search at all depths is too expensive to run at every MCTS expansion. The compromise uses a configurable `exhaustive_depth` parameter (default: 3 plies) that controls which depths search all legal moves vs. only checking moves:

- **Mate-in-1 (depth 1):** exhaustive — all legal moves tried
- **Mate-in-2 (depth 3):** exhaustive — catches quiet-first forced mates like 1.Kg6! Ra8# where the first move doesn't give check
- **Mate-in-3 (depth 5):** checks-only — keeps branching manageable at deeper levels

The exhaustive mate-in-2 is modest because mate-in-2 has limited branching: the attacker's ~30 legal moves each lead to positions where the defender must have *all* replies lead to mate-in-1 — a condition that prunes aggressively. No artificial node budget is needed — depth limiting and checks-only filtering keep the search naturally bounded.

### Fast check detection with gives\_check() pre-filter

On attacker plies with checks-only search, the naive approach generates all ~35 pseudo-legal moves, then for each: `make_move` → `is_legal` → `is_check` → `undo_move`. Since only ~1-5 moves give check, ~30 moves pay the full `make_move`/`undo_move` cost (Zobrist update, history push/pop) just to discover they don't give check.

The optimization calls `gives_check()` *before* `make_move`. This function works on the pre-move board using modified occupancy: direct checks require one magic/table lookup (does the piece on `to` attack the king?), discovered checks require up to two slider lookups (does vacating `from` reveal an attack?). Special moves (promotions, castling, en passant) fall back to `apply_move_to_board` + `is_check`. Non-checking moves skip `make_move` entirely — eliminating both the expensive board mutation and the subsequent `is_check` call.

**Correctness subtlety:** When all moves are filtered out on an attacker ply, `has_legal_move` stays false. But this doesn't mean checkmate/stalemate — it means no *checking* moves exist. The fix: on attacker plies with checks-only, return 0 (no mate found) instead of checking for checkmate/stalemate. Terminal detection only matters on defender plies, where all legal moves are tried.

This reduced mate search cost by 26% (1.08 → 0.80 us/node).

### Stateless search: &Board instead of BoardStack

Even after the `gives_check()` optimization, mate search cost 1.7x more per node than KOTH (0.80 vs 0.47 us/node). Profiling revealed the remaining gap came from `BoardStack` overhead: `make_move`/`undo_move` performs incremental Zobrist hashing, pushes/pops state history, and `is_draw_by_repetition()` scans the position history — none of which is needed for forced mate detection, since a repeated position on the path to checkmate is irrelevant.

The fix: convert mate search from `&mut BoardStack` (stateful make/undo) to `&Board` (stateless `apply_move_to_board`), matching KOTH's approach. Each recursive call creates a new `Board` on the stack — no history tracking, no Zobrist updates, no repetition checks.

Combined with `gives_check()`, this reduced mate search from 1.08 to 0.64 us/node (41% total improvement).

### Pure minimax with legality-first filtering

The remaining gap between mate search (0.64 us/node) and KOTH (0.19 us/node) came from three sources: alpha-beta bookkeeping (score tracking, alpha/beta updates per move), atomic node budgets (`SearchContext` with `AtomicUsize`, `AtomicBool`, `Mutex`), and board clones wasted on illegal pseudo-legal moves.

**Alpha-beta → pure minimax.** Within a single depth of iterative deepening, the mate question is binary: "does a forced mate exist at this depth?" Alpha-beta's score range tracking is unnecessary — the solver only needs short-circuit semantics: attacker returns true on first child success, defender returns false on first child refutation. This matches KOTH's `solve_koth` pattern exactly. Iterative deepening still finds the shortest mate (depth 1 before 3 before 5).

**Atomic node budget removed.** The 100K node budget used `AtomicUsize::fetch_sub` + two `AtomicBool` loads per node (~15-20ns overhead). This was unnecessary — mate search is depth-limited (max depth 5) and heavily pruned by checks-only filtering + geometric pruning. The budget never triggered in practice. Replaced with a plain `&mut i32` counter.

**`is_legal_after_move()` before `apply_move_to_board()`.** Previously: `apply_move_to_board(m)` (clones entire Board, ~400 bytes) → `is_legal(move_gen)` → discard if illegal. Now: `is_legal_after_move(m, move_gen)` (no clone, ~200 cycles) → only `apply_move_to_board` if legal. This saves a full board clone for every illegal pseudo-legal move — ~5 per defender ply.

**Redundant `get_piece()` calls removed.** Two call sites checked `board.get_piece(m.from).is_some()` before processing moves from `gen_pseudo_legal_moves` — but pseudo-legal move generation only produces moves from occupied squares, making these 12-bitboard scans per move pure dead code.

These four changes reduced mate search mean time from 608 us to 224 us (2.7x speedup), primarily by reducing mean nodes per call from 954 to 359. The per-node cost dropped to 0.63 us/node.

### Batched per-piece checkmate detection at depth-0 leaves

After the pure minimax rewrite, depth-0 checkmate detection became the dominant cost within mate search. ~95% of nodes in a checks-only search are depth-0 leaves. The original approach called `gen_pseudo_legal_moves` — generating moves for all 6 piece types, allocating ~10 vectors — then iterated until finding one legal move. In the common case (not checkmate), a king move provides the evasion, but the engine still paid for generating knight, bishop, rook, queen, and pawn moves unnecessarily.

The optimization generates moves by piece type in priority order, aborting early when any legal evasion is found:

1. **King moves** (direct bitboard, zero allocation, skip castling since in check): highest evasion probability (~70-80% of cases), cheapest generation. Uses `k_move_bitboard[king_sq] & !friendly_occ` — the same pattern as KOTH direct king-move generation.
2. **Double check detection** (2 magic lookups + 2 table lookups): if two or more pieces attack the king, only king moves can evade. Since king moves already failed above, it's checkmate — skip all 5 remaining piece-type generations entirely. A slider constraint avoids unnecessary work: double check requires at least one slider, so if no slider attacks the king (0 magic hits), it's a single non-slider check and double-check detection is skipped.
3. **Knight → Bishop → Rook → Queen → Pawn** in increasing generation cost. Each batch aborts on the first legal move found.

An additional optimization: in checks-only mode, the attacker only plays checking moves, so the defender is always in check at depth 0. The `is_check()` call (which scans all 6 attacker piece types for attacks on the king) is skipped entirely.

This reduced mate search from 224 us to 132 us (41% speedup, 0.63 → 0.37 us/node) without changing the search tree — identical node counts confirm no behavioral change. Mate search's share of wall time dropped from 10% to 6%.

**Cumulative mate search optimization:** 608 us → 132 us (4.6x speedup, 78% reduction). Per-node cost: 1.08 → 0.37 us/node (66% reduction). The remaining 2x gap versus KOTH (0.37 vs 0.19 us/node) comes from `gives_check()` filtering on attacker plies and richer move generation needed for full checkmate detection vs. KOTH's simple geometric reachability.

### KOTH geometric pruning

In King of the Hill, a king that can reach {d4, e4, d5, e5} wins. The gate computes: can the side-to-move's king reach any center square in at most 3 moves, considering blocking pieces and opponent interception? This is pure geometry — no search needed.

### KOTH direct king-move generation

On root-side turns in `solve_koth`, when the king is not already in the required distance ring for the current ply, only king moves toward center matter. The original approach generated all ~35 pseudo-legal moves via `gen_pseudo_legal_moves`, then filtered to king moves landing in the target ring — discarding ~30 non-king moves that were generated unnecessarily.

The optimization skips full move generation entirely on these turns. Instead, it computes valid king destinations directly via bitboard intersection: `k_move_bitboard[king_sq] & target_mask & !friendly_occ`. This yields exactly the 1-3 valid destination squares, each producing a `Move::new(king_sq, to_sq, None)` that goes straight to `apply_move_to_board` + `is_legal`. No post-apply ring check is needed since `target_mask` already constrains destinations. When the king is already in the target ring, full movegen is used with a post-apply ring check — non-king moves are valid in this case (king stays put) and the king might move out of the ring.

This reduced KOTH per-node cost from 0.33 to 0.19 us/node (42% improvement, cumulative 63% from the original 0.52 us/node).

### Depth scaling: why mate-in-5 and KOTH-in-3

Both KOTH and mate search depths are configurable (`koth_depth` and `mate_search_depth` in `TacticalMctsConfig`), but profiling shows the defaults (3 and 5 respectively) sit at a sweet spot. Comparing baseline (KOTH-in-3, mate-in-5) against deeper search (KOTH-in-4, mate-in-6) over 10 self-play games at 400 simulations/move:

| Operation | Mean (us) | Mean nodes | Total (ms) | % wall time |
|-----------|----------:|----------:|-----------:|------------:|
| KOTH-in-3 | 149 | 780 | 30,729 | 8% |
| KOTH-in-4 | 8,250 | 56,643 | 1,713,129 | 83% |
| Mate-in-5 | 132 | 359 | 22,967 | 6% |
| Mate-in-6 | 292 | 861 | 50,905 | 12% |

**KOTH-in-4 is catastrophic.** 55x slower per call, 73x more nodes. At depth 4, the root ply's geometric target mask expands to `RING_3` — the entire board border — so the direct king-move pruning that makes KOTH-in-3 fast (intersecting king moves with a small target ring of ~4-12 squares) degenerates to full move generation. KOTH-in-4 alone consumes 5x the NN inference budget, dominating total wall time at 83%. The hit rate for center-in-4 wins that aren't already center-in-3 is low, making the cost unjustifiable.

**Mate-in-6 is cheap but marginal.** Only 2.2x slower (checks-only branching factor ~3-5 keeps each additional depth affordable), moving from 6% to 12% of wall time. However, forced mate-in-6 via pure checks is rare in practice — most deep mates require quiet preparatory moves that checks-only search cannot find.

**Exhaustive mate-in-2: not worth it.** The `exhaustive_mate_depth` parameter enables all-legal-moves search at shallow depths, catching quiet mates (rook lifts, king approaches, clearance moves) that checks-only misses. Profiling exhaustive mate-in-2 (all legal moves for mate-in-1/2, checks-only for mate-in-3+):

| Mate search mode | Mean (us) | Mean nodes | Total (ms) | % wall time |
|------------------|----------:|----------:|-----------:|------------:|
| Checks-only (depth 5) | 133 | 359 | 23,169 | 6% |
| Exhaustive mate-in-2 (depth 5) | 193 | 1,342 | 33,650 | 9% |

Nodes jump 3.7x (from ~3-5 checks to ~35 legal moves at shallow plies), but time only increases 1.46x because the extra nodes are cheap shallow ones. However, the cost is not justified: a quiet mate-in-2 will be found by MCTS within a few simulations anyway — the engine only needs ~2-3 visits to discover the winning line through normal expansion. The safety gate's value comes from catching forced wins that MCTS would take many simulations to prove (deep checks-only mates), not short mates that MCTS handles naturally.

**Current defaults:** Checks-only mate-in-5 and KOTH-in-3, with no exhaustive mate search (`exhaustive_mate_depth: 0`). The geometric pruning that makes KOTH efficient has a hard wall at depth 4, while mate search depth 5 captures the common checks-only forced mates without excessive cost. Exhaustive mate search is available via `--exhaustive-mate-depth` for profiling but disabled in production.

## 3. Tier 2: Quiescence Search and Tactical Ordering

Tier 2's core contribution is `forced_ext_pesto_balance()` — an extended PeSTO piece-square-table quiescence search (depth 20) that runs at every MCTS leaf evaluation to compute $\Delta M$, the tapered positional+material evaluation after all forced tactical sequences resolve. Unlike simple piece counting (`[1,3,3,5,9,0]`), PeSTO uses Texel-tuned piece-square tables (standard RofChade values) that account for piece placement — a knight on e4 is worth more than a knight on a1. This is a classical alpha-beta tree search whose results no neural network can easily replicate: it explores variable-depth exchange sequences to detect hanging pieces, discovered attacks, and forced promotion lines. $\Delta M$ feeds into the value function through two paths: as a direct input feature to the value head's FC layer (position-dependent learned modulation) and via the additive $k \cdot \Delta M$ term (global baseline trust), providing the foundation that the NN builds on top of.

### Extended quiescence: beyond pure captures

The original Q-search (`pesto_qsearch`) only considered captures and promotions. The extended version (`ext_pesto_qsearch_counted`) adds three innovations: tactical quiet moves, null-move threat detection, and mystery-square fork resolution.

#### Tactical quiet moves (budget-limited)

Each side gets one non-capture tactical move per search:

- **Non-capture checks**: any quiet move that gives check, detected via `gives_check()` before move application
- **Pawn forks**: a pawn advance where the pawn's capture bitboard at the destination attacks 2+ enemy pieces in {N, B, R, Q, K}
- **Knight forks**: a knight move where the knight's attack bitboard at the destination attacks 2+ enemy pieces in {R, Q, K}

Budget rules keep the search bounded: each side's single tactical move is tracked via `white_tactic_used`/`black_tactic_used` flags threaded through recursion. Check evasions are free (when in check, all legal moves are generated — no stand-pat).

**Why one tactic per side.** Allowing unlimited tactical moves would explode the search tree — every quiet check or fork spawns a subtree. One per side is sufficient to detect the most common tactical patterns (Nc7+ forking K+R, d5 forking two minor pieces) while keeping node counts manageable. The budget ensures the extended Q-search remains effectively free relative to NN inference (<1% of wall time).

**Why these three move types.** Non-capture checks are the highest-value tactical moves: they force the opponent to deal with check (often losing material). Pawn and knight forks are the next most common tactical patterns that pure capture search misses — a knight fork winning an exchange is invisible to capture-only Q-search because the fork move itself is quiet. Bishop and rook forks are rarer and harder to detect without full sliding-piece attack generation, so they're excluded from the budget-limited search.

#### Null-move threat detection with "deny first choice"

Standard Q-search has a **stand-pat** assumption: at every node, the side to move can "choose" not to capture and accept the static eval. This is a lie when pieces are hanging — if you "do nothing," the opponent captures them. The extended Q-search replaces blind stand-pat with a **null-move probe**:

1. **Pass the turn** to the opponent (one null-move per branch total, tracked by `white_null_used`/`black_null_used`)
2. **Evaluate each opponent response** — all captures and tactical quiets, each scored recursively
3. **Deny the opponent's first choice** — the pass represents making a quiet move that addresses the most urgent threat

The "deny first choice" logic handles three cases:

- **0 threats**: Stand-pat holds. Position is quiet.
- **1 threat**: Pass fully addresses it — you'd retreat the threatened piece. `adjusted_stand_pat = stand_pat`.
- **2+ threats (fork)**: You can only save one piece. Deny the opponent's best capture; they get their second-best. The adjusted stand-pat drops accordingly.

**Why not a full recursive null-move.** The earlier approach ran a single recursive Q-search after the null-move, returning the opponent's best score. This was simpler but had a critical flaw: it assumed "doing nothing" meant losing the *best* piece, when in reality you'd save the best piece and lose the *second-best*. Evaluating captures individually and denying the first choice correctly models fork resolution.

#### Mystery-square recapture

The "deny first choice" alone still underestimates the side-to-move's resources in a fork. After d5 forks Bc4 and Ne4: deny dxc4 (save bishop), opponent plays dxe4 (capture knight). But the evaluation stops there — it doesn't see that the bishop, now on a "mystery square" after retreating, can recapture the pawn on e4 (e.g., via Bd3, dxe4, Bxe4). Without this recapture, the fork costs a full knight; with it, the fork costs knight minus pawn = one exchange.

When there are 2+ opponent captures, the null-move probe adds a **mystery-square recapture step**:

1. Opponent makes their second-choice capture (the one not denied)
2. The saved piece (from the denied first-choice's target square) "teleports" to the capture landing square, recapturing the attacker
3. The resulting position is evaluated with PeSTO

This bends chess geometry — the saved piece moves from its current square to the capture square regardless of whether this is a legal piece move. But it correctly models what happens in practice: the saved piece retreats to *some* square from which it can recapture. In the center fork trick, this is exactly Bd3→Bxe4. In a knight fork of queen and rook, it's Q→(mystery)→Qx(knight's landing square).

The `adjusted_stand_pat` uses the better of the raw second-choice score and the recapture-adjusted score, capped at the original static eval. This means:
- If the recapture improves the evaluation (common in forks): use the recapture score
- If the recapture makes things worse (rare): fall back to the raw second-choice
- If neither threat matters (quiet position): stand-pat holds

**Benchmark: center fork trick** (1.e4 e5 2.Nc3 Nf6 3.Bc4 Nxe4 4.Nxe4, Black to move with d5 fork available):

| Q-search variant | Score (Black POV) | Nodes | Assessment |
|:---|---:|---:|:---|
| Basic (captures only) | -2.90 | 1 | Completely wrong — misses d5 |
| Extended (checks + forks) | -2.90 | 6 | Finds d5 but can't evaluate it |
| + Null-move (deny first choice) | +0.10 | 251 | Correct — fork equalizes |
| + Mystery recapture | +0.10 | 251 | Correct with accurate floor |

The visit-ordering component (MVV-LVA) is a minor addition: captures are visited in Most-Valuable-Victim / Least-Valuable-Attacker order on their first visit. After the first visit, normal UCB selection takes over.

### Earlier attempt: Q-search grafting at expansion

The first approach was more ambitious: at every MCTS expansion, run a Q-search rooted at the expanded position and "graft" the Q-search tree onto the MCTS tree — Q-search leaf values became initial values for MCTS children.

**What went wrong.** Q-search at every expansion was expensive (~10x slower expansions), and the grafted values were noisy — Q-search with alpha-beta in a random MCTS leaf often had poor alpha/beta bounds. The complexity was high (converting Q-search nodes to MCTS nodes, handling transpositions between the two trees) and the benefit was marginal.

**The lesson:** The Q-search doesn't need to run at expansion time or graft into the tree structure. Running it once at leaf evaluation time — as `forced_ext_pesto_balance()` — is simpler, cheaper, and gives the value function a clean positional+material signal. MVV-LVA visit ordering handles the remaining tactical concern (which captures to try first).

## 4. The Value Function: V = tanh(V\_logit + k * DeltaM)

### The insight: separate what's learnable from what's computable

Standard AlphaZero learns everything end-to-end: the NN must discover that material matters, learn piece values, and learn to evaluate positions — all from game outcomes. This works, but it's sample-inefficient. The NN spends millions of games rediscovering that a queen is worth more than a pawn.

**The key idea:** Factor the evaluation into a *computable* component (material after forced exchanges) and a *learnable* component (positional factors the NN must discover).

$$V_{final} = \tanh(V_{logit} + k \cdot \Delta M)$$

- $V_{logit}$ (NN output, unbounded): positional assessment only — piece activity, king safety, pawn structure
- $\Delta M$ (computed exactly): PeSTO evaluation after forced tactical sequences via `forced_ext_pesto_balance()`
- $k$ (NN output, positive): how much material should matter in this position

### Why forced\_pesto\_balance(), not simple piece counting

A position might have equal material (P=P) but after forced exchanges end up +3 (winning a piece). Simple piece counting misses hanging pieces, discovered attacks, and forced exchange sequences. `forced_ext_pesto_balance()` runs an extended PeSTO piece-square-table quiescence search — tapered positional+material evaluation using standard RofChade PST values, no additional bonuses (bishop pair, king safety, mobility) — to resolve forced captures, non-capture checks, and forks. This gives $\Delta M$ a richer evaluation than raw piece counts: a centralized knight is worth more than a cornered one, a passed pawn on the 7th rank scores higher than one on the 3rd, and a knight fork winning an exchange is detected even though the fork move itself is quiet.

### Why the NN returns V\_logit (unbounded), not tanh(V\_logit)

If the NN returned a bounded value in [-1, 1], adding $k \cdot \Delta M$ would saturate quickly. With unbounded $V_{logit}$, the NN can learn arbitrary positional offsets that interact smoothly with the material term. The final $\tanh$ constrains the output at the end.

### Dynamic k: learned confidence in material

Not all positions are equally material-sensitive. In a closed Sicilian with locked pawns, material imbalances matter less than in an open position.

$k$ is computed as $0.47 \cdot \text{Softplus}(k_{logit})$ where $k_{logit}$ is a single learned scalar (`nn.Parameter`). The 0.47 coefficient is Texel-calibrated so that PeSTO centipawn evaluations map through $\tanh$ to calibrated win probabilities. At initialization ($k_{logit} = 0$): $k = 0.47 \cdot \ln 2 \approx 0.326$.

### k architecture evolution: from per-position networks to a global scalar

The first implementation computed $k$ from the backbone's penultimate features (shared with the value head). This caused $k$ to overfit to specific positions — the deep features encoded too much position-specific detail.

The second design used a **separate shallow input network** for $k$: a single 3x3 conv → BN → global average pool → two linear layers. This was too shallow — one 3x3 conv + global average pool is a "bag of local patterns" that knows what local patterns exist but not where they are. It cannot reliably detect king safety (diluted to 1/64 of signal), pawn structure (multi-hop), or open files (long-range).

The third design used **handcrafted scalar features + king patches**: 12 scalar features (pawn counts, piece counts, queen presence, pawn contacts, castling rights, king rank, bishop square-color presence) + Q-search completion flag + two 5×5 king-centered patches compressed via FC(300→32), combined via FC(77→32→1). Total: ~22k parameters. This worked well but had two drawbacks: (1) the dynamic king-patch extraction (per-sample argmax + 5×5 gather) was hostile to CUDA reimplementation for a future GPU-resident MCTS kernel, and (2) the position-dependent modulation it provided could be absorbed by the value head itself.

**Current design: global scalar k + q_result as value head input.** The K-head was removed entirely (−21,536 parameters) and replaced with:
1. A single `nn.Parameter` scalar $k_{logit}$, initialized to 0 → $k = 0.326$ (Texel-calibrated)
2. The Q-search result ($\Delta M$) concatenated as a 65th input to the value head's first FC layer (widened from FC(64→256) to FC(65→256))

This preserves both paths for material information to influence the value:
- **Additive path** ($k \cdot \Delta M$): a global, position-independent baseline trust level
- **Learned path** ($\Delta M$ as input to FC): the value head learns position-dependent adjustments through its existing weights — in quiet endgames it can learn that $\Delta M$ is uninformative, in sharp middlegames it can learn to rely on it heavily

The key insight: making $k$ position-independent is acceptable because the position-dependent part of "how much to trust material" is learned implicitly through the value head FC layers. The scalar $k$ sets the global operating point; the network adjusts around it.

### Classical fallback: V\_logit=0, k=0.326

With no neural network, the engine uses $V_{logit} = 0$ (no positional knowledge) and $k = 0.326$ (Texel-calibrated material weight), giving $V_{final} = \tanh(0.326 \cdot \Delta M)$. This matches the NN's initialization — a freshly initialized network produces identical values to the classical fallback, ensuring smooth bootstrapping.

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

1x1 conv → BN → flatten (64 features) → concat(q_result) → FC(65→256) → FC(256→1). The Q-search result ($\Delta M$) is fed as a direct 65th input feature, giving the value head position-dependent control over material trust. The output is $V_{logit}$ (unbounded). This feeds into the symbolic residual formula with the independently-computed $k$ and $\Delta M$.

### k: global scalar

Single `nn.Parameter` scalar $k_{logit}$, output via $0.47 \cdot \text{softplus}(k_{logit})$ (Texel-calibrated). Sets a global baseline trust level in material. Position-dependent modulation is absorbed by the value head FC which receives $\Delta M$ as a direct input. See Section 4 for the architectural evolution.

### Zero initialization

All output layers ($V_{out}$, policy head) are initialized to zero. $k_{logit}$ is initialized to 0 (giving $k = 0.326$):
- **Policy:** $\text{softmax}(0, 0, ...) = $ uniform distribution (explore everything equally)
- **Value:** $V_{logit} = 0$, so $V_{final} = \tanh(k \cdot \Delta M)$ (pure PeSTO evaluation)
- **k:** $k_{logit} = 0 \Rightarrow k = 0.326$ (Texel-calibrated material weight)

This means a freshly initialized network immediately produces meaningful behavior: uniform exploration with material-aware evaluation.

## 6. Training Pipeline Evolution

### Gating: from fixed threshold to SPRT

**v1: Fixed threshold (55% winrate).** Simple but noisy — a 55% result in 100 games is barely significant. Promising but weak improvements were rejected, while lucky noise was occasionally accepted.

**v2: One-sided binomial test (p < 0.05).** Better statistical rigor but still problematic with small sample sizes. A candidate winning 6/10 games would be rejected, while one winning 30/50 might be accepted despite similar effect sizes.

**v3: SPRT early stopping (current).** Sequential Probability Ratio Test with trinomial GSPRT matching fishtest's implementation. Tests "candidate is ≥ elo1 Elo stronger" vs. "candidate is ≤ elo0 Elo stronger." Clear improvements terminate in ~30 games; marginal cases use up to 800. This saves significant wall-clock time while maintaining statistical rigor.

**Why 800 eval games.** With 100 eval games and a high draw rate (~84% at 100 sims), even a WR of 0.575 yields p ≈ 0.067 — not enough to reject the null hypothesis. A one-sided binomial test needs ~275 games for 80% power to detect a 53% true WR. The default of 800 games provides comfortable statistical power while SPRT early stopping keeps the average cost much lower for clear-cut results.

### Replay buffer: from clear-on-accept to sliding window

**v1: Clear buffer on model acceptance.** Every time a new model was accepted, the entire replay buffer was cleared. Rationale: old data was generated by a weaker model. Problem: the first training iteration after acceptance had very little data, causing severe overfitting.

**v2: Sliding window.** FIFO eviction — oldest games are removed when capacity is exceeded, regardless of model acceptance. The buffer always contains a mix of recent and older data.

**v3: Recency-weighted sampling.** Sliding window with exponential half-life decay across model generations. Recent games sampled more frequently, old games still contribute. Problem: the half-life parameter was arbitrary and disconnected from actual model strength — a 55% winrate gap (barely detectable) produced the same 7:1 sampling ratio as a massive strength improvement if the same number of positions separated them.

**v4: Elo-based strength weighting (current).** Each model acceptance chains the SPRT winrate into a cumulative Elo rating: $\Delta_{Elo} = -400 \cdot \log_{10}(1/WR - 1)$. Training data is weighted by expected score against the current best model:

$$w_i = n_i \cdot 2 \cdot \frac{1}{1 + 10^{(Elo_{max} - Elo_i) / 400}}$$

This produces sampling ratios proportional to actual measured strength differences. A 55% winrate gap (~35 Elo) yields a ~1.2:1 ratio (not 7:1). A 200 Elo gap yields ~0.48x weight. No data is ever fully discarded — even data from much weaker models contributes at a reduced rate. The weighting adapts automatically: rapid improvement produces steeper gradients, while plateaus produce near-uniform sampling. Eval game data from both sides is ingested after each evaluation; the losing side's data is tagged with a lower Elo based on the measured winrate, so the odds-ratio weighting automatically downweights it.

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

### Train-from-latest: reverted, then re-enabled

**The hypothesis.** When a candidate is rejected by SPRT, it's probably slightly better than the current best — just not statistically provably so. Discarding it and training the next generation from the last *accepted* checkpoint wastes this incremental progress. Training from the most recent candidate (accepted or rejected) should produce a continuous improvement trajectory.

**First attempt: reverted.** An early "train-from-latest" run with adaptive epochs underperformed a "train-from-best" run with fixed 2 epochs (152 Elo at gen 15 vs 340 Elo at gen 27). The diagnosis was cumulative overfitting: each rejection compounds training on the same data. This led to reverting to train-from-best.

**Re-evaluation.** Later analysis revealed the comparison was confounded — the two runs differed in epoch count (adaptive vs fixed 2), not just resume strategy. A controlled comparison showed train-from-best caused candidates to repeatedly train from the same base without building on previous learning. In one run, gens 3-5 all resumed from gen_2.pth and produced progressively worse candidates (0.482, 0.449 winrate). Meanwhile, runs that used train-from-latest showed the expected pattern: gen 1 rejected at 0.511, gen 2 built on it and jumped to +50.5 Elo with 0.572 winrate.

**Current approach (train-from-latest).** Always resume from the most recent candidate, whether accepted or rejected. Early stopping prevents cumulative overfitting — if the latest candidate is already well-trained on the buffer, validation loss won't improve and training stops after 1 epoch. The key insight: a rejected candidate at 51% winrate is almost certainly better than the current best, just not provably so. Starting over from the accepted checkpoint throws away real learning.

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

**What the data showed.** Over 23 generations of local testing, policy-only was consistently the weakest variant (average WR 0.474 — *worse* than the previous generation). Value-only and all-heads tied for best. Policy-only never outperformed all-heads in a single generation. The diagnostic value was real but the compute cost was not justified: 3 variants × 800 eval games = 2,400 evaluation games per generation, tripling the eval overhead.

**Current default: single-variant (all-heads only).** Joint training is the default (`--multi-variant` flag available to opt back in). This cuts per-generation wall time by ~60% while losing no measurable strength. The lesson: with a well-designed value function ($V_{logit} + k \cdot \Delta M$), joint training is stable — the feared interference between heads didn't materialize.

### Evaluation: proportional-or-greedy mixing

**The problem.** AlphaZero-style proportional move selection (sampling from visit counts) adds noise that obscures model differences. Two models might produce very different search trees but — by random sampling — end up playing similar moves, making SPRT evaluation less sensitive.

**v1: Fixed cutoff.** Proportional sampling for the first 10 plies, then pure greedy (most-visited child). Simple and effective, but the hard discontinuity is arbitrary and greedy play after ply 10 produces identical games from identical positions — reducing training data diversity.

**v2: Top-p nucleus sampling (tried and replaced).** Decaying nucleus: p = 0.95^(move-1), sampling from the top-p fraction of moves by visit count. Smooth transition, but nucleus boundaries are arbitrary — with p=0.3, the 2nd-best move might be much weaker than the 1st yet still gets sampled 20% of the time. Empirically, this added too much noise: runs with top-p=0.95 showed 14+ consecutive SPRT rejections at plateaus where the old greedy approach had broken through.

**v3: Proportional-or-greedy mix (current).** With probability p = 0.90^(move-1), play a proportional move (visits-1 distribution over all moves); with probability 1-p, play greedy (most-visited). This has clean semantics: greedy moves provide strong SPRT signal (directly testing strength), while proportional moves add diversity for training data. By move 15, ~80% of moves are greedy. No arbitrary nucleus boundaries — when you explore, you explore broadly; when you exploit, you play the best move. Forced wins are always played deterministically regardless of move number.

### Evaluation game data reuse

**The problem.** Each generation runs 50-800 MCTS evaluation games (candidate vs current model) purely for gating. These games run full MCTS searches with policy outputs at every move, but all position data is discarded — only W/L/D is kept. This is wasted training data. Worse, eval games actually produce *higher quality* samples than self-play: each move starts a fresh MCTS search (no subtree reuse between moves), making samples more independent.

**The solution.** Evaluation games now collect training samples by default. At each move, the MCTS root's visit-count distribution becomes the policy target and `forced_ext_pesto_balance()` provides the material scalar — the same extraction used by self-play. Samples are partitioned by which model was side-to-move: candidate's moves go to one vector, current model's moves to another.

**Selective ingestion based on gating outcome.** If the candidate wins SPRT: both sides' data is added to the replay buffer (the candidate is stronger, and the current model's data is still valid). If the candidate loses: only the current model's data is kept — the rejected candidate may be overfit or degenerate, so its policy targets could be harmful. The current model's data is always useful regardless of outcome, since by definition it represents the best known model.

**Elo tagging.** Candidate samples are tagged with the newly accepted model's cumulative Elo, and current samples with the previous model's Elo. This integrates naturally with the Elo-based strength weighting — eval data from a newly accepted model receives proportionally higher sampling weight, while data from the previous model is downweighted by exactly the measured strength gap.

**Cost-benefit.** A typical SPRT evaluation (mean ~75 games, ~100 positions per game) yields ~7,500 training samples — roughly equivalent to 75 self-play games but at zero marginal compute cost (the MCTS searches were already being run for evaluation). With 100 self-play games per generation, this represents a ~75% increase in training data per generation for free.

### Eval-only mode: skipping self-play after gen 1

Since eval games produce training data at zero marginal cost, self-play is redundant after the initial buffer seeding. The MCTS training signal (visit-count policy targets, game-outcome value targets) comes from the engine's own search regardless of opponent — the opponent only determines which positions arise. Eval games actually produce *more* data (up to 800 games vs 100 self-play) at higher quality (greedy play after ply 10, fresh MCTS search per move).

With `--skip-self-play`, the loop after gen 1 becomes: train on buffer → eval (800 games, producing training data) → gate → ingest eval data. Gen 1 always runs self-play to seed the buffer. This cuts per-generation wall time roughly in half by eliminating the self-play phase.

## 7. Performance Optimizations

### Incremental Zobrist hashing

Each move XOR-updates the hash rather than recomputing from scratch. This makes hash updates O(1) instead of O(number of pieces). Critical for MCTS where millions of positions are hashed per second.

### Clone-free legality checking

`is_legal_after_move()` checks if a move leaves the king in check without cloning the board. It applies the move, checks legality, then undoes it in-place. This avoids the allocation overhead of `Board::clone()` which was a hot path in move generation.

### Pseudo-legal move validation

`is_pseudo_legal()` validates transposition table moves without generating all legal moves. Given a move from the TT, it checks basic consistency (right color piece on source square, target square not occupied by friendly piece, etc.) in O(1). If valid, the move can be used directly.

### SeeBoard for SEE

Static Exchange Evaluation (SEE) uses a lightweight `SeeBoard` struct instead of cloning the full `Board`. `SeeBoard` contains only the minimum state needed for exchange evaluation — occupancy bitboards and piece types.

## 8. Tournament Design: Adaptive CI-Targeted Pairing

### The problem: uniform round-robins waste games

A standard round-robin plays equal games across all pairings. With 10 models (45 pairs), playing 200 games each costs 9,000 games total — but most of that budget is spent on lopsided matchups (tiered_gen17 vs vanilla_gen0) where the outcome is obvious after 10 games. Meanwhile, the close matchups that actually determine rankings (tiered_gen4 vs tiered_gen17) remain statistically uncertain.

### Solution: seed then focus

The tournament runs in two phases:

**Seed phase.** An interleaved round-robin plays a small number of games per pair (default 10). This establishes rough rankings cheaply — 450 games for 10 models. Interleaving (playing one batch per pair before starting the next round) means preliminary Elo ratings are available after the first pass through all 45 pairs.

**Focus phase.** The remaining budget targets *ranking uncertainty* directly:

1. Compute MLE Elo ratings, rank models
2. Bootstrap resample all pairwise results (200 iterations), recompute Elo each time
3. For each consecutive-rank pair (#1 vs #2, #2 vs #3, ...), measure the 95% CI width of their Elo gap
4. Play the next batch of games for the consecutive pair with the **widest CI**
5. Repeat until all consecutive CIs fall below a target (default 50 Elo) or the total game budget is exhausted

This focuses games exactly where they matter for ranking confidence. A pair separated by 400 Elo with CI±30 doesn't need more games — their relative ranking is settled. But a pair separated by 25 Elo with CI±160 needs many more games before we can confidently order them.

### Why consecutive pairs, not all pairs

Ranking confidence depends on *adjacent* comparisons. If #1 vs #2 is settled and #2 vs #3 is settled, then #1 vs #3 is transitively settled — playing more #1 vs #3 games adds no ranking information. By focusing exclusively on consecutive-rank gaps, the algorithm targets exactly the comparisons that could change the final ordering. This also means lopsided cross-type matchups (tiered vs vanilla) naturally get deprioritized once the tiered-vanilla boundary is established.

### Bootstrap details

Each bootstrap iteration resamples every pairwise result independently: given (w, l, d) outcomes for a pair, draw `w+l+d` outcomes from a multinomial with probabilities `(w/total, l/total, d/total)`. This preserves the total game count per pair while varying outcomes. MLE Elo is recomputed from scratch for each resample (1000 iterations, fast in pure Python). The consecutive gaps are measured using the *current* ranking order (not the bootstrap ranking), so the CI reflects uncertainty about whether a specific pair's gap might be negative (i.e., whether their ranking should swap).

### Pre-seeding from training eval data

During training, each generation is evaluated against the current best via SPRT (up to 800 games). For consecutive accepted generations, this produces high-quality head-to-head results using the same search config as the tournament. A seed script (`scripts/seed_tournament_from_training.py`) extracts these eval pairs from training logs and writes a pre-populated results JSON. The tournament then resumes from this seeded data, skipping hundreds of games on same-type consecutive pairs and focusing its budget on cross-type matchups (tiered vs vanilla) where no training data exists.

### Resume support

Results are saved to JSON after every batch. The adaptive phase picks up where it left off — re-run bootstrap on existing results, find the widest CI, continue. No special checkpointing needed. Pre-seeded results from training eval data are loaded identically to partially completed tournament results.

### Model-mode pairing

Each model plays with the search configuration it was trained with. Tiered models use all three tiers; vanilla models disable Tier 1 and material evaluation. The tournament script takes model directories and generation numbers as CLI arguments (`--tiered-dir`, `--tiered-gens`, `--vanilla-dir`, `--vanilla-gens`) and enforces correct per-side tier flags automatically, so mixed matchups are valid comparisons of the full training+search systems.

## 9. Applicability Beyond Chess

The three-tier decomposition is not chess-specific. The pattern — injecting exact solutions for tractable subproblems as terminal MCTS nodes — applies wherever a domain has classical solvers for subproblems:

| Domain | Tier 1 (Exact) | Tier 2 (Heuristic) | Tier 3 (Learned) |
|--------|---------------|-------------------|-----------------|
| **Chess** | Mate search, KOTH geometry | Extended quiescence (captures, checks, forks) + MVV-LVA ordering | Neural positional evaluation |
| **Mathematical reasoning** | Automated theorem provers (e.g., Lean's `decide`, `omega`) resolving subgoals | Lemma relevance ranking, proof-term similarity | Neural proof step prediction |
| **Program synthesis** | Type checking, partial evaluation, SMT solvers | API frequency heuristics | Code generation model |
| **Game playing** | Endgame tablebases, solved subgames | Domain heuristics | Value/policy networks |

The key requirement: exactly resolved nodes must be **terminal** in the MCTS tree. If a proven node can be expanded and diluted by approximate child evaluations, the proof is wasted. This terminal semantics insight is the most transferable contribution.

The value function factorization ($V = \tanh(V_{learned} + k \cdot V_{computed})$) also transfers: any domain where part of the evaluation can be computed exactly benefits from separating the learnable residual from the computable component. The learned confidence scalar $k$ provides a global trust level in the exact component, while feeding the computed result as a direct input to the value head enables per-position modulation — in chess, the value head learns to rely more on $\Delta M$ in open tactical positions and less in closed strategic ones; in mathematical reasoning, an analogous architecture could modulate trust in a theorem prover's assessment based on how prover-friendly the current subgoal is.

## 10. Related Work

Caissawary draws on and extends several lines of prior work:

- **AlphaZero** (Silver et al., 2017) established pure neural MCTS for chess, learning everything from self-play. Caissawary uses the same MCTS + self-play framework but decomposes evaluation into learnable and computable components rather than learning end-to-end.

- **MCTS-Solver** (Winands et al., 2008) propagates proven game-theoretic values (wins/losses) through MCTS trees, avoiding the dilution of exact values by approximate backups. Caissawary extends this idea by treating *any* provably resolved node as terminal — not just endgame wins/losses, but also short forced mates and KOTH geometric wins detected mid-search.

- **KataGo** (Wu, 2019) incorporates handcrafted features alongside neural evaluation, including ownership predictions and score estimation. Caissawary's $k$ scalar similarly modulates the influence of a separately computed material balance, but as a global learned parameter rather than a per-position prediction — position-dependent modulation is absorbed by the value head which receives the Q-search result as a direct input feature.

- **MT-MCTS** (Mannen & Wiering, 2012) decomposes games into subgames solved by specialized agents. Caissawary's three-tier structure is a specific instance of this: Tier 1 handles tractable subgames exactly, Tier 2 applies domain heuristics, and Tier 3 handles the uncertain residual with learned evaluation.

The primary novelty is the *combination*: terminal semantics for proven nodes (preventing value dilution), factored value function with a learned confidence scalar, and systematic subgame decomposition — integrated into a single MCTS framework.
