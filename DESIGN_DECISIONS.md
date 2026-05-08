# Design Decisions: A Scientific Journey

This document traces the evolution of Caissawary's architecture — what was tried, what failed, and why the current design emerged. The core pattern — exact subgame resolution injected as terminal MCTS nodes — applies to any domain with tractable subproblems.

### Why King of the Hill?

Standard chess is a poor testbed for Tier 1 safety gates — forced checkmates are rare in typical play. In King of the Hill (KOTH), where moving your king to a central square also wins, the dual tension of checkmate threats and king-march threats is *always in the air*. Both are tractable subgames solvable by Tier 1 gates. KOTH is an ideal stress test: the three-tier system gets far more opportunities to prove its value.

## 1. Three-Tier MCTS: Why Not Pure AlphaZero?

**The problem.** AlphaZero treats every position uniformly: expand, evaluate with the NN, backpropagate. But many positions have known answers. A mate-in-2 doesn't need a neural network — it needs a proof. Pure AlphaZero wastes its most expensive resource (NN calls) on positions where a microsecond of classical analysis gives an exact answer.

**The hypothesis.** Decompose positions into three tiers: (1) tractable subgames with exact solutions, (2) positions with useful heuristic structure, and (3) genuinely uncertain positions requiring learned evaluation. Only Tier 3 needs the neural network.

**Domain-general framing.** This decomposition applies wherever MCTS encounters tractable subproblems. The most natural next domain is mathematical reasoning: automated theorem provers (Lean's `decide`, `omega`, `norm_num`) can resolve certain subgoals exactly, while a neural policy guides high-level proof search. The confidence scalar $k$ maps directly: a global trust level in the computable component.

The pattern is always the same: compute what you can exactly, heuristically order what you understand, learn what remains.

## 2. PUCT Fix: sqrt(max(N, 1)) Instead of sqrt(N)

Vanilla AlphaZero PUCT is $U(s,a) = c \cdot P(s,a) \cdot \sqrt{N(s)} / (1 + n(s,a))$. When a node is first expanded, $N(s) = 0$, so $U$ collapses to zero for every child — the first simulation is selected by whatever tiebreaker the move generator happens to impose, which in chess is systematically biased toward structurally poor moves (a-pawn pushes, etc.). That first simulation returns a noisy, often pessimistically-biased value that gets backed up through every ancestor. Because most nodes in a tree receive only a handful of visits, this first-move noise is a meaningful fraction of the evaluation signal at the majority of nodes.

**Fix:** Replace $\sqrt{N}$ with $\sqrt{\max(N, 1)}$.

This strictly dominates vanilla PUCT:
- **Identical behavior for $N \geq 1$.** No retuning of $c$, no new pathologies.
- **Correct behavior for $N = 0$.** $U$ becomes proportional to $P(s,a)$, so the first simulation follows the policy prior — exactly what the network was trained to recommend.
- **Negligible cost.** One `max()` per selection, trivial on GPU.

Applied in both Rust (`calculate_ucb_value` in `src/mcts/selection.rs`) and CUDA (`select_child_puct` and inline selection loops in `cuda/mcts_kernel.cu`).

## 3. Tier 1: Safety Gates as Terminal Nodes

### The key insight: terminal semantics

The initial design ran mate search at expansion, then used the result as a value prior — but the node was still expanded normally. MCTS diluted the proven values: a proven mate-in-2 (+1.0) would be averaged with dozens of approximate NN evaluations (+0.3, -0.1), converging toward something much weaker.

**Solution:** Gate-resolved nodes became **terminal** — treated identically to checkmate and stalemate. No children are ever created. Every future visit re-uses the cached exact value. Proven values must be *cached*, not *mixed with approximations*. Once an MCTS node is proven, it stays proven forever.

### Mate search: exhaustive mate-in-2 + checks-only mate-in-3

Full-width mate search at all depths is too expensive for every MCTS expansion. The compromise: mate-in-1/2 exhaustive (all legal moves), mate-in-3 checks-only (keeps branching manageable). The exhaustive mate-in-2 works because branching is naturally limited — the attacker's ~30 moves each lead to positions where the defender must have *all* replies lead to mate-in-1, a condition that prunes aggressively.

Cumulative optimization brought mate search from 608 µs to 132 µs per call (4.6×): `gives_check()` pre-filtering to avoid `make_move`/`undo_move` on non-checking moves, stateless `&Board` instead of `&mut BoardStack` (no Zobrist/repetition overhead), pure minimax replacing alpha-beta (binary "does forced mate exist?" needs only short-circuit semantics), `is_legal_after_move()` before `apply_move_to_board()` (skip board clones on illegal moves), and batched per-piece checkmate detection at depth-0 leaves (king moves first for early evasion, double-check detection to skip remaining piece types).

### KOTH geometric pruning

Can the king reach {d4, e4, d5, e5} in at most 3 moves, considering blocking pieces and opponent interception? Pure geometry, no search needed. Direct king-move generation via bitboard intersection (`k_move_bitboard[king_sq] & target_mask & !friendly_occ`) yields exactly the 1-3 valid moves, avoiding full movegen. Per-node cost: 0.19 µs.

### Depth scaling

KOTH-in-4 is catastrophic (55× slower — the target mask expands to the entire board border, defeating geometric pruning). Mate-in-6 is cheap (2.2× slower) but marginal — forced mate-in-6 via pure checks is rare. Current defaults: checks-only mate-in-5 and KOTH-in-3.

## 4. Tier 2: Quiescence Search and Tactical Ordering

Tier 2's core is `forced_ext_pesto_balance()` — an extended PeSTO quiescence search (depth 20) at every MCTS leaf, computing $\Delta M$: tapered positional+material evaluation after forced tactical sequences resolve. Unlike simple piece counting, PeSTO uses Texel-tuned piece-square tables (RofChade values) accounting for piece placement. This classical alpha-beta search detects hanging pieces, discovered attacks, and forced promotion lines. $\Delta M$ feeds into the value function both as a direct input to the value head's FC layer and via the additive $k \cdot \Delta M$ term.

### Extended quiescence: beyond pure captures

The extended Q-search adds three innovations beyond captures/promotions:

**Tactical quiet moves (budget-limited).** Each side gets one non-capture tactical move per search: non-capture checks (via `gives_check()`), pawn forks (capture bitboard attacks 2+ enemy pieces), or knight forks (attack bitboard hits 2+ of {R, Q, K}). One per side keeps node counts manageable while catching the most common patterns.

**Null-move threat detection with "deny first choice."** Standard stand-pat assumes you can safely do nothing — a lie when pieces are hanging. The extended Q-search passes the turn, evaluates each opponent response, then denies the opponent's first choice (you'd retreat the best-threatened piece). With 2+ threats (fork), you save one piece but the opponent gets their second-best capture.

**Mystery-square recapture.** After denying the first-choice capture in a fork, the saved piece "teleports" to recapture on the second-choice square — modeling retreat + recapture (e.g., Bd3 saving bishop, then Bxe4 recapturing the fork pawn). This bends geometry but correctly models practical play.

**Center fork trick benchmark** (1.e4 e5 2.Nc3 Nf6 3.Bc4 Nxe4 4.Nxe4, Black with d5 fork):

| Q-search variant | Score (Black POV) | Assessment |
|:---|---:|:---|
| Basic (captures only) | -2.90 | Misses d5 entirely |
| Extended (checks + forks) | -2.90 | Finds d5 but can't evaluate it |
| + Null-move (deny first choice) | +0.10 | Correct — fork equalizes |

Validated on 200+ positions from the 2026 FIDE Candidates (Sindarov, Rounds 1-5): the extended Q-search diverges from basic in 8% of positions, and all major divergences correspond to real tactical features — knight check-forks, connected threats, hanging pieces after forced sequences.

### Iterative widening

The full extended Q-search can explode (130K nodes worst case). SEE pruning was tried but damaged accuracy — it correctly prunes QxP-when-defended but also prunes NxP-when-defended, which might initiate a fork trick. The solution is iterative widening (controlling captures-per-node): principal exchange first (1 capture, straight line), then cap=1 (adds checks), then cap=2+ (adds forks/null-move), stopping when a node budget is exceeded. Each level gives a complete, valid score; later levels refine it. Stand-pat at every level prevents forcing bad captures.

### Earlier attempt: Q-search grafting

The first approach grafted Q-search trees onto the MCTS tree at expansion. Too expensive (~10× slower expansions), noisy values from poor alpha/beta bounds, and high complexity. **The lesson:** run Q-search once at leaf evaluation time as `forced_ext_pesto_balance()` — simpler, cheaper, and gives the value function a clean signal.

## 5. The Value Function: V = tanh(V\_logit + k · ΔM)

### Separate what's learnable from what's computable

Standard AlphaZero learns everything end-to-end, spending millions of games rediscovering that a queen is worth more than a pawn. The key idea: factor evaluation into computable and learnable components.

$$V_{final} = \tanh(V_{logit} + k \cdot \Delta M)$$

- $V_{logit}$ (NN output, unbounded): positional assessment — piece activity, king safety, pawn structure
- $\Delta M$ (computed exactly): PeSTO evaluation after forced exchanges via `forced_ext_pesto_balance()`
- $k$ (NN output, positive): how much material should matter in this position

The NN returns $V_{logit}$ unbounded (not pre-squashed) so it can learn arbitrary positional offsets that interact smoothly with the material term. The final $\tanh$ constrains output at the end.

### Dynamic k: learned confidence in material

$k = 0.47 \cdot \text{Softplus}(k_{logit})$ where $k_{logit}$ is a single learned scalar. The 0.47 coefficient is Texel-calibrated so PeSTO centipawn evaluations map to calibrated win probabilities. At initialization ($k_{logit} = 0$): $k \approx 0.326$.

### k architecture evolution

**v1: Per-position from backbone features.** Overfitted to specific positions — deep features encoded too much detail.

**v2: Separate shallow input network** (3×3 conv → BN → GAP → FC). Too shallow — a "bag of local patterns" that couldn't reliably detect king safety, pawn structure, or open files.

**v3: Handcrafted features + king patches** (12 scalar features + two 5×5 king-centered patches, ~22K params). Worked well but dynamic king-patch extraction was CUDA-hostile, and the position-dependent modulation could be absorbed by the value head.

**v4 (current): Global scalar k + q_result as value head input.** K-head removed entirely (−21,536 params). Single `nn.Parameter` scalar plus $\Delta M$ concatenated as a 65th input to the value head FC (widened 64→65). The additive path ($k \cdot \Delta M$) sets a global baseline; the learned path (FC weights on $\Delta M$) provides position-dependent modulation. Making $k$ position-independent is acceptable because the position-dependent part is learned implicitly through the value head.

### Classical fallback

With no NN: $V_{logit} = 0$, $k = 0.326$, so $V_{final} = \tanh(0.326 \cdot \Delta M)$. A freshly initialized network produces identical values, ensuring smooth bootstrapping.

## 6. OracleNet Architecture

**SE-ResNet backbone.** Configurable depth and width (default: 6 blocks, 128 channels, ~2M params). SE attention in each block learns per-position channel importance.

**Board encoding.** 17×8×8 in side-to-move perspective. Planes 0-5: STM pieces; 6-11: opponent; 12: en passant; 13-16: castling (STM-relative). When Black moves, ranks are flipped — the network only learns from one perspective.

**Policy head.** 73 planes × 64 squares = 4672 logits (AlphaZero encoding: 56 queen slides + 8 knight + 9 underpromotions).

**Value head.** 1×1 conv → BN → flatten (64) → concat($\Delta M$) → FC(65→256) → FC(256→1). Output is unbounded $V_{logit}$.

**Zero initialization.** All output layers initialized to zero: policy = uniform exploration, value = pure PeSTO evaluation, $k$ = 0.326.

## 7. Training Pipeline Evolution

### Gating: from fixed threshold to SPRT

**v1: Fixed 55% winrate.** Noisy — barely significant in 100 games. **v2: One-sided binomial (p<0.05).** Better rigor but still problematic with small samples. **v3 (current): SPRT.** Sequential Probability Ratio Test with trinomial GSPRT (matching fishtest). Clear improvements terminate in ~30 games; marginal cases use up to 800. Saves wall-clock time while maintaining statistical rigor.

### Replay buffer: from clear-on-accept to Elo-weighted

**v1: Clear buffer on model acceptance.** Caused severe overfitting on the first post-acceptance iteration. **v2: Sliding window (FIFO).** Stable but uninformed. **v3: Recency-weighted.** Exponential half-life — arbitrary and disconnected from actual strength. **v4 (current): Elo-based strength weighting.** Each acceptance chains the SPRT winrate into cumulative Elo. Training data is weighted by expected score: $w_i = n_i \cdot 2 / (1 + 10^{(\text{Elo}_{max} - \text{Elo}_i) / 400})$. A 55% winrate gap (~35 Elo) yields ~1.2:1 ratio (not 7:1). No data is fully discarded. Eval game data from both sides is ingested, with the losing side tagged at lower Elo.

### Training data scheduling

**v1: Fixed minibatch count.** Overfitting early (small buffer), underfitting late (large buffer). **v2: Adaptive count targeting ~1.5 epochs.** Better, but random sampling with replacement still caused redundant/missed positions. **v3 (current): Epoch-based with Elo-weighted inclusion.** Each epoch iterates the buffer once; each position's inclusion probability is the odds ratio of its model's expected score. Max-Elo data: 100% inclusion. 100 Elo weaker: 56%. 200 Elo: 32%. Every max-Elo position is trained on exactly once per epoch — no wasted or redundant data.

### Adaptive epochs

90/10 train/validation split with patience-1 early stopping, up to `--max-epochs` (default 10). Early generations (small buffers) typically select 2-3 epochs; later generations with large buffers need only 1. Automatically adjusts where a fixed count would either underfit early or overfit late.

### Train-from-latest

When a candidate is rejected by SPRT, it's probably slightly better than the current best — just not provably so. Training the next generation from the last *accepted* checkpoint wastes this incremental progress. Initial experiments confounded train-from-latest with other changes, leading to a temporary revert. Controlled comparison showed train-from-best caused repeated training from the same stale checkpoint (gens 3-5 all from gen_2.pth, progressively worse). **Current approach:** always resume from the most recent candidate. Early stopping prevents cumulative overfitting.

### Data augmentation

Chess has horizontal flip symmetry when castling rights are absent; pawnless castling-free positions have the full D4 dihedral group (8 transforms). All equivalent transforms are expanded during loading (not randomly sampled). Positions without castling get 2× training signal; pawnless endgames get 8×. This overweighting compensates for their underrepresentation in self-play data and is especially valuable in KOTH where voluntary loss of castling rights is strategically important.

### Optimizer

Adam → Muon (Momentum + Newton-Schulz orthogonalization). Converges faster on ResNet-style architectures by normalizing gradient updates. Falls back to AdamW for 1D parameters.

### Multi-variant training: tried and retired

Trained three variants per generation (policy-only, value-only, all-heads) to diagnose SPRT failures. Over 23 generations, policy-only was consistently worst (avg WR 0.474). All-heads and value-only tied. The diagnostic value was real but 3× eval cost was not justified. **Current: single-variant (all-heads only).**

### Evaluation: proportional-or-greedy mixing

**v1: Fixed cutoff** (proportional for 10 plies, then greedy). Simple but arbitrary. **v2: Top-p nucleus.** Too noisy — 14+ consecutive SPRT rejections at plateaus. **v3 (current): Proportional-or-greedy mix.** With probability $0.90^{(\text{move}-1)}$, play proportional; otherwise greedy. By move 15, ~80% greedy. Clean semantics: greedy tests strength, proportional adds diversity. Forced wins always played deterministically.

### Evaluation game data reuse

Eval games (50-800 per generation) run full MCTS searches but traditionally discard position data. Now both sides' training samples are collected. If the candidate wins SPRT, both sides' data enters the buffer; if rejected, only the current model's data is kept. A typical SPRT evaluation yields ~7,500 training samples at zero marginal cost — ~75% more data per generation.

### Eval-only mode

Since eval games produce training data for free, self-play is redundant after initial buffer seeding. With `--skip-self-play`, the loop after gen 1 becomes: train → eval (up to 800 games, producing data) → gate → ingest. Cuts per-generation wall time roughly in half.

## 8. GPU-Resident MCTS

### Motivation

In the CPU implementation, each MCTS simulation requires a CPU→GPU round-trip for NN inference. At 400 sims/move, inference dominates wall time (84%). The solution: move the entire MCTS loop onto the GPU as a persistent CUDA kernel — select→expand→evaluate→backup with no CPU interaction during search.

### Architecture simplification

Deep symbolic searches (mate-in-5, KOTH-in-3) are replaced by lightweight GPU-side gates: mate-in-1 (exact terminal detection) and KOTH-in-1 (king on center or one move away). Deeper patterns are expected to be learned by the NN. The quiescence search uses the principal exchange variant (follow single best MVV-LVA capture — straight line, ~1-5 nodes, zero warp divergence, register-resident PeSTO eval). Full iterative-widening Q-search remains available CPU-side.

### SE-ResNet optimization history

The GPU forward pass went through four kernel-level implementations and three MCTS-level scheduling iterations.

**Forward kernel evolution (v0 → v2).** The initial warp-cooperative path (32 threads, im2col to global scratch) proved correctness but was 325× slower than projected. Block-cooperative scalar (256 threads, shared-memory activations, direct conv3x3) achieved 9.52 ms — 13.7× faster. TC im2col (wmma FP16 GEMM replacing scalar conv) reached 3.65 ms but im2col gather still dominated. The TC shifted-copy path decomposes conv3x3 into 9 dense GEMMs by kernel position, building shifted copies via bit operations only — no integer division, no staging. Loop reorder (shifts outer, N_tiles inner) reduced shifted copies from 36→9 per conv. Result: 1.51 ms forward pass, 1.64 ms/sim, **79 samples/sec** at 36 concurrent games (v2 baseline).

**v3 — FP16 inter-layer activation storage.** A late discovery: the shifted-copy path was already FP16+WMMA+FP32-accumulator at the matmul level; the remaining FP32 surface was inter-layer activation buffers (`buf1`/`buf2`, 16 KB each). Converting these to FP16 dropped per-block smem from 81 KB to 49 KB and removed an FP16/FP32 round-trip at every conv boundary. Throughput gain: **86.4 samples/sec** (~9% over v2). The bigger win was the smem headroom that v4 needed.

**v4 — true batched conv primitive (B=2).** A new device function `oracle_net_forward_block_b2` plus underlying `block_conv_3x3_shifted_b2` load each WMMA weight tile A *once* per (n_tile, k_tile) and apply it to both batch elements before advancing — true weight-load amortization. The "naive" sequential B-loop design (call the existing forward twice) was rejected because it provides no per-FLOP improvement. Microbenchmark in isolation (single block, 1000 iterations, CUDA-event timing): **1.19× per-sim NN improvement** at B=2. Smem 99 KB (just under cap). Three runs identical to four sig figs — confirms weight-bandwidth was a real fraction of NN time.

**v5 — 2-explorer virtual-loss scheduling.** Wires v4 into the production MCTS path. Each block runs 2 explorers per round: thread 0 does SELECT for both sequentially (so virtual loss steers the second away from the first's path); EXPAND and BACKUP parallelize across warps 0 and 1; the NN forward is one batched `oracle_net_forward_block_b2` call. The virtual-loss infrastructure (atomic `apply_virtual_loss`, `backprop_value` that decrements VL on backup) was already present from earlier work — v5 just needed a second explorer that exercises it. Per-block static smem grew ~1.1 KB; `sh_path` shrunk from 256 to 128 deep to fit the per-block 99 KB cap. End-to-end speedup: **101.4 samples/sec at 6×128**, **188.6 at 6×64** — same ~1.17× ratio at both architectures, confirming the mechanism is architecture-agnostic.

**Cumulative result (v2 → v5):** 79 → 101.4 samples/sec at 6×128 production (1.28×), 161.9 → 188.6 at 6×64 apples-to-apples (1.17×).

### Transformer: hybrid TC/scalar

A pre-LayerNorm transformer (current: 12 layers, D=128, 4 heads, FFN 128→512→128, ~2.4M params) runs as an alternative architecture via `--arch transformer`. Same input/output/tiered MCTS integration.

The forward pass uses a hybrid TC/scalar approach. Q/K/V projections and FFN use wmma Tensor Cores (FP16 input, FP32 accumulation). QK^T, softmax, and attn×V stay scalar due to workspace memory constraints. The key enabler: `buf_out` (LayerNorm output) is stored in FP16, freeing 16 KB of shared memory for a dedicated TC staging region while keeping total shared memory at 89 KB (under the 99 KB hardware limit). The FP16 intermediate is numerically safe — it only feeds linear projections which convert to FP16 anyway for wmma. LayerNorm and softmax process all 64 tokens in parallel via 4-thread warp-shuffle groups.

### Two architectures, two scaling regimes

GPU-resident inference (one block per tree, 256 threads) wins for small models where activations fit in shared memory. For larger models, batched inference from the host with cuBLAS/cuDNN would better utilize GPU compute.

**Smem-bound batch ceiling.** At 6×128 with FP16 activations the per-block batch is capped at B=2 (97 KB of the 99 KB per-block cap). Each output-channel of a 3×3 conv depends on all 128 input channels, so we cannot reduce the materialized activation tensor between layers without either dropping precision (FP8) or streaming activations through global memory — both of which give up the persistent-activations-in-smem advantage we've built around.

**Apples-to-apples benchmark vs Lc0+Maia.** A temporary 6×64 experiment branch (matching Maia's architecture) measured Caissawary v5 at 188.6 samples/sec vs Lc0+Maia at 368 — about 51% of Lc0's throughput at the same network. The remaining ~1.95× gap decomposes roughly as: ~1.5× from Lc0's larger batch (B=30+ via virtual loss + cross-block batching) → better Tensor Core utilization, ~1.2× from CUTLASS-tuned conv kernels, ~1.15× from CPU/GPU overlap (CPU walks tree while GPU does forward). The first two are reachable with persistent-kernel work; the third is fundamentally incompatible with our design.

### GPU self-play

A host-side game loop calls `gpu_mcts_eval_trees_budget` (or the v5 `_p2` variant) for batches of 36 concurrent games on RTX 5060 Ti. Each batch runs one MCTS evaluation per active position (one block per game, 256 threads in single-explorer mode; 256 threads with 2 explorers per round in v5 p2 mode). The host handles move application, proportional-or-greedy sampling, game termination (checkmate, stalemate, 50-move, threefold repetition via Zobrist), and training data recording.

Selfplay opt-in: `SelfPlayConfig.use_vloss_p2 = true` switches to the v5 path. `bench_throughput 100 200 36 p2` runs the throughput benchmark in v5 mode; without the `p2` argument it uses the single-explorer baseline (untouched, still 86.4 samples/sec).

### Future directions

The persistent-kernel approach has plateaued faster than projected. Each successive optimization — v3 (~9%), v4 (1.19× microbench), v5 (1.17× end-to-end) — is harder than the last, and the smem-bound batch ceiling at B=2 caps the next round of gains. Realistic next steps in order of leverage:

- **FP8 activations** to break through B=2 → B=4. Adds quantization-aware training and an FP8 WMMA path. Estimated ~1.15–1.25× more throughput; brings 6×128 production to ~120 samples/sec.
- **Cross-block batching** (Lc0-style architecture) — splits MCTS workers from a global inference engine that batches across many blocks. This is essentially rebuilding around Lc0's design pattern and gives up the persistent-kernel advantage; the upside is uncapped batch scaling.
- **Hybrid CPU NNUE + GPU big-net** — a fast CPU evaluator (NNUE-style or small distilled net) provides immediate provisional values at every leaf so the tree never stalls on GPU latency; the slow GPU evaluator refines values + provides policy priors in batches. Conceptually similar to speculative decoding. Calibration between the two evaluators is the load-bearing risk; distilling the small net from the big net's value head is the strongest version (eliminates calibration entirely). Could in principle exceed Lc0 at equal hardware, which the persistent-kernel approach essentially can't.

The persistent-kernel design's "no CPU-GPU sync per sim" advantage is real but bounded — at B=2 it costs us ~50% of theoretical Lc0 performance even at apples-to-apples architecture. Closing the rest of the gap requires either FP8 + more kernel work (~80% of Lc0 reachable) or a fundamentally different architecture.

### Test coverage

157 CUDA + 24 Python + 609 Rust = ~790 total tests.

## 9. Tournament Design: Adaptive CI-Targeted Pairing

A standard round-robin wastes games on lopsided matchups. The tournament runs in two phases:

**Seed phase.** Interleaved round-robin with a small number of games per pair (default 10) to establish rough rankings.

**Focus phase.** Bootstrap resampling (200 iterations) computes 95% CI widths for each consecutive-rank Elo gap. The next batch targets the pair with the widest CI. Games focus exactly where they matter — a pair separated by 400 Elo with CI±30 needs no more games, while a pair at 25 Elo with CI±160 does. Consecutive pairs only, because transitive ordering makes non-adjacent comparisons redundant.

Pre-seeding from training SPRT data skips hundreds of games on same-type pairs. Results save to JSON after every batch for resume support. Each model plays with the search configuration it was trained with (tiered models with tiers, vanilla models without).

## 10. Applicability Beyond Chess

The three-tier decomposition is not chess-specific. The pattern applies wherever a domain has classical solvers for subproblems:

| Domain | Tier 1 (Exact) | Tier 2 (Heuristic) | Tier 3 (Learned) |
|--------|---------------|-------------------|-----------------|
| **Chess** | Mate search, KOTH geometry | Extended quiescence + MVV-LVA | Neural positional evaluation |
| **Mathematical reasoning** | Theorem provers (Lean's `decide`, `omega`) | Lemma relevance ranking | Neural proof step prediction |
| **Program synthesis** | Type checking, partial evaluation, SMT | API frequency heuristics | Code generation model |
| **Game playing** | Endgame tablebases, solved subgames | Domain heuristics | Value/policy networks |

The key requirement: exactly resolved nodes must be **terminal** in the MCTS tree. If a proven node can be expanded and diluted by approximate child evaluations, the proof is wasted.

The value function factorization ($V = \tanh(V_{learned} + k \cdot V_{computed})$) also transfers: any domain where part of the evaluation can be computed exactly benefits from separating the learnable residual from the computable component. The learned confidence scalar $k$ provides a global trust level in the exact component, while feeding the computed result as a direct value head input enables per-position modulation.

## 11. Related Work

- **AlphaZero** (Silver et al., 2017): pure neural MCTS. Caissawary uses the same framework but decomposes evaluation into learnable and computable components.
- **MCTS-Solver** (Winands et al., 2008): propagates proven game-theoretic values through MCTS trees. Caissawary extends this to *any* provably resolved node — not just endgame W/L, but also forced mates and KOTH geometric wins detected mid-search.
- **KataGo** (Wu, 2019): incorporates handcrafted features alongside neural evaluation. Caissawary's $k$ scalar similarly modulates material influence, but as a global learned parameter with position-dependent modulation absorbed by the value head.
- **MT-MCTS** (Mannen & Wiering, 2012): decomposes games into subgames solved by specialized agents. Caissawary's three-tier structure is a specific instance.

The primary novelty is the *combination*: terminal semantics for proven nodes, factored value function with learned confidence, and systematic subgame decomposition — integrated into a single MCTS framework.
