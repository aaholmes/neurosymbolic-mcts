# How the search works

The engine evaluates each position with the cheapest method that can evaluate it. This page gives the details; the [README](../README.md) gives the overview.

Each position is handled by the cheapest method that can evaluate it:

| Tier | Method | Result | Mean cost |
|---|---|---|---|
| 1 | Exact search for a forced mate in up to 5 moves (checking moves only); in King of the Hill also a forced king march to the centre in up to 3 | Proven value | 132–149 µs³ |
| 2 | Quiescence search: captures and promotions are played out until the position is quiet, with all replies to check, one quiet check or fork per side, and a test for threats made by letting the side to move pass; the quiet position is scored with PeSTO, a standard hand-tuned evaluation of material and piece placement | Heuristic value | 8 µs |
| 3 | Neural network | Learned value | 1,346 µs |

Costs were measured with `profile_engine` (4 games played one after another,
200 simulations per move). The Tier 3 cost is the time per network call as the search sees it,
including up to 0.5 ms that the inference server waits for other requests to fill a batch.

³ Measured before the mate search was sped up ([MATE_SEARCH.md](MATE_SEARCH.md)), so the current cost is lower.

A position evaluated exactly in Tier 1 is terminal, like checkmate or stalemate, so its proven value
propagates up the tree without being averaged with approximate evaluations of its children.

The final value combines the network with the classical score:

$$V = \tanh(V_{\text{net}} + k \cdot \Delta M)$$

where $\Delta M$ is the Tier 2 score, $V_{\text{net}}$ is the network's unbounded output (which also
receives $\Delta M$ as an input), and $k$ is a learned scalar for how much to trust material,
initialized to 0.326. With no network, $V = \tanh(0.326 \cdot \Delta M)$; this is the untrained
engine in [VARIANT_RESULT.md](VARIANT_RESULT.md). The baseline ("vanilla") runs with Tiers 1 and 2 off: no exact
search, and the network receives $\Delta M = 0$.

Either mode can also use two cheap exact results, so that the network does not have to learn them:

- **Forced mates.** By default, `self_play` and `evaluate_models` look for a mate in up to 5 moves
  in which every attacking move gives check. In network-only mode, a leaf where such a mate exists
  gets the value +1 (tiered search already finds these). The search answers only whether a mate
  exists, not which move mates or how fast, so it can stop at the first mating line of any length.
  It tries first the checks that leave the enemy king fewest escape squares, and for the defender
  captures of the checking piece, then blocks, then king moves, and it remembers the result for
  each position it has searched; it takes about 13 µs per position.
  Tier 1 runs the same search at leaves, to its own depth, and at the root also finds the
  mating move. `--mate-checks N` sets the depth, `--mate-in N` searches every move up to mate in N,
  `--mate-quiet M` allows M non-checking attacking moves per line, and `--no-mate-search` turns it
  off.
- **Endgame tables.** With `--syzygy <dir>`, a leaf with 5 or fewer pieces and no castling rights
  gets its value from Syzygy endgame tables (win/draw/loss tables for every such position, 396 MB).
  Wins and losses that the fifty-move rule would turn into draws count as draws. The tables are not
  used in King of the Hill, where reaching the centre changes the result.

Either kind of position also ends a self-play or evaluation game with its exact result, so no
training position of either kind is recorded. A model should be played with the settings it was
trained with; models trained before the mate search became the default need `--no-mate-search`, or
`--mate-in 1` for the standard-chess pilot.

Moves are selected with AlphaZero's rule (PUCT, a variant of the upper confidence bound for trees),
with one change:

$$U(s,a) = c \cdot P(s,a) \cdot \frac{\sqrt{\max(N(s), 1)}}{1 + n(s,a)}$$

AlphaZero uses $N(s)$, the visit count of the parent, in place of $\max(N(s), 1)$. On a node's first
visit $N = 0$, which makes the exploration bonus zero for every move, so the first move tried is
whichever the move generator lists first. With $\max(N, 1)$ the bonus is proportional to the
network's prior $P(s,a)$ from the first visit, so the network's top choice is tried first. For
$N \geq 1$ the two rules are identical.

After each move the subtree under the move played is kept for the next search. Its visits count
toward the next move's budget, so "200 simulations per move" means the root reaches 200 visits,
not that 200 new simulations run.
