# Speeding up the quiescence search

The quiescence search (Tier 2) plays out captures, promotions, replies to check and one quiet check
or fork per side, tests for threats by letting the side to move pass (the threat probe), and scores
the quiet position with PeSTO. It is timed on one thread by `qsearch_bench`, using 20,000 leaf
positions logged during standard-chess self-play (the pilot's generation-28 network), split into a
tuning half and a held-out half by game. A frozen copy of the original search
(`src/search/quiescence_reference.rs`) is the reference: exact changes must return its score on
every position, checked by a checksum in the benchmark and by a test on 100 held-out leaves.

The cost is concentrated in a few positions. At the start, 1% of calls took 50% of the time, and
calls that hit the 20-ply depth limit took 95% of it; the median call took 120 µs and the mean
3,592 µs. A profile showed about 290 ns per position searched, spent mostly on making and undoing
moves (31%), move generation (23%) and memory allocation (9%), so exact changes could at best
halve the time per position, and a large speed-up needed fewer positions searched.

Times are µs per call, means of 3 runs (standard deviation under 0.5%); each row is its own build.

| Change | Tuning half | Held-out half | Median (tuning) | Positions searched per call (tuning) | Speed-up (tuning, held-out) |
|---|---|---|---|---|---|
| Start | 3,592 | 3,748 | 120 | 14,035 | |
| **Move lists reused**, one set per search depth, instead of allocated at each position | 2,849 | 2,951 | 94 | 14,035 | 1.26×, 1.27× |
| **Threat probe skipped where it cannot change the score**¹ | 1,912 | 1,945 | 72 | 9,287 | 1.88×, 1.93× |
| **No repetition bookkeeping** (the quiescence search never tests repetitions) | 1,760 | 1,750 | 67 | 9,287 | 2.04×, 2.14× |
| **Only captures generated** where no quiet move can be searched | 1,646 | 1,627 | 64 | 9,287 | 2.18×, 2.30× |

¹ The probe can only lower the score of standing pat, so it is skipped when standing pat cannot
raise alpha. Every score is unchanged; the completed flag (whether every line finished within the
depth limit, an input to the network) changes on 1.2% of positions, all from "not completed" to
"completed", because the skipped probes were the only unfinished lines.

## Pruning, which changes scores

Each rule was judged by speed and by agreement with the reference: the share of positions with the
same score, and the share whose value in the engine's units, $V = \tanh(0.326 \cdot \text{score})$,
changes by more than 0.1. Measured in one build, cumulative, 3 runs:

| Pruning added | Tuning half | Held-out half | Same score (tuning, held-out) | Value changes by more than 0.1 | Speed-up over the start |
|---|---|---|---|---|---|
| None (the exact search above, this build) | 1,632 | 1,639 | 100%, 100% | 0%, 0% | 2.2×, 2.3× |
| **Captures that lose material by static exchange evaluation skipped**² | 457 | 522 | 94.5%, 94.6% | 1.7%, 1.7% | 7.9×, 7.2× |
| **Delta pruning**: a capture skipped when the static score plus the captured piece plus 2 pawns cannot reach alpha | 247 | 290 | 92.1%, 92.6% | 2.1%, 2.1% | 14.5×, 12.9× |
| **One threat probe per side per line**³ (the default) | 148 | 153 | 86.6%, 86.1% | 3.9%, 4.1% | 24.2×, 24.4× |

² Static exchange evaluation estimates whether a capture wins material by playing out all
captures on that square, least valuable piece first. It looked for recaptures among the wrong
side's pieces and counted a final recapture that might not exist, so it marked many winning
captures as losing (with it, skipping "losing" captures changed the score of 25% of positions by more
than a pawn).
Both are fixed, with tests of exact values for each side.

³ The search's own documentation said each side gets at most one null move per line, but every
node on a line of captures ran its own probe. Now a side that has probed does not probe again
further down.

Ruled out: a depth limit of 12 instead of 20 (7.3× alone) changed the completed flag on 27% of
positions. With all three rules, the median call takes 24 µs and 96% of calls finish within the
depth limit, compared with 79% at the start.

What this does not measure: whether the changed scores make self-play stronger or weaker. The
reference is the original search, not a ground truth; a pruned score that differs may be better or
worse.

```bash
target/release/qsearch_bench leaves.txt 3 --half tune --impl ref,exact,default,see,delta200+see
```
