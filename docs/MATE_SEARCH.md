# Speeding up the mate search

The mate search is timed on one thread by `mate_bench`, using 20,000 positions from the pilot's
logged evaluation games (generations 20–29), each extended by 1–3 random moves to imitate search
leaves. The answer for every position must stay the same after each change, checked by a checksum
in the benchmark and by a test that compares the search with a slow reference solver on 300 of
these positions. Times are µs per position, means of 5 runs (standard deviation under 0.4%;
rebuilding identical code shifts them by about 1%).

| Change | Checks, mate in 5 | Checks, mate in 6 | Checks, mate in 3, one quiet move | Every move, mate in 2 |
|---|---|---|---|---|
| Start | 44.7 | 151.6 | 78.7 | 29.0 |
| **Move lists reused**, one per ply, instead of allocated at each position | 37.6 | 127.8 | 63.6 | 25.7 |
| **Checking moves generated directly** | 22.9 | 81.9 | 32.2 | 26.1 |
| **Only moves out of check generated**, and the checkmate test stops at the first legal one | 18.4 | 70.2 | 28.1 | 25.9 |
| **Checking moves tried in order of the enemy king's escape squares**² | 16.0 | 56.0 | 28.7 | 25.9 |
| **Positions already searched remembered** (a transposition table)² | 12.0 | 32.0 | 28.1 | 26.0 |
| Total speed-up | 3.7× | 4.7× | 2.8× | 1.1× |

² Absolute times drifted between later runs by more than their standard deviation (up to 16% for
mate in 6), so these rows apply the change measured within one run to the row above: −12.8%,
−20.2%, +2.1% and +0.1% for the move order, and −25.2%, −42.9%, −2.2% and +0.5% for the table
(which includes the defender's captures by its most valuable piece first, −0.1% to −0.5%).

- **Checking moves.** In a checks-only search the attacker plays only checks, about 3 of a
  position's 35 moves, but the search generated all of them and tested each with `gives_check`
  (23% of the time). Direct checks now come from intersecting each piece's reachable squares with
  the squares from which that piece would attack the enemy king. Moves that could uncover a check,
  promotions, en passant and castling are rare and are still tested one by one, so the generated
  set is exactly the one filtering gave.
- **Moves out of check.** A defender in check can only move the king, capture the checking piece or
  block it. Generating only these replaced generating every move and testing each for legality.
  Their order matters: listing king moves first made the search visit 5× more positions, because
  captures of the checking piece refute most mating attempts soonest; captures, then blocks, then
  king moves restored it.
- **A false mate.** The old checkmate test treated an enemy pawn diagonally behind the king as a
  second checker and declared mate without trying blocks or captures. It reported a mate in 2 that
  does not exist in 1 of the 20,000 positions; tier 1 used the same code.
- **Searches over every move** gained little: their time goes to making moves and testing their
  legality, which these changes do not touch.

**Move order.** The search stops at the first mate at the attacker's move and at the first
refutation at the defender's, so the order of moves changes how many positions it visits but never
its answer. 77 pairs of an attacker and a defender order were compared on half of the 20,000
positions, all giving identical answers, and the best checked on the other half:

| Attacker order, defender order | Positions searched, mate in 5 | Time, mate in 5 | Time, mate in 6 |
|---|---|---|---|
| As generated (pieces in a fixed order), as generated (captures, blocks, king moves) | | 18.9 µs | 72.2 µs |
| As generated, captures, promotions, blocks, king moves | +0.0% | +8.6% | +8.0% |
| Double checks, captures, promotions, queen moves closest to the king, knight moves, the rest closest first; captures, promotions, blocks, king moves | −5.6% | +27.0% | +28.5% |
| Farthest from the king first, as generated | +13.0% | +43.5% | +56.4% |
| As generated, king moves first | +581% | +595% | +1,087% |
| Fewest king escape squares after the move (move made), as generated | −36.8% | +25.5% | +7.1% |
| Fewest escape squares, estimated without making the move, as generated | −19.6% | −2.3% | −8.9% |
| **As above for checking moves only, not on the attacker's last move**, as generated | **−19.0%** | **−13.4%** | **−19.8%** |
| The same, on the held-out half | −17.2% | −12.1% | −18.5% |

Differences are relative to the first row, from runs on the same half (5 runs each, standard
deviations under 0.2%).

- **Computing an order costs more than most orders save.** Sorting alone, with no change in the
  positions searched, cost 8.6%, and every order built from move types (double checks, captures,
  piece, distance to the king) took longer than generated order.
- **What pays is the king's escape squares**: a mating attempt that leaves the enemy king fewer
  squares to run to is more likely to mate. Counting them by making each move halved the positions
  searched at mate in 6 but doubled the cost per position. An estimate from one map of the
  squares the attacker controls, plus the moved piece's attacks from its new square, keeps most of
  the saving at a fraction of the cost.
- **Order matters only where a search stops early.** On the attacker's last move every checking
  move is tried unless one mates, so sorting there cost more than it saved; dropping it turned
  −2.3% into −13.4%. With a quiet move per line or every move searched, lists are long and stop
  rarely, so they stay as generated.
- **The defender's generated order was beaten only slightly**: captures of the checking piece,
  then blocks, then king moves, with captures by the most valuable piece first (rather than pawns
  first) 0.2–0.6% faster.
- **Orders in which moves are generated** cost nothing, but none of 72 combinations helped beyond
  that: generating queen checks first was 1.5–2.3% slower at mate in 5, capturing checks first
  10–11% slower, and sorting only queen checks by escape squares 11% slower than sorting every
  check.
- **Trying first the move that ended the previous sibling's search** (the killer heuristic, common
  in chess engines) was 44% slower at mate in 5: it overrides the escape-square and
  captures-first orders, which were already better.

**Remembering positions.** Checking lines revisit positions often: a check, a king step and a
check back return to an earlier position, and different orders of the same checks meet. The
search now keeps, for the duration of one search, the result for each position it has searched
with at least two moves left: 16,384 entries of 16 bytes (256 KB) per search thread, replaced on
collision. A mate found with fewer moves left also answers a search with more, and no mate with
more moves left answers one with fewer. This cut positions searched at mate in 5 by 33% and time
by 25%; at mate in 7 time fell 59%. On positions where a quiet move is allowed or every move is
searched, it gained 0–2%.

**Benchmark positions.** The positions above extend logged games by random moves. To check that
they represent what the search actually sees, `self_play` can log every Nth position at which it
searches a leaf for mates (`KF_LOG_MATE_LEAVES=<file>:<N>`), with its game. 20,000 positions
sampled from 1,920 games of the generation-28 model, split into halves by game, ranked the 72
generation orders almost exactly as the old positions did (Spearman correlation 0.995 at mate in
5). On the half not used for tuning, the table cut time by 26.6% at mate in 5, 44.6% at mate in
6 and 64.2% at mate in 7.

**In self-play**, with endgame tables on and two runs each (± half their difference), the faster
search makes no difference at the default mate in 5, where the GPU is the bottleneck: 764.5 ± 0.7
moves per second, compared with 764.9 ± 2.5. At mate in 7, where the old search ran the CPU out
of spare time, it raises throughput 18%, from 643.3 ± 0.6 to 760.0 ± 0.3, and cuts CPU time per
move from 35.6 to 21.5 ms. Adding the table left mate in 5 unchanged (761.8 ± 0.9, compared
with 762.4 ± 1.4) and raised mate in 7 from 759.3 ± 0.6 to 767.8 ± 0.7, now 0.8% faster than mate
in 5; CPU time per move at mate in 7 fell from 21.4 to 19.8 ms. Deeper mate searches are now
nearly free, which the rerun of the depth sweep will measure.
