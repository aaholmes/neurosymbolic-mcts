# Result in King of the Hill

King of the Hill is a chess variant in which moving the king to one of the four centre squares also wins. It was used for the first experiments because positions with exact solutions are common in it. The tiered engine also searches for a forced king march to the centre in up to 3 moves (Tier 1), and endgame tables are not used, because reaching the centre changes the result.

Both engines were trained by self-play in King of the Hill, a variant chosen because positions
with exact solutions are common in it ([DESIGN_DECISIONS.md](../DESIGN_DECISIONS.md)). Both use the
same network (6 residual blocks, 128 channels, ~2M parameters) and 200 search simulations per move;
the baseline uses the neural network alone, as AlphaZero does.

After 18 generations the tiered engine rates **2663 Elo [2565, 2763]**, and the baseline after 28
generations **2038 [1957, 2111]**.

![Elo by training generation for both runs](../tournament_results_800eval_elo_plot.png)

Ratings come from 6,579 games over 17 pairings, fitted with the Bradley–Terry model, with 95%
bootstrap intervals (1,000 resamples) relative to the baseline's first generation. Three things
limit what this shows:

- **Compute is not matched.** Both engines get 200 simulations per move, but the tiered engine's
  classical search is extra computation that is not counted.
- **The two runs are joined by one 100-game match**: the tiered engine before any training (its
  network initialized to zero, so it evaluates positions classically) versus the baseline after 28
  generations, +30 Elo [−20, +74]. Each run was rated as a ladder of adjacent generations, and no
  bootstrap interval on the gap between the final models has been computed.
- **Part of the gap may be a head start**: the untrained tiered engine is already about level with
  the fully trained baseline.

A comparison in standard chess at equal wall-clock time, including the untrained tiered engine, is
the next measurement (see the [README](../README.md#next-steps)).

<details>
<summary>All 18 ratings</summary>

| Model | Elo | 95% interval |
|---|---|---|
| tiered_gen18 | 2663 | [2565, 2763] |
| tiered_gen12 | 2595 | [2502, 2692] |
| tiered_gen8 | 2542 | [2448, 2633] |
| tiered_gen6 | 2478 | [2391, 2567] |
| tiered_gen5 | 2347 | [2260, 2428] |
| tiered_gen4 | 2312 | [2223, 2395] |
| tiered_gen3 | 2228 | [2144, 2306] |
| tiered_gen1 | 2167 | [2082, 2242] |
| tiered_gen0 | 2068 | [1984, 2148] |
| vanilla_gen28 | 2038 | [1957, 2111] |
| vanilla_gen23 | 1996 | [1920, 2063] |
| vanilla_gen21 | 1866 | [1801, 1933] |
| vanilla_gen13 | 1833 | [1772, 1902] |
| vanilla_gen10 | 1794 | [1734, 1859] |
| vanilla_gen7 | 1726 | [1672, 1781] |
| vanilla_gen4 | 1639 | [1598, 1684] |
| vanilla_gen1 | 1600 | [1566, 1636] |
| vanilla_gen0 | 1500 | (anchor) |

`tiered` is this engine and `vanilla` the baseline.

</details>
