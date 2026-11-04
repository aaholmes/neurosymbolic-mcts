# Training

Two loops are available. Both train on positions from self-play, with a policy target (the
search's visit counts at the root) and a value target (the game's result). The search visits every
root move at least once, so the target and the choice of the move played both use each move's
visits minus one: a move whose only visit showed it was bad gets no weight.

**In generations** (`orchestrate.py`), as in AlphaGo Zero: self-play with the current model, train
a candidate on a replay buffer, and accept the candidate only if it beats the current model.

- **Optimizer:** Muon (learning rate 0.02), with adaptive epochs and early stopping. It has not yet
  been compared with AdamW.
- **Accepting a new generation:** a sequential probability ratio test (SPRT) over up to 800
  evaluation games by default (200 in the pilot), which stops early when the result is clear.
- **Data:** after generation 1 the evaluation games also supply training data, so separate self-play
  is optional. Positions are weighted by the strength of the model that played them (a 200-Elo gap
  gives about 32% inclusion).

**Continuously** (`continuous.py`), as in AlphaZero and KataGo: one network, updated while
self-play runs, with no acceptance step.

- **Self-play** keeps a fixed number of games going (`--concurrent-games`). Each new game loads the
  newest weights the trainer has published (`self_play --watch <pointer file>`), so a game never
  mixes two networks. Game files are written under a temporary name and renamed when complete, and
  their names record the weights version.
- **The trainer** moves finished games into a sliding window of recent positions (`--window`) and
  trains so that each generated position is used `--replay-ratio` times on average. It publishes
  weights every `--publish-every` steps.
- **If training falls behind** by more than `--max-owed` steps, self-play is paused until it
  catches up, so the replay ratio holds whatever the relative speeds of the two.
- **Logged at each publish:** losses, positions and games so far, how many versions old the
  finished games' weights were, and the time self-play spent paused.
- **A 9-minute check** (fresh network, 32 games at a time, 100 simulations per move, replay ratio
  4): 388 games were played with weights versions 0 to 11, each version loaded once, and training
  kept up without pausing self-play.
- **Batches stay on one network** because each version has its own inference server; while games
  started on an older version finish, the games in flight are split across servers, so batches
  are smaller. This has not been measured.

**Profile of the trainer** (6 residual blocks, 128 channels, Muon, on 38,502 positions saved by
the check above, one RTX 5060 Ti; two runs of 300 steps per setting, ± under 0.2 ms). The GPU
step is the forward pass, backward pass and optimizer step:

| Precision and layout | GPU step, batch 256 | GPU step, batch 1024 | Per position, batch 1024 |
|---|---|---|---|
| f32 (convolutions in TF32, the PyTorch default) | 24.8 ms | 81.7 ms | 0.080 ms |
| f32, channels-last | 24.6 ms | 75.2 ms | 0.073 ms |
| bf16 autocast | 15.0 ms | 44.6 ms | 0.044 ms |
| bf16 autocast, channels-last (default) | 14.1 ms | 39.2 ms | 0.038 ms |

Reading positions and the symmetry transforms take a further 7.9 ms per 256 positions on the CPU.
On the same batches, the training loss after 320 steps was 1.0425 in f32 and 1.0429 in bf16 with
channels-last, a difference about as large as changing only the layout in f32 (1.0408). The
weights stay in f32 and are published in the layout the Rust loader reads.

**Sharing the GPU with self-play** (network-only self-play of 1,920 games at 200 simulations per
move, mate search and endgame tables on; each trainer paced to train each position 4 times, and
the ratios achieved were 3.98 to 4.02; two runs per setting, ± half their difference):

| Trainer running alongside | Self-play moves/s | Trainer steps/s | Trainer GPU step while sharing |
|---|---|---|---|
| None | 767.8 ± 2.4 | | |
| f32, batch 256 | 586.3 ± 0.3 (−23.6%) | 9.2 | 49.6 ms |
| bf16, channels-last, batch 256 | 653.6 ± 0.3 (−14.9%) | 10.2 | 29.7 ms |
| bf16, channels-last, batch 1024 | 687.4 ± 2.8 (−10.5%) | 2.7 | 75.4 ms |

The two take turns on the GPU, so self-play loses roughly the GPU time the trainer uses. bf16 with
channels-last makes self-play 11% faster than with the f32 trainer, and batch 1024 makes it 17%
faster. The larger batch takes 4 times fewer optimizer steps for the same data; whether that
trains as well per position is not yet measured, so the default batch stays 256.

**Tuning** (`python/hpo.py`, ratings in `python/ratings.py`). Each trial trains a network from
scratch for one hour: network-only search with forced mates and endgame tables, 1,024 games in
flight, the learning rate decaying to 0 on a cosine. Its final weights then play 800 games, at 200
simulations per move with both sides searching as in training, split among the first trial (the
default settings, fixed at 0 Elo) and up to four engines spread across the current ratings. All
engines are rated jointly with a Bradley–Terry fit, with standard errors from resampling the games
in each match. Each engine plays with the exploration constant it was trained with. Starting
trials: the defaults, the defaults with another seed, the defaults with Dirichlet noise at the root
(ε = 0.25; root noise is not tuned, and this trial checks whether it helps), fewer simulations with
more data, the defaults with side positions at half of all moves, and eight Sobol points; after
that, Optuna's Gaussian-process sampler proposes each trial from the ratings so far, using only the
trials without root noise or side positions.

Results of the starting trials (Elo relative to the defaults, standard errors from the bootstrap):
the defaults with seed 1, +22 ± 6; fewer simulations with more data, +6 ± 8; root noise, −57 ± 6;
side positions at half of all moves, −110 ± 8 (ratings at the end of the search). Each side position is searched with the full 200
simulations, so that trial finished 12,852 games in the hour compared with 24,723 at the defaults.
Side positions were then fixed off, and eight new Sobol points cover the remaining six settings.
The first three runs the optimizer proposed rated 103–205 Elo above the defaults, all at the
highest learning rate and lowest replay ratio allowed, and two at the lowest exploration constant.
A replay ratio of 1 leaves most of the GPU to self-play, which then plays 1.5–2 times as many games
in the hour. The ranges for those three settings were then widened to those below. The ledger
(`trials.jsonl`, `matches.jsonl`) lets an interrupted search resume.

| Setting | Range |
|---|---|
| Learning rate (Muon) | 0.005–0.2, log scale |
| Replay ratio | 0.25–8, log scale |
| Simulations per move | 100–400 |
| Moves for the temperature to halve (decay 0.79–0.989 per move; 0.8–0.95 before) | 3–60, log scale |
| Exploration constant in PUCT (default 1.414) | 0.5–3, log scale |
| Window (positions kept) | 100,000–1,000,000, log scale |

Runs past the old limits did no better: learning rates 0.1 and 0.2 rated about 120 Elo, replay
ratios below 0.5 rated well below the best, and no run beat the best at the old limits. The
optimizer kept proposing runs at the limits, where its model is least certain. From trial
`bo13`, new trials are proposed only inside narrower ranges around the best runs (learning rate
0.02–0.1, replay ratio 0.5–2, 100–250 simulations, half-life 4–40 moves, exploration constant
0.6–2, window 150,000–1,000,000), while the model is still fitted on every rated run in the
ranges above, so runs outside the narrower ranges still inform it.

**Results after 26 optimizer trials** (`bo00`–`bo25`, 43 rated runs in all; Elo relative to the
defaults with bootstrap standard errors, ratings as of 2026-11-04):

| Run | Elo | Learning rate | Replay ratio | Simulations | Half-life (moves) | Exploration constant | Window | Positions | Games |
|---|---|---|---|---|---|---|---|---|---|
| `bo19` | +233 ± 8 | 0.02 | 0.65 | 100 | 40 | 0.68 | 150k | 7.77M | 78,024 |
| `bo23` | +213 ± 12 | 0.055 | 0.65 | 100 | 40 | 0.60 | 150k | 7.62M | 76,182 |
| `bo15` | +208 ± 8 | 0.054 | 1.16 | 172 | 4 | 0.87 | 150k | 4.16M | 42,121 |
| `bo25` | +194 ± 10 | 0.092 | 1.01 | 100 | 40 | 0.69 | 150k | 6.64M | 62,402 |
| `bo00` | +192 ± 6 | 0.05 | 1.00 | 152 | 12.8 | 1.00 | 746k | 4.48M | 44,776 |
| `bo01` | +183 ± 8 | 0.05 | 1.00 | 205 | 9.8 | 1.00 | 218k | 3.59M | 36,203 |
| `bo20` | +177 ± 11 | 0.02 | 0.84 | 100 | 40 | 0.60 | 150k | 6.94M | 61,600 |
| `bo21` | +113 ± 11 | 0.02 | 0.50 | 100 | 40 | 0.72 | 150k | 7.99M | 78,475 |
| `bo24` | −32 ± 10 | 0.02 | 1.13 | 100 | 4 | 0.95 | 150k | 7.04M | 76,384 |
| `t00` (defaults) | 0 | 0.02 | 4.00 | 200 | 6.6 | 1.414 | 200k | 2.39M | 24,723 |

- Differences, with 95% intervals from resampling the games (seed variation not included):
  `bo19` − `bo00` +41 [23, 59]; `bo19` − `bo23` +20 [−6, 42]; `bo15` − `bo00` +15 [−3, 36].
- Runs with similar settings differ far more than two default runs with different seeds did
  (+22 ± 6 at the end of the search): `bo20` rated 55 [33, 77] below `bo19`, and `bo21` 120 below.
  Until settings are repeated with new seeds, the order within the top group is not established.

**How the replay ratio divides the GPU.** The trainer is paced so that each run takes exactly
`steps = replay_ratio × positions / 256` optimizer steps (batch size 256), and a game averages
98 ± 5 positions. Each position costs GPU time for self-play, roughly in proportion to the
simulations per move, and for training, in proportion to the replay ratio, so positions generated
in one hour fit

$$\text{positions (millions)} = \frac{1}{a \cdot \text{sims} + b \cdot \text{ratio} + c}$$

with $a = 1.43 \times 10^{-3} \pm 0.05 \times 10^{-3}$, $b = 0.036 \pm 0.003$ and
$c = -0.033 \pm 0.006$ (37 runs without side positions or root noise, $R^2 = 0.989$, residuals
about 7%). One unit of replay ratio costs about as much GPU time as 25 simulations per move: at 100
simulations, raising the ratio from 0.65 to 1 cuts positions by 9% and adds 41% more steps. The
negative $c$ means the cost is not quite linear in simulations; the fit holds within the tested
range.

**What the ratings follow.** In weighted linear fits of Elo over the same 37 runs, judged by
leave-one-out error, the log of positions generated explains the most (error 105 Elo, compared
with 134 for the mean alone; about +106 Elo per doubling of positions). The number of optimizer
steps adds nothing once positions are known, and adding the learning rate and the exploration
constant brings the error to 95. These are correlations: more positions mostly come from fewer
simulations, so more data and a shallower search cannot be separated in these runs, and the fit
leaves about 95 Elo unexplained, as large as the differences it would rank.

With 1,024 games in flight, self-play alone runs at 707 moves/s, compared with 768 at 1,920 and
561 at 512 (one run each), and the first games finish after about 110 s rather than 200 s.

**Board symmetries**, in both loops: a position without castling rights is equivalent to its
mirror image, and a pawnless one to all eight rotations and reflections of the board. Each sampled
position is trained on one of its equivalent images, drawn at random, so every position counts
once whatever its symmetry and repeated draws see different images. Adding every image instead
would count a pawnless ending eight times as often as a position with castling rights.

## Next experiments

The best 1-hour settings rate about 190–230 Elo above the defaults, mostly because a replay ratio
near 1 or below and fewer simulations leave the GPU to self-play, which then generates 1.7–3.3
times as many positions. These ratings only compare the runs with each other, and one hour rewards fast
early learning, so the result may shrink in longer runs. Each experiment below is a matched
comparison at the best settings with two seeds, not another setting for the optimizer.

1. **Repeat seeds** of the top runs (`bo19`, `bo23`, `bo15`, `bo00`) and of `bo21`, since runs
   with similar settings differed by 50–120 Elo, more than the seed variation measured at the
   defaults.
2. **The exploration constant at match time**: each engine plays its rating matches with the
   constant it was trained with, so the ratings mix how well a network trained with how well that
   constant plays at 200 simulations. Replaying the best networks at constants 0.7–2 separates
   the two.
3. **4-hour reruns** of the best one or two settings and the defaults, each also played with a
   fixed opponent (the pilot's `gen_28`, or the classical alpha-beta engine), to check that the
   order holds and to measure strength on a reference that does not change.
4. **Full searches on only some moves** (playout cap randomization,
   [Wu, 2019](https://arxiv.org/abs/1902.10565)): a full search on about a quarter of moves and a
   quick one on the rest, with policy targets recorded only from full searches. Games per hour
   mattered most in the tuning runs, and this raises them.
5. **Network size**: 10 blocks of 128 channels, or 6 of 192, compared with 6 of 128, at 4 hours.
   A larger network plays fewer games per hour, so 1-hour runs would always favour the smaller one.
6. **Gumbel AlphaZero** ([Danihelka et al., ICLR 2022](https://iclr.cc/virtual/2022/poster/6418)) in
   place of PUCT at the root, temperature and root noise: actions are sampled without replacement,
   and the policy target comes from the search's value estimates. Its authors report clear gains
   when planning with few simulations. It is a larger change to the search.
7. **A value target mixing the search's value and the game's result**, so that positions in drawn
   games still carry a signal once most games are draws.

Lower priority:

- **Root noise in place of the temperature schedule**: root noise lost 60 Elo at the defaults, and
  every root move already gets one visit, which covers part of its purpose.
- **Simpler variants of PUCT** (an exploration constant that grows with visits, or a lower first
  estimate for unvisited moves): the forced first visit to every root move and tactical move
  already plays part of their role.
- **Batch size 1024** (self-play 5% faster in the profile above) **and the learning rate
  schedule's form**: small expected gains.
- **How deep the exact search goes** (mate search settings other than C5) is the project's main
  research question, not a setting to tune; it is tested by matched runs with and without exact
  endings once the settings are fixed.

```bash
python python/orchestrate.py --enable-koth \
  --sims-schedule "0:100,5:200,10:400,20:800"    # full training loop, King of the Hill
python python/orchestrate.py --disable-tier1       # ablation: no exact search
python python/orchestrate.py --disable-material    # ablation: no classical material score
python python/orchestrate.py --skip-self-play      # train on evaluation games only
python python/orchestrate.py --disable-tier1 --disable-material \
  --syzygy-path /data/syzygy                       # network only, plus forced mates and endgame tables
python python/orchestrate.py --mate-checks 0       # no mate search (--mate-in N, --mate-quiet M also exist)
python python/orchestrate.py --arch transformer --enable-koth
python python/continuous.py --work-dir runs/c1 \
  --replay-ratio 4 --window 200000                 # continuous training; every setting is a flag
```

## Long-term goal: training only the policy

The engine's value would come from classical methods alone (Tier 1 exact results, otherwise
$V = \tanh(k \cdot \Delta M)$ from quiescence search, with the network's value output fixed at
0), and the network would learn only the policy. Every training target would then come from the
search, which depends only on the position and the current network, not on how the game ends.

- **For:** the value target is the noisiest part of self-play training. Over minutes 15–20 to
  55–60 of the 1-hour runs (median of 37), the value loss fell 0.032 with publish-to-publish
  noise of 0.019, while the policy loss fell 0.40 with noise of 0.029. All positions in a game
  share its result, so a run has only 45,000–78,000 independent value labels. The classical
  score also gives a head start: in the chess variant, the untrained tiered engine played about
  as well as a trained network-only baseline.
- **Against:** strength is capped by a shallow search on a hand-tuned evaluation (PeSTO), however
  good the policy becomes, and MCTS averages leaf values rather than taking the minimax, so a
  weak evaluation costs more than in alpha-beta search. Self-play with the full classical search
  is limited by the CPU (about 110 moves/s, compared with 765 for the network alone) until
  quiescence search is made cheaper.
- **Related work:** training an evaluation toward search values rather than game results is
  established: TreeStrap ([Veness et al., NeurIPS 2009](https://papers.nips.cc/paper/3722-bootstrapping-from-game-tree-search))
  and TDLeaf(λ) ([Baxter et al., 2000](https://arxiv.org/abs/cs/9901001)) in classical chess, and
  value targets built from the search's values in AlphaZero
  ([Willemsen et al., 2021](https://link.springer.com/content/pdf/10.1007/s00521-021-05928-5.pdf)).
  No published comparison of a learned policy and a fixed evaluation with full AlphaZero was
  found.
- **Test:** matched 4-hour runs from scratch, two seeds each, of the network alone, the full
  tiered engine and the policy-only tiered engine, rated with a fixed opponent. A 12-hour pair
  would follow if the policy-only engine is close, since 4 hours may favour its head start.
