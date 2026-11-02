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
in each match. Starting trials: the defaults, the defaults with another seed, two hand-picked
settings and five Sobol points; after that, Optuna's Gaussian-process sampler proposes each trial
from the ratings so far. The ledger (`trials.jsonl`, `matches.jsonl`) lets an interrupted search
resume.

| Setting | Range |
|---|---|
| Learning rate (Muon) | 0.005–0.05, log scale |
| Replay ratio | 1–8, log scale |
| Simulations per move | 100–400 |
| Temperature decay per move | 0.8–0.95 |
| Root noise ε | 0–0.25 |

With 1,024 games in flight, self-play alone runs at 707 moves/s, compared with 768 at 1,920 and
561 at 512 (one run each), and the first games finish after about 110 s rather than 200 s.

**Board symmetries**, in both loops: a position without castling rights is equivalent to its
mirror image, and a pawnless one to all eight rotations and reflections of the board. Each sampled
position is trained on one of its equivalent images, drawn at random, so every position counts
once whatever its symmetry and repeated draws see different images. Adding every image instead
would count a pawnless ending eight times as often as a position with castling rights.

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
