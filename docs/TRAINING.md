# Training

Two loops are available. Both train on positions from self-play, with a policy target (the
search's visit counts) and a value target (the game's result).

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

**Profile of the trainer** (6 residual blocks, 128 channels, Muon, batch 256, f32, on 38,502
positions saved by the check above, one RTX 5060 Ti; two runs of 500 steps, ± the standard
deviation over blocks of 100 steps): 32.3 ms per step, or 30.9 steps per second. The step on the
GPU (forward, backward and optimizer) takes 24.4 ± 0.01 ms, reading positions 3.5 ms and the
symmetry transforms 4.4 ms. A replay ratio of 4 at 760 positions per second needs about 12 steps
per second, so the trainer alone has 2.6 times what it needs; how much self-play slows when both
share the GPU is being measured.

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
