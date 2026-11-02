"""Continuous training: self-play and training run together, with no SPRT test.

Rust self-play keeps `concurrent_games` games going and starts each new game
with the newest weights the trainer has published (`self_play --watch`). The
trainer moves finished games into a sliding window of recent positions and
trains so that each generated position is trained on `replay_ratio` times on
average. It publishes weights every `publish_every` steps. If training falls
more than `max_owed` steps behind, self-play is paused until it catches up, so
the replay ratio holds whatever the relative speeds.

Usage: python python/continuous.py --work-dir runs/continuous [options]
"""

import argparse
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentation import random_transform
from model import SeresNet, TransformerNet
from replay_buffer import BYTES_PER_SAMPLE, INPUT_CHANNELS, BOARD_SIZE, POLICY_SIZE
from train import batch_loss, make_optimizer


@dataclass
class ContinuousConfig:
    work_dir: str = "runs/continuous"
    # Network and optimizer
    arch: str = "seresnet"
    num_blocks: int = 6
    hidden_dim: int = 128
    optimizer: str = "muon"
    lr: float = 0.02
    batch_size: int = 256
    augment: bool = True
    precision: str = "bf16"  # "bf16" runs the forward and backward passes under bf16 autocast
    channels_last: bool = True  # NHWC memory layout for the convolutions
    device: str = ""  # "" = cuda if available
    seed: int = 0
    resume_from: str = ""  # a cont_N.pth to continue from
    # Data flow
    replay_ratio: float = 4.0  # times each generated position is trained on, on average
    window: int = 200_000  # positions kept, oldest games dropped first
    min_positions: int = 20_000  # no training until the window holds this many
    publish_every: int = 200  # training steps between published weights
    max_owed: int = 0  # pause self-play beyond this many owed steps (0 = 4 * publish_every)
    keep_published: int = 5
    max_steps: int = 0  # 0 = no limit
    max_seconds: float = 0.0  # wall-clock limit for the run, 0 = no limit
    lr_schedule: str = "constant"  # "cosine" decays to 0 over max_seconds (or else max_steps)
    max_positions: int = 0  # 0 = no limit
    poll_seconds: float = 1.0
    stall_seconds: float = 300.0  # stop if self-play runs this long without finishing a game (0 = never)
    # Self-play
    concurrent_games: int = 128
    simulations_per_move: int = 200
    enable_koth: bool = False
    enable_tier1: bool = True
    enable_material: bool = True
    qsearch: str = "extended"
    mate_in: int = 0
    mate_checks: int = 5
    mate_quiet: int = 0
    syzygy_path: str = ""
    self_play_bin: str = ""  # a built self_play binary; "" = cargo run
    inference_batch_size: int = 128
    inference_workers: int = 4
    c_puct: float = 1.414  # exploration constant in the search's PUCT rule
    # Side positions: at this fraction of moves, 1 or 2 random moves off the main line
    # (2 with probability side_two_moves) are searched once for a policy-only sample
    side_fraction: float = 0.0
    side_two_moves: float = 0.5
    # Move played: sampled in proportion to (visits - 1)^(1/tau),
    # tau = tau0 * tau_decay^(move - 1), at least tau_min (0 plays the most-visited move)
    tau0: float = 1.0
    tau_decay: float = 0.9
    tau_min: float = 0.0
    # Dirichlet noise mixed into each search's root priors (epsilon 0 = off)
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0


def lr_factor(schedule, progress):
    """The learning rate's multiplier at `progress` (0 to 1) through the run."""
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    raise ValueError(f"unknown learning-rate schedule {schedule!r}")


def run_progress(cfg, elapsed, steps):
    """How far through the run, from the time limit if there is one, else the step limit."""
    if cfg.max_seconds:
        return elapsed / cfg.max_seconds
    if cfg.max_steps:
        return steps / cfg.max_steps
    return 0.0


def steps_owed(positions_generated, steps_done, replay_ratio, batch_size):
    """Training steps needed so far to train on each position replay_ratio times."""
    return max(0, int(positions_generated * replay_ratio // batch_size) - steps_done)


def should_pause(paused, owed, max_owed):
    """Pause self-play when training is more than max_owed steps behind, and
    resume it only once training has caught up."""
    if owed > max_owed:
        return True
    return paused and owed > 0


class SelfPlayStalled(RuntimeError):
    """Self-play ran for stall_seconds without finishing a game."""


def stalled(now, last_game, paused_s, stall_seconds):
    """Whether self-play has run, not counting time paused, for stall_seconds
    since the last game finished."""
    return bool(stall_seconds) and now - last_game - paused_s >= stall_seconds


def write_published(pointer, version, path):
    """Write `version path` through a rename, the format self-play --watch reads."""
    tmp = pointer + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{version} {path}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, pointer)


def game_version(filename):
    """The published weights version in a game file's name, or None."""
    m = re.search(r"_v(\d+)\.bin$", filename)
    return int(m.group(1)) if m else None


class GameWindow:
    """The most recent finished games, as files moved into one directory."""

    def __init__(self, window_dir, capacity):
        self.dir = window_dir
        self.capacity = capacity
        self.files = []  # [(path, positions)], oldest first
        os.makedirs(window_dir, exist_ok=True)

    def positions(self):
        return sum(n for _, n in self.files)

    def ingest(self, games_dir):
        """Move every finished game from games_dir into the window and drop the
        oldest games beyond capacity. Returns (positions added, game versions)."""
        added, versions = 0, []
        for src in sorted(glob.glob(os.path.join(games_dir, "*.bin"))):
            n = os.path.getsize(src) // BYTES_PER_SAMPLE
            dst = os.path.join(self.dir, os.path.basename(src))
            os.replace(src, dst)
            if n == 0:
                os.remove(dst)
                continue
            self.files.append((dst, n))
            added += n
            v = game_version(src)
            if v is not None:
                versions.append(v)
        while self.positions() > self.capacity and self.files:
            path, _ = self.files.pop(0)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        return added, versions

    def sample(self, batch_size, rng):
        """Positions drawn uniformly, with replacement, from the window."""
        counts = np.array([n for _, n in self.files], dtype=np.float64)
        file_idx = rng.choice(len(self.files), size=batch_size, p=counts / counts.sum())
        board_end = INPUT_CHANNELS * BOARD_SIZE
        boards = np.empty((batch_size, INPUT_CHANNELS, 8, 8), dtype=np.float32)
        scalars = np.empty((batch_size, 2), dtype=np.float32)
        values = np.empty((batch_size, 1), dtype=np.float32)
        policies = np.empty((batch_size, POLICY_SIZE), dtype=np.float32)
        for i, fi in enumerate(file_idx):
            path, n = self.files[fi]
            with open(path, "rb") as f:
                f.seek(int(rng.integers(n)) * BYTES_PER_SAMPLE)
                raw = np.frombuffer(f.read(BYTES_PER_SAMPLE), dtype=np.float32)
            boards[i] = raw[:board_end].reshape(INPUT_CHANNELS, 8, 8)
            scalars[i] = raw[board_end:board_end + 2]
            values[i, 0] = raw[board_end + 2]
            policies[i] = raw[board_end + 3:]
        return boards, scalars, values, policies


PRECISIONS = {"f32": torch.float32, "bf16": torch.bfloat16}


class Trainer:
    """Trains one model and publishes its weights for self-play."""

    def __init__(self, model, weights_dir, optimizer="muon", lr=0.02, batch_size=256,
                 keep_published=5, device="", seed=0, augment=True, disable_material=False,
                 precision="f32", channels_last=False):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, not {precision!r}")
        self.precision = precision
        self.channels_last = channels_last
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.optimizer = make_optimizer(self.model, optimizer, lr)
        self._groups = list(self.optimizer.param_groups)
        if getattr(self.optimizer, "backend", None) is not None:
            self._groups += self.optimizer.backend.param_groups
        self._base_lrs = [g["lr"] for g in self._groups]
        self.weights_dir = weights_dir
        self.pointer = os.path.join(weights_dir, "published.txt")
        self.batch_size = batch_size
        self.keep_published = keep_published
        self.augment = augment
        self.disable_material = disable_material
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.version = -1
        os.makedirs(weights_dir, exist_ok=True)

    def set_lr_factor(self, factor):
        """Scale every learning rate, Muon's backend optimizer included, from its initial value."""
        for group, base in zip(self._groups, self._base_lrs):
            group["lr"] = base * factor

    def resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.steps = ckpt.get("global_minibatch", 0)
        self.version = ckpt.get("version", -1)

    def train_steps(self, n, window):
        """n optimizer steps on batches from the window; returns mean losses."""
        self.model.train()
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "k_mean": 0.0}
        for _ in range(n):
            boards, scalars, values, policies = window.sample(self.batch_size, self.rng)
            if self.augment:
                for i in range(len(boards)):
                    boards[i], _, _, policies[i] = random_transform(
                        boards[i], scalars[i, 0], values[i], policies[i], self.rng)
            boards = torch.from_numpy(boards)
            if self.channels_last:
                boards = boards.to(memory_format=torch.channels_last)
            with torch.autocast(torch.device(self.device).type, dtype=PRECISIONS[self.precision],
                                enabled=self.precision != "f32"):
                loss, p_loss, v_loss, k = batch_loss(
                    self.model, boards, torch.from_numpy(scalars),
                    torch.from_numpy(values), torch.from_numpy(policies),
                    self.device, self.disable_material)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.steps += 1
            totals["loss"] += loss.item()
            totals["policy_loss"] += p_loss.item()
            totals["value_loss"] += v_loss.item()
            totals["k_mean"] += k.mean().item()
        return {key: v / max(1, n) for key, v in totals.items()}

    def publish(self):
        """Save the weights as the next version, point self-play at them, and
        delete all but the newest keep_published versions."""
        self.version += 1
        path = os.path.join(self.weights_dir, f"cont_{self.version}.pth")
        tmp = path + ".tmp"
        torch.save({
            # Contiguous (NCHW) tensors, the layout the Rust loader reads.
            "model_state_dict": {k: v.contiguous() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_minibatch": self.steps,
            "version": self.version,
        }, tmp)
        os.replace(tmp, path)
        write_published(self.pointer, self.version, os.path.abspath(path))
        old = self.version - self.keep_published
        for f in glob.glob(os.path.join(self.weights_dir, "cont_*.pth")):
            v = int(re.search(r"cont_(\d+)\.pth$", f).group(1))
            if v <= old:
                os.remove(f)
        return path


def create_model(cfg):
    if cfg.arch == "transformer":
        return TransformerNet(num_blocks=cfg.num_blocks, hidden_dim=cfg.hidden_dim)
    return SeresNet(num_blocks=cfg.num_blocks, hidden_dim=cfg.hidden_dim)


def make_trainer(cfg, weights_dir):
    torch.manual_seed(cfg.seed)
    return Trainer(create_model(cfg), weights_dir, optimizer=cfg.optimizer, lr=cfg.lr,
                   batch_size=cfg.batch_size, keep_published=cfg.keep_published,
                   device=cfg.device, seed=cfg.seed, augment=cfg.augment,
                   disable_material=not cfg.enable_material, precision=cfg.precision,
                   channels_last=cfg.channels_last)


def self_play_command(cfg, pointer, first_weights, games_dir):
    """The Rust self-play command: many games, each with the newest weights."""
    from orchestrate import RUST_NN_FEATURES, exact_ending_args
    launcher = [cfg.self_play_bin] if cfg.self_play_bin else [
        "cargo", "run", "--release", "--features", RUST_NN_FEATURES, "--bin", "self_play", "--"]
    cmd = launcher + [
        str(10**9),  # games: run until stopped
        str(cfg.simulations_per_move),
        games_dir,
        first_weights,
        str(cfg.enable_koth).lower(),
        str(cfg.enable_tier1).lower(),
        str(cfg.enable_material).lower(),
        "none",
        "--batch-size", str(cfg.inference_batch_size),
        "--inference-workers", str(cfg.inference_workers),
        "--seed-offset", str(cfg.seed * 10**9),
        "--threads", str(cfg.concurrent_games),
        "--tau0", str(cfg.tau0),
        "--tau-decay", str(cfg.tau_decay),
        "--tau-min", str(cfg.tau_min),
        "--dirichlet-alpha", str(cfg.dirichlet_alpha),
        "--dirichlet-epsilon", str(cfg.dirichlet_epsilon),
        "--c-puct", str(cfg.c_puct),
        "--side-fraction", str(cfg.side_fraction), "--side-two-moves", str(cfg.side_two_moves),
        "--watch", pointer,
    ]
    if cfg.qsearch != "extended":
        cmd += ["--qsearch", cfg.qsearch]
    return cmd + exact_ending_args(cfg)


def launch_self_play(cfg, pointer, first_weights, games_dir):
    from orchestrate import get_rust_env
    log = open(os.path.join(cfg.work_dir, "self_play.log"), "a")
    return subprocess.Popen(self_play_command(cfg, pointer, first_weights, games_dir),
                            env=get_rust_env(), stdout=log, stderr=subprocess.STDOUT)


def stop(proc, paused):
    if proc.poll() is not None:
        return
    if paused:
        proc.send_signal(signal.SIGCONT)
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run(cfg, launch=launch_self_play):
    """Run self-play and training together until a limit is reached.
    Returns a summary of the run."""
    weights_dir = os.path.join(cfg.work_dir, "weights")
    games_dir = os.path.join(cfg.work_dir, "games")
    os.makedirs(games_dir, exist_ok=True)
    with open(os.path.join(cfg.work_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    log_path = os.path.join(cfg.work_dir, "continuous_log.jsonl")

    trainer = make_trainer(cfg, weights_dir)
    if cfg.resume_from:
        trainer.resume(cfg.resume_from)
    first = trainer.publish()
    window = GameWindow(os.path.join(cfg.work_dir, "window"), cfg.window)
    max_owed = cfg.max_owed or 4 * cfg.publish_every

    start = time.time()
    start_steps = trainer.steps
    generated = games = lag_total = 0
    paused, paused_since, paused_total = False, 0.0, 0.0
    last_publish = trainer.steps
    losses = {}

    def publish_and_log():
        nonlocal last_publish
        trainer.publish()
        last_publish = trainer.steps
        entry = {
            "version": trainer.version,
            "steps": trainer.steps,
            "positions_generated": generated,
            "games": games,
            "window_positions": window.positions(),
            "mean_game_lag_versions": lag_total / games if games else None,
            "elapsed_s": round(time.time() - start, 1),
            "paused_s": round(paused_total + (time.time() - paused_since if paused else 0), 1),
            "lr": trainer.optimizer.param_groups[0]["lr"],
            **losses,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    proc = launch(cfg, trainer.pointer, first, games_dir)
    last_game, paused_at_last_game = time.time(), 0.0
    try:
        while not (cfg.max_steps and trainer.steps - start_steps >= cfg.max_steps) and \
                not (cfg.max_positions and generated >= cfg.max_positions) and \
                not (cfg.max_seconds and time.time() - start >= cfg.max_seconds):
            added, versions = window.ingest(games_dir)
            generated += added
            games += len(versions)
            lag_total += sum(trainer.version - v for v in versions)
            paused_now = paused_total + (time.time() - paused_since if paused else 0)
            if versions:
                last_game, paused_at_last_game = time.time(), paused_now
            elif stalled(time.time(), last_game, paused_now - paused_at_last_game,
                         cfg.stall_seconds):
                raise SelfPlayStalled(
                    f"no game finished in {cfg.stall_seconds:.0f} s of self-play")

            ready = window.positions() >= cfg.min_positions
            owed = steps_owed(generated, trainer.steps - start_steps,
                              cfg.replay_ratio, cfg.batch_size) if ready else 0
            if cfg.max_steps:
                owed = min(owed, cfg.max_steps - (trainer.steps - start_steps))
            pause = should_pause(paused, owed, max_owed)
            if pause != paused:
                proc.send_signal(signal.SIGSTOP if pause else signal.SIGCONT)
                if pause:
                    paused_since = time.time()
                else:
                    paused_total += time.time() - paused_since
                paused = pause

            if owed == 0:
                if proc.poll() is not None and added == 0:
                    print("Self-play stopped; ending the run.")
                    break
                time.sleep(cfg.poll_seconds)
                continue

            n = min(owed, cfg.publish_every - (trainer.steps - last_publish))
            trainer.set_lr_factor(lr_factor(cfg.lr_schedule, run_progress(
                cfg, time.time() - start, trainer.steps - start_steps)))
            losses = trainer.train_steps(n, window)
            if trainer.steps - last_publish >= cfg.publish_every:
                publish_and_log()
    finally:
        stop(proc, paused)
    if trainer.steps > last_publish:
        publish_and_log()
    weights = os.path.join(weights_dir, f"cont_{trainer.version}.pth")
    return {"steps": trainer.steps, "version": trainer.version, "weights": weights,
            "positions_generated": generated, "games": games}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, default in asdict(ContinuousConfig()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=default,
                           metavar="true|false")
        else:
            p.add_argument(flag, type=type(default), default=default)
    return ContinuousConfig(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    print(run(parse_args()))
