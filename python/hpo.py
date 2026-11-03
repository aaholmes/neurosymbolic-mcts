"""Hyperparameter search over 1-hour continuous training runs.

Each trial trains a network from scratch for one hour (network-only search
with forced mates and endgame tables, standard chess), then plays matches with
a spread of earlier trials. All engines are rated jointly by Elo, with the
first trial (the default settings) fixed at 0. A Gaussian-process optimizer
(Optuna's GPSampler) proposes each later trial from the ratings so far.

Usage:
  python python/hpo.py --work-dir runs/hpo --phase start   # default, repeat, hand-picked, Sobol
  python python/hpo.py --work-dir runs/hpo --phase bo --n-bo 16
  python python/hpo.py --work-dir runs/hpo --phase table
The ledger (trials.jsonl, matches.jsonl) makes every phase resumable.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ratings as rating_fit

# name: (low, high, log scale, type)
SPACE = {
    "lr": (0.005, 0.05, True, float),
    "replay_ratio": (1.0, 8.0, True, float),
    "simulations_per_move": (100, 400, False, int),
    "tau_decay": (0.8, 0.95, False, float),
    "c_puct": (1.0, 3.0, True, float),
    "window": (100_000, 1_000_000, True, int),
}
DEFAULTS = {"lr": 0.02, "replay_ratio": 4.0, "simulations_per_move": 200,
            "tau_decay": 0.9, "c_puct": 1.414, "window": 200_000,
            "side_fraction": 0.0, "side_two_moves": 0.5}
# Side positions are off in every tuned trial: at half of all moves they cost
# about half the games and lost 120 Elo in one hour.
FIXED = {"side_fraction": 0.0, "side_two_moves": 0.5}
ANCHOR = "t00_default"
# (name, settings, Dirichlet noise at the root). Root noise is not tuned; one
# trial at the defaults checks whether it helps.
HAND_PICKED = [
    ("t02_root_noise", dict(DEFAULTS), 0.25),
    ("t03_more_data", dict(DEFAULTS, replay_ratio=2.0, simulations_per_move=100), 0.0),
    ("t04_side_positions", dict(DEFAULTS, side_fraction=0.5), 0.0),
]


def _from_unit(u):
    params = dict(FIXED)
    for (name, (lo, hi, log, kind)), x in zip(SPACE.items(), u):
        v = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if log else lo + x * (hi - lo)
        params[name] = int(round(v)) if kind is int else min(hi, max(lo, round(v, 5)))
    return params


def starting_trials(n_sobol=8, seed=0):
    """The default settings, the defaults again with another seed (to measure
    run-to-run noise), hand-picked settings, and n_sobol space-filling points."""
    from scipy.stats import qmc
    trials = [{"name": ANCHOR, "params": dict(DEFAULTS), "seed": 0},
              {"name": "t01_default_seed1", "params": dict(DEFAULTS), "seed": 1}]
    trials += [{"name": n, "params": p, "seed": 0, "epsilon": eps} for n, p, eps in HAND_PICKED]
    points = qmc.Sobol(d=len(SPACE), scramble=True, seed=seed).random(n_sobol)
    trials += [{"name": f"sobol6d_{i}", "params": _from_unit(u), "seed": 0}
               for i, u in enumerate(points)]
    return trials


def trial_config(params, seed, work_dir, syzygy_path, self_play_bin, dirichlet_epsilon=0.0):
    """A 1-hour continuous run from scratch, with the learning rate decaying to 0."""
    from continuous import ContinuousConfig
    return ContinuousConfig(
        work_dir=work_dir, seed=seed, max_seconds=3600, lr_schedule="cosine",
        enable_tier1=False, enable_material=False, mate_checks=5, syzygy_path=syzygy_path,
        self_play_bin=self_play_bin, concurrent_games=1024,
        lr=params["lr"], replay_ratio=params["replay_ratio"],
        simulations_per_move=params["simulations_per_move"],
        tau_decay=params["tau_decay"], c_puct=params["c_puct"], window=params["window"],
        side_fraction=params["side_fraction"], side_two_moves=params["side_two_moves"],
        dirichlet_epsilon=dirichlet_epsilon,
    )


def match_command(a_weights, b_weights, games, sims, seed_offset, syzygy_path, evaluate_bin,
                  a_c_puct=DEFAULTS["c_puct"], b_c_puct=DEFAULTS["c_puct"]):
    """Both engines search as in training: the network, forced mates and endgame
    tables, each with the exploration constant it was trained with."""
    cmd = [evaluate_bin, a_weights, b_weights, str(games), str(sims),
           "--candidate-c-puct", str(a_c_puct), "--current-c-puct", str(b_c_puct),
           "--disable-tier1", "--disable-material", "--mate-in", "0", "--mate-checks", "5",
           "--no-save-training-data", "--seed-offset", str(seed_offset),
           "--threads", str(games), "--batch-size", "128", "--inference-workers", "4"]
    if syzygy_path:
        cmd += ["--syzygy", syzygy_path]
    return cmd


def parse_match(stdout):
    m = re.search(r"WINS=(\d+)\s+LOSSES=(\d+)\s+DRAWS=(\d+)", stdout)
    if not m:
        raise ValueError("no WINS= line in the match output")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def suggest(observed, seed):
    """The next settings from a Gaussian-process model of (params, Elo) pairs."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dists = {n: optuna.distributions.IntDistribution(lo, hi) if kind is int
             else optuna.distributions.FloatDistribution(lo, hi, log=log)
             for n, (lo, hi, log, kind) in SPACE.items()}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GPSampler(seed=seed))
    for params, value in observed:
        tuned = {n: params[n] for n in SPACE}
        study.add_trial(optuna.trial.create_trial(params=tuned, distributions=dists, value=value))
    return dict(FIXED, **study.ask(dists).params)


def observed_for_bo(trials, table):
    """(tuned settings, Elo) for the rated trials run with the fixed settings and
    no root noise; the others measured something the optimizer cannot change."""
    out = []
    for name, r in table.items():
        t = trials.get(name)
        if t is None or t.get("epsilon") or any(t["params"].get(k) != v for k, v in FIXED.items()):
            continue
        out.append(({n: t["params"][n] for n in SPACE}, r["elo"]))
    return out


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Search:
    """Trains trials, plays each new engine with a spread of earlier ones, and
    keeps a ledger so an interrupted search resumes where it stopped.

    train(trial, work_dir) -> weights path; play(a, b, games, seed_offset) ->
    (wins, losses, draws) for trial a."""

    def __init__(self, work_dir, train, play, games_per_engine=400, opponents=5):
        self.dir = work_dir
        self.train, self.play = train, play
        self.games_per_engine, self.opponents = games_per_engine, opponents
        os.makedirs(work_dir, exist_ok=True)
        self.trials_path = os.path.join(work_dir, "trials.jsonl")
        self.matches_path = os.path.join(work_dir, "matches.jsonl")

    def trials(self):
        return {t["name"]: t for t in _read_jsonl(self.trials_path)}

    def matches(self):
        return _read_jsonl(self.matches_path)

    def ratings(self, n_boot=200):
        """{name: {"elo", "se"}} for every trained engine, the anchor at 0."""
        matches = self.matches()
        if not matches:
            return {ANCHOR: {"elo": 0.0, "se": 0.0}}
        elo = rating_fit.fit(matches, ANCHOR)
        se = rating_fit.bootstrap_se(matches, ANCHOR, n_boot=n_boot, rng=np.random.default_rng(0))
        return {n: {"elo": elo[n], "se": se[n]} for n in elo}

    def run_trials(self, trials):
        for trial in trials:
            done = self.trials()
            if trial["name"] not in done:
                weights = self.train(trial, os.path.join(self.dir, trial["name"]))
                record = dict(trial, weights=weights)
                _append_jsonl(self.trials_path, record)
                done[trial["name"]] = record
            self._rate(done[trial["name"]], done)
            self.print_table()

    def _rate(self, trial, done):
        if trial["name"] == ANCHOR:
            return
        played = {m["b"] for m in self.matches() if m["a"] == trial["name"]}
        pool = {n: r["elo"] for n, r in self.ratings(n_boot=2).items() if n != trial["name"]}
        pool.setdefault(ANCHOR, 0.0)
        opponents = rating_fit.choose_opponents(pool, ANCHOR, self.opponents)
        games = max(2, 2 * (self.games_per_engine // (2 * len(opponents))))
        for opp in opponents:
            if opp in played:
                continue
            seed_offset = 10_000 * (len(self.matches()) + 1)
            w, l, d = self.play(trial, done[opp], games, seed_offset)
            _append_jsonl(self.matches_path, {"a": trial["name"], "b": opp,
                                              "wins": w, "losses": l, "draws": d})

    def run_bo(self, n, seed=0):
        for i in range(n):
            table = self.ratings()
            done = self.trials()
            observed = observed_for_bo(done, table)
            k = sum(1 for name in done if name.startswith("bo"))
            params = suggest(observed, seed=seed + k)
            self.run_trials([{"name": f"bo{k:02d}", "params": params, "seed": 100 + k}])

    def print_table(self):
        table = self.ratings()
        done = self.trials()
        print(f"\n{'engine':<20}{'Elo':>8}{'± se':>8}  settings", flush=True)
        for name in sorted(table, key=lambda n: -table[n]["elo"]):
            p = done.get(name, {}).get("params", {})
            print(f"{name:<20}{table[name]['elo']:>8.0f}{table[name]['se']:>8.0f}  {p}", flush=True)


def real_train(args):
    from continuous import run

    def train(trial, work_dir):
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)  # an interrupted run starts again from scratch
        cfg = trial_config(trial["params"], trial["seed"], work_dir, args.syzygy,
                           args.self_play_bin, trial.get("epsilon", 0.0))
        summary = run(cfg)
        engine = os.path.join(args.work_dir, "engines", trial["name"] + ".pth")
        os.makedirs(os.path.dirname(engine), exist_ok=True)
        shutil.copy(summary["weights"], engine)
        return engine
    return train


def real_play(args):
    from orchestrate import get_rust_env

    def play(a, b, games, seed_offset):
        cmd = match_command(a["weights"], b["weights"], games, args.eval_sims, seed_offset,
                            args.syzygy, args.evaluate_bin,
                            a["params"]["c_puct"], b["params"]["c_puct"])
        out = subprocess.run(cmd, env=get_rust_env(), capture_output=True, text=True)
        with open(os.path.join(args.work_dir, "matches.log"), "a") as f:
            f.write(f"$ {' '.join(cmd)}\n{out.stdout[-2000:]}{out.stderr[-2000:]}\n")
        return parse_match(out.stdout)
    return play


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work-dir", required=True)
    p.add_argument("--phase", choices=["start", "bo", "table"], required=True)
    p.add_argument("--n-sobol", type=int, default=8)
    p.add_argument("--n-bo", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--syzygy", default="/data/syzygy")
    p.add_argument("--self-play-bin", required=False, default="")
    p.add_argument("--evaluate-bin", default="target/release/evaluate_models")
    p.add_argument("--games-per-engine", type=int, default=800)
    p.add_argument("--opponents", type=int, default=5)
    p.add_argument("--eval-sims", type=int, default=200)
    args = p.parse_args(argv)
    args.work_dir = os.path.abspath(args.work_dir)
    search = Search(args.work_dir, real_train(args), real_play(args),
                    games_per_engine=args.games_per_engine, opponents=args.opponents)
    if args.phase == "start":
        search.run_trials(starting_trials(args.n_sobol, args.seed))
    elif args.phase == "bo":
        search.run_bo(args.n_bo, seed=args.seed)
    search.print_table()


if __name__ == "__main__":
    main()
