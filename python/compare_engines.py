"""A matched comparison of the three engines: network only, the full engine and
the policy-only engine, each trained from scratch for one hour with the best
tuned settings (bo19, tuned for the network-only engine), then rated by a round
robin with each side searching as it was trained.

Usage:
  python python/compare_engines.py --work-dir runs/compare_1h --seeds 0 1
The ledger (trials.jsonl, matches.jsonl) makes a run resumable.
"""

import argparse
import itertools
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hpo
import ratings as rating_fit

# bo19's settings, the best of the 1-hour tuning runs.
BO19 = {"side_fraction": 0.0, "side_two_moves": 0.5, "lr": 0.02,
        "replay_ratio": 0.6461686278896777, "simulations_per_move": 100,
        "tau_half_life": 40.0, "c_puct": 0.6753554310115629, "window": 150000}
ENGINES = ("network", "full", "policy")
# (tiers on, classical score on, policy only)
SEARCH = {"network": (False, False, False), "full": (True, True, False),
          "policy": (True, True, True)}


def trials(seeds=(0, 1)):
    return [{"name": f"{e}_s{s}", "engine": e, "seed": s, "params": dict(BO19)}
            for s in seeds for e in ENGINES]


def config(trial, work_dir, syzygy_path, self_play_bin, qsearch_config=""):
    tier1, material, policy_only = SEARCH[trial["engine"]]
    cfg = hpo.trial_config(trial["params"], trial["seed"], work_dir, syzygy_path, self_play_bin)
    cfg.enable_tier1, cfg.enable_material, cfg.policy_only = tier1, material, policy_only
    cfg.qsearch_config = qsearch_config
    return cfg


def side_flags(trial, side):
    tier1, material, policy_only = SEARCH[trial["engine"]]
    flags = []
    if not tier1:
        flags.append(f"--{side}-disable-tier1")
    if not material:
        flags.append(f"--{side}-disable-material")
    if policy_only:
        flags += [f"--{side}-policy-only", "revisit"]
    return flags


def match_command(a, b, games, sims, seed_offset, syzygy_path, evaluate_bin, qsearch_config=""):
    """a plays as the candidate, b as the current engine; each searches as in its training."""
    cmd = [evaluate_bin, a["weights"], b["weights"], str(games), str(sims),
           "--candidate-c-puct", str(a["params"]["c_puct"]),
           "--current-c-puct", str(b["params"]["c_puct"]),
           "--mate-in", "0", "--mate-checks", "5", "--no-save-training-data",
           "--seed-offset", str(seed_offset), "--threads", str(games),
           "--batch-size", "128", "--inference-workers", "4"]
    cmd += side_flags(a, "candidate") + side_flags(b, "current")
    if syzygy_path:
        cmd += ["--syzygy", syzygy_path]
    if qsearch_config:
        cmd += ["--qsearch-config", qsearch_config]
    return cmd


def round_robin(names):
    return list(itertools.combinations(names, 2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--syzygy", default="/data/syzygy")
    p.add_argument("--self-play-bin", required=True)
    p.add_argument("--evaluate-bin", required=True)
    p.add_argument("--qsearch-config", default="")
    p.add_argument("--games", type=int, default=100, help="games per pair")
    p.add_argument("--eval-sims", type=int, default=100)
    p.add_argument("--anchor", default="network_s0")
    p.add_argument("--extra", nargs="*", default=[],
                   help="name=engine:weights of other engines to rate, e.g. bo19=network:path")
    p.add_argument("--rate-only", action="store_true")
    args = p.parse_args(argv)
    from continuous import run
    from orchestrate import get_rust_env
    work = os.path.abspath(args.work_dir)
    os.makedirs(work, exist_ok=True)
    trials_path = os.path.join(work, "trials.jsonl")
    matches_path = os.path.join(work, "matches.jsonl")

    done = {t["name"]: t for t in hpo._read_jsonl(trials_path)}
    if not args.rate_only:
        for trial in trials(tuple(args.seeds)):
            if trial["name"] in done:
                continue
            run_dir = os.path.join(work, trial["name"])
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)  # an interrupted run starts again from scratch
            summary = run(config(trial, run_dir, args.syzygy, args.self_play_bin,
                                 args.qsearch_config))
            engine = os.path.join(work, "engines", trial["name"] + ".pth")
            os.makedirs(os.path.dirname(engine), exist_ok=True)
            shutil.copy(summary["weights"], engine)
            record = dict(trial, weights=engine)
            hpo._append_jsonl(trials_path, record)
            done[trial["name"]] = record
            print(f"trained {trial['name']}", flush=True)

    for spec in args.extra:
        name, rest = spec.split("=", 1)
        engine, weights = rest.split(":", 1)
        done.setdefault(name, {"name": name, "engine": engine, "params": dict(BO19),
                               "weights": weights})

    played = {frozenset((m["a"], m["b"])) for m in hpo._read_jsonl(matches_path)}
    for a, b in round_robin(sorted(done)):
        if frozenset((a, b)) in played:
            continue
        seed_offset = 10_000 * (len(played) + 1)
        cmd = match_command(done[a], done[b], args.games, args.eval_sims, seed_offset,
                            args.syzygy, args.evaluate_bin, args.qsearch_config)
        out = subprocess.run(cmd, env=get_rust_env(), capture_output=True, text=True)
        with open(os.path.join(work, "matches.log"), "a") as f:
            f.write(f"$ {' '.join(cmd)}\n{out.stdout[-2000:]}{out.stderr[-2000:]}\n")
        w, l, d = hpo.parse_match(out.stdout)
        hpo._append_jsonl(matches_path, {"a": a, "b": b, "wins": w, "losses": l, "draws": d})
        played.add(frozenset((a, b)))
        print(f"{a} vs {b}: +{w} -{l} ={d}", flush=True)

    matches = hpo._read_jsonl(matches_path)
    if matches:
        elo = rating_fit.fit(matches, args.anchor)
        se = rating_fit.bootstrap_se(matches, args.anchor, n_boot=200,
                                     rng=np.random.default_rng(0))
        for n in sorted(elo, key=lambda n: -elo[n]):
            print(f"{n:<16}{elo[n]:>8.0f}{se[n]:>8.0f}", flush=True)


if __name__ == "__main__":
    main()
