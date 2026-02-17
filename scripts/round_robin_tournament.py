#!/usr/bin/env python3
"""Round-robin tournament between tiered and vanilla models.

Each model plays with its training search configuration:
- Tiered models: tiers enabled (default)
- Vanilla models: --candidate-disable-tier1 --candidate-disable-material (or --current-)

10 models from the first ~20 generations of each run.
45 pairings, 200 games each, 200 sims/move.
"""

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "target", "release", "evaluate_models")

TIERED_DIR = os.path.join(ROOT, "runs", "long_run", "scaleup_2m_tiered_propgreedy", "weights")
VANILLA_DIR = os.path.join(ROOT, "runs", "long_run", "scaleup_2m_vanilla_propgreedy", "weights")

# Model definitions: (name, path, type)
MODELS = [
    ("tiered_gen0",  os.path.join(TIERED_DIR, "gen_0.pt"),  "tiered"),
    ("tiered_gen1",  os.path.join(TIERED_DIR, "gen_1.pt"),  "tiered"),
    ("tiered_gen4",  os.path.join(TIERED_DIR, "gen_4.pt"),  "tiered"),
    ("tiered_gen17", os.path.join(TIERED_DIR, "gen_17.pt"), "tiered"),
    ("vanilla_gen0",  os.path.join(VANILLA_DIR, "gen_0.pt"),  "vanilla"),
    ("vanilla_gen2",  os.path.join(VANILLA_DIR, "gen_2.pt"),  "vanilla"),
    ("vanilla_gen6",  os.path.join(VANILLA_DIR, "gen_6.pt"),  "vanilla"),
    ("vanilla_gen9",  os.path.join(VANILLA_DIR, "gen_9.pt"),  "vanilla"),
    ("vanilla_gen13", os.path.join(VANILLA_DIR, "gen_13.pt"), "vanilla"),
    ("vanilla_gen18", os.path.join(VANILLA_DIR, "gen_18.pt"), "vanilla"),
]

RESULTS_DIR = os.path.join(ROOT, "runs", "tournaments")


def build_cmd(cand_path, curr_path, cand_type, curr_type, num_games, simulations,
              batch_size, threads, explore_base, seed_offset):
    """Build the evaluate_models command with per-side tier flags."""
    cmd = [
        BINARY,
        cand_path, curr_path,
        str(num_games), str(simulations),
        "--enable-koth",
        "--no-save-training-data",
        "--explore-base", str(explore_base),
        "--batch-size", str(batch_size),
        "--threads", str(threads),
        "--seed-offset", str(seed_offset),
    ]

    # Per-side flags based on model type
    if cand_type == "vanilla":
        cmd += ["--candidate-disable-tier1", "--candidate-disable-material"]
    if curr_type == "vanilla":
        cmd += ["--current-disable-tier1", "--current-disable-material"]

    return cmd


def parse_result(stdout):
    """Parse WINS=X LOSSES=Y DRAWS=Z from stdout."""
    m = re.search(r"WINS=(\d+)\s+LOSSES=(\d+)\s+DRAWS=(\d+)", stdout)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def run_match(cand_name, cand_path, cand_type, curr_name, curr_path, curr_type,
              num_games, simulations, batch_size, threads, explore_base, seed_offset):
    """Run a single match and return (wins, losses, draws) from candidate's perspective."""
    cmd = build_cmd(cand_path, curr_path, cand_type, curr_type,
                    num_games, simulations, batch_size, threads, explore_base, seed_offset)

    print(f"\n{'='*70}")
    print(f"  {cand_name} (candidate) vs {curr_name} (current)")
    print(f"  Candidate type: {cand_type}, Current type: {curr_type}")
    print(f"  {num_games} games, {simulations} sims/move")
    print(f"{'='*70}")
    sys.stdout.flush()

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start

    # Print stderr (game-by-game progress)
    if proc.stderr:
        # Only print last few lines to avoid spam
        lines = proc.stderr.strip().split("\n")
        if len(lines) > 5:
            print(f"  ... ({len(lines) - 3} lines of game output) ...")
            for line in lines[-3:]:
                print(f"  {line}")
        else:
            for line in lines:
                print(f"  {line}")

    result = parse_result(proc.stdout)
    if result is None:
        print(f"  ERROR: Could not parse output: {proc.stdout[:200]}")
        print(f"  stderr: {proc.stderr[-500:]}")
        return None

    w, l, d = result
    total = w + l + d
    wr = (w + 0.5 * d) / total if total > 0 else 0.5
    print(f"  Result: W={w} L={l} D={d} WR={wr:.3f} ({elapsed:.0f}s)")
    sys.stdout.flush()
    return w, l, d


def compute_elo_mle(results, model_names, anchor_idx=0, iterations=1000):
    """Compute Elo ratings via iterative maximum likelihood estimation.

    results: dict of (i, j) -> (wins_i, wins_j, draws) where i is "candidate", j is "current"
    anchor_idx: index of the model anchored at Elo 0
    """
    n = len(model_names)
    elos = [0.0] * n

    for _ in range(iterations):
        for i in range(n):
            if i == anchor_idx:
                continue

            numerator = 0.0
            denominator = 0.0

            for j in range(n):
                if i == j:
                    continue

                # Get results between i and j
                if (i, j) in results:
                    wi, wj, d = results[(i, j)]
                elif (j, i) in results:
                    wj, wi, d = results[(j, i)]
                else:
                    continue

                # Score for player i against j
                score_i = wi + 0.5 * d
                total = wi + wj + d
                if total == 0:
                    continue

                # Expected score based on current Elo estimates
                expected = 1.0 / (1.0 + 10.0 ** ((elos[j] - elos[i]) / 400.0))

                numerator += score_i
                denominator += total * expected

            if denominator > 0:
                # Update: Elo_i += 400 * log10(numerator / denominator)
                elos[i] += 400.0 * math.log10(numerator / denominator)

    # Re-anchor
    anchor_elo = elos[anchor_idx]
    elos = [e - anchor_elo for e in elos]

    return elos


def print_results_table(model_names, elos, results):
    """Print sorted Elo table and full cross-table."""
    n = len(model_names)

    # Sorted Elo table
    ranked = sorted(range(n), key=lambda i: elos[i], reverse=True)
    print(f"\n{'='*60}")
    print(f"  ELO RANKINGS (anchored to {model_names[0]}=0)")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'Model':<20} {'Elo':>8} {'W':>5} {'L':>5} {'D':>5} {'WR':>7}")
    print(f"  {'-'*55}")

    for rank, i in enumerate(ranked):
        total_w, total_l, total_d = 0, 0, 0
        for j in range(n):
            if i == j:
                continue
            if (i, j) in results:
                w, l, d = results[(i, j)]
                total_w += w
                total_l += l
                total_d += d
            elif (j, i) in results:
                l2, w2, d2 = results[(j, i)]
                total_w += w2
                total_l += l2
                total_d += d2
        total = total_w + total_l + total_d
        wr = (total_w + 0.5 * total_d) / total if total > 0 else 0.5
        print(f"  {rank+1:<5} {model_names[i]:<20} {elos[i]:>+8.1f} {total_w:>5} {total_l:>5} {total_d:>5} {wr:>7.3f}")

    # Cross-table
    print(f"\n{'='*60}")
    print(f"  CROSS-TABLE (win rate from row's perspective)")
    print(f"{'='*60}")

    # Header
    short_names = [name[:8] for name in model_names]
    header = f"  {'':>16} " + " ".join(f"{s:>8}" for s in short_names)
    print(header)
    print(f"  {'-'*(16 + 9 * n)}")

    for i in ranked:
        row = f"  {model_names[i]:>16} "
        for j in ranked:
            if i == j:
                row += f"{'---':>8}"
                continue
            if (i, j) in results:
                w, l, d = results[(i, j)]
            elif (j, i) in results:
                l, w, d = results[(j, i)]
            else:
                row += f"{'?':>8}"
                continue
            total = w + l + d
            wr = (w + 0.5 * d) / total if total > 0 else 0.5
            row += f"{wr:>8.3f}"
        print(row)


def save_results(model_names, elos, results, output_dir):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "models": model_names,
        "elos": elos,
        "results": {f"{i},{j}": list(v) for (i, j), v in results.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(output_dir, "round_robin_results.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


def load_partial_results(output_dir):
    """Load partial results from a previous run."""
    path = os.path.join(output_dir, "round_robin_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    results = {}
    for key, val in data.get("results", {}).items():
        i, j = map(int, key.split(","))
        results[(i, j)] = tuple(val)
    return results


def main():
    parser = argparse.ArgumentParser(description="Round-robin tournament")
    parser.add_argument("--games", type=int, default=200, help="Games per match")
    parser.add_argument("--sims", type=int, default=200, help="Simulations per move")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--threads", type=int, default=28, help="Game threads")
    parser.add_argument("--explore-base", type=float, default=1.0, help="Explore base (1.0=always proportional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "round_robin_10model")
    os.makedirs(output_dir, exist_ok=True)

    model_names = [m[0] for m in MODELS]
    n = len(model_names)

    # Verify all model files exist
    for name, path, mtype in MODELS:
        if not os.path.exists(path):
            print(f"ERROR: Model file not found: {path} ({name})")
            sys.exit(1)

    # Verify binary exists
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary not found: {BINARY}")
        print("Run: cargo build --release --bin evaluate_models")
        sys.exit(1)

    # Load partial results if resuming
    results = load_partial_results(output_dir) if args.resume else {}

    # All pairings (i < j to avoid duplicates)
    pairings = list(itertools.combinations(range(n), 2))
    total_matches = len(pairings)
    completed = sum(1 for p in pairings if p in results)

    print(f"Tournament: {n} models, {total_matches} matches, {args.games} games each")
    print(f"Models: {', '.join(model_names)}")
    if completed > 0:
        print(f"Resuming: {completed}/{total_matches} matches already completed")
    print()

    match_num = completed
    for i, j in pairings:
        if (i, j) in results:
            continue

        match_num += 1
        cand_name, cand_path, cand_type = MODELS[i]
        curr_name, curr_path, curr_type = MODELS[j]

        print(f"\n[Match {match_num}/{total_matches}]")

        # Use a unique seed offset per pairing so results are reproducible
        seed_offset = i * 1000 + j * 10
        result = run_match(
            cand_name, cand_path, cand_type,
            curr_name, curr_path, curr_type,
            args.games, args.sims, args.batch_size, args.threads,
            args.explore_base, seed_offset,
        )

        if result is not None:
            results[(i, j)] = result
            # Save after each match for resume capability
            elos = compute_elo_mle(results, model_names)
            save_results(model_names, elos, results, output_dir)
        else:
            print(f"  SKIPPING failed match {cand_name} vs {curr_name}")

    # Final results
    if results:
        elos = compute_elo_mle(results, model_names)
        print_results_table(model_names, elos, results)
        save_results(model_names, elos, results, output_dir)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
