#!/usr/bin/env python3
"""Compute MLE Elo ratings from tournament CSV with bootstrap confidence intervals."""

import csv
import math
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_pairwise_results(csv_path):
    """Load pairwise results from the tournament CSV (stops at blank line)."""
    results = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip():
                break
            model_a, model_b = row[0], row[1]
            wins_a, draws, wins_b = int(row[2]), int(row[3]), int(row[4])
            results.append((model_a, model_b, wins_a, draws, wins_b))
    return results


def expand_to_games(results):
    """Expand pairwise summary results into individual game outcomes.

    Returns list of (model_a, model_b, outcome) where outcome is
    1.0 (a wins), 0.5 (draw), or 0.0 (b wins).
    """
    games = []
    for model_a, model_b, wins_a, draws, wins_b in results:
        games.extend([(model_a, model_b, 1.0)] * wins_a)
        games.extend([(model_a, model_b, 0.5)] * draws)
        games.extend([(model_a, model_b, 0.0)] * wins_b)
    return games


def mle_elo(results, iterations=2000, lr=10.0):
    """Compute MLE Elo ratings via gradient ascent on Bradley-Terry log-likelihood."""
    # Collect all model names
    names = set()
    for a, b, *_ in results:
        names.add(a)
        names.add(b)
    names = sorted(names)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    ratings = [1500.0] * n

    for _ in range(iterations):
        grad = [0.0] * n
        for model_a, model_b, wins_a, draws, wins_b in results:
            i, j = idx[model_a], idx[model_b]
            n_games = wins_a + draws + wins_b
            if n_games == 0:
                continue
            s_ij = wins_a + draws / 2.0  # score for model_a
            expected = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
            g = (math.log(10) / 400.0) * (s_ij - n_games * expected)
            grad[i] += g
            grad[j] -= g

        for k in range(n):
            ratings[k] += lr * grad[k]

        # Re-anchor mean to 1500
        mean = sum(ratings) / n
        for k in range(n):
            ratings[k] += 1500.0 - mean

    return {name: ratings[idx[name]] for name in names}


def log_likelihood(results, ratings):
    """Compute the log-likelihood of the observed results given ratings."""
    ll = 0.0
    for model_a, model_b, wins_a, draws, wins_b in results:
        n_games = wins_a + draws + wins_b
        if n_games == 0:
            continue
        s_ij = wins_a + draws / 2.0
        expected = 1.0 / (1.0 + 10.0 ** ((ratings[model_b] - ratings[model_a]) / 400.0))
        # Avoid log(0)
        expected = max(min(expected, 1.0 - 1e-10), 1e-10)
        ll += s_ij * math.log(expected) + (n_games - s_ij) * math.log(1.0 - expected)
    return ll


def bootstrap_elo(results, n_resamples=1000, seed=42):
    """Bootstrap resample pairwise game results and compute Elo CIs.

    For each pair, resamples individual games with replacement, then
    recomputes MLE Elo. Returns dict of {model: (elo_mean, ci_low, ci_high)}.
    """
    rng = np.random.RandomState(seed)

    # Expand results to individual games grouped by pair
    pair_games = {}
    for model_a, model_b, wins_a, draws, wins_b in results:
        key = (model_a, model_b)
        # Each game is a score for model_a: 1.0, 0.5, or 0.0
        outcomes = [1.0] * wins_a + [0.5] * draws + [0.0] * wins_b
        pair_games[key] = np.array(outcomes)

    # Collect model names from point estimate
    point_ratings = mle_elo(results)
    names = sorted(point_ratings.keys())

    # Run bootstrap resamples
    all_ratings = {name: [] for name in names}

    for _ in range(n_resamples):
        # Resample within each pair
        resampled_results = []
        for (model_a, model_b), outcomes in pair_games.items():
            n_games = len(outcomes)
            if n_games == 0:
                continue
            # Resample with replacement
            boot_outcomes = outcomes[rng.randint(0, n_games, size=n_games)]
            boot_score = boot_outcomes.sum()
            boot_wins_a = int((boot_outcomes == 1.0).sum())
            boot_draws = int((boot_outcomes == 0.5).sum())
            boot_wins_b = n_games - boot_wins_a - boot_draws
            resampled_results.append((model_a, model_b, boot_wins_a, boot_draws, boot_wins_b))

        # Compute MLE Elo on resampled data
        boot_ratings = mle_elo(resampled_results)
        for name in names:
            if name in boot_ratings:
                all_ratings[name].append(boot_ratings[name])

    # Compute 95% CIs
    ci_results = {}
    for name in names:
        samples = np.array(all_ratings[name])
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        ci_results[name] = (point_ratings[name], ci_low, ci_high)

    return ci_results


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "tournament_results_14way.csv"
    results = load_pairwise_results(csv_path)
    ratings = mle_elo(results)

    # Bootstrap confidence intervals
    print("Computing bootstrap confidence intervals (1000 resamples)...")
    ci_results = bootstrap_elo(results, n_resamples=1000)

    # Print ratings with CIs
    ranked = sorted(ci_results.items(), key=lambda x: -x[1][0])
    print("\n=== MLE Elo Ratings with 95% Bootstrap CI ===")
    print(f"{'Model':<20} {'Elo':>7}  {'95% CI':>16}")
    print("-" * 48)
    for name, (elo, ci_low, ci_high) in ranked:
        half_width = (ci_high - ci_low) / 2
        print(f"{name:<20} {elo:>+7.1f}  [{ci_low:>+7.1f}, {ci_high:>+7.1f}]  (±{half_width:.0f})")

    # Print markdown table for README
    print("\n=== Markdown Table (for README) ===")
    print("| Rank | Model | Elo | Type |")
    print("|------|-------|-----|------|")
    for i, (name, (elo, ci_low, ci_high)) in enumerate(ranked, 1):
        half_width = (ci_high - ci_low) / 2
        model_type = "Tiered" if name.startswith("tiered_") else "Vanilla"
        print(f"| {i} | {name} | {elo:.0f} ± {half_width:.0f} | {model_type} |")

    ll = log_likelihood(results, ratings)
    print(f"\nLog-likelihood: {ll:.2f}")

    # Parse generation numbers and group by run
    tiered = {}
    vanilla = {}
    tiered_ci = {}
    vanilla_ci = {}
    for name, (elo, ci_low, ci_high) in ci_results.items():
        gen = int(name.split("gen")[1])
        if name.startswith("tiered_"):
            tiered[gen] = elo
            tiered_ci[gen] = (ci_low, ci_high)
        elif name.startswith("vanilla_"):
            vanilla[gen] = elo
            vanilla_ci[gen] = (ci_low, ci_high)

    # Sort by generation
    tiered_gens = sorted(tiered.keys())
    tiered_elos = [tiered[g] for g in tiered_gens]
    tiered_ci_low = [tiered_ci[g][0] for g in tiered_gens]
    tiered_ci_high = [tiered_ci[g][1] for g in tiered_gens]
    vanilla_gens = sorted(vanilla.keys())
    vanilla_elos = [vanilla[g] for g in vanilla_gens]
    vanilla_ci_low = [vanilla_ci[g][0] for g in vanilla_gens]
    vanilla_ci_high = [vanilla_ci[g][1] for g in vanilla_gens]

    # Plot as step functions with error bands
    fig, ax = plt.subplots(figsize=(8, 5))

    # Tiered: step plot with shaded CI band
    ax.fill_between(tiered_gens, tiered_ci_low, tiered_ci_high,
                     step="post", alpha=0.15, color="#2563eb", zorder=1)
    ax.step(tiered_gens, tiered_elos, where="post", color="#2563eb", linewidth=2,
            label="Tiered (safety gates + material)", zorder=3)
    ax.plot(tiered_gens, tiered_elos, "o", color="#2563eb", markersize=6, zorder=4)

    # Vanilla: step plot with shaded CI band
    ax.fill_between(vanilla_gens, vanilla_ci_low, vanilla_ci_high,
                     step="post", alpha=0.15, color="#dc2626", zorder=1)
    ax.step(vanilla_gens, vanilla_elos, where="post", color="#dc2626", linewidth=2,
            label="Vanilla (NN only)", zorder=3)
    ax.plot(vanilla_gens, vanilla_elos, "s", color="#dc2626", markersize=6, zorder=4)

    ax.set_xlabel("Accepted Generation", fontsize=12)
    ax.set_ylabel("Elo Rating (MLE)", fontsize=12)
    ax.set_title("Caissawary: Tiered vs Vanilla MCTS on KOTH Chess", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, max(max(tiered_gens), max(vanilla_gens)) + 2)

    plt.tight_layout()
    out_path = csv_path.replace(".csv", "_elo_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
