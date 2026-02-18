#!/usr/bin/env python3
"""Plot Elo ratings from round-robin tournament JSON results with bootstrap CIs."""

import json
import math
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path):
    """Load results from round_robin_results.json format."""
    with open(json_path) as f:
        data = json.load(f)

    models = data["models"]
    results = {}
    for key, val in data.get("results", {}).items():
        i, j = map(int, key.split(","))
        results[(i, j)] = tuple(val)

    # Convert to pairwise format: (model_a, model_b, wins_a, draws, wins_b)
    pairwise = []
    for (i, j), (wi, wj, d) in results.items():
        pairwise.append((models[i], models[j], wi, d, wj))

    return models, pairwise


def mle_elo(results, anchor_name=None, anchor_elo=1500, iterations=2000, lr=10.0):
    """Compute MLE Elo ratings via gradient ascent on Bradley-Terry log-likelihood."""
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
            s_ij = wins_a + draws / 2.0
            expected = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
            g = (math.log(10) / 400.0) * (s_ij - n_games * expected)
            grad[i] += g
            grad[j] -= g

        for k in range(n):
            ratings[k] += lr * grad[k]

        # Re-anchor
        if anchor_name and anchor_name in idx:
            offset = anchor_elo - ratings[idx[anchor_name]]
        else:
            offset = 1500.0 - sum(ratings) / n
        for k in range(n):
            ratings[k] += offset

    return {name: ratings[idx[name]] for name in names}


def bootstrap_elo(results, anchor_name=None, anchor_elo=1500, n_resamples=1000, seed=42):
    """Bootstrap resample pairwise results and compute Elo CIs."""
    rng = np.random.RandomState(seed)

    pair_games = {}
    for model_a, model_b, wins_a, draws, wins_b in results:
        key = (model_a, model_b)
        outcomes = [1.0] * wins_a + [0.5] * draws + [0.0] * wins_b
        pair_games[key] = np.array(outcomes)

    point_ratings = mle_elo(results, anchor_name, anchor_elo)
    names = sorted(point_ratings.keys())

    all_ratings = {name: [] for name in names}

    for _ in range(n_resamples):
        resampled_results = []
        for (model_a, model_b), outcomes in pair_games.items():
            n_games = len(outcomes)
            if n_games == 0:
                continue
            boot_outcomes = outcomes[rng.randint(0, n_games, size=n_games)]
            boot_wins_a = int((boot_outcomes == 1.0).sum())
            boot_draws = int((boot_outcomes == 0.5).sum())
            boot_wins_b = n_games - boot_wins_a - boot_draws
            resampled_results.append((model_a, model_b, boot_wins_a, boot_draws, boot_wins_b))

        boot_ratings = mle_elo(resampled_results, anchor_name, anchor_elo)
        for name in names:
            if name in boot_ratings:
                all_ratings[name].append(boot_ratings[name])

    ci_results = {}
    for name in names:
        samples = np.array(all_ratings[name])
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        ci_results[name] = (point_ratings[name], ci_low, ci_high)

    return ci_results


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "runs/tournaments/round_robin_800eval/round_robin_results.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tournament_results_800eval_elo_plot.png"
    anchor_name = "vanilla_gen0"

    models, pairwise = load_results(json_path)

    print(f"Loaded {len(pairwise)} pairwise results for {len(models)} models")
    total_games = sum(w + d + l for _, _, w, d, l in pairwise)
    print(f"Total games: {total_games}")

    # Bootstrap CIs
    print("Computing bootstrap confidence intervals (1000 resamples)...")
    ci_results = bootstrap_elo(pairwise, anchor_name=anchor_name, n_resamples=1000)

    # Print ratings
    ranked = sorted(ci_results.items(), key=lambda x: -x[1][0])
    print(f"\n{'Model':<20} {'Elo':>7}  {'95% CI':>16}")
    print("-" * 48)
    for name, (elo, ci_low, ci_high) in ranked:
        half_width = (ci_high - ci_low) / 2
        print(f"{name:<20} {elo:>+7.1f}  [{ci_low:>+7.1f}, {ci_high:>+7.1f}]  (±{half_width:.0f})")

    # Parse generation numbers and group by type
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

    tiered_gens = sorted(tiered.keys())
    tiered_elos = [tiered[g] for g in tiered_gens]
    tiered_ci_low = [tiered_ci[g][0] for g in tiered_gens]
    tiered_ci_high = [tiered_ci[g][1] for g in tiered_gens]
    vanilla_gens = sorted(vanilla.keys())
    vanilla_elos = [vanilla[g] for g in vanilla_gens]
    vanilla_ci_low = [vanilla_ci[g][0] for g in vanilla_gens]
    vanilla_ci_high = [vanilla_ci[g][1] for g in vanilla_gens]

    # Extend lines to total generations (step function holds last value)
    tiered_max_gen = int(sys.argv[3]) if len(sys.argv) > 3 else tiered_gens[-1]
    vanilla_max_gen = int(sys.argv[4]) if len(sys.argv) > 4 else vanilla_gens[-1]
    if tiered_max_gen > tiered_gens[-1]:
        tiered_gens.append(tiered_max_gen)
        tiered_elos.append(tiered_elos[-1])
        tiered_ci_low.append(tiered_ci_low[-1])
        tiered_ci_high.append(tiered_ci_high[-1])
    if vanilla_max_gen > vanilla_gens[-1]:
        vanilla_gens.append(vanilla_max_gen)
        vanilla_elos.append(vanilla_elos[-1])
        vanilla_ci_low.append(vanilla_ci_low[-1])
        vanilla_ci_high.append(vanilla_ci_high[-1])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Markers only on real data points (not extension points)
    tiered_real = sorted(tiered.keys())
    vanilla_real = sorted(vanilla.keys())

    # Tiered
    ax.fill_between(tiered_gens, tiered_ci_low, tiered_ci_high,
                     step="post", alpha=0.15, color="#2563eb", zorder=1)
    ax.step(tiered_gens, tiered_elos, where="post", color="#2563eb", linewidth=2,
            label="Caissawary (MCTS with forced win and quiescence search)", zorder=3)
    ax.plot(tiered_real, [tiered[g] for g in tiered_real], "o", color="#2563eb", markersize=6, zorder=4)

    # Vanilla
    ax.fill_between(vanilla_gens, vanilla_ci_low, vanilla_ci_high,
                     step="post", alpha=0.15, color="#dc2626", zorder=1)
    ax.step(vanilla_gens, vanilla_elos, where="post", color="#dc2626", linewidth=2,
            label="Vanilla MCTS (AlphaZero-style)", zorder=3)
    ax.plot(vanilla_real, [vanilla[g] for g in vanilla_real], "s", color="#dc2626", markersize=6, zorder=4)

    ax.set_xlabel("Accepted Generation", fontsize=12)
    ax.set_ylabel("Elo Rating (MLE)", fontsize=12)
    ax.set_title("Caissawary: Tiered vs Vanilla MCTS on KOTH Chess", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, max(max(tiered_gens), max(vanilla_gens)) + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
