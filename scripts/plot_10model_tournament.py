#!/usr/bin/env python3
"""Plot Elo ratings from 10-model adaptive tournament with bootstrap CIs."""

import json
import math
import random
import matplotlib.pyplot as plt
import numpy as np

def compute_mle_elos(models, results, anchor_idx=4, anchor_elo=1500.0, iterations=1000):
    """Compute MLE Elo ratings using Bradley-Terry iteration."""
    n = len(models)
    elos = [anchor_elo] * n

    for _ in range(iterations):
        for i in range(n):
            if i == anchor_idx:
                continue
            wins_i = 0.0
            games_i = 0.0
            expected_i = 0.0
            for j in range(n):
                if i == j:
                    continue
                key_ij = f"{min(i,j)},{max(i,j)}"
                if key_ij not in results:
                    continue
                w, l, d = results[key_ij]
                if i < j:
                    wi, wj = w, l
                else:
                    wi, wj = l, w
                total = wi + wj + d
                if total == 0:
                    continue
                wins_i += wi + 0.5 * d
                games_i += total
                exp_score = 1.0 / (1.0 + 10.0 ** ((elos[j] - elos[i]) / 400.0))
                expected_i += total * exp_score

            if expected_i > 0 and games_i > 0:
                ratio = wins_i / expected_i
                if ratio > 0:
                    elos[i] += 400.0 * math.log10(ratio) * 0.5

    # Re-anchor
    offset = anchor_elo - elos[anchor_idx]
    elos = [e + offset for e in elos]
    return elos


def bootstrap_elos(models, results, n_bootstrap=500, anchor_idx=4):
    """Bootstrap resample results and compute Elo CIs."""
    all_elos = []
    for _ in range(n_bootstrap):
        resampled = {}
        for key, (w, l, d) in results.items():
            total = w + l + d
            if total == 0:
                resampled[key] = [0, 0, 0]
                continue
            outcomes = random.choices(['w', 'l', 'd'], weights=[w, l, d], k=total)
            rw = outcomes.count('w')
            rl = outcomes.count('l')
            rd = outcomes.count('d')
            resampled[key] = [rw, rl, rd]
        elos = compute_mle_elos(models, resampled, anchor_idx=anchor_idx)
        all_elos.append(elos)

    all_elos = np.array(all_elos)
    means = np.mean(all_elos, axis=0)
    ci_low = np.percentile(all_elos, 2.5, axis=0)
    ci_high = np.percentile(all_elos, 97.5, axis=0)
    return means, ci_low, ci_high


def main():
    with open("runs/tournaments/round_robin_10model/round_robin_results.json") as f:
        data = json.load(f)

    models = data["models"]
    elos_mle = data["elos"]
    results = {k: v for k, v in data["results"].items()}

    # Bootstrap CIs
    random.seed(42)
    _, ci_low, ci_high = bootstrap_elos(models, results, n_bootstrap=500, anchor_idx=4)

    # Separate tiered and vanilla
    tiered_gens = []
    tiered_elos = []
    tiered_ci_lo = []
    tiered_ci_hi = []
    vanilla_gens = []
    vanilla_elos = []
    vanilla_ci_lo = []
    vanilla_ci_hi = []

    for i, name in enumerate(models):
        gen = int(name.split("_gen")[1])
        if name.startswith("tiered"):
            tiered_gens.append(gen)
            tiered_elos.append(elos_mle[i])
            tiered_ci_lo.append(ci_low[i])
            tiered_ci_hi.append(ci_high[i])
        else:
            vanilla_gens.append(gen)
            vanilla_elos.append(elos_mle[i])
            vanilla_ci_lo.append(ci_low[i])
            vanilla_ci_hi.append(ci_high[i])

    # Sort by generation
    tiered_order = np.argsort(tiered_gens)
    tiered_gens = [tiered_gens[i] for i in tiered_order]
    tiered_elos = [tiered_elos[i] for i in tiered_order]
    tiered_ci_lo = [tiered_ci_lo[i] for i in tiered_order]
    tiered_ci_hi = [tiered_ci_hi[i] for i in tiered_order]

    vanilla_order = np.argsort(vanilla_gens)
    vanilla_gens = [vanilla_gens[i] for i in vanilla_order]
    vanilla_elos = [vanilla_elos[i] for i in vanilla_order]
    vanilla_ci_lo = [vanilla_ci_lo[i] for i in vanilla_order]
    vanilla_ci_hi = [vanilla_ci_hi[i] for i in vanilla_order]

    max_gen = 20

    # Build step-function data (horizontal lines between generations, extending to gen 20)
    def make_step(gens, vals, ci_lo, ci_hi):
        xs, ys, lo, hi = [], [], [], []
        for i in range(len(gens)):
            x_start = gens[i]
            x_end = gens[i + 1] if i + 1 < len(gens) else max_gen
            xs.extend([x_start, x_end])
            ys.extend([vals[i], vals[i]])
            lo.extend([ci_lo[i], ci_lo[i]])
            hi.extend([ci_hi[i], ci_hi[i]])
        return xs, ys, lo, hi

    t_xs, t_ys, t_lo, t_hi = make_step(tiered_gens, tiered_elos, tiered_ci_lo, tiered_ci_hi)
    v_xs, v_ys, v_lo, v_hi = make_step(vanilla_gens, vanilla_elos, vanilla_ci_lo, vanilla_ci_hi)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tiered (blue)
    ax.plot(t_xs, t_ys, color='#1f77b4', linewidth=2, zorder=3)
    ax.fill_between(t_xs, t_lo, t_hi, color='#1f77b4', alpha=0.15, zorder=1)
    ax.scatter(tiered_gens, tiered_elos, color='#1f77b4', s=50, zorder=4,
               label='Caissawary (MCTS with forced win and quiescence search)')

    # Vanilla (red)
    ax.plot(v_xs, v_ys, color='#d62728', linewidth=2, zorder=3)
    ax.fill_between(v_xs, v_lo, v_hi, color='#d62728', alpha=0.15, zorder=1)
    ax.scatter(vanilla_gens, vanilla_elos, color='#d62728', s=50, zorder=4, marker='s',
               label='Vanilla MCTS (AlphaZero-style)')

    ax.set_xlabel('Accepted Generation', fontsize=12)
    ax.set_ylabel('Elo Rating (MLE)', fontsize=12)
    ax.set_title('Caissawary: Tiered vs Vanilla MCTS on KOTH Chess', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(-0.5, max_gen + 0.5)
    ax.set_xticks(range(0, max_gen + 1, 5))
    ax.set_ylim(1400, None)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tournament_results_10model_elo_plot.png', dpi=150)
    print("Saved tournament_results_10model_elo_plot.png")

    # Print summary
    print("\nElo Ratings with 95% CI:")
    for i, name in enumerate(models):
        print(f"  {name:20s}: {elos_mle[i]:7.1f}  [{ci_low[i]:.1f}, {ci_high[i]:.1f}]")


if __name__ == "__main__":
    main()
