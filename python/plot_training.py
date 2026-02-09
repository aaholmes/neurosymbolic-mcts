"""Plot training comparison between baseline (pure AlphaZero) and Caissawary (3-tier MCTS).

Usage:
    python3 python/plot_training.py \
        --baseline runs/xxx/baseline/training_log.jsonl \
        --caissawary runs/xxx/caissawary/training_log.jsonl \
        --output runs/xxx/plots/
"""

import argparse
import json
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


def load_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def plot_winrate(ax, baseline, caissawary, threshold=0.52):
    if baseline:
        gens = [e["gen"] for e in baseline]
        wr = [e["eval_winrate"] for e in baseline]
        ax.plot(gens, wr, "o-", label="Baseline (Pure AZ)", color="tab:blue", markersize=3)
    if caissawary:
        gens = [e["gen"] for e in caissawary]
        wr = [e["eval_winrate"] for e in caissawary]
        ax.plot(gens, wr, "s-", label="Caissawary (3-tier)", color="tab:orange", markersize=3)
    ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")
    ax.axhline(y=0.5, color="lightgray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Eval Win Rate")
    ax.set_title("Eval Win Rate vs Generation")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)


def plot_loss(ax, baseline, caissawary):
    for data, label, color in [(baseline, "Baseline", "tab:blue"), (caissawary, "Caissawary", "tab:orange")]:
        if not data:
            continue
        gens = [e["gen"] for e in data if e.get("training_loss") is not None]
        loss = [e["training_loss"] for e in data if e.get("training_loss") is not None]
        ploss = [e["training_policy_loss"] for e in data if e.get("training_policy_loss") is not None]
        vloss = [e["training_value_loss"] for e in data if e.get("training_value_loss") is not None]
        if gens:
            ax.plot(gens, loss, "o-", label=f"{label} Total", color=color, markersize=2)
            ax.plot(gens, ploss, "^--", label=f"{label} Policy", color=color, markersize=2, alpha=0.6)
            ax.plot(gens, vloss, "v:", label=f"{label} Value", color=color, markersize=2, alpha=0.6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Generation")
    ax.legend(fontsize=7)


def plot_acceptance_rate(ax, baseline, caissawary, window=5):
    for data, label, color in [(baseline, "Baseline", "tab:blue"), (caissawary, "Caissawary", "tab:orange")]:
        if not data or len(data) < window:
            continue
        gens = [e["gen"] for e in data]
        accepted = [1.0 if e["accepted"] else 0.0 for e in data]
        rolling = []
        for i in range(len(accepted)):
            start = max(0, i - window + 1)
            rolling.append(sum(accepted[start:i+1]) / (i - start + 1))
        ax.plot(gens, rolling, "o-", label=f"{label} ({window}-gen rolling)", color=color, markersize=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title(f"Rolling Acceptance Rate ({window}-gen window)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)


def plot_k_evolution(ax, baseline, caissawary):
    for data, label, color in [(baseline, "Baseline", "tab:blue"), (caissawary, "Caissawary", "tab:orange")]:
        if not data:
            continue
        gens = [e["gen"] for e in data if e.get("training_k_mean") is not None]
        k_vals = [e["training_k_mean"] for e in data if e.get("training_k_mean") is not None]
        if gens:
            ax.plot(gens, k_vals, "o-", label=label, color=color, markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("k (material confidence)")
    ax.set_title("K (Material Confidence) Evolution")
    ax.legend(fontsize=8)


def compute_cumulative_elo(entries):
    """Track best model's Elo over generations.
    Gen 0 = Elo 0. Each eval match gives an Elo diff vs current best.
    On acceptance, best Elo jumps up. On rejection, stays flat."""
    elos = [0.0]  # gen 0
    current_best_elo = 0.0
    for e in entries:
        wins, losses, draws = e["eval_wins"], e["eval_losses"], e["eval_draws"]
        total = wins + losses + draws
        score = (wins + 0.5 * draws) / total
        score = max(0.01, min(0.99, score))  # clamp to avoid log(0)
        elo_diff = 400 * math.log10(score / (1 - score))
        if e["accepted"]:
            current_best_elo += elo_diff
        elos.append(current_best_elo)
    return elos


def plot_elo(ax, baseline, caissawary):
    for data, label, color in [(baseline, "Baseline", "tab:blue"), (caissawary, "Caissawary", "tab:orange")]:
        if not data:
            continue
        elos = compute_cumulative_elo(data)
        gens = list(range(len(elos)))
        ax.step(gens, elos, where="post", label=label, color=color, linewidth=1.5)
        # Mark accepted gens with dots
        for i, e in enumerate(data):
            if e["accepted"]:
                ax.plot(i + 1, elos[i + 1], "o", color=color, markersize=4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative Elo")
    ax.set_title("Self-Play Elo (Cumulative)")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="lightgray", linestyle=":", alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description="Plot training comparison")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline log JSONL")
    parser.add_argument("--caissawary", type=str, default=None, help="Caissawary log JSONL")
    parser.add_argument("--output", type=str, default="plots/", help="Output directory")
    args = parser.parse_args()

    if not args.baseline and not args.caissawary:
        print("Provide at least one of --baseline or --caissawary")
        sys.exit(1)

    baseline = load_log(args.baseline) if args.baseline else None
    caissawary = load_log(args.caissawary) if args.caissawary else None

    os.makedirs(args.output, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Training Comparison: Pure AlphaZero vs Caissawary", fontsize=14)

    plot_winrate(axes[0, 0], baseline, caissawary)
    plot_loss(axes[0, 1], baseline, caissawary)
    plot_acceptance_rate(axes[1, 0], baseline, caissawary)
    plot_k_evolution(axes[1, 1], baseline, caissawary)
    plot_elo(axes[2, 0], baseline, caissawary)
    axes[2, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.output, "training_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # Also save individual plots
    for name, plot_fn in [("winrate", plot_winrate), ("loss", plot_loss),
                           ("acceptance", plot_acceptance_rate), ("k_evolution", plot_k_evolution),
                           ("elo", plot_elo)]:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        if name == "winrate":
            plot_fn(ax2, baseline, caissawary)
        elif name == "loss":
            plot_fn(ax2, baseline, caissawary)
        elif name == "acceptance":
            plot_fn(ax2, baseline, caissawary)
        else:
            plot_fn(ax2, baseline, caissawary)
        plt.tight_layout()
        path = os.path.join(args.output, f"{name}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig2)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
