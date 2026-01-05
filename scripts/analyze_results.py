#!/usr/bin/env python3
"""
Analyze experiment results and generate publication-quality figures.

Usage:
    python scripts/analyze_results.py results/ablation_results.json
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Use publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_results(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def plot_nn_reduction(results: list, output_dir: Path):
    """Bar chart showing NN call reduction by configuration."""
    configs = [r['config_name'] for r in results]
    reductions = [r['aggregated']['mean_nn_reduction_percent'] for r in results]
    errors = [
        1.96 * r['aggregated']['std_nn_reduction'] / np.sqrt(r['aggregated']['num_positions'])
        for r in results
    ]
    
    fig, ax = plt.subplots()
    bars = ax.bar(configs, reductions, yerr=errors, capsize=5, 
                  color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'],
                  edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Neural Network Call Reduction (%)')
    ax.set_xlabel('Configuration')
    ax.set_title('Sample Efficiency: NN Calls Saved by Tier System')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars, reductions):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'nn_reduction.pdf')
    plt.savefig(output_dir / 'nn_reduction.png')
    plt.close()
    print(f"✓ Saved nn_reduction.pdf/png")


def plot_tier_breakdown(results: list, output_dir: Path):
    """Stacked bar chart showing contribution of each tier."""
    configs = [r['config_name'] for r in results]
    tier1 = [r['aggregated']['total_tier1_activations'] for r in results]
    tier2 = [r['aggregated']['total_tier2_grafts'] for r in results]
    nn = [r['aggregated']['total_nn_evaluations'] for r in results]
    
    x = np.arange(len(configs))
    width = 0.6
    
    fig, ax = plt.subplots()
    
    p1 = ax.bar(x, tier1, width, label='Tier 1 (Safety Gates)', color='#e74c3c')
    p2 = ax.bar(x, tier2, width, bottom=tier1, label='Tier 2 (Tactical Grafts)', color='#f39c12')
    p3 = ax.bar(x, nn, width, bottom=np.array(tier1) + np.array(tier2), 
                label='Tier 3 (Neural Network)', color='#3498db')
    
    ax.set_ylabel('Number of Evaluations')
    ax.set_xlabel('Configuration')
    ax.set_title('Evaluation Source Breakdown by Tier')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tier_breakdown.pdf')
    plt.savefig(output_dir / 'tier_breakdown.png')
    plt.close()
    print(f"✓ Saved tier_breakdown.pdf/png")

def plot_safety_metrics(results: list, output_dir: Path):
    """Plot safety metrics - tactical accuracy."""
    configs = [r['config_name'] for r in results]
    safety_rates = [r['safety']['forced_mates_found'] / max(r['safety']['positions_with_forced_mate'], 1) * 100 
                    for r in results]
    
    fig, ax = plt.subplots()
    bars = ax.bar(configs, safety_rates, 
                  color=['#27ae60' if rate > 80 else '#e74c3c' for rate in safety_rates],
                  edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Forced Mate Detection Rate (%)')
    ax.set_xlabel('Configuration')
    ax.set_title('Safety: Ability to Find Forced Mates')
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Safety')
    
    for bar, val in zip(bars, safety_rates):
        height = bar.get_height()
        ax.annotate(f'{val:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'safety_metrics.pdf')
    plt.savefig(output_dir / 'safety_metrics.png')
    plt.close()
    print(f"✓ Saved safety_metrics.pdf/png")

def plot_efficiency_vs_accuracy(results: list, output_dir: Path):
    """Scatter plot showing trade-off between efficiency and accuracy."""
    fig, ax = plt.subplots()
    
    for r in results:
        nn_reduction = r['aggregated']['mean_nn_reduction_percent']
        # Calculate accuracy as percentage of correct moves
        correct = sum(1 for p in r['position_results'] if p['correct'])
        total = len(r['position_results'])
        accuracy = (correct / total * 100) if total > 0 else 0
        
        ax.scatter(nn_reduction, accuracy, s=150, alpha=0.8, 
                   label=r['config_name'], edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Neural Network Call Reduction (%)')
    ax.set_ylabel('Move Accuracy (%)')
    ax.set_title('Efficiency vs Accuracy Trade-off')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add ideal region annotation
    ax.annotate('Ideal\nRegion', xy=(70, 90), fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_vs_accuracy.pdf')
    plt.savefig(output_dir / 'efficiency_vs_accuracy.png')
    plt.close()
    print(f"✓ Saved efficiency_vs_accuracy.pdf/png")

def generate_summary_table(results: list, output_dir: Path):
    """Generate a summary markdown table."""
    lines = [
        "# Ablation Study Results Summary\n",
        "| Configuration | NN Reduction | Tier 1 | Tier 2 | NN Calls | Safety |",
        "|---------------|--------------|--------|--------|----------|--------|",
    ]
    
    for r in results:
        agg = r['aggregated']
        safety_rate = r['safety']['forced_mates_found'] / max(r['safety']['positions_with_forced_mate'], 1) * 100
        lines.append(
            f"| {r['config_name']} | {agg['mean_nn_reduction_percent']:.1f}% | "
            f"{agg['total_tier1_activations']} | {agg['total_tier2_grafts']} | "
            f"{agg['total_nn_evaluations']} | {safety_rate:.0f}% |"
        )
    
    with open(output_dir / 'summary.md', 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ Saved summary.md")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py results/ablation_results.json")
        sys.exit(1)
    
    results_path = sys.argv[1]
    results = load_results(results_path)
    
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Analyzing {len(results)} experiment configurations...\n")
    
    plot_nn_reduction(results, output_dir)
    plot_tier_breakdown(results, output_dir)
    plot_safety_metrics(results, output_dir)
    plot_efficiency_vs_accuracy(results, output_dir)
    generate_summary_table(results, output_dir)
    
    print(f"\n✅ All figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
