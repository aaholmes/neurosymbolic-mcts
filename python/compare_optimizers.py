"""
Compare Adam, AdamW, and Muon optimizers on chess training data.

Runs training with each optimizer and generates comparison plots.

Usage:
    python3 python/compare_optimizers.py [data_dir] [--epochs N] [--runs N]
"""
import subprocess
import sys
import re
import csv
from pathlib import Path

OPTIMIZERS = ['adam', 'adamw', 'muon']


def run_training(optimizer, run_id, data_dir, output_dir, epochs):
    """Run training with specified optimizer and capture loss per epoch."""
    model_path = f'{output_dir}/{optimizer}_run{run_id}.pth'
    cmd = [
        sys.executable, 'python/train.py',
        data_dir,
        model_path,
        '--optimizer', optimizer,
        '--epochs', str(epochs),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse per-epoch average losses from output
    epoch_results = []
    for line in result.stdout.split('\n'):
        # Match: "Epoch N Average: Policy=X.XXXX Value=X.XXXX K=X.XXXX"
        m = re.search(r'Epoch (\d+) Average: Policy=([\d.]+) Value=([\d.]+) K=([\d.]+)', line)
        if m:
            epoch_results.append({
                'epoch': int(m.group(1)),
                'policy_loss': float(m.group(2)),
                'value_loss': float(m.group(3)),
                'k': float(m.group(4)),
                'total_loss': float(m.group(2)) + float(m.group(3)),
            })

    if result.returncode != 0:
        print(f"  WARNING: Training failed (exit code {result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")

    return epoch_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare optimizers on chess training")
    parser.add_argument('data_dir', type=str, nargs='?', default='data/training',
                        help='Training data directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs per run (default: 20)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Runs per optimizer (default: 3)')
    args = parser.parse_args()

    output_dir = Path('results/optimizer_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for optimizer in OPTIMIZERS:
        print(f"\n{'='*60}")
        print(f"Training with {optimizer.upper()}")
        print(f"{'='*60}")

        for run in range(args.runs):
            print(f"\nRun {run+1}/{args.runs}...")
            epoch_results = run_training(
                optimizer, run, args.data_dir, str(output_dir), args.epochs
            )

            for er in epoch_results:
                all_results.append({
                    'optimizer': optimizer,
                    'run': run,
                    **er,
                })

    # Save CSV results
    csv_path = output_dir / 'results.csv'
    if all_results:
        fieldnames = ['optimizer', 'run', 'epoch', 'policy_loss', 'value_loss', 'total_loss', 'k']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")
    else:
        print("\nNo results collected. Check that training data exists.")
        return

    # Generate plots (optional — only if matplotlib available)
    try:
        generate_plots(all_results, output_dir)
    except ImportError:
        print("matplotlib/seaborn not available — skipping plots.")
        print("Install with: pip install matplotlib seaborn")

    print_summary(all_results, args.epochs)


def generate_plots(results, output_dir):
    """Generate comparison plots if matplotlib is available."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for optimizer in OPTIMIZERS:
        opt_data = [r for r in results if r['optimizer'] == optimizer]
        if not opt_data:
            continue

        # Group by epoch, compute mean and std across runs
        epochs = sorted(set(r['epoch'] for r in opt_data))
        for metric_idx, (metric, title) in enumerate([
            ('total_loss', 'Total Loss'),
            ('policy_loss', 'Policy Loss'),
            ('value_loss', 'Value Loss'),
        ]):
            means, stds = [], []
            for ep in epochs:
                vals = [r[metric] for r in opt_data if r['epoch'] == ep]
                means.append(sum(vals) / len(vals))
                if len(vals) > 1:
                    mean = means[-1]
                    stds.append((sum((v - mean)**2 for v in vals) / (len(vals) - 1))**0.5)
                else:
                    stds.append(0)

            ax = axes[metric_idx]
            ax.plot(epochs, means, label=optimizer, linewidth=2)
            ax.fill_between(
                epochs,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2,
            )
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'comparison.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


def print_summary(results, total_epochs):
    """Print summary table of final epoch results."""
    print(f"\n{'='*60}")
    print("SUMMARY — Final Epoch Results")
    print(f"{'='*60}")
    print(f"{'Optimizer':<12} {'Total Loss':>12} {'Policy Loss':>12} {'Value Loss':>12}")
    print(f"{'-'*48}")

    for optimizer in OPTIMIZERS:
        final = [r for r in results if r['optimizer'] == optimizer and r['epoch'] == total_epochs]
        if not final:
            # Try last available epoch
            opt_epochs = [r['epoch'] for r in results if r['optimizer'] == optimizer]
            if opt_epochs:
                last_ep = max(opt_epochs)
                final = [r for r in results if r['optimizer'] == optimizer and r['epoch'] == last_ep]

        if final:
            avg_total = sum(r['total_loss'] for r in final) / len(final)
            avg_policy = sum(r['policy_loss'] for r in final) / len(final)
            avg_value = sum(r['value_loss'] for r in final) / len(final)
            print(f"{optimizer:<12} {avg_total:>12.4f} {avg_policy:>12.4f} {avg_value:>12.4f}")
        else:
            print(f"{optimizer:<12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")


if __name__ == '__main__':
    main()
