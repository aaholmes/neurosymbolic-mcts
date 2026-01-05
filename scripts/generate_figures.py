import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def generate_figures(results_dir):
    # 1. Elo Tournament Results
    csv_path = os.path.join(results_dir, 'tournament_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Processing {csv_path}...")
        
        # Plot head-to-head Elo differences
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='engine_b', y='elo_diff', hue='engine_a')
        plt.title('Elo Difference (A vs B)')
        plt.ylabel('Elo Difference')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'elo_differences.png'))
        
    # 2. Add more plotting logic as needed for tactical efficiency etc.
    print("Figures generated in", results_dir)

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    generate_figures(results_dir)
