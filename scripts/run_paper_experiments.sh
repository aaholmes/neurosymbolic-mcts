#!/bin/bash
# Run all experiments for the research paper

set -e

echo "═══════════════════════════════════════════════════════════"
echo "   CAISSAWARY: Safe & Sample-Efficient RL Experiments"
echo "═══════════════════════════════════════════════════════════"

# Build optimized binary
echo -e "\n📦 Building release binary..."
cargo build --release

# Create results directory
mkdir -p results/figures

# Run ablation study
echo -e "\n🔬 Running ablation study..."
cargo run --release --bin run_experiments -- --config ablation

# Generate figures
echo -e "\n📊 Generating figures..."
python3 scripts/analyze_results.py results/ablation_results.json

# Generate LaTeX document snippets
echo -e "\n📄 Generating LaTeX snippets..."
if [ -f results/ablation_table.tex ]; then
    cat results/ablation_table.tex
fi

echo -e "\n═══════════════════════════════════════════════════════════"
echo "   ✅ Experiments complete! Results in results/"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Key files:"
echo "  - results/ablation_results.json    (raw data)"
echo "  - results/ablation_table.tex       (LaTeX table)"
echo "  - results/figures/*.png            (publication figures)"
echo "  - results/figures/summary.md       (markdown summary)"