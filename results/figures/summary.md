# Ablation Study Results Summary

| Configuration | NN Reduction | Tier 1 | Tier 2 | NN Calls | Safety |
|---------------|--------------|--------|--------|----------|--------|
| baseline_mcts | 25.0% | 0 | 0 | 0 | 33% |
| tier1_only | 25.0% | 585 | 0 | 0 | 67% |
| classical_hybrid | 37.5% | 587 | 76 | 0 | 67% |
| neural_only | 25.0% | 0 | 0 | 0 | 33% |
| full_system | 37.5% | 587 | 76 | 0 | 67% |