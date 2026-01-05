// src/experiments/config.rs
//! Experiment configuration for ablation studies and benchmarking.
//!
//! This module provides structured configurations for running reproducible
//! experiments comparing different tiers of the tactical-first MCTS.

use serde::{Deserialize, Serialize};

/// Complete experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub description: String,
    pub search_config: SearchConfig,
    pub ablation: AblationConfig,
    pub evaluation: EvaluationConfig,
    pub output: OutputConfig,
}

/// Search algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub max_iterations: u32,
    pub time_limit_ms: u64,
    pub exploration_constant: f64,
    pub mate_search_depth: i32,
    pub quiescence_depth: i32,
}

/// Ablation flags for controlled experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    /// Enable Tier 1: Safety gates (mate search + KOTH)
    pub enable_tier1_gate: bool,
    /// Enable Tier 2: Tactical grafting from quiescence search
    pub enable_tier2_graft: bool,
    /// Enable Tier 3: Neural network policy/value
    pub enable_tier3_neural: bool,
    /// Enable Q-value initialization from tactical analysis
    pub enable_q_init: bool,
}

impl AblationConfig {
    /// Full system with all tiers enabled
    pub fn full() -> Self {
        AblationConfig {
            enable_tier1_gate: true,
            enable_tier2_graft: true,
            enable_tier3_neural: true,
            enable_q_init: true,
        }
    }

    /// Pure MCTS baseline (no enhancements)
    pub fn baseline_mcts() -> Self {
        AblationConfig {
            enable_tier1_gate: false,
            enable_tier2_graft: false,
            enable_tier3_neural: false,
            enable_q_init: false,
        }
    }

    /// Tier 1 only (safety gates)
    pub fn tier1_only() -> Self {
        AblationConfig {
            enable_tier1_gate: true,
            enable_tier2_graft: false,
            enable_tier3_neural: false,
            enable_q_init: false,
        }
    }

    /// Tiers 1+2 (classical hybrid, no neural)
    pub fn classical_hybrid() -> Self {
        AblationConfig {
            enable_tier1_gate: true,
            enable_tier2_graft: true,
            enable_tier3_neural: false,
            enable_q_init: true,
        }
    }

    /// Neural only (like standard AlphaZero)
    pub fn neural_only() -> Self {
        AblationConfig {
            enable_tier1_gate: false,
            enable_tier2_graft: false,
            enable_tier3_neural: true,
            enable_q_init: false,
        }
    }
}

/// Evaluation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Test positions to evaluate
    pub test_suite: TestSuite,
    /// Number of games for self-play evaluation
    pub self_play_games: u32,
    /// Seeds for reproducibility
    pub random_seeds: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestSuite {
    /// Built-in tactical test positions
    TacticalSuite,
    /// Standard benchmark (e.g., BT2630)
    StandardBenchmark,
    /// Custom positions from file
    CustomFile(String),
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub results_dir: String,
    pub save_search_trees: bool,
    pub save_detailed_stats: bool,
    pub export_latex: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        ExperimentConfig {
            name: "default_experiment".to_string(),
            description: "Default experimental configuration".to_string(),
            search_config: SearchConfig {
                max_iterations: 800,
                time_limit_ms: 1000,
                exploration_constant: 1.414,
                mate_search_depth: 5,
                quiescence_depth: 8,
            },
            ablation: AblationConfig::full(),
            evaluation: EvaluationConfig {
                test_suite: TestSuite::TacticalSuite,
                self_play_games: 100,
                random_seeds: vec![42, 123, 456, 789, 1011],
            },
            output: OutputConfig {
                results_dir: "results".to_string(),
                save_search_trees: false,
                save_detailed_stats: true,
                export_latex: true,
            },
        }
    }
}

/// Generate standard ablation experiment configurations
pub fn generate_ablation_configs() -> Vec<ExperimentConfig> {
    let base = ExperimentConfig::default();
    
    vec![
        ExperimentConfig {
            name: "baseline_mcts".to_string(),
            description: "Pure MCTS with Pesto evaluation (no tiers)".to_string(),
            ablation: AblationConfig::baseline_mcts(),
            ..base.clone()
        },
        ExperimentConfig {
            name: "tier1_only".to_string(),
            description: "MCTS with Tier 1 safety gates only".to_string(),
            ablation: AblationConfig::tier1_only(),
            ..base.clone()
        },
        ExperimentConfig {
            name: "classical_hybrid".to_string(),
            description: "MCTS with Tiers 1+2 (no neural network)".to_string(),
            ablation: AblationConfig::classical_hybrid(),
            ..base.clone()
        },
        ExperimentConfig {
            name: "neural_only".to_string(),
            description: "MCTS with neural network only (AlphaZero-style)".to_string(),
            ablation: AblationConfig::neural_only(),
            ..base.clone()
        },
        ExperimentConfig {
            name: "full_system".to_string(),
            description: "Complete Caissawary with all three tiers".to_string(),
            ablation: AblationConfig::full(),
            ..base.clone()
        },
    ]
}
