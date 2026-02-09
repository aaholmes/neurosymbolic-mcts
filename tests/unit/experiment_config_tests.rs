//! Tests for experiments::config module

use kingfisher::experiments::config::{
    AblationConfig, ExperimentConfig, generate_ablation_configs,
};

// === AblationConfig Preset Tests ===

#[test]
fn test_ablation_full_all_enabled() {
    let config = AblationConfig::full();
    assert!(config.enable_tier1_gate);
    assert!(config.enable_tier3_neural);
    assert!(config.enable_q_init);
}

#[test]
fn test_ablation_baseline_all_disabled() {
    let config = AblationConfig::baseline_mcts();
    assert!(!config.enable_tier1_gate);
    assert!(!config.enable_tier3_neural);
    assert!(!config.enable_q_init);
}

#[test]
fn test_ablation_tier1_only() {
    let config = AblationConfig::tier1_only();
    assert!(config.enable_tier1_gate);
    assert!(!config.enable_tier3_neural);
    assert!(!config.enable_q_init);
}

#[test]
fn test_ablation_classical_hybrid() {
    let config = AblationConfig::classical_hybrid();
    assert!(config.enable_tier1_gate);
    assert!(!config.enable_tier3_neural);
    assert!(config.enable_q_init);
}

#[test]
fn test_ablation_neural_only() {
    let config = AblationConfig::neural_only();
    assert!(!config.enable_tier1_gate);
    assert!(config.enable_tier3_neural);
    assert!(!config.enable_q_init);
}

// === ExperimentConfig Default Tests ===

#[test]
fn test_experiment_config_default() {
    let config = ExperimentConfig::default();
    assert_eq!(config.name, "default_experiment");
    assert!(!config.description.is_empty());
    assert_eq!(config.search_config.max_iterations, 800);
    assert_eq!(config.search_config.mate_search_depth, 5);
    assert!(config.search_config.exploration_constant > 1.0);
    // Full ablation by default
    assert!(config.ablation.enable_tier1_gate);
    assert!(config.ablation.enable_tier3_neural);
}

#[test]
fn test_experiment_config_default_has_seeds() {
    let config = ExperimentConfig::default();
    assert!(!config.evaluation.random_seeds.is_empty());
}

#[test]
fn test_experiment_config_default_output() {
    let config = ExperimentConfig::default();
    assert_eq!(config.output.results_dir, "results");
    assert!(config.output.save_detailed_stats);
    assert!(config.output.export_latex);
}

// === generate_ablation_configs Tests ===

#[test]
fn test_generate_ablation_configs_count() {
    let configs = generate_ablation_configs();
    assert_eq!(configs.len(), 5, "Should generate 5 ablation configurations");
}

#[test]
fn test_generate_ablation_configs_names() {
    let configs = generate_ablation_configs();
    let names: Vec<&str> = configs.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"baseline_mcts"));
    assert!(names.contains(&"tier1_only"));
    assert!(names.contains(&"classical_hybrid"));
    assert!(names.contains(&"neural_only"));
    assert!(names.contains(&"full_system"));
}

#[test]
fn test_generate_ablation_configs_unique_names() {
    let configs = generate_ablation_configs();
    let names: Vec<&str> = configs.iter().map(|c| c.name.as_str()).collect();
    let mut unique_names = names.clone();
    unique_names.sort();
    unique_names.dedup();
    assert_eq!(names.len(), unique_names.len(), "All config names should be unique");
}

#[test]
fn test_generate_ablation_configs_descriptions_nonempty() {
    let configs = generate_ablation_configs();
    for config in &configs {
        assert!(!config.description.is_empty(), "Config '{}' should have a description", config.name);
    }
}

#[test]
fn test_generate_ablation_configs_baseline_has_no_tiers() {
    let configs = generate_ablation_configs();
    let baseline = configs.iter().find(|c| c.name == "baseline_mcts").unwrap();
    assert!(!baseline.ablation.enable_tier1_gate);
    assert!(!baseline.ablation.enable_tier3_neural);
    assert!(!baseline.ablation.enable_q_init);
}

#[test]
fn test_generate_ablation_configs_full_has_all_tiers() {
    let configs = generate_ablation_configs();
    let full = configs.iter().find(|c| c.name == "full_system").unwrap();
    assert!(full.ablation.enable_tier1_gate);
    assert!(full.ablation.enable_tier3_neural);
    assert!(full.ablation.enable_q_init);
}

#[test]
fn test_generate_ablation_configs_share_search_settings() {
    let configs = generate_ablation_configs();
    // All configs should have the same search settings (only ablation flags differ)
    let base_iterations = configs[0].search_config.max_iterations;
    for config in &configs {
        assert_eq!(
            config.search_config.max_iterations, base_iterations,
            "Config '{}' should share the same search settings", config.name
        );
    }
}
