//! Tests for experiments::metrics module

use kingfisher::experiments::metrics::{
    AggregatedMetrics, SafetyMetrics, SearchMetrics,
};
use std::time::Duration;

// === SearchMetrics Tests ===

#[test]
fn test_nn_call_reduction_no_calls() {
    let metrics = SearchMetrics::default();
    assert_eq!(metrics.nn_call_reduction_percent(), 0.0);
}

#[test]
fn test_nn_call_reduction_all_saved_by_tier1() {
    let metrics = SearchMetrics {
        nn_evaluations: 0,
        nn_calls_saved_tier1: 100,
        nn_calls_saved_tier2: 0,
        ..Default::default()
    };
    assert_eq!(metrics.nn_call_reduction_percent(), 100.0);
}

#[test]
fn test_nn_call_reduction_none_saved() {
    let metrics = SearchMetrics {
        nn_evaluations: 100,
        nn_calls_saved_tier1: 0,
        nn_calls_saved_tier2: 0,
        ..Default::default()
    };
    assert_eq!(metrics.nn_call_reduction_percent(), 0.0);
}

#[test]
fn test_nn_call_reduction_partial() {
    let metrics = SearchMetrics {
        nn_evaluations: 60,
        nn_calls_saved_tier1: 30,
        nn_calls_saved_tier2: 10,
        ..Default::default()
    };
    // saved = 40, potential = 100
    assert!((metrics.nn_call_reduction_percent() - 40.0).abs() < 0.01);
}

#[test]
fn test_nodes_per_second_zero_time() {
    let metrics = SearchMetrics {
        nodes_expanded: 100,
        search_time: Duration::from_secs(0),
        ..Default::default()
    };
    assert_eq!(metrics.nodes_per_second(), 0.0);
}

#[test]
fn test_nodes_per_second_normal() {
    let metrics = SearchMetrics {
        nodes_expanded: 5000,
        search_time: Duration::from_secs(2),
        ..Default::default()
    };
    assert_eq!(metrics.nodes_per_second(), 2500.0);
}

#[test]
fn test_nodes_per_second_fractional() {
    let metrics = SearchMetrics {
        nodes_expanded: 1000,
        search_time: Duration::from_millis(500),
        ..Default::default()
    };
    assert_eq!(metrics.nodes_per_second(), 2000.0);
}

#[test]
fn test_selection_confidence_no_second_best() {
    let metrics = SearchMetrics {
        best_move_visits: 100,
        second_best_visits: 0,
        ..Default::default()
    };
    assert!(metrics.selection_confidence().is_infinite());
}

#[test]
fn test_selection_confidence_equal_visits() {
    let metrics = SearchMetrics {
        best_move_visits: 50,
        second_best_visits: 50,
        ..Default::default()
    };
    assert_eq!(metrics.selection_confidence(), 1.0);
}

#[test]
fn test_selection_confidence_clear_best() {
    let metrics = SearchMetrics {
        best_move_visits: 200,
        second_best_visits: 50,
        ..Default::default()
    };
    assert_eq!(metrics.selection_confidence(), 4.0);
}

// === AggregatedMetrics Tests ===

#[test]
fn test_aggregated_from_empty_metrics() {
    let agg = AggregatedMetrics::from_search_metrics("empty", &[]);
    assert_eq!(agg.config_name, "empty");
    assert_eq!(agg.num_positions, 0);
    assert_eq!(agg.mean_iterations, 0.0);
}

#[test]
fn test_aggregated_from_single_metric() {
    let metrics = vec![SearchMetrics {
        total_iterations: 100,
        nodes_expanded: 200,
        search_time: Duration::from_millis(50),
        tier1_activations: 3,
        nn_evaluations: 80,
        nn_calls_saved_tier1: 20,
        ..Default::default()
    }];

    let agg = AggregatedMetrics::from_search_metrics("single", &metrics);
    assert_eq!(agg.num_positions, 1);
    assert_eq!(agg.mean_iterations, 100.0);
    assert_eq!(agg.mean_nodes, 200.0);
    assert_eq!(agg.total_tier1_activations, 3);
    assert_eq!(agg.total_nn_evaluations, 80);
    // std_dev should be 0 for single metric
    assert_eq!(agg.std_iterations, 0.0);
}

#[test]
fn test_aggregated_from_multiple_metrics() {
    let metrics = vec![
        SearchMetrics {
            total_iterations: 100,
            nodes_expanded: 200,
            search_time: Duration::from_millis(50),
            tier1_activations: 2,
            nn_evaluations: 80,
            ..Default::default()
        },
        SearchMetrics {
            total_iterations: 200,
            nodes_expanded: 400,
            search_time: Duration::from_millis(100),
            tier1_activations: 5,
            nn_evaluations: 150,
            ..Default::default()
        },
    ];

    let agg = AggregatedMetrics::from_search_metrics("multi", &metrics);
    assert_eq!(agg.num_positions, 2);
    assert_eq!(agg.mean_iterations, 150.0);
    assert_eq!(agg.mean_nodes, 300.0);
    assert_eq!(agg.total_tier1_activations, 7);
    assert_eq!(agg.total_nn_evaluations, 230);
    // std should be 50 (population std of [100, 200])
    assert!((agg.std_iterations - 50.0).abs() < 0.01);
}

#[test]
fn test_aggregated_latex_row_format() {
    let metrics = vec![SearchMetrics {
        total_iterations: 100,
        nodes_expanded: 200,
        nn_evaluations: 80,
        nn_calls_saved_tier1: 20,
        search_time: Duration::from_millis(50),
        ..Default::default()
    }];

    let agg = AggregatedMetrics::from_search_metrics("test_config", &metrics);
    let latex = agg.to_latex_row();
    assert!(latex.contains("test_config"));
    assert!(latex.contains(r"\\"));
}

// === SafetyMetrics Tests ===

#[test]
fn test_tactical_safety_rate_no_mates() {
    let safety = SafetyMetrics::default();
    assert_eq!(safety.tactical_safety_rate(), 1.0);
}

#[test]
fn test_tactical_safety_rate_all_found() {
    let safety = SafetyMetrics {
        positions_with_forced_mate: 10,
        forced_mates_found: 10,
        ..Default::default()
    };
    assert_eq!(safety.tactical_safety_rate(), 1.0);
}

#[test]
fn test_tactical_safety_rate_partial() {
    let safety = SafetyMetrics {
        positions_with_forced_mate: 10,
        forced_mates_found: 7,
        ..Default::default()
    };
    assert!((safety.tactical_safety_rate() - 0.7).abs() < 0.001);
}

#[test]
fn test_defensive_accuracy_no_positions() {
    let safety = SafetyMetrics::default();
    assert_eq!(safety.defensive_accuracy(), 1.0);
}

#[test]
fn test_defensive_accuracy_partial() {
    let safety = SafetyMetrics {
        positions_being_mated: 5,
        best_defenses_found: 3,
        ..Default::default()
    };
    assert!((safety.defensive_accuracy() - 0.6).abs() < 0.001);
}

#[test]
fn test_blunder_rate_no_positions() {
    let safety = SafetyMetrics::default();
    assert_eq!(safety.blunder_rate(), 0.0);
}

#[test]
fn test_blunder_rate_with_blunders() {
    let safety = SafetyMetrics {
        positions_analyzed: 100,
        material_blunders: 3,
        hanging_piece_misses: 2,
        ..Default::default()
    };
    assert!((safety.blunder_rate() - 0.05).abs() < 0.001);
}

#[test]
fn test_blunder_rate_no_blunders() {
    let safety = SafetyMetrics {
        positions_analyzed: 100,
        material_blunders: 0,
        hanging_piece_misses: 0,
        ..Default::default()
    };
    assert_eq!(safety.blunder_rate(), 0.0);
}
