// src/experiments/metrics.rs
//! Metrics collection for experimental evaluation.
//!
//! Tracks key metrics that validate our research claims:
//! - Sample efficiency (NN calls saved)
//! - Safety (tactical errors avoided)
//! - Playing strength (Elo estimation)

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Comprehensive metrics from a single search
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchMetrics {
    // Core efficiency metrics
    pub total_iterations: u32,
    pub nodes_expanded: u32,
    pub search_time: Duration,
    
    // Tier 1 metrics (Safety)
    pub tier1_activations: u32,
    pub mates_found_by_gate: u32,
    pub koth_wins_detected: u32,
    pub nn_calls_saved_tier1: u32,
    
    // Tier 2 metrics (Tactical Grafting)
    pub tier2_grafts: u32,
    pub positions_with_tactical_moves: u32,
    pub avg_graft_depth: f32,
    pub nn_calls_saved_tier2: u32,
    
    // Tier 3 metrics (Neural)
    pub nn_evaluations: u32,
    pub nn_policy_queries: u32,
    pub avg_nn_inference_time_us: f32,
    
    // Quality metrics
    pub best_move_visits: u32,
    pub second_best_visits: u32,
    pub root_value: f64,
    pub value_confidence: f64,
}

impl SearchMetrics {
    /// Calculate the percentage of NN calls saved by tiers 1 and 2
    pub fn nn_call_reduction_percent(&self) -> f64 {
        let potential_calls = self.nn_evaluations + self.nn_calls_saved_tier1 + self.nn_calls_saved_tier2;
        if potential_calls == 0 {
            return 0.0;
        }
        let saved = self.nn_calls_saved_tier1 + self.nn_calls_saved_tier2;
        100.0 * (saved as f64) / (potential_calls as f64)
    }
    
    /// Calculate nodes per second
    pub fn nodes_per_second(&self) -> f64 {
        let secs = self.search_time.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.nodes_expanded as f64 / secs
    }
    
    /// Calculate move selection confidence (ratio of best to second-best visits)
    pub fn selection_confidence(&self) -> f64 {
        if self.second_best_visits == 0 {
            return f64::INFINITY;
        }
        self.best_move_visits as f64 / self.second_best_visits as f64
    }
}

/// Aggregated metrics across multiple positions/games
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub config_name: String,
    pub num_positions: u32,
    
    // Means
    pub mean_iterations: f64,
    pub mean_nodes: f64,
    pub mean_time_ms: f64,
    pub mean_nn_reduction_percent: f64,
    
    // Tier-specific
    pub total_tier1_activations: u32,
    pub total_tier2_grafts: u32,
    pub total_nn_evaluations: u32,
    
    // Standard deviations for confidence intervals
    pub std_iterations: f64,
    pub std_nn_reduction: f64,
    
    // Playing strength
    pub tactical_accuracy: f64,  // % of tactical puzzles solved
    pub positional_score: f64,   // Average eval difference from optimal
    pub estimated_elo: f64,
    pub elo_confidence_interval: (f64, f64),
}

impl AggregatedMetrics {
    pub fn from_search_metrics(config_name: &str, metrics: &[SearchMetrics]) -> Self {
        let n = metrics.len() as f64;
        if n == 0.0 {
            return AggregatedMetrics {
                config_name: config_name.to_string(),
                ..Default::default()
            };
        }
        
        // Calculate means
        let mean_iterations = metrics.iter().map(|m| m.total_iterations as f64).sum::<f64>() / n;
        let mean_nodes = metrics.iter().map(|m| m.nodes_expanded as f64).sum::<f64>() / n;
        let mean_time_ms = metrics.iter().map(|m| m.search_time.as_millis() as f64).sum::<f64>() / n;
        let mean_nn_reduction = metrics.iter().map(|m| m.nn_call_reduction_percent()).sum::<f64>() / n;
        
        // Calculate standard deviations
        let std_iterations = (metrics.iter()
            .map(|m| (m.total_iterations as f64 - mean_iterations).powi(2))
            .sum::<f64>() / n).sqrt();
        let std_nn_reduction = (metrics.iter()
            .map(|m| (m.nn_call_reduction_percent() - mean_nn_reduction).powi(2))
            .sum::<f64>() / n).sqrt();
        
        // Sum totals
        let total_tier1 = metrics.iter().map(|m| m.tier1_activations).sum();
        let total_tier2 = metrics.iter().map(|m| m.tier2_grafts).sum();
        let total_nn = metrics.iter().map(|m| m.nn_evaluations).sum();
        
        AggregatedMetrics {
            config_name: config_name.to_string(),
            num_positions: metrics.len() as u32,
            mean_iterations,
            mean_nodes,
            mean_time_ms,
            mean_nn_reduction_percent: mean_nn_reduction,
            total_tier1_activations: total_tier1,
            total_tier2_grafts: total_tier2,
            total_nn_evaluations: total_nn,
            std_iterations,
            std_nn_reduction,
            tactical_accuracy: 0.0,  // Computed separately
            positional_score: 0.0,
            estimated_elo: 0.0,
            elo_confidence_interval: (0.0, 0.0),
        }
    }
    
    /// Generate LaTeX table row
    pub fn to_latex_row(&self) -> String {
        format!(
            r"{} & {:.0} & {:.0} & {:.1}\% $\pm$ {:.1} & {} & {} & {} \\",
            self.config_name,
            self.mean_iterations,
            self.mean_nodes,
            self.mean_nn_reduction_percent,
            1.96 * self.std_nn_reduction / (self.num_positions as f64).sqrt(), // 95% CI
            self.total_tier1_activations,
            self.total_tier2_grafts,
            self.total_nn_evaluations,
        )
    }
}

/// Safety metrics - tracking tactical blunders
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SafetyMetrics {
    pub positions_analyzed: u32,
    
    /// Positions where a forced mate existed
    pub positions_with_forced_mate: u32,
    /// Positions where the engine found the forced mate
    pub forced_mates_found: u32,
    
    /// Positions where engine was getting mated
    pub positions_being_mated: u32,
    /// Positions where engine found best defense
    pub best_defenses_found: u32,
    
    /// Material blunders (losing significant material)
    pub material_blunders: u32,
    /// Hanging piece oversights
    pub hanging_piece_misses: u32,
}

impl SafetyMetrics {
    pub fn tactical_safety_rate(&self) -> f64 {
        if self.positions_with_forced_mate == 0 {
            return 1.0;
        }
        self.forced_mates_found as f64 / self.positions_with_forced_mate as f64
    }
    
    pub fn defensive_accuracy(&self) -> f64 {
        if self.positions_being_mated == 0 {
            return 1.0;
        }
        self.best_defenses_found as f64 / self.positions_being_mated as f64
    }
    
    pub fn blunder_rate(&self) -> f64 {
        if self.positions_analyzed == 0 {
            return 0.0;
        }
        (self.material_blunders + self.hanging_piece_misses) as f64 / self.positions_analyzed as f64
    }
}

/// Complete experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    pub config_name: String,
    pub timestamp: String,
    pub search_metrics: Vec<SearchMetrics>,
    pub aggregated: AggregatedMetrics,
    pub safety: SafetyMetrics,
    pub position_results: Vec<PositionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionResult {
    pub fen: String,
    pub position_type: String,
    pub expected_move: Option<String>,
    pub engine_move: String,
    pub correct: bool,
    pub time_ms: u64,
    pub tier_used: TierUsed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TierUsed {
    Tier1Gate,
    Tier2Graft,
    Tier3Neural,
    ClassicalEval,
}
