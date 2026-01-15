//! Elo Rating Estimation
//!
//! Provides utilities to estimate Elo rating differences between engine variants
//! based on head-to-head match results and position analysis performance.

use std::collections::HashMap;
use crate::benchmarks::strength_testing::{EngineVariant, PositionResult};

/// Elo calculation utilities
pub struct EloCalculator;

impl Default for EloCalculator {
    fn default() -> Self {
        EloCalculator
    }
}

impl EloCalculator {
    pub fn new() -> Self {
        EloCalculator
    }
    
    /// Calculate expected score based on rating difference
    pub fn expected_score(rating_a: f64, rating_b: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
    }
    
    /// Estimate Elo difference from win rate
    pub fn elo_from_win_rate(win_rate: f64) -> f64 {
        if win_rate <= 0.0 {
            return -800.0; // Cap at -800 Elo
        }
        if win_rate >= 1.0 {
            return 800.0; // Cap at +800 Elo
        }
        
        -400.0 * (1.0 / win_rate - 1.0).log10()
    }
    
    /// Calculate performance-based Elo estimates
    pub fn estimate_performance_ratings(&self, results: &[PositionResult]) -> HashMap<EngineVariant, EloEstimate> {
        let mut engine_results: HashMap<EngineVariant, Vec<&PositionResult>> = HashMap::new();
        
        // Group results by engine
        for result in results {
            engine_results.entry(result.engine).or_insert_with(Vec::new).push(result);
        }
        
        let mut estimates = HashMap::new();
        
        // Calculate baseline rating (Alpha-Beta = 1500)
        let baseline_rating = 1500.0;
        
        for (engine, engine_results) in &engine_results {
            let estimate = self.calculate_engine_rating(*engine, engine_results, baseline_rating);
            estimates.insert(*engine, estimate);
        }
        
        estimates
    }
    
    /// Calculate rating for a specific engine
    fn calculate_engine_rating(&self, engine: EngineVariant, results: &[&PositionResult], baseline: f64) -> EloEstimate {
        if results.is_empty() {
            return EloEstimate::default();
        }
        
        // Performance-based metrics
        let total_positions = results.len() as f64;
        let correct_moves = results.iter().filter(|r| r.correct_move).count() as f64;
        let accuracy = correct_moves / total_positions;
        
        let average_quality = results.iter().map(|r| r.move_quality_score).sum::<f64>() / total_positions;
        let average_time = results.iter().map(|r| r.time_taken_ms as f64).sum::<f64>() / total_positions;
        
        // Estimate rating based on move accuracy and quality
        let accuracy_component = (accuracy - 0.5) * 200.0; // ±100 Elo for ±25% accuracy
        let quality_component = (average_quality - 0.6) * 300.0; // ±150 Elo for ±0.5 quality
        
        // Time bonus (faster is better, within reason)
        let time_component = if average_time < 500.0 {
            20.0 // Small bonus for being fast
        } else if average_time > 2000.0 {
            -30.0 // Penalty for being very slow
        } else {
            0.0
        };
        
        let estimated_rating = baseline + accuracy_component + quality_component + time_component;
        
        // Calculate confidence based on sample size and consistency
        let consistency = self.calculate_consistency(results);
        let confidence = (total_positions.sqrt() / 10.0).min(1.0) * consistency;
        
        EloEstimate {
            engine,
            estimated_rating,
            confidence,
            sample_size: results.len(),
            accuracy,
            average_quality,
            average_time_ms: average_time,
            rating_components: RatingComponents {
                accuracy_component,
                quality_component,
                time_component,
            },
        }
    }
    
    /// Calculate consistency score (0.0 to 1.0)
    fn calculate_consistency(&self, results: &[&PositionResult]) -> f64 {
        if results.len() < 2 {
            return 0.5;
        }
        
        let qualities: Vec<f64> = results.iter().map(|r| r.move_quality_score).collect();
        let mean = qualities.iter().sum::<f64>() / qualities.len() as f64;
        let variance = qualities.iter().map(|q| (q - mean).powi(2)).sum::<f64>() / qualities.len() as f64;
        let std_dev = variance.sqrt();
        
        // Lower standard deviation = higher consistency
        (1.0 - std_dev).max(0.0).min(1.0)
    }
    
    /// Compare two engines and estimate Elo difference
    pub fn compare_engines(&self, engine_a: &EloEstimate, engine_b: &EloEstimate) -> EngineComparison {
        let rating_diff = engine_a.estimated_rating - engine_b.estimated_rating;
        let combined_confidence = (engine_a.confidence + engine_b.confidence) / 2.0;
        
        // Calculate significance based on rating difference and confidence
        let significance = if rating_diff.abs() > 50.0 && combined_confidence > 0.7 {
            ComparisonSignificance::Significant
        } else if rating_diff.abs() > 25.0 && combined_confidence > 0.5 {
            ComparisonSignificance::Moderate
        } else {
            ComparisonSignificance::Insignificant
        };
        
        let better_engine = if rating_diff > 0.0 {
            engine_a.engine
        } else {
            engine_b.engine
        };
        
        EngineComparison {
            engine_a: engine_a.engine,
            engine_b: engine_b.engine,
            rating_difference: rating_diff,
            better_engine,
            significance,
            confidence: combined_confidence,
        }
    }
    
    /// Generate comprehensive Elo report
    pub fn generate_elo_report(&self, estimates: &HashMap<EngineVariant, EloEstimate>) -> EloReport {
        let mut rankings: Vec<&EloEstimate> = estimates.values().collect();
        rankings.sort_by(|a, b| b.estimated_rating.partial_cmp(&a.estimated_rating).unwrap());
        
        let baseline_engine = EngineVariant::AlphaBeta;
        let baseline_rating = estimates.get(&baseline_engine)
            .map(|e| e.estimated_rating)
            .unwrap_or(1500.0);
        
        let mut comparisons = Vec::new();
        for estimate in &rankings {
            if estimate.engine != baseline_engine {
                if let Some(baseline) = estimates.get(&baseline_engine) {
                    comparisons.push(self.compare_engines(estimate, baseline));
                }
            }
        }
        
        EloReport {
            rankings: rankings.into_iter().cloned().collect(),
            baseline_engine,
            baseline_rating,
            comparisons,
            overall_improvement: self.calculate_overall_improvement(estimates),
        }
    }
    
    /// Calculate overall improvement of advanced engines over baseline
    fn calculate_overall_improvement(&self, estimates: &HashMap<EngineVariant, EloEstimate>) -> f64 {
        let baseline_rating = estimates.get(&EngineVariant::AlphaBeta)
            .map(|e| e.estimated_rating)
            .unwrap_or(1500.0);
        
        let best_rating = estimates.values()
            .map(|e| e.estimated_rating)
            .fold(f64::NEG_INFINITY, f64::max);
        
        best_rating - baseline_rating
    }
}

/// Elo estimate for an engine variant
#[derive(Debug, Clone)]
pub struct EloEstimate {
    pub engine: EngineVariant,
    pub estimated_rating: f64,
    pub confidence: f64,
    pub sample_size: usize,
    pub accuracy: f64,
    pub average_quality: f64,
    pub average_time_ms: f64,
    pub rating_components: RatingComponents,
}

impl Default for EloEstimate {
    fn default() -> Self {
        EloEstimate {
            engine: EngineVariant::AlphaBeta,
            estimated_rating: 1500.0,
            confidence: 0.0,
            sample_size: 0,
            accuracy: 0.5,
            average_quality: 0.5,
            average_time_ms: 1000.0,
            rating_components: RatingComponents::default(),
        }
    }
}

/// Components that contribute to rating calculation
#[derive(Debug, Clone)]
pub struct RatingComponents {
    pub accuracy_component: f64,
    pub quality_component: f64,
    pub time_component: f64,
}

impl Default for RatingComponents {
    fn default() -> Self {
        RatingComponents {
            accuracy_component: 0.0,
            quality_component: 0.0,
            time_component: 0.0,
        }
    }
}

/// Comparison between two engines
#[derive(Debug, Clone)]
pub struct EngineComparison {
    pub engine_a: EngineVariant,
    pub engine_b: EngineVariant,
    pub rating_difference: f64,
    pub better_engine: EngineVariant,
    pub significance: ComparisonSignificance,
    pub confidence: f64,
}

/// Statistical significance of comparison
#[derive(Debug, Clone, Copy)]
pub enum ComparisonSignificance {
    Significant,     // >50 Elo difference, high confidence
    Moderate,        // 25-50 Elo difference, medium confidence
    Insignificant,   // <25 Elo difference or low confidence
}

/// Complete Elo analysis report
#[derive(Debug)]
pub struct EloReport {
    pub rankings: Vec<EloEstimate>,
    pub baseline_engine: EngineVariant,
    pub baseline_rating: f64,
    pub comparisons: Vec<EngineComparison>,
    pub overall_improvement: f64,
}

impl EloReport {
    /// Print comprehensive Elo report
    pub fn print_report(&self) {
        println!("\n📊 ELO RATING ANALYSIS");
        println!("======================");
        
        println!("\n🏆 ESTIMATED RATINGS:");
        for (i, estimate) in self.rankings.iter().enumerate() {
            let diff_from_baseline = estimate.estimated_rating - self.baseline_rating;
            println!("  {}. {} - {:.0} Elo ({:+.0} vs baseline) [confidence: {:.1}%]",
                    i + 1,
                    estimate.engine,
                    estimate.estimated_rating,
                    diff_from_baseline,
                    estimate.confidence * 100.0);
        }
        
        println!("\n📈 KEY IMPROVEMENTS:");
        for comparison in &self.comparisons {
            if matches!(comparison.significance, ComparisonSignificance::Significant | ComparisonSignificance::Moderate) {
                println!("  {} vs {}: {:+.0} Elo ({:?})",
                        comparison.better_engine,
                        comparison.engine_b,
                        comparison.rating_difference.abs(),
                        comparison.significance);
            }
        }
        
        println!("\n📋 DETAILED BREAKDOWN:");
        for estimate in &self.rankings {
            println!("\n  🔧 {}:", estimate.engine);
            println!("     Estimated Rating: {:.0} Elo", estimate.estimated_rating);
            println!("     Accuracy: {:.1}% ({:+.0} Elo)", 
                    estimate.accuracy * 100.0, 
                    estimate.rating_components.accuracy_component);
            println!("     Move Quality: {:.3} ({:+.0} Elo)", 
                    estimate.average_quality, 
                    estimate.rating_components.quality_component);
            println!("     Speed: {:.0}ms ({:+.0} Elo)", 
                    estimate.average_time_ms, 
                    estimate.rating_components.time_component);
            println!("     Confidence: {:.1}% ({} positions)", 
                    estimate.confidence * 100.0, 
                    estimate.sample_size);
        }
        
        println!("\n🎯 OVERALL ASSESSMENT:");
        if self.overall_improvement > 50.0 {
            println!("  ✅ Significant improvement: {:+.0} Elo gain from innovations!", self.overall_improvement);
        } else if self.overall_improvement > 0.0 {
            println!("  📈 Moderate improvement: {:+.0} Elo gain", self.overall_improvement);
        } else {
            println!("  ⚠️  No significant improvement detected ({:+.0} Elo)", self.overall_improvement);
        }
        
        println!("\n📝 INTERPRETATION:");
        println!("  - Elo ratings are estimates based on position analysis performance");
        println!("  - Confidence levels indicate reliability of estimates");
        println!("  - Differences >50 Elo are generally meaningful");
        println!("  - Consider running more positions for higher confidence");
    }
}