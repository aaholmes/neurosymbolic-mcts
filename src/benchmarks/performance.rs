//! Performance comparison framework for demonstrating mate-search-first advantage

use super::*;
use crate::benchmarks::tactical_suite::run_tactical_benchmark;
use std::time::Duration;

/// Compare performance between different engine configurations
pub struct PerformanceComparison {
    pub results: Vec<(String, BenchmarkSummary)>,
}

impl PerformanceComparison {
    pub fn new() -> Self {
        PerformanceComparison {
            results: Vec::new(),
        }
    }
    
    pub fn add_benchmark(&mut self, engine_name: String, summary: BenchmarkSummary) {
        self.results.push((engine_name, summary));
    }
    
    pub fn print_comparison(&self) {
        println!("\n🏆 PERFORMANCE COMPARISON RESULTS");
        println!("=====================================");
        
        // Print header
        println!("{:<20} | {:>8} | {:>8} | {:>10} | {:>12} | {:>10}", 
                "Engine", "Solved", "Accuracy", "Avg Time", "Total Nodes", "Speed");
        println!("{}", "-".repeat(80));
        
        // Print results
        for (engine_name, summary) in &self.results {
            let speed_metric = if summary.average_time.as_millis() > 0 {
                format!("{:.0} n/s", summary.total_nodes as f64 / summary.average_time.as_secs_f64() / summary.total_positions as f64)
            } else {
                "N/A".to_string()
            };
            
            println!("{:<20} | {:>8} | {:>7.1}% | {:>9.1}ms | {:>12} | {:>10}", 
                    engine_name,
                    summary.positions_solved,
                    summary.mate_accuracy,
                    summary.average_time.as_millis(),
                    summary.total_nodes,
                    speed_metric);
        }
        
        println!("\n📊 KEY INSIGHTS:");
        self.analyze_results();
    }
    
    fn analyze_results(&self) {
        if self.results.len() < 2 {
            println!("   Need at least 2 engines for comparison");
            return;
        }
        
        // Find best accuracy
        let best_accuracy = self.results.iter()
            .max_by(|a, b| a.1.mate_accuracy.partial_cmp(&b.1.mate_accuracy).unwrap())
            .unwrap();
        
        // Find fastest average time
        let fastest = self.results.iter()
            .min_by(|a, b| a.1.average_time.cmp(&b.1.average_time))
            .unwrap();
        
        // Find most efficient (accuracy per time)
        let most_efficient = self.results.iter()
            .max_by(|a, b| {
                let eff_a = a.1.mate_accuracy / a.1.average_time.as_millis() as f64;
                let eff_b = b.1.mate_accuracy / b.1.average_time.as_millis() as f64;
                eff_a.partial_cmp(&eff_b).unwrap()
            })
            .unwrap();
        
        println!("   🎯 Highest Accuracy: {} ({:.1}%)", best_accuracy.0, best_accuracy.1.mate_accuracy);
        println!("   ⚡ Fastest: {} ({:.1}ms avg)", fastest.0, fastest.1.average_time.as_millis());
        println!("   🏆 Most Efficient: {} ({:.2} acc/ms)", most_efficient.0, 
                most_efficient.1.mate_accuracy / most_efficient.1.average_time.as_millis() as f64);
        
        // Calculate improvements
        if let Some(baseline) = self.results.iter().find(|(name, _)| name.contains("AlphaBeta")) {
            for (engine_name, summary) in &self.results {
                if engine_name != &baseline.0 {
                    let accuracy_improvement = summary.mate_accuracy - baseline.1.mate_accuracy;
                    let speed_improvement = if summary.average_time < baseline.1.average_time {
                        let speedup = baseline.1.average_time.as_millis() as f64 / summary.average_time.as_millis() as f64;
                        format!("{:.1}x faster", speedup)
                    } else {
                        let slowdown = summary.average_time.as_millis() as f64 / baseline.1.average_time.as_millis() as f64;
                        format!("{:.1}x slower", slowdown)
                    };
                    
                    println!("   📈 {} vs {}: {:.1}% accuracy improvement, {}", 
                            engine_name, baseline.0, accuracy_improvement, speed_improvement);
                }
            }
        }
    }
}

/// Run comprehensive performance comparison
pub fn run_performance_comparison(time_limit_per_position: Duration) -> PerformanceComparison {
    let mut comparison = PerformanceComparison::new();
    
    println!("\n🚀 COMPREHENSIVE PERFORMANCE COMPARISON");
    println!("Testing mate-search-first advantage...\n");
    
    // Test 1: Traditional AlphaBeta (baseline)
    println!("1️⃣ Testing Traditional Alpha-Beta Search");
    let mut simple_agent = create_simple_agent();
    let ab_results = run_tactical_benchmark(&mut simple_agent, "AlphaBeta", time_limit_per_position);
    let ab_summary = BenchmarkSummary::from_results(&ab_results);
    ab_summary.print_summary("Traditional AlphaBeta");
    comparison.add_benchmark("AlphaBeta".to_string(), ab_summary);
    
    // Test 2: Humanlike Agent with Mate-Search-First MCTS
    println!("2️⃣ Testing Mate-Search-First MCTS");
    let mut humanlike_agent = create_humanlike_agent();
    let mcts_results = run_tactical_benchmark(&mut humanlike_agent, "MateSearchFirst-MCTS", time_limit_per_position);
    let mcts_summary = BenchmarkSummary::from_results(&mcts_results);
    mcts_summary.print_summary("MateSearchFirst-MCTS");
    comparison.add_benchmark("MateSearchFirst-MCTS".to_string(), mcts_summary);
    
    // Test 3: AlphaBeta with deeper mate search
    println!("3️⃣ Testing Deep Mate Search AlphaBeta");
    let mut deep_mate_agent = SimpleAgent::new(
        5,     // deeper mate_search_depth
        6,     // shallower ab_search_depth to compensate
        16,    // q_search_max_depth
        false, // verbose
        &*BENCH_MOVE_GEN,
        &*BENCH_PESTO_EVAL,
    );
    let deep_results = run_tactical_benchmark(&mut deep_mate_agent, "DeepMate-AlphaBeta", time_limit_per_position);
    let deep_summary = BenchmarkSummary::from_results(&deep_results);
    deep_summary.print_summary("DeepMate-AlphaBeta");
    comparison.add_benchmark("DeepMate-AlphaBeta".to_string(), deep_summary);
    
    comparison
}

/// Specialized mate-finding speed test
pub fn mate_speed_benchmark() {
    println!("\n⚡ MATE FINDING SPEED BENCHMARK");
    println!("================================");

    let time_limits = vec![
        Duration::from_millis(100),
        Duration::from_millis(500), 
        Duration::from_millis(1000),
        Duration::from_millis(2000),
    ];
    
    for time_limit in time_limits {
        println!("\n⏱️  Testing with {}ms time limit:", time_limit.as_millis());
        
        let mut simple_agent = create_simple_agent();
        let mut humanlike_agent = create_humanlike_agent();
        
        let ab_results = run_tactical_benchmark(&mut simple_agent, "AlphaBeta", time_limit);
        let mcts_results = run_tactical_benchmark(&mut humanlike_agent, "MateSearchFirst", time_limit);
        
        let ab_summary = BenchmarkSummary::from_results(&ab_results);
        let mcts_summary = BenchmarkSummary::from_results(&mcts_results);
        
        println!("   AlphaBeta: {}/{} solved ({:.1}%)", 
                ab_summary.positions_solved, ab_summary.total_positions, ab_summary.mate_accuracy);
        println!("   MateSearchFirst: {}/{} solved ({:.1}%)", 
                mcts_summary.positions_solved, mcts_summary.total_positions, mcts_summary.mate_accuracy);
        
        if mcts_summary.mate_accuracy > ab_summary.mate_accuracy {
            let improvement = mcts_summary.mate_accuracy - ab_summary.mate_accuracy;
            println!("   🎯 MateSearchFirst advantage: +{:.1}% accuracy", improvement);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_comparison_structure() {
        let mut comparison = PerformanceComparison::new();
        assert_eq!(comparison.results.len(), 0);
        
        let dummy_summary = BenchmarkSummary {
            total_positions: 10,
            positions_solved: 8,
            average_time: Duration::from_millis(100),
            total_nodes: 1000,
            mate_accuracy: 80.0,
        };
        
        comparison.add_benchmark("Test Engine".to_string(), dummy_summary);
        assert_eq!(comparison.results.len(), 1);
    }
}