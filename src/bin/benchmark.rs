//! Benchmark binary for demonstrating Kingfisher's mate-search-first advantage

use kingfisher::benchmarks::performance::{run_performance_comparison, mate_speed_benchmark};
use kingfisher::benchmarks::tactical_suite::run_tactical_benchmark;
use kingfisher::benchmarks::{create_simple_agent, create_humanlike_agent};
use std::env;
use std::time::Duration;

fn print_banner() {
    println!("🏰 KINGFISHER CHESS ENGINE BENCHMARK SUITE");
    println!("===========================================");
    println!("Demonstrating Novel Mate-Search-First MCTS Approach\n");
}

fn print_usage() {
    println!("Usage: cargo run --bin benchmark [command]");
    println!("\nCommands:");
    println!("  tactical     - Run tactical position benchmark");
    println!("  comparison   - Run comprehensive engine comparison");
    println!("  speed        - Run mate-finding speed tests");
    println!("  all          - Run all benchmarks");
    println!("  help         - Show this help message");
    println!("\nExamples:");
    println!("  cargo run --release --bin benchmark tactical");
    println!("  cargo run --release --bin benchmark comparison");
    println!("  cargo run --release --bin benchmark all");
}

fn run_tactical_benchmark_cmd() {
    println!("🎯 TACTICAL POSITION BENCHMARK");
    println!("==============================\n");
    
    let time_limit = Duration::from_millis(2000);
    
    // Run benchmarks
    let mut simple_agent = create_simple_agent();
    let mut humanlike_agent = create_humanlike_agent();

    let ab_results = run_tactical_benchmark(&mut simple_agent, "Traditional AlphaBeta", time_limit);
    let mcts_results = run_tactical_benchmark(&mut humanlike_agent, "MateSearchFirst MCTS", time_limit);
    
    // Print summaries
    use kingfisher::benchmarks::BenchmarkSummary;
    let ab_summary = BenchmarkSummary::from_results(&ab_results);
    let mcts_summary = BenchmarkSummary::from_results(&mcts_results);
    
    ab_summary.print_summary("Traditional AlphaBeta");
    mcts_summary.print_summary("MateSearchFirst MCTS");
    
    // Quick comparison
    println!("📊 QUICK COMPARISON:");
    if mcts_summary.mate_accuracy > ab_summary.mate_accuracy {
        let improvement = mcts_summary.mate_accuracy - ab_summary.mate_accuracy;
        println!("✅ MateSearchFirst MCTS: +{:.1}% better accuracy", improvement);
    }
    
    if mcts_summary.average_time < ab_summary.average_time {
        let speedup = ab_summary.average_time.as_millis() as f64 / mcts_summary.average_time.as_millis() as f64;
        println!("⚡ MateSearchFirst MCTS: {:.1}x faster on average", speedup);
    }
}

fn run_comparison_cmd() {
    let time_limit = Duration::from_millis(2000);
    let comparison = run_performance_comparison(time_limit);
    comparison.print_comparison();
}

fn run_speed_cmd() {
    mate_speed_benchmark();
}

fn run_all_benchmarks() {
    print_banner();
    
    println!("🔥 RUNNING COMPLETE BENCHMARK SUITE");
    println!("====================================\n");
    
    println!("Phase 1: Tactical Position Tests");
    run_tactical_benchmark_cmd();
    
    println!("\n{}\n", "=".repeat(60));
    
    println!("Phase 2: Comprehensive Comparison");
    run_comparison_cmd();
    
    println!("\n{}\n", "=".repeat(60));
    
    println!("Phase 3: Speed Analysis");
    run_speed_cmd();
    
    println!("\n🏆 BENCHMARK SUITE COMPLETE!");
    println!("============================");
    println!("Results demonstrate Kingfisher's novel mate-search-first approach");
    println!("provides significant advantages in tactical position solving.");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_banner();
        print_usage();
        return;
    }
    
    match args[1].as_str() {
        "tactical" => {
            print_banner();
            run_tactical_benchmark_cmd();
        },
        "comparison" => {
            print_banner();
            run_comparison_cmd();
        },
        "speed" => {
            print_banner();
            run_speed_cmd();
        },
        "all" => {
            run_all_benchmarks();
        },
        "help" | "--help" | "-h" => {
            print_banner();
            print_usage();
        },
        _ => {
            println!("❌ Unknown command: {}", args[1]);
            print_usage();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kingfisher::benchmarks::tactical_suite::get_tactical_test_suite;
    
    #[test]
    fn test_benchmark_suite_loads() {
        let positions = get_tactical_test_suite();
        assert!(!positions.is_empty());
        println!("✅ Loaded {} tactical positions", positions.len());
    }
    
    #[test]
    fn test_agents_create() {
        let _simple = create_simple_agent();
        let _humanlike = create_humanlike_agent();
        println!("✅ Both agents created successfully");
    }
}