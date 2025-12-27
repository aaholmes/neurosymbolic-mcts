//! Performance Profiler for Tactical-Enhanced MCTS
//!
//! This tool profiles the performance of the tactical MCTS implementation
//! to identify bottlenecks and optimization opportunities.

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search, TacticalMctsConfig};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct ProfileResult {
    pub total_time: Duration,
    pub iterations: u32,
    pub nodes_expanded: u32,
    pub mate_searches: u32,
    pub tactical_moves_identified: u32,
    pub nn_evaluations: u32,
    pub avg_time_per_iteration: Duration,
    pub nodes_per_second: f64,
}

/// Profile the tactical MCTS on a specific position
fn profile_position(
    board: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    config: TacticalMctsConfig,
) -> ProfileResult {
    let mut nn_policy = None;
    
    let start_time = Instant::now();
    let (_, stats, _) = tactical_mcts_search(
        board,
        move_gen,
        pesto_eval,
        &mut nn_policy,
        config,
    );
    let total_time = start_time.elapsed();
    
    let avg_time_per_iteration = if stats.iterations > 0 {
        Duration::from_nanos(total_time.as_nanos() as u64 / stats.iterations as u64)
    } else {
        Duration::from_nanos(0)
    };
    
    let nodes_per_second = if total_time.as_secs_f64() > 0.0 {
        stats.nodes_expanded as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };
    
    ProfileResult {
        total_time,
        iterations: stats.iterations,
        nodes_expanded: stats.nodes_expanded,
        mate_searches: stats.mates_found, // Approximate
        tactical_moves_identified: stats.tactical_moves_explored,
        nn_evaluations: stats.nn_policy_evaluations,
        avg_time_per_iteration,
        nodes_per_second,
    }
}

/// Run comprehensive profiling across different configurations
fn run_profiling_suite() {
    println!("🔍 Tactical-Enhanced MCTS Performance Profiler");
    println!("{}", "=".repeat(60));
    
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();
    
    // Test positions
    let test_positions = vec![
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middle Game", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Tactical Position", "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3"),
        ("Endgame", "8/8/8/3k4/8/3K4/8/8 w - - 0 1"),
    ];
    
    // Test configurations
    let test_configs = vec![
        ("Quick Search", TacticalMctsConfig {
            max_iterations: 100,
            time_limit: Duration::from_millis(500),
            mate_search_depth: 1,
            exploration_constant: 1.414,
            use_neural_policy: false,
        }),
        ("Standard Search", TacticalMctsConfig {
            max_iterations: 500,
            time_limit: Duration::from_millis(2000),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: false,
        }),
        ("Deep Search", TacticalMctsConfig {
            max_iterations: 1000,
            time_limit: Duration::from_millis(5000),
            mate_search_depth: 5,
            exploration_constant: 1.414,
            use_neural_policy: false,
        }),
    ];
    
    for (pos_name, fen) in test_positions {
        println!("\n🎯 Profiling Position: {}", pos_name);
        println!("   FEN: {}", fen);
        
        let board = Board::new_from_fen(fen);
        
        for (config_name, config) in &test_configs {
            println!("\n   📊 Configuration: {}", config_name);
            
            let result = profile_position(board.clone(), &move_gen, &pesto_eval, config.clone());
            print_profile_result(&result);
        }
    }
    
    // Scaling analysis
    println!("\n🚀 SCALING ANALYSIS");
    println!("{}", "=".repeat(60));
    
    let board = Board::new(); // Starting position
    let iteration_counts = vec![50, 100, 200, 500, 1000];
    
    for &iterations in &iteration_counts {
        let config = TacticalMctsConfig {
            max_iterations: iterations,
            time_limit: Duration::from_millis(10000), // High time limit
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: false,
        };
        
        println!("\n📈 {} iterations:", iterations);
        let result = profile_position(board.clone(), &move_gen, &pesto_eval, config);
        print_scaling_result(&result, iterations);
    }
}

fn print_profile_result(result: &ProfileResult) {
    println!("      Total time:           {}ms", result.total_time.as_millis());
    println!("      Iterations:           {}", result.iterations);
    println!("      Nodes expanded:       {}", result.nodes_expanded);
    println!("      Tactical moves:       {}", result.tactical_moves_identified);
    println!("      NN evaluations:       {}", result.nn_evaluations);
    println!("      Avg time/iteration:   {:.2}ms", result.avg_time_per_iteration.as_secs_f64() * 1000.0);
    println!("      Nodes/second:         {:.0}", result.nodes_per_second);
    
    // Calculate efficiency metrics
    if result.iterations > 0 {
        let tactical_per_iteration = result.tactical_moves_identified as f64 / result.iterations as f64;
        let nodes_per_iteration = result.nodes_expanded as f64 / result.iterations as f64;
        println!("      Tactical/iteration:   {:.1}", tactical_per_iteration);
        println!("      Nodes/iteration:      {:.1}", nodes_per_iteration);
    }
}

fn print_scaling_result(result: &ProfileResult, target_iterations: u32) {
    let efficiency = if target_iterations > 0 {
        (result.iterations as f64 / target_iterations as f64) * 100.0
    } else {
        0.0
    };
    
    println!("   Completed: {}/{} iterations ({:.1}%)", 
             result.iterations, target_iterations, efficiency);
    println!("   Time: {}ms, Nodes: {}, NPS: {:.0}", 
             result.total_time.as_millis(), result.nodes_expanded, result.nodes_per_second);
}

/// Memory usage analysis
fn analyze_memory_usage() {
    println!("\n💾 MEMORY USAGE ANALYSIS");
    println!("{}", "=".repeat(60));
    
    // This is a simplified analysis - in a real implementation,
    // you would use tools like valgrind or built-in memory profilers
    println!("Memory analysis would require external profiling tools.");
    println!("Suggested tools:");
    println!("  - valgrind --tool=massif for heap profiling");
    println!("  - cargo-profiler for Rust-specific profiling");
    println!("  - perf for system-level performance analysis");
}

/// Bottleneck identification
fn identify_bottlenecks() {
    println!("\n🎯 POTENTIAL BOTTLENECKS");
    println!("{}", "=".repeat(60));
    
    println!("Based on the tactical MCTS implementation:");
    println!("1. 🔍 Mate Search:");
    println!("   - Called for every leaf node");
    println!("   - Optimization: Cache results, reduce depth for non-tactical positions");
    
    println!("\n2. 🎯 Tactical Move Identification:");
    println!("   - MVV-LVA calculation for each move");
    println!("   - Fork detection algorithms");
    println!("   - Optimization: Cache tactical moves per position");
    
    println!("\n3. 🌳 Node Creation and Management:");
    println!("   - RefCell borrowing overhead");
    println!("   - HashMap operations for move priorities");
    println!("   - Optimization: Object pooling, reduce allocations");
    
    println!("\n4. 🧮 Position Evaluation:");
    println!("   - Pesto evaluation for each leaf");
    println!("   - Board cloning for mate search");
    println!("   - Optimization: Incremental evaluation, board copying optimization");
}

fn main() {
    println!("🔬 Starting Tactical-Enhanced MCTS Performance Profiler\n");
    
    run_profiling_suite();
    analyze_memory_usage();
    identify_bottlenecks();
    
    println!("\n✅ Profiling Complete!");
    println!("\n💡 Next Steps:");
    println!("   1. Run with release build for accurate performance numbers");
    println!("   2. Use external profiling tools for detailed analysis");
    println!("   3. Implement suggested optimizations");
    println!("   4. Re-run profiling to measure improvements");
}