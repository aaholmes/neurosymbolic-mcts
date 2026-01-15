//! Tactical Move Cache Performance Demo
//!
//! This demo showcases the performance improvement achieved by implementing
//! position-based caching for tactical move detection in the Kingfisher Chess Engine.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::tactical::{identify_tactical_moves, get_tactical_cache_stats, clear_tactical_cache};
use std::time::Instant;

fn main() {
    println!("🏆 Kingfisher Chess Engine - Tactical Move Cache Performance Demo");
    println!("=================================================================\n");

    // Test positions with varying tactical complexity
    let test_positions = vec![
        (
            "Starting Position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ),
        (
            "Complex Middlegame",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
        ),
        (
            "Tactical Position",
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        ),
        (
            "Sharp Sicilian",
            "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQkq - 3 6"
        ),
        (
            "Endgame Position",
            "8/2k5/3p4/p2P1p2/P4P2/1K6/8/8 w - - 0 1"
        ),
    ];

    let move_gen = MoveGen::new();
    
    println!("📊 Phase 1: Cache Performance Analysis");
    println!("=====================================\n");
    
    benchmark_cache_performance(&test_positions, &move_gen);
    
    println!("\n⚡ Phase 2: Repeated Position Analysis");
    println!("=====================================\n");
    
    benchmark_repeated_positions(&test_positions, &move_gen);
    
    println!("\n🎯 Phase 3: Cache Statistics Summary");
    println!("===================================\n");
    
    display_final_statistics();
    
    println!("\n✅ Demo completed! The tactical move cache provides significant");
    println!("   performance improvements for repeated position analysis.");
}

/// Benchmark cache performance with cold vs warm cache
fn benchmark_cache_performance(positions: &[(&str, &str)], move_gen: &MoveGen) {
    println!("Testing cache performance with {} positions:", positions.len());
    
    // Cold cache test (no caching)
    clear_tactical_cache();
    let cold_start = Instant::now();
    let mut total_tactical_moves_cold = 0;
    
    for (name, fen) in positions {
        let board = Board::new_from_fen(fen);
        let tactical_moves = identify_tactical_moves(&board, move_gen);
        total_tactical_moves_cold += tactical_moves.len();
        println!("   • {}: {} tactical moves", name, tactical_moves.len());
    }
    let cold_time = cold_start.elapsed();
    
    // Warm cache test (repeat same positions)
    let warm_start = Instant::now();
    let mut total_tactical_moves_warm = 0;
    
    for (_, fen) in positions {
        let board = Board::new_from_fen(fen);
        let tactical_moves = identify_tactical_moves(&board, move_gen);
        total_tactical_moves_warm += tactical_moves.len();
    }
    let warm_time = warm_start.elapsed();
    
    // Results
    println!("\n📈 Cache Performance Results:");
    println!("   🔸 Cold Cache Time: {:?}", cold_time);
    println!("   🔸 Warm Cache Time: {:?}", warm_time);
    
    if warm_time.as_nanos() > 0 {
        let speedup = cold_time.as_nanos() as f64 / warm_time.as_nanos() as f64;
        println!("   🔸 Speedup Factor: {:.2}x", speedup);
    }
    
    assert_eq!(total_tactical_moves_cold, total_tactical_moves_warm);
    
    let (cache_size, _, hits, misses, hit_rate) = get_tactical_cache_stats();
    println!("   🔸 Cache Hits: {}", hits);
    println!("   🔸 Cache Misses: {}", misses);
    println!("   🔸 Hit Rate: {:.1}%", hit_rate * 100.0);
    println!("   🔸 Cache Size: {} positions", cache_size);
}

/// Benchmark performance with repeated position analysis
fn benchmark_repeated_positions(positions: &[(&str, &str)], move_gen: &MoveGen) {
    println!("Testing repeated position analysis:");
    
    clear_tactical_cache();
    
    let iterations = 20;
    let boards: Vec<Board> = positions.iter()
        .map(|(_, fen)| Board::new_from_fen(fen))
        .collect();
    
    // Benchmark repeated analysis
    let start = Instant::now();
    let mut total_moves = 0;
    
    for i in 0..iterations {
        for (j, board) in boards.iter().enumerate() {
            let tactical_moves = identify_tactical_moves(board, move_gen);
            total_moves += tactical_moves.len();
            
            // Show progress every few iterations
            if i % 5 == 0 && j == 0 {
                let (_, _, hits, misses, hit_rate) = get_tactical_cache_stats();
                println!("   Iteration {}: {:.1}% hit rate ({} hits, {} misses)", 
                         i + 1, hit_rate * 100.0, hits, misses);
            }
        }
    }
    
    let total_time = start.elapsed();
    
    println!("\n📊 Repeated Analysis Results:");
    println!("   🔸 Total Iterations: {}", iterations);
    println!("   🔸 Positions per Iteration: {}", positions.len());
    println!("   🔸 Total Analyses: {}", iterations * positions.len());
    println!("   🔸 Total Time: {:?}", total_time);
    println!("   🔸 Average Time per Analysis: {:?}", 
             total_time / (iterations * positions.len()) as u32);
    println!("   🔸 Total Tactical Moves Found: {}", total_moves);
    
    let (cache_size, max_size, hits, misses, hit_rate) = get_tactical_cache_stats();
    println!("   🔸 Final Hit Rate: {:.1}%", hit_rate * 100.0);
    println!("   🔸 Cache Utilization: {}/{} ({:.1}%)", 
             cache_size, max_size, (cache_size as f64 / max_size as f64) * 100.0);
}

/// Display final cache statistics and performance insights
fn display_final_statistics() {
    let (cache_size, max_size, hits, misses, hit_rate) = get_tactical_cache_stats();
    let total_requests = hits + misses;
    
    println!("📋 Final Cache Statistics:");
    println!("   🔸 Total Requests: {}", total_requests);
    println!("   🔸 Cache Hits: {} ({:.1}%)", hits, (hits as f64 / total_requests as f64) * 100.0);
    println!("   🔸 Cache Misses: {} ({:.1}%)", misses, (misses as f64 / total_requests as f64) * 100.0);
    println!("   🔸 Overall Hit Rate: {:.1}%", hit_rate * 100.0);
    println!("   🔸 Cache Size: {}/{} positions", cache_size, max_size);
    
    // Performance insights
    println!("\n💡 Performance Insights:");
    
    if hit_rate > 0.8 {
        println!("   ✅ Excellent cache performance! Hit rate > 80%");
    } else if hit_rate > 0.6 {
        println!("   ✅ Good cache performance! Hit rate > 60%");
    } else if hit_rate > 0.4 {
        println!("   ⚠️  Moderate cache performance. Consider increasing cache size.");
    } else {
        println!("   ⚠️  Low cache performance. Many unique positions analyzed.");
    }
    
    if cache_size as f64 / max_size as f64 > 0.9 {
        println!("   ⚠️  Cache is nearly full. Consider increasing max_size for better performance.");
    }
    
    // Calculate estimated performance gain
    if hits > 0 && misses > 0 {
        let cache_efficiency = hits as f64 / total_requests as f64;
        let estimated_speedup = 1.0 / (1.0 - cache_efficiency + cache_efficiency * 0.1);
        println!("   📈 Estimated overall speedup: {:.1}x", estimated_speedup);
    }
    
    println!("\n🔧 Optimization Recommendations:");
    if hit_rate < 0.7 {
        println!("   • Consider increasing cache size for workloads with many unique positions");
    }
    if cache_size == max_size {
        println!("   • Cache is full - consider implementing LRU eviction for better performance");
    }
    println!("   • The tactical cache reduces redundant computation in MCTS tree search");
    println!("   • Greatest benefits seen in deep searches of similar position types");
}