//! Transposition Table and Mate Search Demo
//!
//! This demo showcases how the transposition table improves the efficiency of
//! tactical-first MCTS by caching mate search results and allowing them to
//! be reused across different branches of the search tree.

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search_with_tt, TacticalMctsConfig};
use kingfisher::transposition::TranspositionTable;
use std::time::{Duration, Instant};

fn main() {
    println!("🧪 Transposition Table Mate Search Demo");
    println!("=======================================");
    
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();
    
    // Sample tactical positions
    let positions = vec![
        ("Back Rank Mate", "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"),
        ("Smothered Mate", "6rk/5Npp/8/8/8/8/8/7K b - - 0 1"),
        ("Complex Mate-in-3", "r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1"),
    ];
    
    // 1. Show speedup from cold vs warm TT
    show_tt_speedup(&positions, &move_gen, &pesto_eval);
    
    // 2. Analyze cache efficiency
    analyze_mate_cache_efficiency(&positions, &move_gen, &pesto_eval);
    
    // 3. Benchmark repeated searches
    benchmark_repeated_searches(&positions, &move_gen, &pesto_eval);
}

fn show_tt_speedup(
    positions: &[(&str, &str)],
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
) {
    println!("\n--- Part 1: Cold vs Warm TT Speedup ---");
    
    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_millis(500),
        mate_search_depth: 4,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: None,
        logger: None,
        ..Default::default()
    };
    
    // Cold transposition table test
    let cold_start = Instant::now();
    let mut cold_total_tt_hits = 0;
    let mut cold_total_tt_misses = 0;
    let mut cold_total_mates = 0;
    
    for (_, fen) in positions {
        let board = Board::new_from_fen(fen);
        let mut nn_policy = None;
        let mut tt = TranspositionTable::new(); // New TT each time
        
        let (_, stats, _) = tactical_mcts_search_with_tt(
            board, move_gen, pesto_eval, &mut nn_policy, config.clone(), &mut tt
        );
        
        cold_total_tt_hits += stats.tt_mate_hits;
        cold_total_tt_misses += stats.tt_mate_misses;
        cold_total_mates += stats.mates_found;
    }
    let cold_time = cold_start.elapsed();
    
    // Warm transposition table test (shared TT)
    let warm_start = Instant::now();
    let mut warm_total_tt_hits = 0;
    let mut warm_total_tt_misses = 0;
    let mut warm_total_mates = 0;
    let mut shared_transposition_table = TranspositionTable::new(); // Shared TT
    
    for (_, fen) in positions {
        let board = Board::new_from_fen(fen);
        let mut nn_policy = None;
        
        let (_, stats, _) = tactical_mcts_search_with_tt(
            board, move_gen, pesto_eval, &mut nn_policy, config.clone(), &mut shared_transposition_table
        );
        
        warm_total_tt_hits += stats.tt_mate_hits;
        warm_total_tt_misses += stats.tt_mate_misses;
        warm_total_mates += stats.mates_found;
    }
    let warm_time = warm_start.elapsed();
    
    println!("\n📈 Performance Comparison:");
    println!("   🔸 Cold TT Time: {:?}", cold_time);
    println!("   🔸 Warm TT Time: {:?}", warm_time);
    
    if warm_time.as_nanos() > 0 {
        let speedup = cold_time.as_nanos() as f64 / warm_time.as_nanos() as f64;
        println!("   🔸 Speedup Factor: {:.2}x", speedup);
    }
}

fn analyze_mate_cache_efficiency(
    positions: &[(&str, &str)],
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
) {
    println!("\n--- Part 2: Cache Efficiency Analysis ---");
    
    let config = TacticalMctsConfig {
        max_iterations: 200,
        time_limit: Duration::from_millis(1000),
        mate_search_depth: 4,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: None,
        logger: None,
        ..Default::default()
    };
    
    let mut shared_tt = TranspositionTable::new();
    
    for (i, (name, fen)) in positions.iter().enumerate() {
        let board = Board::new_from_fen(fen);
        let mut nn_policy = None;
        
        let start = Instant::now();
        let (best_move, stats, _) = tactical_mcts_search_with_tt(
            board, move_gen, pesto_eval, &mut nn_policy, config.clone(), &mut shared_tt
        );
        let elapsed = start.elapsed();
        
        let hit_rate = if stats.tt_mate_hits + stats.tt_mate_misses > 0 {
            stats.tt_mate_hits as f64 / (stats.tt_mate_hits + stats.tt_mate_misses) as f64 * 100.0
        } else {
            0.0
        };
        
        println!("   Position {}: {}", i + 1, name);
        println!("      • Time: {:?}, Iterations: ?{}", elapsed, stats.iterations);
        println!("      • Mates Found: {}, Best Move: ?{:?}", stats.mates_found, best_move.is_some());
        println!("      • TT Mate Hit Rate: {:.1}% ({}/{})", 
                 hit_rate, stats.tt_mate_hits, stats.tt_mate_hits + stats.tt_mate_misses);
    }
}

fn benchmark_repeated_searches(
    positions: &[(&str, &str)],
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
) {
    println!("\n--- Part 3: Repeated Position Benchmark ---");
    
    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_millis(300),
        mate_search_depth: 3,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: None,
        logger: None,
        ..Default::default()
    };
    
    let rounds = 5;
    let board = Board::new_from_fen(positions[0].1);
    let mut shared_tt = TranspositionTable::new();
    
    println!("   Running {} rounds on position: ?{}", rounds, positions[0].0);
    
    for round in 1..=rounds {
        let mut nn_policy = None;
        
        let start = Instant::now();
        let (_best_move, stats, _) = tactical_mcts_search_with_tt(
            board.clone(), move_gen, pesto_eval, &mut nn_policy, config.clone(), &mut shared_tt
        );
        let elapsed = start.elapsed();
        
        let hit_rate = if stats.tt_mate_hits + stats.tt_mate_misses > 0 {
            stats.tt_mate_hits as f64 / (stats.tt_mate_hits + stats.tt_mate_misses) as f64 * 100.0
        } else {
            0.0
        };
        
        let (tt_size, mate_entries) = shared_tt.stats();
        
        println!("   Round {}: {:?}, TT hit rate: {:.1}%, TT size: {} ({} mate)", 
                 round, elapsed, hit_rate, tt_size, mate_entries);
    }
}
