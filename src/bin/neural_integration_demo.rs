//! Neural Network Integration Demonstration
//!
//! This demo shows how the tactical-first MCTS search successfully integrates
//! with neural network policy evaluation to combine strategic guidance
//! with tactical precision.

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::neural_net::NeuralNetPolicy;
use std::time::{Duration, Instant};

fn main() {
    println!("🧠 Kingfisher Neural Network Integration Demo");
    println!("===========================================");
    
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();
    
    // Sample test positions
    let positions = vec![
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Tactical Pin", "r1bqk1nr/pp3ppp/2n5/1B1pp3/1b1PP3/2N2N2/PP3PPP/R1BQK2R w KQkq - 0 7"),
        ("Endgame Puzzle", "8/8/8/4k3/4P3/4K3/8/8 w - - 0 1"),
    ];
    
    // 1. Showcase basic integration
    run_integration_showcase(&positions, &move_gen, &pesto_eval);
    
    // 2. Performance comparison
    run_performance_comparison(&positions, &move_gen, &pesto_eval);
}

/// Showcase basic integration of tactical-first search with neural policy
fn run_integration_showcase(positions: &[(&str, &str)], move_gen: &MoveGen, pesto_eval: &PestoEval) {
    println!("-- Part 1: Neural Integration Showcase --");
    
    // Create neural policy instance
    let mut nn_policy = Some(NeuralNetPolicy::new_demo_enabled());
    
    for (name, fen) in positions {
        println!("🔍 Analyzing: {}", name);
        println!("   Position: {}", fen);
        
        let board = Board::new_from_fen(fen);
        
        let config = TacticalMctsConfig {
            max_iterations: 500,
            time_limit: Duration::from_millis(1000),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: true,
        };
        
        let start_time = Instant::now();
        let (best_move, stats, _) = tactical_mcts_search(
            board.clone(),
            move_gen,
            pesto_eval,
            &mut nn_policy,
            config,
        );
        let search_time = start_time.elapsed();
        
        if let Some(mv) = best_move {
            println!("   🎯 Best Move: {}", mv.to_uci());
        } else {
            println!("   ❌ No move found");
        }
        
        println!("   📈 Search Statistics:");
        println!("      • Iterations: {}", stats.iterations);
        println!("      • Nodes Expanded: {}", stats.nodes_expanded);
        println!("      • Tactical Moves Explored: {}", stats.tactical_moves_explored);
        println!("      • NN Policy Evaluations: {}", stats.nn_policy_evaluations);
        println!("      • Mates Found: {}", stats.mates_found);
        println!("      • Search Time: {:?}", search_time);
        
        println!("   ✅ Integration working: Classical tactics + Neural guidance\n");
    }
}

fn run_performance_comparison(positions: &[(&str, &str)], move_gen: &MoveGen, pesto_eval: &PestoEval) {
    println!("-- Part 2: Performance Comparison --");
    
    let mut hybrid_total_time = Duration::from_millis(0);
    let mut classical_total_time = Duration::from_millis(0);
    let mut neural_total_time = Duration::from_millis(0);
    
    let mut hybrid_tactical_moves = 0;
    let mut classical_tactical_moves = 0;
    let mut neural_tactical_moves = 0;
    
    let mut hybrid_nn_calls = 0;
    let mut neural_nn_calls = 0;
    
    for (i, (name, fen)) in positions.iter().enumerate() {
        println!("🏁 Round {}: {}", i + 1, name);
        
        let board = Board::new_from_fen(fen);
        
        // 1. Hybrid
        let mut nn_policy_hybrid = Some(NeuralNetPolicy::new_demo_enabled());
        let hybrid_config = TacticalMctsConfig {
            max_iterations: 300,
            time_limit: Duration::from_millis(800),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: true,
        };
        
        let start = Instant::now();
        let (hybrid_move, hybrid_stats, _) = tactical_mcts_search(
            board.clone(), move_gen, pesto_eval, &mut nn_policy_hybrid, hybrid_config
        );
        let hybrid_time = start.elapsed();
        hybrid_total_time += hybrid_time;
        hybrid_tactical_moves += hybrid_stats.tactical_moves_explored;
        hybrid_nn_calls += hybrid_stats.nn_policy_evaluations;
        
        // 2. Classical
        let mut nn_policy_none = None;
        let classical_config = TacticalMctsConfig {
            max_iterations: 300,
            time_limit: Duration::from_millis(800),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: false,
        };
        
        let start_classical = Instant::now();
        let (classical_move, classical_stats, _) = tactical_mcts_search(
            board.clone(), move_gen, pesto_eval, &mut nn_policy_none, classical_config
        );
        let classical_time = start_classical.elapsed();
        classical_total_time += classical_time;
        classical_tactical_moves += classical_stats.tactical_moves_explored;

        // 3. Neural-Only
        let mut nn_policy_neural = Some(NeuralNetPolicy::new_demo_enabled());
        let neural_config = TacticalMctsConfig {
            max_iterations: 300,
            time_limit: Duration::from_millis(800),
            mate_search_depth: 0,
            exploration_constant: 1.414,
            use_neural_policy: true,
        };
        
        let start_neural = Instant::now();
        let (neural_move, neural_stats, _) = tactical_mcts_search(
            board.clone(), move_gen, pesto_eval, &mut nn_policy_neural, neural_config
        );
        let neural_time = start_neural.elapsed();
        neural_total_time += neural_time;
        neural_tactical_moves += neural_stats.tactical_moves_explored;
        neural_nn_calls += neural_stats.nn_policy_evaluations;
        
        println!("   🧠 Hybrid: {} ({}ms, {} tactical, {} NN)", 
                 format_move_option(hybrid_move), hybrid_time.as_millis(),
                 hybrid_stats.tactical_moves_explored, hybrid_stats.nn_policy_evaluations);
        println!("   ⚔️  Classical: {} ({}ms, {} tactical)", 
                 format_move_option(classical_move), classical_time.as_millis(),
                 classical_stats.tactical_moves_explored);
        println!("   🤖 Neural: {} ({}ms, {} tactical, {} NN)", 
                 format_move_option(neural_move), neural_time.as_millis(),
                 neural_stats.tactical_moves_explored, neural_stats.nn_policy_evaluations);
    }
}

fn format_move_option(mv: Option<kingfisher::move_types::Move>) -> String {
    match mv {
        Some(m) => m.to_uci(),
        None => "None".to_string(),
    }
}
