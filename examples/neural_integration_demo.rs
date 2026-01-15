//! Neural Integration Demo
//! 
//! This binary demonstrates the integration of the neural network policy
//! into the tactical MCTS search. It compares:
//! 1. Pure MCTS (Baseline)
//! 2. Tactical-First MCTS (No NN)
//! 3. Neural-Enhanced MCTS (With NN)

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::mcts::inference_server::InferenceServer;
use kingfisher::neural_net::NeuralNetPolicy;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    println!("🧠 Neural Integration Demo");
    println!("========================");
    
    // Paths
    let model_path = "models/latest.pt"; // Adjust as needed
    
    // Initialize components
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();
    
    // Try to load the model
    let mut nn_policy = NeuralNetPolicy::new();
    let model_loaded = match nn_policy.load(model_path) {
        Ok(_) => {
            println!("✅ Model loaded successfully from {}", model_path);
            true
        },
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            println!("   Running in fallback mode (Tactical MCTS only)");
            false
        }
    };
    
    // Test positions
    let positions = vec![
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Tactical Midgame", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Sharp Position", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ];
    
    for (name, fen) in positions {
        println!("\n🔍 Analyzing: {}", name);
        let board = Board::new_from_fen(fen);
        
        let config = TacticalMctsConfig {
            max_iterations: 100,
            time_limit: Duration::from_secs(1),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: true,
            inference_server: None,
            logger: None,
            ..Default::default()
        };
        
        // 1. Direct Inference (if model loaded)
        if model_loaded {
            println!("   Running with Direct Inference...");
            let start = Instant::now();
            let mut policy_opt = Some(nn_policy.clone());
            let (best_move, stats, _) = tactical_mcts_search(
                board.clone(),
                &move_gen,
                &pesto_eval,
                &mut policy_opt,
                config.clone(),
            );
            println!("   Direct: {:?} ({} iter, {:?})", 
                     best_move, stats.iterations, start.elapsed());
        }
        
        // 2. Inference Server (Batched) - Simulation
        if model_loaded {
            println!("   Running with Inference Server (Batched)...");
            // Create a new policy instance for the server
            let mut server_policy = NeuralNetPolicy::new();
            let _ = server_policy.load(model_path);
            
            let server = Arc::new(InferenceServer::new(
                server_policy, 
                1 // batch size
            ));
            
            // Wait for server to start
            std::thread::sleep(Duration::from_millis(500));
            
            let server_config = TacticalMctsConfig {
                inference_server: Some(server.clone()),
                ..config.clone()
            };
            
            let start = Instant::now();
            let mut policy_opt = Some(nn_policy.clone());
            let (best_move, stats, _) = tactical_mcts_search(
                board.clone(),
                &move_gen,
                &pesto_eval,
                &mut policy_opt,
                server_config,
            );
            println!("   Server: {:?} ({} iter, {:?})", 
                     best_move, stats.iterations, start.elapsed());
        }
    }
    
    // Compare Configurations
    if model_loaded {
        println!("\n📊 Configuration Comparison");
        let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        
        let mut server_policy = NeuralNetPolicy::new();
        let _ = server_policy.load(model_path);
        let server = Arc::new(InferenceServer::new(server_policy, 4));
        
        let hybrid_config = TacticalMctsConfig {
            max_iterations: 200,
            time_limit: Duration::from_secs(2),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: true,
            inference_server: Some(server.clone()),
            logger: None,
            ..Default::default()
        };
        
        println!("   Running Hybrid Search...");
        let start = Instant::now();
        let mut policy = Some(nn_policy.clone());
        let (_, hybrid_stats, _) = tactical_mcts_search(
            board.clone(), &move_gen, &pesto_eval, &mut policy, hybrid_config
        );
        println!("   Hybrid: {} nodes/sec", 
                 hybrid_stats.nodes_expanded as f64 / start.elapsed().as_secs_f64());

        let classical_config = TacticalMctsConfig {
            max_iterations: 200,
            time_limit: Duration::from_secs(2),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: false,
            inference_server: None,
            logger: None,
            ..Default::default()
        };
        
        println!("   Running Classical Search...");
        let start = Instant::now();
        let mut no_policy = None;
        let (_, classical_stats, _) = tactical_mcts_search(
            board.clone(), &move_gen, &pesto_eval, &mut no_policy, classical_config
        );
        println!("   Classical: {} nodes/sec", 
                 classical_stats.nodes_expanded as f64 / start.elapsed().as_secs_f64());
                 
        let neural_config = TacticalMctsConfig {
            max_iterations: 200,
            time_limit: Duration::from_secs(2),
            mate_search_depth: 0, // Disable mate search to test pure neural MCTS
            exploration_constant: 1.414,
            use_neural_policy: true,
            inference_server: Some(server.clone()),
            logger: None,
            ..Default::default()
        };
        
        println!("   Running Pure Neural Search...");
        let start = Instant::now();
        let mut policy = Some(nn_policy.clone());
        let (_, neural_stats, _) = tactical_mcts_search(
            board.clone(), &move_gen, &pesto_eval, &mut policy, neural_config
        );
        println!("   Neural: {} nodes/sec", 
                 neural_stats.nodes_expanded as f64 / start.elapsed().as_secs_f64());
    }
}