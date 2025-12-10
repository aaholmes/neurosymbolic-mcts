//! Neural Network Integration Test Binary
//!
//! Tests the neural network policy integration with the chess engine

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::mcts::neural_mcts_search;
use kingfisher::move_generation::MoveGen;
use kingfisher::neural_net::NeuralNetPolicy;
use std::time::Duration;

fn main() {
    println!("🧠 Kingfisher Neural Network Integration Test");
    println!("=============================================");
    
    // Initialize components
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();
    
    // Test neural network availability
    let mut nn_policy = Some(NeuralNetPolicy::new());
    
    // Test basic availability
    if let Some(ref nn) = nn_policy {
        println!("Neural Network available: {}", nn.is_available());
        
        // Tensor conversion not implemented in stub
        // let tensor = nn.board_to_tensor(&board);
        // println!("Tensor shape: {:?}", tensor.size());
    }
    
    // Test board-to-tensor conversion
    println!("\n🎯 Testing board representation...");
    let board = Board::new();
    
    // Board to tensor test disabled for stub
    /*
    if let Some(ref nn) = nn_policy {
        let tensor = nn.board_to_tensor(&board);
        println!("✅ Board tensor shape: {} values (expected: {})", tensor.len(), 12 * 8 * 8);
        
        // Count pieces in starting position
        let piece_count: f32 = tensor.iter().sum();
        println!("✅ Total pieces in tensor: {} (expected: 32)", piece_count);
    }
    */
    
    // Test neural network prediction
    println!("\n🔮 Testing neural network prediction...");
    if let Some(ref mut nn) = nn_policy {
        if let Some((policy, value)) = nn.predict(&board) {
            println!("✅ Policy prediction: {} values", policy.len());
            println!("✅ Value prediction: {:.4}", value);
            
            // Get top moves
            let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
            let mut all_moves = captures;
            all_moves.extend(moves);
            
            let legal_moves: Vec<_> = all_moves
                .into_iter()
                .filter(|&mv| {
                    let new_board = board.apply_move_to_board(mv);
                    new_board.is_legal(&move_gen)
                })
                .collect();
            
            // Show policy statistics instead of top moves
            println!("✅ Policy statistics:");
            let policy_sum: f32 = policy.iter().sum();
            let policy_max = policy.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
            println!("   Policy sum: {:.4}, Max probability: {:.4}%", policy_sum, policy_max * 100.0);
            
            let move_priors = nn.policy_to_move_priors(&policy, &legal_moves);
            println!("✅ Move priors for {} legal moves", move_priors.len());
        } else {
            println!("⚠️  Neural network prediction failed");
        }
    }
    
    // Test neural MCTS search
    println!("\n🌲 Testing Neural MCTS search...");
    
    let start_time = std::time::Instant::now();
    
    let best_move = neural_mcts_search(
        board,
        &move_gen,
        &pesto_eval,
        &mut nn_policy,
        2,  // Mate search depth
        Some(50),  // Limited iterations for testing
        Some(Duration::from_millis(1000)),  // 1 second limit
    );
    
    let search_time = start_time.elapsed();
    
    match best_move {
        Some(mv) => {
            println!("✅ Neural MCTS found move: {:?}", mv);
            println!("✅ Search completed in {:?}", search_time);
        }
        None => {
            println!("❌ No move found (checkmate/stalemate?)");
        }
    }
    
    // Test cache performance
    if let Some(ref nn) = nn_policy {
        let (cache_size, max_size) = nn.cache_stats();
        println!("✅ NN Cache: {}/{} positions", cache_size, max_size);
    }
    
    println!("\n🎉 Neural network integration test complete!");
    println!("\nNext steps:");
    println!("1. Install PyTorch for full neural network support");
    println!("2. Train a chess model using the training pipeline");
    println!("3. Integrate with stronger MCTS for competitive play");
}