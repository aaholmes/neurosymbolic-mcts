//! Self-Play Data Generation Binary
//! 
//! This binary plays games of the engine against itself to generate training data
//! for the neural network. It outputs binary files compatible with the Python training script.

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search_for_training, TacticalMctsConfig};
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::piece_types::{PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK};
use kingfisher::tensor::move_to_index;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;

// Data structure for holding a single training sample in memory before serialization
struct TrainingSample {
    board: Board,
    policy: Vec<(u16, f32)>, // (Move Index, Probability)
    value_target: f32,       // +1 (Win), -1 (Loss), 0 (Draw)
    material_scalar: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games = if args.len() > 1 { args[1].parse().unwrap_or(1) } else { 1 };
    let simulations = if args.len() > 2 { args[2].parse().unwrap_or(800) } else { 800 };
    let output_dir = if args.len() > 3 { &args[3] } else { "data/training" };
    let model_path = if args.len() > 4 { Some(args[4].clone()) } else { None };

    println!("🤖 Self-Play Generator Starting...");
    println!("   Games: {}", num_games);
    println!("   Simulations/Move: {}", simulations);
    println!("   Output Dir: {}", output_dir);
    println!("   Model Path: {:?}", model_path);

    std::fs::create_dir_all(output_dir).unwrap();

    let completed_games = Mutex::new(0);

    // Run games in parallel
    (0..num_games).into_par_iter().for_each(|i| {
        let samples = play_game(i, simulations, model_path.clone());
        
        if !samples.is_empty() {
            // Save binary data
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            let filename = format!("{}/game_{}_{}.bin", output_dir, timestamp, i);
            if let Err(e) = save_binary_data(&filename, &samples) {
                eprintln!("❌ Failed to save game data: {}", e);
            } else {
                let mut count = completed_games.lock().unwrap();
                *count += 1;
                println!("✅ Game {}/{} finished. Saved {} samples.", *count, num_games, samples.len());
            }
        }
    });
}

fn play_game(_game_num: usize, simulations: u32, model_path: Option<String>) -> Vec<TrainingSample> {
    let mut board = Board::new();
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();
    
    // Each thread gets its own NN instance
    let mut nn_policy = if let Some(path) = model_path {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(&path) {
            eprintln!("⚠️ Failed to load model from {}: {}", path, e);
            None
        } else {
            Some(nn)
        }
    } else {
        None
    };

    let mut samples = Vec::new();
    let mut move_count = 0;
    
    loop {
        // 1. MCTS Search
        let config = TacticalMctsConfig {
            max_iterations: simulations,
            time_limit: Duration::from_secs(60), 
            mate_search_depth: 1, 
            exploration_constant: 1.414,
            use_neural_policy: true, 
            inference_server: None,
        };

        let result = tactical_mcts_search_for_training(
            board.clone(),
            &move_gen,
            &pesto_eval,
            &mut nn_policy,
            config,
        );

        if result.best_move.is_none() {
            break; // Game Over
        }

        // 2. Prepare Sample (Value unknown yet)
        let total_visits: u32 = result.root_policy.iter().map(|(_, v)| *v).sum();
        let mut policy_dist = Vec::new();
        if total_visits > 0 {
            for (mv, visits) in &result.root_policy {
                let idx = move_to_index(*mv) as u16;
                let prob = *visits as f32 / total_visits as f32;
                policy_dist.push((idx, prob));
            }
        }

        // Calculate Material Scalar
        let material_scalar = board.material_imbalance() as f32;

        samples.push(TrainingSample {
            board: board.clone(),
            policy: policy_dist,
            value_target: 0.0, // Placeholder
            material_scalar,
        });

        // 3. Play Move
        let selected_move = result.best_move.unwrap(); 
        
        // Apply move
        board = board.apply_move_to_board(selected_move);
        move_count += 1;

        // Check for draw/end
        if move_count > 200 { 
            break; 
        }
        
        let (mate, _stalemate) = board.is_checkmate_or_stalemate(&move_gen);
        if mate || _stalemate {
            break;
        }
    }

    // 4. Assign Outcomes
    let (mate, _stalemate) = board.is_checkmate_or_stalemate(&move_gen);
    let final_score_white = if mate {
        if board.w_to_move {
            -1.0 // Black wins
        } else {
            1.0 // White wins
        }
    } else {
        0.0 // Draw/Stalemate
    };

    // Backpropagate Z (Value Target)
    for (i, sample) in samples.iter_mut().enumerate() {
        // i=0 is White's move. i=1 is Black's move.
        let white_to_move_at_sample = i % 2 == 0;
        
        if white_to_move_at_sample {
            sample.value_target = final_score_white;
        } else {
            sample.value_target = -final_score_white;
        }
    }

    samples
}

fn save_binary_data(filename: &str, samples: &[TrainingSample]) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    let mut buffer = Vec::new();

    for sample in samples {
        // 1. Board Features [17, 8, 8] -> 1088 floats
        let mut board_planes = vec![0.0f32; 17 * 8 * 8];
        
        // 0-11: Pieces
        for color in 0..2 {
            for pt in 0..6 {
                let offset = (color * 6 + pt) * 64;
                let bb = sample.board.get_piece_bitboard(color, pt);
                for i in 0..64 {
                    if (bb >> i) & 1 == 1 {
                        let rank = i / 8;
                        let file = i % 8;
                        board_planes[offset + (7 - rank) * 8 + file] = 1.0;
                    }
                }
            }
        }

        // 12: En Passant
        if let Some(sq) = sample.board.en_passant() {
            let offset = 12 * 64;
            let rank = sq / 8;
            let file = sq % 8;
            board_planes[offset + (usize::from(7 - rank)) * 8 + usize::from(file)] = 1.0;
        }

        // 13-16: Castling
        let rights = [
            sample.board.castling_rights.white_kingside,
            sample.board.castling_rights.white_queenside,
            sample.board.castling_rights.black_kingside,
            sample.board.castling_rights.black_queenside,
        ];
        for (i, &allowed) in rights.iter().enumerate() {
            if allowed {
                let offset = (13 + i) * 64;
                for j in 0..64 {
                    board_planes[offset + j] = 1.0;
                }
            }
        }

        for val in board_planes {
            buffer.extend_from_slice(&val.to_le_bytes());
        }

        // 2. Material Scalar [1 float]
        buffer.extend_from_slice(&sample.material_scalar.to_le_bytes());

        // 3. Value Target [1 float]
        buffer.extend_from_slice(&sample.value_target.to_le_bytes());

        // 4. Policy Target [4672 floats]
        // Initialize zero vector
        let mut policy_vec = vec![0.0f32; 4672];
        for (idx, prob) in &sample.policy {
            if (*idx as usize) < 4672 {
                policy_vec[*idx as usize] = *prob;
            }
        }
        for p in policy_vec {
            buffer.extend_from_slice(&p.to_le_bytes());
        }
    }

    file.write_all(&buffer)?;
    Ok(())
}