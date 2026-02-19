//! Self-Play Data Generation Binary
//!
//! This binary plays games of the engine against itself to generate training data
//! for the neural network. It outputs binary files compatible with the Python training script.

use kingfisher::boardstack::BoardStack;
use kingfisher::mcts::{
    reuse_subtree, tactical_mcts_search_for_training_with_reuse, InferenceServer,
    TacticalMctsConfig,
};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::search::mate_search;
use kingfisher::search::quiescence::forced_material_balance;
use kingfisher::search::{koth_best_move, koth_center_in_3};
use kingfisher::tensor::move_to_index;
use kingfisher::training_data::{save_binary_data, TrainingSample};
use kingfisher::transposition::TranspositionTable;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games = if args.len() > 1 {
        args[1].parse().unwrap_or(1)
    } else {
        1
    };
    let simulations = if args.len() > 2 {
        args[2].parse().unwrap_or(800)
    } else {
        800
    };
    let output_dir = if args.len() > 3 {
        &args[3]
    } else {
        "data/training"
    };
    let model_path = if args.len() > 4 {
        Some(args[4].clone())
    } else {
        None
    };
    let enable_koth = if args.len() > 5 {
        args[5].parse().unwrap_or(false)
    } else {
        false
    };
    let enable_tier1 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(true);
    let enable_material = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(true);
    // "all" = log every game, "first" (default) = log game 0 only, "none" = no game logs
    let log_games = args.get(8).map(|s| s.as_str()).unwrap_or("first");
    let log_all = log_games == "all";
    let log_first = log_games == "first";

    let inference_batch_size: usize = args
        .iter()
        .position(|a| a == "--batch-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);

    let num_threads: Option<usize> = args
        .iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());

    let seed_offset: u64 = args
        .iter()
        .position(|a| a == "--seed-offset")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let explore_base: f64 = args
        .iter()
        .position(|a| a == "--explore-base")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.80);

    println!("Self-Play Generator Starting...");
    println!("   Games: {}", num_games);
    println!("   Simulations/Move: {}", simulations);
    println!("   Output Dir: {}", output_dir);
    println!("   Model Path: {:?}", model_path);
    println!("   KOTH Mode: {}", enable_koth);
    println!("   Tier1 Gate: {}", enable_tier1);
    println!("   Material Value: {}", enable_material);
    println!("   Log Games: {}", log_games);
    println!("   Inference Batch Size: {}", inference_batch_size);
    println!("   Explore Base: {}", explore_base);

    // Set thread count if specified (overrides RAYON_NUM_THREADS env var)
    if let Some(threads) = num_threads {
        println!("   Game Threads: {}", threads);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok(); // Fails silently if already initialized
    } else {
        println!(
            "   Game Threads: {} (from RAYON_NUM_THREADS or default)",
            rayon::current_num_threads()
        );
    }

    std::fs::create_dir_all(output_dir).unwrap();

    // Create a single shared InferenceServer for all game threads
    let shared_server: Option<Arc<InferenceServer>> = if let Some(ref path) = model_path {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(path) {
            eprintln!("Failed to load model from {}: {}", path, e);
            None
        } else {
            println!(
                "   Shared InferenceServer: batch_size={}",
                inference_batch_size
            );
            Some(Arc::new(InferenceServer::new(nn, inference_batch_size)))
        }
    } else {
        None
    };

    let completed_games = Mutex::new(0);

    // Run games in parallel
    (0..num_games).into_par_iter().for_each(|i| {
        let verbose = log_all || (log_first && i == 0);
        let seed = seed_offset + i as u64;
        let samples = play_game(
            i,
            simulations,
            seed,
            shared_server.clone(),
            enable_koth,
            enable_tier1,
            enable_material,
            verbose,
            explore_base,
        );

        if !samples.is_empty() {
            // Save binary data
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let filename = format!("{}/game_{}_{}.bin", output_dir, timestamp, i);
            if let Err(e) = save_binary_data(&filename, &samples) {
                eprintln!("Failed to save game data: {}", e);
            } else {
                let mut count = completed_games.lock().unwrap();
                *count += 1;
                println!(
                    "Game {}/{} finished. Saved {} samples.",
                    *count,
                    num_games,
                    samples.len()
                );
            }
        }
    });
}

#[derive(Clone, Copy)]
enum EarlyWinType {
    Mate,
    Koth,
}

fn play_game(
    game_num: usize,
    simulations: u32,
    seed: u64,
    inference_server: Option<Arc<InferenceServer>>,
    enable_koth: bool,
    enable_tier1: bool,
    enable_material: bool,
    verbose: bool,
    explore_base: f64,
) -> Vec<TrainingSample> {
    let mut rng = StdRng::seed_from_u64(seed);
    let move_gen = MoveGen::new();
    let mut game_moves: Vec<String> = Vec::new();

    let has_nn = inference_server.is_some();

    // Shared config and transposition table across moves (Phase 2/3 optimization)
    let config = TacticalMctsConfig {
        max_iterations: simulations,
        time_limit: Duration::from_secs(60),
        mate_search_depth: if enable_tier1 { 5 } else { 0 },
        exploration_constant: 1.414,
        use_neural_policy: has_nn,
        inference_server,
        logger: None,
        dirichlet_alpha: 0.0,
        dirichlet_epsilon: 0.0,
        enable_koth,
        enable_tier1_gate: enable_tier1,
        enable_material_value: enable_material,
        enable_tier3_neural: has_nn,
        randomize_move_order: true,
        ..Default::default()
    };
    let mut transposition_table = TranspositionTable::new();

    // Use BoardStack for repetition and 50-move detection (Phase 3)
    let mut board_stack = BoardStack::new();
    let mut samples = Vec::new();
    let mut move_count = 0;
    let mut previous_root = None;
    let mut early_outcome: Option<f32> = None;
    let mut early_win_type: Option<EarlyWinType> = None;

    loop {
        let board = board_stack.current_state().clone();

        // --- Tier 1 pre-checks: skip MCTS for positions resolved by safety gates ---
        // Check both KOTH and mate, pick the fastest forced win (fewest plies).
        if enable_tier1 {
            let mut best_gate: Option<(i32, EarlyWinType, f32)> = None; // (plies, type, score_white)

            // Check KOTH gate
            if enable_koth {
                if let Some(n) = koth_center_in_3(&board, &move_gen) {
                    let plies = (n as i32) * 2 - 1; // n=1 -> 1 ply, n=2 -> 3 plies, etc.
                    let score_white = if board.w_to_move { 1.0 } else { -1.0 };
                    best_gate = Some((plies, EarlyWinType::Koth, score_white));
                }
            }

            // Check mate gate
            {
                let (score, _, _) = mate_search(&board, &move_gen, 5, false, 3);
                if score >= 1_000_000 {
                    let mate_plies = score - 1_000_000;
                    let score_white = if board.w_to_move { 1.0 } else { -1.0 };
                    // Pick mate if faster (fewer plies), or on ties (prefer mate as more definitive)
                    if best_gate.is_none() || mate_plies <= best_gate.unwrap().0 {
                        best_gate = Some((mate_plies, EarlyWinType::Mate, score_white));
                    }
                } else if score <= -1_000_000 {
                    // STM is being mated — KOTH can't help (it's opponent's forced win)
                    let mate_plies = (-score) - 1_000_000;
                    let score_white = if board.w_to_move { -1.0 } else { 1.0 };
                    if best_gate.is_none() || mate_plies <= best_gate.unwrap().0 {
                        best_gate = Some((mate_plies, EarlyWinType::Mate, score_white));
                    }
                }
            }

            if let Some((_, win_type, score_white)) = best_gate {
                early_outcome = Some(score_white);
                early_win_type = Some(win_type);
                break;
            }
        }

        // Try to reuse subtree from previous search (Phase 2)
        let reused_root = previous_root.take();

        let result = tactical_mcts_search_for_training_with_reuse(
            board.clone(),
            &move_gen,
            config.clone(),
            reused_root,
            &mut transposition_table,
        );

        if result.best_move.is_none() {
            break; // Game Over
        }

        // Prepare Sample (Value unknown yet)
        let total_visits: u32 = result.root_policy.iter().map(|(_, v)| *v).sum();
        let mut policy_dist = Vec::new();
        let mut policy_moves = Vec::new();
        if total_visits > 0 {
            for (mv, visits) in &result.root_policy {
                let relative_mv = if board.w_to_move {
                    *mv
                } else {
                    mv.flip_vertical()
                };
                let idx = move_to_index(relative_mv) as u16;
                let prob = *visits as f32 / total_visits as f32;
                policy_dist.push((idx, prob));
                policy_moves.push((*mv, prob));
            }
        }

        let mut temp_stack = BoardStack::with_board(board.clone());
        let (material_balance, qsearch_completed) =
            forced_material_balance(&mut temp_stack, &move_gen);
        let material_scalar = material_balance as f32;

        samples.push(TrainingSample {
            board: board.clone(),
            policy: policy_dist,
            policy_moves,
            value_target: 0.0, // Placeholder
            material_scalar,
            qsearch_completed,
            w_to_move: board.w_to_move,
        });

        // Play Move — proportional-or-greedy with decaying exploration
        let move_number = (move_count / 2) + 1;
        let explore_prob = explore_base.powi((move_number as i32) - 1);
        let selected_move = if rng.gen::<f64>() < explore_prob {
            sample_proportional(&result.root_policy, &mut rng)
                .unwrap_or_else(|| result.best_move.unwrap())
        } else {
            select_greedy(&result.root_policy)
                .unwrap_or_else(|| result.best_move.unwrap())
        };

        if verbose {
            game_moves.push(selected_move.to_san(&board, &move_gen));
        }

        // Reuse the subtree for the opponent's next move
        previous_root = reuse_subtree(result.root_node, selected_move);

        // Apply move via BoardStack (tracks position history for repetition)
        board_stack.make_move(selected_move);
        move_count += 1;

        // Check KOTH win after each move
        if enable_koth {
            let (white_won, black_won) = board_stack.current_state().is_koth_win();
            if white_won || black_won {
                break;
            }
        }

        // Check for draws (Phase 3: repetition + 50-move rule)
        if board_stack.is_draw_by_repetition() {
            break; // 3-fold repetition draw
        }
        if board_stack.current_state().halfmove_clock() >= 100 {
            break; // 50-move rule (100 half-moves)
        }
        if move_count > 200 {
            break;
        }

        let (mate, stalemate) = board_stack
            .current_state()
            .is_checkmate_or_stalemate(&move_gen);
        if mate || stalemate {
            break;
        }
    }

    // Assign Outcomes
    let final_board = board_stack.current_state();
    let (mate, _stalemate) = final_board.is_checkmate_or_stalemate(&move_gen);
    let is_repetition = board_stack.is_draw_by_repetition();
    let is_50_move = final_board.halfmove_clock() >= 100;
    let (koth_white, koth_black) = if enable_koth {
        final_board.is_koth_win()
    } else {
        (false, false)
    };

    let final_score_white = if let Some(score) = early_outcome {
        score
    } else if koth_white {
        1.0 // White wins by KOTH
    } else if koth_black {
        -1.0 // Black wins by KOTH
    } else if mate {
        if final_board.w_to_move {
            -1.0 // Black wins
        } else {
            1.0 // White wins
        }
    } else if is_repetition || is_50_move || _stalemate {
        0.0 // Draw
    } else {
        0.0 // Draw (move limit or other)
    };

    // Backpropagate Z (Value Target)
    for sample in samples.iter_mut() {
        if sample.w_to_move {
            sample.value_target = final_score_white;
        } else {
            sample.value_target = -final_score_white;
        }
    }

    if verbose {
        // If game ended by Tier1 gate, play out the forced sequence for display
        let mut playout_stack = None;
        if early_outcome.is_some() {
            let mut temp_stack = BoardStack::with_board(final_board.clone());
            match early_win_type {
                Some(EarlyWinType::Mate) => {
                    // Play out forced mate: mate_search for winning side, first legal for losing
                    for _ in 0..20 {
                        let board = temp_stack.current_state();
                        let (is_mate, is_stalemate) = board.is_checkmate_or_stalemate(&move_gen);
                        if is_mate || is_stalemate {
                            break;
                        }

                        let (score, mv, _) = mate_search(temp_stack.current_state(), &move_gen, 5, false, 3);
                        if score >= 1_000_000 {
                            game_moves.push(mv.to_san(temp_stack.current_state(), &move_gen));
                            temp_stack.make_move(mv);
                        } else {
                            // Losing side — pick first legal move
                            let board = temp_stack.current_state();
                            let (caps, quiets) = move_gen.gen_pseudo_legal_moves(board);
                            let legal: Vec<Move> = caps
                                .iter()
                                .chain(quiets.iter())
                                .filter(|&&m| board.apply_move_to_board(m).is_legal(&move_gen))
                                .cloned()
                                .collect();
                            if let Some(&response) = legal.first() {
                                game_moves
                                    .push(response.to_san(temp_stack.current_state(), &move_gen));
                                temp_stack.make_move(response);
                            } else {
                                break; // Checkmate
                            }
                        }
                    }
                }
                Some(EarlyWinType::Koth) => {
                    // Play out forced KOTH: koth_best_move for winning side, first legal for losing
                    for _ in 0..20 {
                        let board = temp_stack.current_state();
                        let (is_mate, is_stalemate) = board.is_checkmate_or_stalemate(&move_gen);
                        if is_mate || is_stalemate {
                            break;
                        }
                        if enable_koth {
                            let (wk, bk) = board.is_koth_win();
                            if wk || bk {
                                break;
                            }
                        }

                        if let Some(mv) = koth_best_move(board, &move_gen) {
                            game_moves.push(mv.to_san(temp_stack.current_state(), &move_gen));
                            temp_stack.make_move(mv);
                        } else {
                            // Losing side or no forced KOTH move — pick first legal
                            let board = temp_stack.current_state();
                            let (caps, quiets) = move_gen.gen_pseudo_legal_moves(board);
                            let legal: Vec<Move> = caps
                                .iter()
                                .chain(quiets.iter())
                                .filter(|&&m| board.apply_move_to_board(m).is_legal(&move_gen))
                                .cloned()
                                .collect();
                            if let Some(&response) = legal.first() {
                                game_moves
                                    .push(response.to_san(temp_stack.current_state(), &move_gen));
                                temp_stack.make_move(response);
                            } else {
                                break;
                            }
                        }
                    }
                }
                None => {}
            }
            playout_stack = Some(temp_stack);
        }

        // Use the post-play-out board for FEN display
        let display_board = if let Some(ref ps) = playout_stack {
            ps.current_state()
        } else {
            final_board
        };

        // Determine result label from the actual terminal state after play-out
        let result_str = if playout_stack.is_some() {
            let db = display_board;
            let (is_mate, _) = db.is_checkmate_or_stalemate(&move_gen);
            let (dkw, dkb) = if enable_koth {
                db.is_koth_win()
            } else {
                (false, false)
            };
            if dkw {
                "White wins (KOTH)"
            } else if dkb {
                "Black wins (KOTH)"
            } else if is_mate {
                if db.w_to_move {
                    "Black wins (checkmate)"
                } else {
                    "White wins (checkmate)"
                }
            } else {
                // Play-out didn't reach terminal — fall back to gate info
                if final_score_white > 0.0 {
                    "White wins (Tier1)"
                } else {
                    "Black wins (Tier1)"
                }
            }
        } else if koth_white {
            "White wins (KOTH)"
        } else if koth_black {
            "Black wins (KOTH)"
        } else if mate {
            if final_board.w_to_move {
                "Black wins (checkmate)"
            } else {
                "White wins (checkmate)"
            }
        } else if is_repetition {
            "Draw (repetition)"
        } else if is_50_move {
            "Draw (50-move rule)"
        } else if _stalemate {
            "Draw (stalemate)"
        } else {
            "Draw (move limit)"
        };

        // Print per-sample training data for auditability
        eprintln!(
            "\n=== Training samples for Game {} ({} samples) ===",
            game_num,
            samples.len()
        );
        for (i, s) in samples.iter().enumerate() {
            let fen = s.board.to_fen().unwrap_or_default();
            let stm = if s.w_to_move { "W" } else { "B" };
            // Top 5 policy moves by probability (using move names)
            let mut top_policy: Vec<(Move, f32)> = s
                .policy_moves
                .iter()
                .filter(|(_, p)| *p > 0.0)
                .cloned()
                .collect();
            top_policy.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top_policy.truncate(5);
            let policy_str: Vec<String> = top_policy
                .iter()
                .map(|(mv, p)| format!("{}:{:.3}", mv.to_san(&s.board, &move_gen), p))
                .collect();
            eprintln!(
                "  [{:3}] {} STM={} V={:+.1} M={:+.1} policy=[{}]",
                i,
                fen,
                stm,
                s.value_target,
                s.material_scalar,
                policy_str.join(", ")
            );
        }
        eprintln!();

        // Format as numbered move pairs
        let mut move_str = String::new();
        for (i, san) in game_moves.iter().enumerate() {
            if i % 2 == 0 {
                if !move_str.is_empty() {
                    move_str.push(' ');
                }
                move_str.push_str(&format!("{}.", i / 2 + 1));
            }
            move_str.push(' ');
            move_str.push_str(san);
        }
        println!(
            "\n--- Game {} ({} moves) -> {} ---",
            game_num,
            game_moves.len(),
            result_str
        );
        println!("{}", move_str);
        println!("Final FEN: {}", display_board.to_fen().unwrap_or_default());
        println!();
    }

    samples
}

/// Sample a move proportionally from visit counts (visits-1 distribution).
/// Falls back to greedy if all moves have 0 or 1 visit.
fn sample_proportional(policy: &[(Move, u32)], rng: &mut impl Rng) -> Option<Move> {
    if policy.is_empty() {
        return None;
    }
    let total: u32 = policy.iter().map(|(_, v)| v.saturating_sub(1)).sum();
    if total == 0 {
        return select_greedy(policy);
    }
    let threshold = rng.gen_range(0..total);
    let mut cumulative = 0u32;
    for (mv, visits) in policy {
        cumulative += visits.saturating_sub(1);
        if cumulative > threshold {
            return Some(*mv);
        }
    }
    policy.last().map(|(mv, _)| *mv)
}

/// Select the most-visited move.
fn select_greedy(policy: &[(Move, u32)]) -> Option<Move> {
    policy.iter().max_by_key(|(_, v)| *v).map(|(mv, _)| *mv)
}
