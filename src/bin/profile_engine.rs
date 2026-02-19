//! Profiling Binary for MCTS Operation Timing
//!
//! Runs N self-play games and reports aggregate timing statistics for each
//! operation in evaluate_leaf_node: KOTH-in-3, mate search, Q-search, and NN inference.

use kingfisher::boardstack::BoardStack;
use kingfisher::mcts::{
    reuse_subtree, tactical_mcts_search_for_training_with_reuse, StatsAccumulator,
    TacticalMctsConfig, TimingAccumulator,
};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::transposition::TranspositionTable;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

#[cfg(feature = "neural")]
use kingfisher::mcts::InferenceServer;
#[cfg(feature = "neural")]
use kingfisher::neural_net::NeuralNetPolicy;
#[cfg(feature = "neural")]
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path: Option<String> = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned());

    let num_games: usize = args
        .iter()
        .position(|a| a == "--games")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let simulations: u32 = args
        .iter()
        .position(|a| a == "--simulations")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);

    let enable_koth = args.iter().any(|a| a == "--koth");
    let disable_tier1 = args.iter().any(|a| a == "--disable-tier1");
    let disable_material = args.iter().any(|a| a == "--disable-material");

    let batch_size: usize = args
        .iter()
        .position(|a| a == "--batch-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    println!("=== MCTS Operation Profiler ===");
    println!("  Games: {}", num_games);
    println!("  Simulations/move: {}", simulations);
    println!("  KOTH: {}", enable_koth);
    println!("  Tier1: {}", !disable_tier1);
    println!("  Material: {}", !disable_material);
    println!("  Model: {:?}", model_path);
    println!("  Batch size: {}", batch_size);
    println!();

    // Set up inference server if model provided
    #[cfg(feature = "neural")]
    let inference_server: Option<Arc<InferenceServer>> = if let Some(ref path) = model_path {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(path) {
            eprintln!("Failed to load model from {}: {}", path, e);
            None
        } else {
            println!("  Loaded model, batch_size={}", batch_size);
            Some(Arc::new(InferenceServer::new(nn, batch_size)))
        }
    } else {
        None
    };

    #[cfg(not(feature = "neural"))]
    let inference_server: Option<()> = None;

    let has_nn = inference_server.is_some();

    // Accumulators across all games
    let mut koth_acc = TimingAccumulator::default();
    let mut mate_acc = TimingAccumulator::default();
    let mut qsearch_acc = TimingAccumulator::default();
    let mut nn_acc = TimingAccumulator::default();
    let mut koth_nodes_acc = StatsAccumulator::default();
    let mut mate_nodes_acc = StatsAccumulator::default();
    let mut qsearch_nodes_acc = StatsAccumulator::default();
    let mut qsearch_depth_acc = StatsAccumulator::default();
    let mut qsearch_completed: u64 = 0;
    let mut qsearch_hit_limit: u64 = 0;
    let mut total_moves: u64 = 0;
    let mut total_iterations: u64 = 0;

    let overall_start = Instant::now();

    for game_idx in 0..num_games {
        let move_gen = MoveGen::new();

        let config = TacticalMctsConfig {
            max_iterations: simulations,
            time_limit: Duration::from_secs(300),
            mate_search_depth: if disable_tier1 { 0 } else { 5 },
            exploration_constant: 1.414,
            use_neural_policy: has_nn,
            #[cfg(feature = "neural")]
            inference_server: inference_server.clone(),
            #[cfg(not(feature = "neural"))]
            inference_server: None,
            logger: None,
            enable_koth,
            enable_tier1_gate: !disable_tier1,
            enable_material_value: !disable_material,
            enable_tier3_neural: has_nn,
            randomize_move_order: true,
            ..Default::default()
        };

        let mut rng = StdRng::seed_from_u64(game_idx as u64);
        let mut board_stack = BoardStack::new();
        let mut tt = TranspositionTable::new();
        let mut previous_root = None;
        let mut move_count = 0u32;
        let explore_base: f64 = 0.80;

        loop {
            let board = board_stack.current_state().clone();

            let result = tactical_mcts_search_for_training_with_reuse(
                board.clone(),
                &move_gen,
                config.clone(),
                previous_root.take(),
                &mut tt,
            );

            if result.best_move.is_none() {
                break;
            }

            // Merge timing and node stats from this search
            koth_acc.merge(&result.stats.koth_timing);
            mate_acc.merge(&result.stats.mate_search_timing);
            qsearch_acc.merge(&result.stats.qsearch_timing);
            nn_acc.merge(&result.stats.nn_timing);
            koth_nodes_acc.merge(&result.stats.koth_nodes);
            mate_nodes_acc.merge(&result.stats.mate_search_nodes);
            qsearch_nodes_acc.merge(&result.stats.qsearch_nodes);
            qsearch_depth_acc.merge(&result.stats.qsearch_depth);
            qsearch_completed += result.stats.qsearch_completed_count;
            qsearch_hit_limit += result.stats.qsearch_hit_limit_count;
            total_iterations += result.stats.iterations as u64;

            // Proportional-or-greedy move selection (same as self_play)
            let move_number = (move_count / 2) + 1;
            let explore_prob = explore_base.powi((move_number as i32) - 1);
            let selected_move = if rng.gen::<f64>() < explore_prob {
                sample_proportional(&result.root_policy, &mut rng)
                    .unwrap_or_else(|| result.best_move.unwrap())
            } else {
                result.best_move.unwrap()
            };
            previous_root = reuse_subtree(result.root_node, selected_move);

            board_stack.make_move(selected_move);
            move_count += 1;

            // Check termination conditions
            if enable_koth {
                let (wk, bk) = board_stack.current_state().is_koth_win();
                if wk || bk {
                    break;
                }
            }
            if board_stack.is_draw_by_repetition() {
                break;
            }
            if board_stack.current_state().halfmove_clock() >= 100 {
                break;
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

        total_moves += move_count as u64;
        println!(
            "  Game {}/{}: {} moves, {} iterations",
            game_idx + 1,
            num_games,
            move_count,
            total_iterations
        );
    }

    let overall_elapsed = overall_start.elapsed();

    // Print results
    println!();
    println!(
        "=== Profiling Results ({} games, {} total moves, {} total iterations) ===",
        num_games, total_moves, total_iterations
    );
    println!(
        "Total wall time: {:.1}s",
        overall_elapsed.as_secs_f64()
    );
    println!();
    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>12}",
        "Operation", "Count", "Total(ms)", "Mean(us)", "Std(us)"
    );
    println!("{}", "-".repeat(68));

    print_row("KOTH-in-3", &koth_acc);
    print_row("Mate search", &mate_acc);
    print_row("Q-search", &qsearch_acc);
    print_row("NN inference", &nn_acc);

    println!();
    println!("=== Nodes per call ===");
    println!(
        "{:<20} {:>10} {:>12} {:>12}",
        "Operation", "Count", "Mean nodes", "Std nodes"
    );
    println!("{}", "-".repeat(56));
    print_stats_row("KOTH-in-3", &koth_nodes_acc);
    print_stats_row("Mate search", &mate_nodes_acc);
    print_stats_row("Q-search", &qsearch_nodes_acc);

    println!();
    println!("=== Q-search depth stats ===");
    let qsearch_total = qsearch_completed + qsearch_hit_limit;
    if qsearch_total > 0 {
        println!(
            "Completed naturally: {:>10} ({:.1}%)",
            qsearch_completed,
            100.0 * qsearch_completed as f64 / qsearch_total as f64
        );
        println!(
            "Hit depth limit:    {:>10} ({:.1}%)",
            qsearch_hit_limit,
            100.0 * qsearch_hit_limit as f64 / qsearch_total as f64
        );
        println!(
            "Depth used: mean={:.1}, std={:.1}",
            qsearch_depth_acc.mean_val(),
            qsearch_depth_acc.std_val()
        );
    } else {
        println!("No Q-search calls recorded.");
    }
}

fn print_stats_row(name: &str, acc: &StatsAccumulator) {
    if acc.count == 0 {
        println!(
            "{:<20} {:>10} {:>12} {:>12}",
            name, 0, "-", "-"
        );
    } else {
        println!(
            "{:<20} {:>10} {:>12.1} {:>12.1}",
            name,
            acc.count,
            acc.mean_val(),
            acc.std_val()
        );
    }
}

/// Sample a move proportionally from visit counts (visits-1 distribution).
fn sample_proportional(policy: &[(Move, u32)], rng: &mut impl Rng) -> Option<Move> {
    if policy.is_empty() {
        return None;
    }
    let total: u32 = policy.iter().map(|(_, v)| v.saturating_sub(1)).sum();
    if total == 0 {
        return policy.iter().max_by_key(|(_, v)| *v).map(|(mv, _)| *mv);
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

fn print_row(name: &str, acc: &TimingAccumulator) {
    if acc.count == 0 {
        println!(
            "{:<20} {:>10} {:>12} {:>12} {:>12}",
            name, 0, "-", "-", "-"
        );
    } else {
        println!(
            "{:<20} {:>10} {:>12.1} {:>12.1} {:>12.1}",
            name,
            acc.count,
            acc.total.as_secs_f64() * 1000.0,
            acc.mean_us(),
            acc.std_us()
        );
    }
}
