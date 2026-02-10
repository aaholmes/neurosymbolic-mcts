//! Verbose Game - Play a full game between two classical MCTS agents
//!
//! Both sides use neurosymbolic MCTS with classical fallback only (no NN),
//! KOTH enabled, 20 iterations/move. The Debug-level search logger prints
//! every rollout's evaluation cascade.
//!
//! Usage: verbose_game [--iterations <n>] [--max-moves <n>] [--seed <n>] [--no-emoji]

use std::env;
use std::sync::Arc;
use std::time::Duration;

use kingfisher::boardstack::BoardStack;
use kingfisher::mcts::{
    tactical_mcts_search_for_training, TacticalMctsConfig,
};
use kingfisher::mcts::search_logger::{SearchLogger, Verbosity};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = env::args().collect();
    let iterations = parse_arg(&args, "--iterations").unwrap_or(20);
    let max_moves = parse_arg(&args, "--max-moves").unwrap_or(200);
    let seed = parse_arg_u64(&args, "--seed").unwrap_or(42);
    let no_emoji = args.iter().any(|a| a == "--no-emoji");

    println!("=== Caissawary Verbose Game ===");
    println!("Iterations/move: {}", iterations);
    println!("Max moves: {}", max_moves);
    println!("Seed: {}", seed);
    println!("KOTH: enabled");
    println!("NN: disabled (classical fallback)");
    println!();

    let mut rng = StdRng::seed_from_u64(seed);

    let move_gen = MoveGen::new();
    let mut board_stack = BoardStack::new();
    let mut move_count = 0u32;

    loop {
        let board = board_stack.current_state().clone();
        let side = if board.w_to_move { "White" } else { "Black" };
        let fen = board.to_fen().unwrap_or_else(|| "???".to_string());

        println!("════════════════════════════════════════════════════════════");
        println!("Move {}: {} to move", move_count / 2 + 1, side);
        println!("FEN: {}", fen);
        board.print();
        println!("════════════════════════════════════════════════════════════");

        // Create a fresh logger for each move search
        let logger = Arc::new(
            SearchLogger::new(Verbosity::Debug).with_emoji(!no_emoji),
        );

        let config = TacticalMctsConfig {
            max_iterations: iterations,
            time_limit: Duration::from_secs(300),
            mate_search_depth: 5,
            exploration_constant: 1.414,
            use_neural_policy: false,
            inference_server: None,
            logger: Some(logger),
            enable_koth: true,
            ..Default::default()
        };

        let result = tactical_mcts_search_for_training(
            board.clone(),
            &move_gen,
            config,
        );

        // Print root children summary
        println!();
        println!("--- Root children ---");
        println!("{:<10} {:>6}  {:>8}", "Move", "Visits", "Q-value");
        let mut children_info: Vec<(Move, u32, f64)> = Vec::new();
        {
            let root = result.root_node.borrow();
            for child in &root.children {
                let c = child.borrow();
                if let Some(mv) = c.action {
                    // Q from the choosing side's perspective: positive = good for side to move
                    let q = if c.visits > 0 {
                        c.total_value / c.visits as f64
                    } else {
                        0.0
                    };
                    children_info.push((mv, c.visits, q));
                }
            }
        }
        children_info.sort_by(|a, b| b.1.cmp(&a.1));
        for (mv, visits, q) in &children_info {
            println!("{:<10} {:>6}  {:>+8.3}", mv.to_uci(), visits, q);
        }

        if result.best_move.is_none() {
            println!("\nNo legal moves — game over.");
            break;
        }

        // Sample proportionally from visit counts
        let selected = sample_proportional(&result.root_policy, &mut rng)
            .unwrap_or_else(|| result.best_move.unwrap());
        println!("\nSelected: {} (proportional sampling)", selected.to_uci());
        println!();

        board_stack.make_move(selected);
        move_count += 1;

        // Check termination conditions
        let current = board_stack.current_state();
        let (mate, stalemate) = current.is_checkmate_or_stalemate(&move_gen);
        if mate {
            let winner = if current.w_to_move { "Black" } else { "White" };
            println!("========================================");
            println!("CHECKMATE! {} wins after {} moves.", winner, (move_count + 1) / 2);
            println!("Final FEN: {}", current.to_fen().unwrap_or_default());
            break;
        }
        if stalemate {
            println!("========================================");
            println!("STALEMATE! Draw after {} moves.", (move_count + 1) / 2);
            break;
        }

        // KOTH check
        let (w_koth, b_koth) = current.is_koth_win();
        if w_koth {
            println!("========================================");
            println!("KOTH WIN! White reaches the center after {} moves.", (move_count + 1) / 2);
            break;
        }
        if b_koth {
            println!("========================================");
            println!("KOTH WIN! Black reaches the center after {} moves.", (move_count + 1) / 2);
            break;
        }

        if board_stack.is_draw_by_repetition() {
            println!("========================================");
            println!("DRAW by threefold repetition after {} moves.", (move_count + 1) / 2);
            break;
        }
        if current.halfmove_clock() >= 100 {
            println!("========================================");
            println!("DRAW by 50-move rule after {} moves.", (move_count + 1) / 2);
            break;
        }
        if move_count >= max_moves as u32 {
            println!("========================================");
            println!("DRAW by move limit ({} half-moves).", max_moves);
            break;
        }
    }
}

/// Sample a move proportionally from visit counts (temperature = 1).
fn sample_proportional(policy: &[(Move, u32)], rng: &mut impl Rng) -> Option<Move> {
    let total: u32 = policy.iter().map(|(_, v)| v.saturating_sub(1)).sum();
    if total == 0 {
        return None;
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

fn parse_arg(args: &[String], flag: &str) -> Option<u32> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_arg_u64(args: &[String], flag: &str) -> Option<u64> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
