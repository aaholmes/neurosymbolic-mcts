//! Verbose Search - MCTS with Stream of Consciousness output
//!
//! Usage: verbose_search <fen> [--verbosity <level>] [--iterations <n>]

use std::env;
use std::sync::Arc;
use std::time::Duration;

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::mcts::inference_server::InferenceServer;
use kingfisher::mcts::search_logger::{SearchLogger, Verbosity};
use kingfisher::move_generation::MoveGen;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return;
    }
    
    let fen = &args[1];
    let verbosity = parse_verbosity(&args).unwrap_or(Verbosity::Normal);
    let iterations = parse_arg(&args, "--iterations").unwrap_or(200);
    let no_emoji = args.iter().any(|a| a == "--no-emoji");
    
    println!("🎙️ Caissawary Verbose MCTS Search");
    println!("======================");
    println!("FEN: {}", fen);
    println!("Verbosity: {:?}", verbosity);
    println!("Iterations: {}", iterations);
    println!();
    
    // Create logger
    let logger = SearchLogger::new(verbosity)
        .with_emoji(!no_emoji);
    
    // Initialize components
    let board = Board::new_from_fen(fen);
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let server = InferenceServer::new_mock();
    
    let config = TacticalMctsConfig {
        max_iterations: iterations,
        time_limit: Duration::from_secs(60),
        mate_search_depth: 5,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: Some(Arc::new(server)),
        logger: Some(Arc::new(logger)),
        ..Default::default()
    };
    
    println!("--- Search begins ---\n");
    
    let (best_move, stats, _root) = tactical_mcts_search(
        board,
        &move_gen,
        &pesto,
        &mut None,
        config,
    );
    
    println!("\n--- Search complete ---");
    println!();
    if let Some(mv) = best_move {
        println!("🏆 Best Move: {}", mv.to_uci());
    }
    println!("   Iterations: {}", stats.iterations);
    println!("   Time: {:?}", stats.search_time);
    println!("   Nodes: {}", stats.nodes_expanded);
    println!("   Mates found: {}", stats.mates_found);
}

fn parse_arg(args: &[String], flag: &str) -> Option<u32> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_verbosity(args: &[String]) -> Option<Verbosity> {
    args.iter()
        .position(|a| a == "--verbosity" || a == "-v")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| match v.to_lowercase().as_str() {
            "silent" | "0" => Some(Verbosity::Silent),
            "minimal" | "1" => Some(Verbosity::Minimal),
            "normal" | "2" => Some(Verbosity::Normal),
            "verbose" | "3" => Some(Verbosity::Verbose),
            "debug" | "4" => Some(Verbosity::Debug),
            _ => None,
        })
}

fn print_usage() {
    println!("Verbose MCTS Search - Stream of Consciousness Output");
    println!();
    println!("Usage: verbose_search <fen> [options]");
    println!();
    println!("Options:");
    println!("  --verbosity <level>  Output level: silent|minimal|normal|verbose|debug");
    println!("  --iterations <n>     MCTS iterations to run (default: 200)");
    println!("  --no-emoji           Disable emoji in output");
    println!();
    println!("Verbosity Levels:");
    println!("  silent  (0) - No output");
    println!("  minimal (1) - Only tier overrides and results");
    println!("  normal  (2) - Selection decisions and evaluations");
    println!("  verbose (3) - Full trace including QS details");
    println!("  debug   (4) - Internal state dumps");
    println!();
    println!("Example:");
    println!("  verbose_search \"6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1\" --verbosity verbose");
}
