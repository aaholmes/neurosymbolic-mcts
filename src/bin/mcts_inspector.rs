//! MCTS Tree Inspector - Visualize search trees
//!
//! Usage: mcts_inspector <fen> [--depth <n>] [--min-visits <n>] [--output <file.dot>]

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

use kingfisher::board::Board;
use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return;
    }
    
    let fen = &args[1];
    let depth_limit = parse_arg(&args, "--depth").unwrap_or(4);
    let min_visits = parse_arg(&args, "--min-visits").unwrap_or(1);
    let output_file = parse_string_arg(&args, "--output")
        .unwrap_or_else(|| "mcts_tree.dot".to_string());
    let iterations = parse_arg(&args, "--iterations").unwrap_or(500);
    let mate_depth = parse_arg(&args, "--mate-depth").unwrap_or(3);
    
    println!("🔍 MCTS Inspector");
    println!("==================");
    println!("FEN: {}", fen);
    println!("Depth limit: {}", depth_limit);
    println!("Min visits: {}", min_visits);
    println!("Iterations: {}", iterations);
    println!("Mate search depth: {}", mate_depth);
    println!();
    
    // Initialize components
    let board = Board::new_from_fen(fen);
    let move_gen = MoveGen::new();

    let config = TacticalMctsConfig {
        max_iterations: iterations,
        time_limit: Duration::from_secs(60), // Generous limit for inspection
        mate_search_depth: mate_depth as i32,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: None,
        logger: None,
        ..Default::default()
    };
    
    println!("🔄 Running MCTS search...");
    let (best_move, stats, root_node) = tactical_mcts_search(
        board,
        &move_gen,
        config,
    );
    
    println!("✅ Search complete!");
    println!("   Iterations: {}", stats.iterations);
    println!("   Time: {:?}", stats.search_time);
    println!("   Nodes: {}", stats.nodes_expanded);
    if let Some(mv) = best_move {
        println!("   Best move: {}", mv.to_uci());
    }
    println!();
    
    // Export DOT
    println!("📊 Generating Graphviz DOT...");
    let dot_output = root_node.borrow().export_dot(depth_limit as usize, min_visits);
    
    // Write to file
    let mut file = File::create(&output_file).expect("Failed to create output file");
    file.write_all(dot_output.as_bytes()).expect("Failed to write DOT file");
    
    println!("✅ DOT file written to: {}", output_file);
    println!();
    println!("To render PNG:");
    println!("  dot -Tpng {} -o mcts_tree.png", output_file);
    println!();
    println!("To render SVG (interactive):");
    println!("  dot -Tsvg {} -o mcts_tree.svg", output_file);
}

fn parse_arg(args: &[String], flag: &str) -> Option<u32> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn print_usage() {
    println!("MCTS Tree Inspector");
    println!();
    println!("Usage: mcts_inspector <fen> [options]");
    println!();
    println!("Options:");
    println!("  --depth <n>       Maximum tree depth to export (default: 4)");
    println!("  --min-visits <n>  Minimum visits to include node (default: 1)");
    println!("  --output <file>   Output DOT file (default: mcts_tree.dot)");
    println!("  --iterations <n>  MCTS iterations to run (default: 500)");
    println!();
    println!("Example:");
    println!("  mcts_inspector \"6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1\" --depth 3 --iterations 200");
}
