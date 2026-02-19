use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;
use kingfisher::search::koth_center_in_3;
use kingfisher::search::mate_search;
use std::time::Instant;

fn main() {
    let move_gen = MoveGen::new();

    println!("🚀 Benchmarking Safety Gates & MCTS Inspector...");

    // --- 1. MCTS Search + Inspector Test ---
    println!("\n--- MCTS Inspector Test ---");
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/4Q3/PPPP1PPP/RNB1KBNR w KQkq - 0 1";
    let board = Board::new_from_fen(fen);
    let config = TacticalMctsConfig {
        max_iterations: 100,
        ..Default::default()
    };

    let (_mv, _stats, root) = tactical_mcts_search(board, &move_gen, config);
    let dot = root.borrow().export_dot(3, 0);

    match std::fs::write("test_tree.dot", dot) {
        Ok(_) => println!("✅ MCTS Inspector: test_tree.dot generated successfully."),
        Err(e) => println!("❌ MCTS Inspector: Failed to generate dot file: {}", e),
    }

    // --- 2. Mate Search Tests ---
    println!("\n--- Tier 1: Mate Search (Checks Only) ---");
    let mate_puzzles = vec![
        ("4k3/Q7/4K3/8/8/8/8/8 w - - 0 1", "Mate in 1"),
        ("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1", "Mate in 2"),
    ];

    for (fen, name) in mate_puzzles {
        let board = Board::new_from_fen(fen);

        let start = Instant::now();
        let (score, mv, nodes) = mate_search(&board, &move_gen, 3, false, 3);
        let duration = start.elapsed();

        println!(
            "{}: {} ({} nodes) in {:?}",
            name,
            if score >= 1_000_000 {
                format!("FOUND ({})", mv.to_uci())
            } else {
                "NOT FOUND".to_string()
            },
            nodes,
            duration
        );
    }

    // --- 3. KOTH Gate Tests ---
    println!("\n--- Tier 1: KOTH-in-3 Gate ---");
    let koth_puzzles = vec![
        ("8/8/8/8/8/3K4/8/7k w - - 0 1", "KOTH in 1", true),
        ("8/8/8/8/8/8/3K4/7k w - - 0 1", "KOTH in 2", true),
        ("8/8/8/8/8/8/8/3K3k w - - 0 1", "KOTH in 3", true),
        ("8/8/8/8/8/8/1k6/K7 w - - 0 1", "KOTH Blocked", false),
    ];

    for (fen, name, expected) in koth_puzzles {
        let board = Board::new_from_fen(fen);

        let start = Instant::now();
        let found = koth_center_in_3(&board, &move_gen).is_some();
        let duration = start.elapsed();

        println!(
            "{}: {} (Expected {}) in {:?}",
            name,
            if found { "FOUND" } else { "NOT FOUND" },
            if expected { "FOUND" } else { "NOT FOUND" },
            duration
        );
    }
}
