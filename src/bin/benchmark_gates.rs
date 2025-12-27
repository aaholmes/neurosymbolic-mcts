use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::{mate_search, koth_center_in_3};
use kingfisher::boardstack::BoardStack;
use std::time::Instant;

fn main() {
    let move_gen = MoveGen::new();
    
    println!("🚀 Benchmarking Safety Gates...");

    // --- 1. Mate Search Tests ---
    println!("\n--- Tier 1: Mate Search (Checks Only) ---");
    let mate_puzzles = vec![
        // Mate in 1
        ("4k3/Q7/4K3/8/8/8/8/8 w - - 0 1", "Mate in 1"),
        // Mate in 2 (Back rank)
        ("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1", "Mate in 2"),
        // Smothered Mate (needs more than checks only? Let's see)
        // Actually smothered mate usually involves checks until the end.
    ];

    for (fen, name) in mate_puzzles {
        let board = Board::new_from_fen(fen);
        let mut stack = BoardStack::with_board(board);
        
        let start = Instant::now();
        let (score, mv, nodes) = mate_search(&mut stack, &move_gen, 3, false);
        let duration = start.elapsed();
        
        println!("{}: {} ({} nodes) in {:?}", 
            name, 
            if score >= 1_000_000 { format!("FOUND ({})", mv.to_uci()) } else { "NOT FOUND".to_string() },
            nodes,
            duration
        );
    }

    // --- 2. KOTH Gate Tests ---
    println!("\n--- Tier 1: KOTH-in-3 Gate ---");
    let koth_puzzles = vec![
        // White King on d3, can reach d4 in 1 move.
        ("8/8/8/8/8/3K4/8/7k w - - 0 1", "KOTH in 1", true),
        // White King on d2, can reach d4 in 2 moves
        ("8/8/8/8/8/8/3K4/7k w - - 0 1", "KOTH in 2", true),
        // White King on d1, can reach d4 in 3 moves
        ("8/8/8/8/8/8/8/3K3k w - - 0 1", "KOTH in 3", true),
        // White King on a1, blocked by Black King on b2.
        // a1 needs to go to b2/c2/c3 to progress. b2 is blocked.
        // But a1 can go to b1 (dist 3) or a2 (dist 3).
        // The pruning requires reaching Ring 2 (dist 2) on move 1.
        // Squares at dist 2 from d4 (center) are c2, c3, b3...
        // Wait, a1->b2 is the ONLY way to get to Dist 2?
        // a1 (0,0). d4 (3,3). Dist 3.
        // Neighbors of a1: a2 (0,1), b2 (1,1), b1 (1,0).
        // a2 to d4: dx=3, dy=2. Dist 3. (Not Ring 2)
        // b1 to d4: dx=2, dy=3. Dist 3. (Not Ring 2)
        // b2 to d4: dx=2, dy=2. Dist 2. (Ring 2).
        // So b2 is the ONLY square that progresses to Ring 2.
        // If b2 is occupied/blocked, we fail the pruning check.
        ("8/8/8/8/8/8/1k6/K7 w - - 0 1", "KOTH Blocked", false),
    ];

    for (fen, name, expected) in koth_puzzles {
        let board = Board::new_from_fen(fen);
        
        let start = Instant::now();
        let found = koth_center_in_3(&board, &move_gen);
        let duration = start.elapsed();
        
        println!("{}: {} (Expected {}) in {:?}", 
            name, 
            if found { "FOUND" } else { "NOT FOUND" },
            if expected { "FOUND" } else { "NOT FOUND" },
            duration
        );
    }
}
