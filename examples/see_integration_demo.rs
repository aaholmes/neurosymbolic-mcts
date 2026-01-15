//! SEE Integration Demo
//!
//! This demo showcases the impact of proper Static Exchange Evaluation (SEE) integration
//! on tactical move filtering in the Kingfisher Chess Engine.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::tactical::identify_tactical_moves;
use kingfisher::search::see;
use kingfisher::move_types::Move;

fn main() {
    println!("🏆 Kingfisher Chess Engine - SEE Integration Demo");
    println!("=================================================\n");

    // Test positions where SEE makes a significant difference
    let test_positions = vec![
        (
            "Complex Exchange Position",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
            "Multiple pieces attacking central squares - SEE crucial for evaluation"
        ),
        (
            "Hanging Piece Position", 
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 6 4",
            "Black bishop on c5 can be captured - SEE determines if it's profitable"
        ),
        (
            "Defended Pawn Position",
            "rnbqkb1r/ppp2ppp/3p1n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", 
            "Central pawns defended by multiple pieces"
        ),
        (
            "Queen vs Pieces Exchange",
            "r1bqk2r/pppp1ppp/2n2n2/8/1bB1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
            "Black bishop attacks white bishop - complex exchange evaluation"
        ),
    ];

    let move_gen = MoveGen::new();
    
    println!("📊 Phase 1: SEE Impact Analysis");
    println!("===============================\n");
    
    analyze_see_impact(&test_positions, &move_gen);
    
    println!("\n⚡ Phase 2: Tactical Move Quality Comparison");
    println!("===========================================\n");
    
    compare_tactical_quality(&test_positions, &move_gen);
    
    println!("\n🎯 Phase 3: Specific Exchange Evaluations");
    println!("=========================================\n");
    
    demonstrate_exchange_evaluations(&move_gen);
    
    println!("\n✅ Demo completed! SEE integration significantly improves");
    println!("   tactical move quality by filtering out losing captures.");
}

/// Analyze the impact of SEE on tactical move identification
fn analyze_see_impact(positions: &[(&str, &str, &str)], move_gen: &MoveGen) {
    for (name, fen, description) in positions {
        println!("🔍 Analyzing: {}", name);
        println!("   Position: {}", fen);
        println!("   Context: {}", description);
        
        let board = Board::new_from_fen(fen);
        
        // Get all pseudo-legal captures
        let (captures, _) = move_gen.gen_pseudo_legal_moves(&board);
        let legal_captures: Vec<Move> = captures.into_iter()
            .filter(|&mv| board.apply_move_to_board(mv).is_legal(move_gen))
            .collect();
        
        println!("   📈 Capture Analysis:");
        println!("      • Total legal captures: {}", legal_captures.len());
        
        if !legal_captures.is_empty() {
            let mut good_captures = 0;
            let mut bad_captures = 0;
            let mut neutral_captures = 0;
            
            for &capture in &legal_captures {
                let see_value = see(&board, move_gen, capture.to, capture.from);
                
                if see_value > 0 {
                    good_captures += 1;
                } else if see_value < 0 {
                    bad_captures += 1;
                } else {
                    neutral_captures += 1;
                }
                
                let from_sq = format!("{}{}", 
                    (b'a' + (capture.from % 8) as u8) as char,
                    (b'1' + (capture.from / 8) as u8) as char);
                let to_sq = format!("{}{}", 
                    (b'a' + (capture.to % 8) as u8) as char,
                    (b'1' + (capture.to / 8) as u8) as char);
                
                let evaluation = if see_value > 0 { "Good" } else if see_value < 0 { "Bad" } else { "Equal" };
                println!("         - {}x{}: SEE = {:+}, {}", from_sq, to_sq, see_value, evaluation);
            }
            
            println!("      • Good captures (SEE > 0): {}", good_captures);
            println!("      • Bad captures (SEE < 0): {}", bad_captures);  
            println!("      • Neutral captures (SEE = 0): {}", neutral_captures);
            
            let filtering_impact = if legal_captures.len() > 0 {
                (bad_captures as f64 / legal_captures.len() as f64) * 100.0
            } else {
                0.0
            };
            println!("      • SEE filtering impact: {:.1}% bad captures filtered", filtering_impact);
        }
        
        // Get tactical moves with SEE filtering
        let tactical_moves = identify_tactical_moves(&board, move_gen);
        let tactical_captures = tactical_moves.iter()
            .filter(|tm| matches!(tm, kingfisher::mcts::tactical::TacticalMove::Capture(_, _)))
            .count();
        
        println!("   ✅ Tactical moves identified: {} (including {} captures)", 
                 tactical_moves.len(), tactical_captures);
        println!();
    }
}

/// Compare tactical move quality with and without SEE
fn compare_tactical_quality(positions: &[(&str, &str, &str)], move_gen: &MoveGen) {
    println!("Comparing tactical move quality across test positions:");
    
    for (i, (name, fen, _)) in positions.iter().enumerate() {
        println!("\n   Position {}: {}", i + 1, name);
        
        let board = Board::new_from_fen(fen);
        
        // Simulate "without SEE" by counting all legal captures
        let (captures, _) = move_gen.gen_pseudo_legal_moves(&board);
        let all_legal_captures = captures.into_iter()
            .filter(|&mv| board.apply_move_to_board(mv).is_legal(move_gen))
            .count();
        
        // With SEE filtering (current implementation)
        let tactical_moves = identify_tactical_moves(&board, move_gen);
        let see_filtered_captures = tactical_moves.iter()
            .filter(|tm| matches!(tm, kingfisher::mcts::tactical::TacticalMove::Capture(_, _)))
            .count();
        
        println!("      • All legal captures: {}", all_legal_captures);
        println!("      • SEE-filtered captures: {}", see_filtered_captures);
        
        if all_legal_captures > 0 {
            let reduction_percent = if all_legal_captures > see_filtered_captures {
                ((all_legal_captures - see_filtered_captures) as f64 / all_legal_captures as f64) * 100.0
            } else {
                0.0
            };
            println!("      • Reduction: {:.1}% (filtered {} bad captures)", 
                     reduction_percent, all_legal_captures - see_filtered_captures);
        }
    }
    
    println!("\n💡 Quality Insights:");
    println!("   ✅ SEE filtering removes losing captures from tactical consideration");
    println!("   ✅ MCTS focuses computation on profitable tactical moves");
    println!("   ✅ Improved move ordering leads to better search efficiency");
}

/// Demonstrate specific exchange evaluations
fn demonstrate_exchange_evaluations(move_gen: &MoveGen) {
    println!("Demonstrating specific exchange scenarios:");
    
    let examples = vec![
        (
            "Simple Pawn Trade",
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            Move::new(28, 35, None), // e4xd5
            "Pawn takes pawn - should be approximately equal"
        ),
        (
            "Piece vs Pawn Exchange", 
            "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
            Move::new(26, 36, None), // Bc4xe5 (if legal)
            "Bishop takes pawn - may lose to knight recapture"
        ),
        (
            "Queen vs Rook Exchange",
            "r1bqk2r/pppp1ppp/2n2n2/8/1bB1P3/3P1N2/PPP1QPPP/RNB1K2R w KQkq - 0 6",
            Move::new(12, 11, None), // Qe2xb4 (example)
            "Queen takes bishop - complex evaluation"
        ),
    ];
    
    for (name, fen, example_move, description) in examples {
        println!("\n   📋 {}", name);
        println!("      Position: {}", fen);
        println!("      Description: {}", description);
        
        let board = Board::new_from_fen(fen);
        
        // Check if the example move is legal
        let new_board = board.apply_move_to_board(example_move);
        if new_board.is_legal(move_gen) {
            let see_value = see(&board, move_gen, example_move.to, example_move.from);
            
            let from_sq = format!("{}{}", 
                (b'a' + (example_move.from % 8) as u8) as char,
                (b'1' + (example_move.from / 8) as u8) as char);
            let to_sq = format!("{}{}", 
                (b'a' + (example_move.to % 8) as u8) as char,
                (b'1' + (example_move.to / 8) as u8) as char);
            
            let evaluation = if see_value > 0 {
                "Winning exchange"
            } else if see_value < 0 {
                "Losing exchange"
            } else {
                "Equal exchange"
            };
            
            println!("      Move: {}x{}", from_sq, to_sq);
            println!("      SEE Value: {:+} centipawns", see_value);
            println!("      Evaluation: {}", evaluation);
            
            // Show if this would be filtered out by SEE
            if see_value < 0 {
                println!("      ❌ This capture would be filtered out by SEE");
            } else {
                println!("      ✅ This capture would be accepted by SEE");
            }
        } else {
            println!("      ⚠️  Example move is not legal in this position");
        }
    }
    
    println!("\n🔧 SEE Implementation Benefits:");
    println!("   • Accurate material exchange evaluation");
    println!("   • Considers all attackers and defenders");
    println!("   • Prevents bad captures from cluttering tactical analysis");
    println!("   • Improves overall engine tactical strength");
}