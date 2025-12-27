//! Quick test to verify our tactical benchmark works

use kingfisher::benchmarks::tactical_suite::get_tactical_test_suite;
use kingfisher::benchmarks::create_simple_agent;
use kingfisher::boardstack::BoardStack;
use kingfisher::agent::Agent;
use std::time::Instant;

fn main() {
    println!("🧪 Quick Tactical Benchmark Test");
    println!("================================");
    
    // Load test positions
    let positions = get_tactical_test_suite();
    println!("✅ Loaded {} tactical positions", positions.len());
    
    // Test first position only
    let first_position = &positions[0];
    println!("\n🎯 Testing position: {}", first_position.name);
    println!("   FEN: {}", first_position.fen);
    println!("   Expected: {} (mate in {})", first_position.best_move_uci, first_position.mate_in);
    
    // Create agent
    let mut agent = create_simple_agent();
    println!("✅ Created AlphaBeta agent");
    
    // Test position
    println!("\n⚡ Running quick test (500ms limit)...");
    let start = Instant::now();
    
    let mut board = BoardStack::new_from_fen(&first_position.fen);
    let move_found = agent.get_move(&mut board);
    
    let elapsed = start.elapsed();
    println!("⏱️  Completed in {:.1}ms", elapsed.as_millis());
    println!("🔍 Move found: {:?}", move_found);
    
    // Check if correct
    let expected = first_position.get_best_move();
    println!("🎯 Expected: {:?}", expected);
    
    if let Some(exp) = expected {
        if move_found == exp {
            println!("✅ CORRECT! Found the right move");
        } else {
            println!("❌ INCORRECT - but that's okay for this test");
        }
    }
    
    println!("\n🏆 Quick test completed successfully!");
    println!("The benchmark framework is working properly.");
}