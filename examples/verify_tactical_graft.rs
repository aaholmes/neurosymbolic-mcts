use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::eval::PestoEval;
use kingfisher::mcts::TacticalMctsConfig;
use kingfisher::mcts::tactical_mcts::tactical_mcts_search;
use std::time::Duration;

fn main() {
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    
    println!("🧪 Verifying Tier 2 Tactical Grafting (Queen Sac Test)...");

    // Position: White Queen can take a protected Pawn on e5.
    // r1bqkbnr/pppp1ppp/2n5/4p3/4P3/4Q3/PPPP1PPP/RNB1KBNR w KQkq - 0 1
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/4Q3/PPPP1PPP/RNB1KBNR w KQkq - 0 1";
    let board = Board::new_from_fen(fen);
    
    // We want to see if MCTS avoids Qxe5 (which is 37 to 36 index? No, e3 to e5).
    // e3 is 20, e5 is 36.
    
    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(2),
        ..Default::default()
    };

    let mut nn = None;
    let (best_move, stats, _) = tactical_mcts_search(board, &move_gen, &pesto, &mut nn, config);

    println!("\nSearch Results:");
    println!("  Best Move: {}", best_move.map_or("None".to_string(), |m| m.to_uci()));
    println!("  Nodes Expanded: {}", stats.nodes_expanded);
    
    if let Some(mv) = best_move {
        if mv.to_uci() == "e3e5" {
            println!("❌ FAILURE: Engine chose the Queen Sacrifice!");
        } else {
            println!("✅ SUCCESS: Engine avoided the Queen Sacrifice.");
        }
    }
}
