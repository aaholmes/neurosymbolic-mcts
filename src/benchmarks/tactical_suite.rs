//! Tactical test suite for demonstrating mate-search-first superiority

use super::*;
use crate::move_types::Move;

#[derive(Debug, Clone)]
pub struct TacticalPosition {
    pub name: String,
    pub fen: String,
    pub mate_in: u8,
    pub best_move_uci: String,
    pub description: String,
}

impl TacticalPosition {
    pub fn get_best_move(&self) -> Option<Move> {
        Move::from_uci(&self.best_move_uci)
    }
}

/// Famous tactical positions for benchmarking
pub fn get_tactical_test_suite() -> Vec<TacticalPosition> {
    vec![
        // Mate in 1 positions
        TacticalPosition {
            name: "Back Rank Mate 1".to_string(),
            fen: "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1".to_string(),
            mate_in: 1,
            best_move_uci: "e1e8".to_string(),
            description: "Simple back rank mate".to_string(),
        },
        TacticalPosition {
            name: "Queen Mate 1".to_string(),
            fen: "k7/8/1K6/8/8/8/8/Q7 w - - 0 1".to_string(),
            mate_in: 1,
            best_move_uci: "a1a8".to_string(),
            description: "Queen delivers mate".to_string(),
        },
        TacticalPosition {
            name: "Smothered Mate Setup".to_string(),
            fen: "6k1/5ppp/8/8/8/8/8/6QK w - - 0 1".to_string(),
            mate_in: 1,
            best_move_uci: "g1g8".to_string(),
            description: "Queen mate on back rank".to_string(),
        },
        
        // Mate in 2 positions
        TacticalPosition {
            name: "Légal's Mate Pattern".to_string(),
            fen: "r1bqk2r/pppp1ppp/2n2n2/2b5/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1".to_string(),
            mate_in: 2,
            best_move_uci: "f3e5".to_string(),
            description: "Classic sacrifice leading to mate".to_string(),
        },
        TacticalPosition {
            name: "Scholar's Mate Defense Failure".to_string(),
            fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 1".to_string(),
            mate_in: 2,
            best_move_uci: "f6d5".to_string(),
            description: "Counter-attack leading to mate".to_string(),
        },
        
        // Mate in 3 positions  
        TacticalPosition {
            name: "Anastasia's Mate".to_string(),
            fen: "2kr4/ppp5/8/8/8/8/8/4RR1K w - - 0 1".to_string(),
            mate_in: 3,
            best_move_uci: "e1e8".to_string(),
            description: "Classic rook and knight mate pattern".to_string(),
        },
        TacticalPosition {
            name: "Boden's Mate Pattern".to_string(),
            fen: "2kr4/ppp5/8/8/8/8/8/2KR4 w - - 0 1".to_string(),
            mate_in: 3,
            best_move_uci: "d1d8".to_string(),
            description: "Two bishops deliver mate".to_string(),
        },
        
        // Complex tactical positions
        TacticalPosition {
            name: "WAC.001".to_string(),
            fen: "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3QP/PPB4P/R4RK1 w - - 0 1".to_string(),
            mate_in: 3,
            best_move_uci: "g3g6".to_string(),
            description: "Win At Chess position #1".to_string(),
        },
        TacticalPosition {
            name: "WAC.002".to_string(),
            fen: "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1".to_string(),
            mate_in: 2,
            best_move_uci: "b3b2".to_string(),
            description: "Win At Chess position #2".to_string(),
        },
        TacticalPosition {
            name: "Deep Tactical Shot".to_string(),
            fen: "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2QK2R w KQkq - 0 1".to_string(),
            mate_in: 4,
            best_move_uci: "d5f6".to_string(),
            description: "Complex sacrificial attack".to_string(),
        },
    ]
}

/// Benchmark a single position with the given agent
pub fn benchmark_position<T: Agent>(
    position: &TacticalPosition,
    agent: &mut T,
    _time_limit: Duration, // TODO: Pass time limit to agent when Agent trait supports it
) -> BenchmarkResult {
    let start_time = Instant::now();
    let mut board_stack = BoardStack::new_from_fen(&position.fen);
    
    // Get the move from the agent
    let found_move = agent.get_move(&mut board_stack);
    let time_taken = start_time.elapsed();
    
    // Check if the move matches the expected best move
    let expected_move = position.get_best_move();
    let found_mate = if let Some(expected) = expected_move {
        found_move == expected
    } else {
        false
    };
    
    BenchmarkResult {
        position_name: position.name.clone(),
        engine_name: "Unknown".to_string(), // Will be set by caller
        time_taken,
        nodes_searched: 0, // TODO: Add node counting to agents
        best_move: Some(found_move),
        found_mate,
        mate_depth: if found_mate { Some(position.mate_in) } else { None },
    }
}

/// Run the complete tactical benchmark suite
pub fn run_tactical_benchmark<T: Agent>(
    agent: &mut T, 
    engine_name: &str, 
    time_limit_per_position: Duration
) -> Vec<BenchmarkResult> {
    let _positions = get_tactical_test_suite();
    let mut results = Vec::new();
    
    println!("\n🎯 Running Tactical Benchmark for {}", engine_name);
    println!("Testing {} positions with {}ms time limit per position\n", 
             _positions.len(), time_limit_per_position.as_millis());
    
    for (i, position) in _positions.iter().enumerate() {
        print!("Position {}/{}: {} ... ", i + 1, _positions.len(), position.name);
        
        let mut result = benchmark_position(position, agent, time_limit_per_position);        result.engine_name = engine_name.to_string();
        
        if result.found_mate {
            println!("✅ SOLVED in {:.1}ms", result.time_taken.as_millis());
        } else {
            println!("❌ FAILED in {:.1}ms (found: {:?})", 
                    result.time_taken.as_millis(), result.best_move);
        }
        
        results.push(result);
    }
    
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    
    #[test]
    fn test_tactical_positions_load() {
        let positions = get_tactical_test_suite();
        assert!(!positions.is_empty());
        
        // Verify all positions have valid FENs
        for position in &positions {
            let board = Board::new_from_fen(&position.fen);
            assert!(board.zobrist_hash != 0, "Invalid FEN: {}", position.fen);
        }
    }
    
    #[test]
    fn test_best_move_parsing() {
        let positions = get_tactical_test_suite();
        for position in &positions {
            let best_move = position.get_best_move();
            assert!(best_move.is_some(), 
                   "Could not parse best move for position: {}", position.name);
        }
    }
}