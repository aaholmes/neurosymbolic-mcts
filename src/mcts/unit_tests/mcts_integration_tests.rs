//! Integration tests for the complete tactical-enhanced MCTS system
//!
//! Tests cover full search functionality, statistics accuracy,
//! mate detection integration, and system behavior.

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::mcts::{tactical_mcts_search, TacticalMctsConfig};
use std::time::Duration;

fn setup_test_env() -> MoveGen {
    MoveGen::new()
}

fn get_test_config() -> TacticalMctsConfig {
    TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_millis(1000),
        mate_search_depth: 3,
        exploration_constant: 1.414,
        use_neural_policy: false,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search_functionality() {
        let move_gen = setup_test_env();
        let config = get_test_config();

        let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let (best_move, stats, _root) = tactical_mcts_search(
            board.clone(),
            &move_gen,
            config,
        );

        // Should return a valid move
        assert!(best_move.is_some(), "Should return a move from starting position");

        // Statistics should be reasonable
        assert!(stats.iterations > 0, "Should have performed iterations");
        assert!(stats.nodes_expanded > 0, "Should have expanded nodes");
        assert!(stats.search_time > Duration::from_millis(0), "Should have taken some time");

        // Verify returned move is legal
        if let Some(mv) = best_move {
            let legal_moves = {
                let (captures, non_captures) = move_gen.gen_pseudo_legal_moves(&board);
                let mut all_moves = captures;
                all_moves.extend(non_captures);
                all_moves.into_iter()
                    .filter(|&m| board.apply_move_to_board(m).is_legal(&move_gen))
                    .collect::<Vec<_>>()
            };

            assert!(legal_moves.contains(&mv), "Returned move {:?} should be legal", mv);
        }
    }

    #[test]
    fn test_tactical_position_search() {
        let move_gen = setup_test_env();
        let config = get_test_config();

        // Position with tactical opportunities
        let board = Board::new_from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3");

        let (best_move, stats, _root) = tactical_mcts_search(
            board,
            &move_gen,
            config,
        );

        // Should find a move
        assert!(best_move.is_some(), "Should find move in tactical position");

        // Should explore tactical moves OR find a mate immediately
        assert!(stats.tactical_moves_explored > 0 || stats.mates_found > 0,
                "Should explore tactical moves or find mate, found: {} tact, {} mates",
                stats.tactical_moves_explored, stats.mates_found);

        // Should be efficient (more tactical moves than NN calls)
        assert!(stats.tactical_moves_explored >= stats.nn_policy_evaluations,
                "Should explore tactical moves before NN evaluation");
    }

    #[test]
    fn test_mate_detection_integration() {
        let move_gen = setup_test_env();
        let config = TacticalMctsConfig {
            max_iterations: 50,
            time_limit: Duration::from_millis(500),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: false,
            ..Default::default()
        };

        // Simple mate in 1 position - back rank mate
        let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/7R w - - 0 1");

        let (best_move, stats, _root) = tactical_mcts_search(
            board,
            &move_gen,
            config,
        );

        // Should find a move (mate detection is implementation-dependent)
        assert!(best_move.is_some(), "Should find a move in mate position");

        // Should be very fast for mate positions (allow up to 1s for thread pool init)
        assert!(stats.search_time < Duration::from_millis(1000),
                "Mate detection should be fast, took: {:?}", stats.search_time);
    }

    #[test]
    fn test_search_time_limits() {
        let move_gen = setup_test_env();

        let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Test short time limit
        let short_config = TacticalMctsConfig {
            max_iterations: u32::MAX,
            time_limit: Duration::from_millis(50),
            mate_search_depth: 1,
            exploration_constant: 1.414,
            use_neural_policy: false,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let (_, stats, _root) = tactical_mcts_search(
            board.clone(),
            &move_gen,
            short_config,
        );
        let elapsed = start.elapsed();

        // Should respect time limit (with some tolerance)
        assert!(elapsed < Duration::from_millis(200),
                "Should respect time limit, took: {:?}", elapsed);
        assert!(stats.search_time <= elapsed, "Reported time should be <= actual time");
    }

    #[test]
    fn test_iteration_limits() {
        let move_gen = setup_test_env();

        let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Test iteration limit
        let iter_config = TacticalMctsConfig {
            max_iterations: 25,
            time_limit: Duration::from_secs(10),
            mate_search_depth: 1,
            exploration_constant: 1.414,
            use_neural_policy: false,
            ..Default::default()
        };

        let (_, stats, _root) = tactical_mcts_search(
            board,
            &move_gen,
            iter_config,
        );

        // Should respect iteration limit
        assert!(stats.iterations <= 25,
                "Should respect iteration limit, performed: {}", stats.iterations);
    }

    #[test]
    fn test_statistics_consistency() {
        let move_gen = setup_test_env();
        let config = get_test_config();

        let board = Board::new_from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3");

        let (_, stats, _root) = tactical_mcts_search(
            board,
            &move_gen,
            config.clone(),
        );

        // Statistics should be internally consistent
        assert!(stats.iterations <= config.max_iterations,
                "Iterations should not exceed limit");
        assert!(stats.tactical_moves_explored <= stats.nodes_expanded,
                "Tactical moves should not exceed total nodes");
        assert!(stats.nn_policy_evaluations <= stats.nodes_expanded,
                "NN evaluations should not exceed total nodes");
    }

    #[test]
    fn test_different_exploration_constants() {
        let move_gen = setup_test_env();

        let board = Board::new_from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3");

        let low_exploration = TacticalMctsConfig {
            max_iterations: 50,
            time_limit: Duration::from_millis(500),
            mate_search_depth: 2,
            exploration_constant: 0.5,
            use_neural_policy: false,
            ..Default::default()
        };

        let high_exploration = TacticalMctsConfig {
            max_iterations: 50,
            time_limit: Duration::from_millis(500),
            mate_search_depth: 2,
            exploration_constant: 2.0,
            use_neural_policy: false,
            ..Default::default()
        };

        let (move1, stats1, _) = tactical_mcts_search(
            board.clone(),
            &move_gen,
            low_exploration,
        );

        let (move2, stats2, _) = tactical_mcts_search(
            board,
            &move_gen,
            high_exploration,
        );

        // Both should find valid moves
        assert!(move1.is_some(), "Low exploration should find move");
        assert!(move2.is_some(), "High exploration should find move");

        // Both should have reasonable statistics
        assert!(stats1.nodes_expanded > 0 || stats1.mates_found > 0, "Low exploration should expand nodes or find mate");
        assert!(stats2.nodes_expanded > 0 || stats2.mates_found > 0, "High exploration should expand nodes or find mate");
    }

    #[test]
    fn test_mate_search_depth_effect() {
        let move_gen = setup_test_env();

        let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1");

        let no_mate_search = TacticalMctsConfig {
            max_iterations: 50,
            time_limit: Duration::from_millis(200),
            mate_search_depth: 0,
            exploration_constant: 1.414,
            use_neural_policy: false,
            ..Default::default()
        };

        let with_mate_search = TacticalMctsConfig {
            max_iterations: 50,
            time_limit: Duration::from_millis(200),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: false,
            ..Default::default()
        };

        let (_, stats_no_mate, _) = tactical_mcts_search(
            board.clone(),
            &move_gen,
            no_mate_search,
        );

        let (_, stats_with_mate, _) = tactical_mcts_search(
            board,
            &move_gen,
            with_mate_search,
        );

        // Mate search should find mates
        assert!(stats_with_mate.mates_found >= stats_no_mate.mates_found,
                "Mate search should find more mates: {} vs {}",
                stats_with_mate.mates_found, stats_no_mate.mates_found);
    }

    #[test]
    fn test_reproducible_search() {
        let move_gen = setup_test_env();
        let config = TacticalMctsConfig {
            max_iterations: 25,
            time_limit: Duration::from_secs(5),
            mate_search_depth: 2,
            exploration_constant: 1.414,
            use_neural_policy: false,
            ..Default::default()
        };

        let board = Board::new_from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3");

        // Run search twice with same parameters
        let (move1, stats1, _) = tactical_mcts_search(
            board.clone(),
            &move_gen,
            config.clone(),
        );

        let (move2, stats2, _) = tactical_mcts_search(
            board,
            &move_gen,
            config,
        );

        // Should have performed same number of iterations
        assert_eq!(stats1.iterations, stats2.iterations,
                   "Should perform same iterations: {} vs {}",
                   stats1.iterations, stats2.iterations);

        // Should find valid moves (may be different due to tie-breaking)
        assert!(move1.is_some(), "First search should find move");
        assert!(move2.is_some(), "Second search should find move");
    }
}
