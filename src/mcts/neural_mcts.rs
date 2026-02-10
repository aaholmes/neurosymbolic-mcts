//! Neural Network enhanced MCTS implementation
//!
//! This module provides MCTS search with neural network policy and value guidance,
//! combining the mate-search-first innovation with modern neural network evaluation.

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use crate::mcts::inference_server::InferenceServer;
use std::sync::Arc;
use std::time::Duration;

/// Enhanced MCTS search with neural network policy guidance
pub fn neural_mcts_search(
    root_state: Board,
    move_gen: &MoveGen,
    inference_server: Option<Arc<InferenceServer>>,
    mate_search_depth: i32,
    iterations: Option<u32>,
    time_limit: Option<Duration>,
) -> Option<Move> {
    let has_nn = inference_server.is_some();
    let config = TacticalMctsConfig {
        max_iterations: iterations.unwrap_or(1000),
        time_limit: time_limit.unwrap_or(Duration::from_secs(5)),
        mate_search_depth,
        exploration_constant: 1.414,
        use_neural_policy: has_nn,
        inference_server,
        logger: None,
        enable_tier3_neural: has_nn,
        ..Default::default()
    };

    let (best_move, _stats, _root) = tactical_mcts_search(
        root_state,
        move_gen,
        config
    );

    best_move
}
