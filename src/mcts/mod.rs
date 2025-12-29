//! Monte Carlo Tree Search (MCTS) Implementation
//!
//! This module provides the infrastructure for MCTS search guided by both traditional
//! evaluation functions and neural network policies.

pub mod node;
pub mod simulation;
pub mod selection;
pub mod policy;
pub mod tactical;
pub mod neural_mcts;
pub mod tactical_mcts;
pub mod nn_counter;
pub mod inference_server;
pub mod search_logger;

use crate::board::Board;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use std::time::Duration;

// Re-export common components
pub use self::node::{MctsNode, MoveCategory, NodeOrigin, select_leaf_for_expansion};
pub use self::tactical_mcts::{
    tactical_mcts_search, tactical_mcts_search_for_training, tactical_mcts_search_with_tt,
    TacticalMctsConfig, TacticalMctsStats,
};
pub use self::neural_mcts::neural_mcts_search;
pub use self::inference_server::InferenceServer;
pub use self::search_logger::{SearchLogger, Verbosity, GateReason, SelectionReason};

/// Exploration constant for UCB (sqrt(2))
pub const EXPLORATION_CONSTANT: f64 = 1.41421356237;

/// Wrapper search function that maintains backward compatibility but uses the new tactical search
pub fn mcts_pesto_search(
    root_state: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    mate_search_depth: i32,
    iterations: Option<u32>,
    time_limit: Option<Duration>,
) -> Option<Move> {
    let config = TacticalMctsConfig {
        max_iterations: iterations.unwrap_or(1000),
        time_limit: time_limit.unwrap_or(Duration::from_secs(5)),
        mate_search_depth,
        exploration_constant: EXPLORATION_CONSTANT,
        use_neural_policy: false,
        inference_server: None,
        logger: None,
    };

    let mut nn = None;
    let (best_move, _stats, _root) = tactical_mcts_search(
        root_state,
        move_gen,
        pesto_eval,
        &mut nn,
        config
    );

    best_move
}
