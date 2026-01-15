//! Policy network interface for MCTS.
//!
//! This module defines the trait interface for policy networks used in MCTS.
//! The actual implementation is in `neural_net.rs` using LibTorch.

use crate::board::Board;
use crate::move_types::Move;
use std::collections::HashMap;

/// Trait for policy networks used in MCTS.
///
/// A policy network evaluates a position and returns:
/// 1. A prior probability distribution over legal moves
/// 2. A value estimate for the current position
pub trait PolicyNetwork {
    /// Evaluates a position given the legal moves and returns (policy, value).
    ///
    /// # Arguments
    /// * `board` - The chess position to evaluate.
    /// * `legal_moves` - A slice containing the legal moves for the current position.
    ///
    /// # Returns
    /// * `HashMap<Move, f64>` - Prior probabilities for each legal move provided.
    /// * `f64` - Value estimate for the position from the perspective of the player to move.
    ///           Should be in range [-1.0, 1.0] where 1.0 means certain win for current player.
    fn evaluate(&self, board: &Board, legal_moves: &[Move]) -> (HashMap<Move, f64>, f64);
}
