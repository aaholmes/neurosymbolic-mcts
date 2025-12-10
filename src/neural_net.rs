//! Stub for Neural Network Policy
//!
//! This module provides a placeholder implementation for the NeuralNetPolicy trait
//! to allow compilation while the actual neural network integration is being developed.

use crate::board::Board;
use crate::move_types::Move;

#[derive(Debug, Clone)]
pub struct NeuralNetPolicy {
    // Placeholder field
    _dummy: u8,
}

impl NeuralNetPolicy {
    pub fn new() -> Self {
        NeuralNetPolicy { _dummy: 0 }
    }

    /// Creates a new instance for demo purposes (stub)
    pub fn new_demo_enabled() -> Self {
        NeuralNetPolicy::new()
    }

    pub fn is_available(&self) -> bool {
        false // Neural network is currently unavailable
    }

    pub fn predict(&mut self, _board: &Board) -> Option<(Vec<f32>, f32)> {
        None // No prediction available
    }

    pub fn policy_to_move_priors(&self, _policy: &[f32], _moves: &[Move]) -> Vec<(Move, f32)> {
        Vec::new() // No priors
    }

    pub fn get_position_value(&mut self, _board: &Board) -> Option<i32> {
        None // No value available
    }

    /// Returns cache statistics (stub)
    pub fn cache_stats(&self) -> (usize, usize) {
        (0, 0)
    }
}