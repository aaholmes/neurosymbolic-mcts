//! Neural Network Call Counter for Measuring Efficiency
//!
//! This module provides utilities to track and measure neural network
//! evaluation calls to validate the efficiency claims of lazy policy evaluation.

use crate::board::Board;
use std::cell::RefCell;
use std::rc::Rc;

/// A wrapper that counts neural network evaluation calls for efficiency measurement
pub struct CountingNeuralNetPolicy {
    call_count: Rc<RefCell<u32>>,
    total_positions: Rc<RefCell<u32>>,
}

impl CountingNeuralNetPolicy {
    pub fn new() -> Self {
        CountingNeuralNetPolicy {
            call_count: Rc::new(RefCell::new(0)),
            total_positions: Rc::new(RefCell::new(0)),
        }
    }
    
    pub fn get_call_count(&self) -> u32 {
        *self.call_count.borrow()
    }
    
    pub fn get_total_positions(&self) -> u32 {
        *self.total_positions.borrow()
    }
    
    pub fn get_efficiency_ratio(&self) -> f64 {
        let calls = *self.call_count.borrow() as f64;
        let positions = *self.total_positions.borrow() as f64;
        if positions > 0.0 {
            calls / positions
        } else {
            0.0
        }
    }
    
    pub fn reset_counters(&self) {
        *self.call_count.borrow_mut() = 0;
        *self.total_positions.borrow_mut() = 0;
    }
    
    /// Simulate a neural network policy evaluation (for testing)
    pub fn mock_evaluate(&self, _board: &Board) -> Result<(Vec<f32>, f32), String> {
        // Increment counters
        *self.call_count.borrow_mut() += 1;
        *self.total_positions.borrow_mut() += 1;
        
        // Return mock policy and value
        // In a real implementation, this would call the actual NN
        let policy = vec![0.05; 64]; // Uniform policy over 64 squares (simplified)
        let value = 0.5; // Neutral evaluation
        
        Ok((policy, value))
    }
    
    /// Track position evaluation without NN call (for comparison)
    pub fn track_position_without_nn(&self) {
        *self.total_positions.borrow_mut() += 1;
    }
}

/// Utility for comparing NN call efficiency between different MCTS implementations
pub struct EfficiencyComparison {
    pub tactical_mcts_calls: u32,
    pub tactical_mcts_positions: u32,
    pub classical_mcts_calls: u32,
    pub classical_mcts_positions: u32,
}

impl EfficiencyComparison {
    pub fn new() -> Self {
        EfficiencyComparison {
            tactical_mcts_calls: 0,
            tactical_mcts_positions: 0,
            classical_mcts_calls: 0,
            classical_mcts_positions: 0,
        }
    }
    
    pub fn tactical_efficiency(&self) -> f64 {
        if self.tactical_mcts_positions > 0 {
            self.tactical_mcts_calls as f64 / self.tactical_mcts_positions as f64
        } else {
            0.0
        }
    }
    
    pub fn classical_efficiency(&self) -> f64 {
        if self.classical_mcts_positions > 0 {
            self.classical_mcts_calls as f64 / self.classical_mcts_positions as f64
        } else {
            0.0
        }
    }
    
    pub fn improvement_percentage(&self) -> f64 {
        let classical_eff = self.classical_efficiency();
        let tactical_eff = self.tactical_efficiency();
        
        if classical_eff > 0.0 {
            ((classical_eff - tactical_eff) / classical_eff) * 100.0
        } else {
            0.0
        }
    }
    
    pub fn print_comparison(&self) {
        println!("🔬 Neural Network Call Efficiency Comparison");
        println!("{}", "=".repeat(50));
        println!("Tactical-Enhanced MCTS:");
        println!("  NN calls: {} / {} positions", self.tactical_mcts_calls, self.tactical_mcts_positions);
        println!("  Efficiency: {:.3} calls per position", self.tactical_efficiency());
        
        println!("\nClassical MCTS:");
        println!("  NN calls: {} / {} positions", self.classical_mcts_calls, self.classical_mcts_positions);
        println!("  Efficiency: {:.3} calls per position", self.classical_efficiency());
        
        println!("\nImprovement: {:.1}% reduction in NN calls", self.improvement_percentage());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    
    #[test]
    fn test_counting_neural_net_policy() {
        let counter = CountingNeuralNetPolicy::new();
        let board = Board::new();
        
        // Simulate some evaluations
        counter.mock_evaluate(&board).unwrap();
        counter.mock_evaluate(&board).unwrap();
        counter.track_position_without_nn();
        
        assert_eq!(counter.get_call_count(), 2);
        assert_eq!(counter.get_total_positions(), 3);
        assert!((counter.get_efficiency_ratio() - 0.666).abs() < 0.01);
    }
    
    #[test]
    fn test_efficiency_comparison() {
        let mut comparison = EfficiencyComparison::new();
        
        // Simulate tactical MCTS (lazy evaluation)
        comparison.tactical_mcts_calls = 50;
        comparison.tactical_mcts_positions = 100;
        
        // Simulate classical MCTS (eager evaluation)
        comparison.classical_mcts_calls = 90;
        comparison.classical_mcts_positions = 100;
        
        assert_eq!(comparison.tactical_efficiency(), 0.5);
        assert_eq!(comparison.classical_efficiency(), 0.9);
        assert!((comparison.improvement_percentage() - 44.44).abs() < 0.1);
    }
}