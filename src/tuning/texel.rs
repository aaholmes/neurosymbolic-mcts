//! Texel tuning implementation for optimizing evaluation parameters

use super::*;
use crate::eval::{EvalWeights, PestoEval};
use crate::move_generation::MoveGen;

pub struct TexelTuner {
    positions: Vec<TexelPosition>,
    current_weights: EvalWeights,
    best_weights: EvalWeights,
    learning_rate: f64,
    k_factor: f64, // Scaling factor for sigmoid function
    best_error: f64,
}

impl TexelTuner {
    pub fn new(positions: Vec<TexelPosition>, initial_weights: EvalWeights) -> Self {
        let best_error = f64::INFINITY;

        TexelTuner {
            positions,
            current_weights: initial_weights.clone(),
            best_weights: initial_weights,
            learning_rate: 0.1,
            k_factor: 400.0, // Standard chess evaluation scaling
            best_error,
        }
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    pub fn set_k_factor(&mut self, k: f64) {
        self.k_factor = k;
    }

    /// Calculate evaluation error using mean squared error
    pub fn calculate_error(&self, weights: &EvalWeights, move_gen: &MoveGen) -> f64 {
        let mut total_error = 0.0;
        let evaluator = PestoEval::with_weights(weights.clone());

        for position in &self.positions {
            // Get evaluation in centipawns
            let eval_cp = evaluator.eval(&position.board, move_gen);

            // Convert to win probability using sigmoid: 1 / (1 + exp(-eval/k))
            let eval_prob = sigmoid(eval_cp as f64, self.k_factor);

            // Calculate squared error
            let error = (eval_prob - position.game_result).powi(2);
            total_error += error;
        }

        total_error / self.positions.len() as f64
    }

    /// Perform one iteration of gradient descent
    pub fn optimize_iteration(&mut self, move_gen: &MoveGen) -> f64 {
        let current_error = self.calculate_error(&self.current_weights, move_gen);

        // Try small adjustments to each parameter
        let mut improved_weights = self.current_weights.clone();
        let mut best_iteration_error = current_error;

        // Optimize material values
        self.optimize_material_values(&mut improved_weights, &mut best_iteration_error, move_gen);

        // Optimize piece-square table values
        self.optimize_pst_values(&mut improved_weights, &mut best_iteration_error, move_gen);

        // Optimize positional bonuses
        self.optimize_positional_bonuses(
            &mut improved_weights,
            &mut best_iteration_error,
            move_gen,
        );

        // Update weights if improvement found
        if best_iteration_error < self.best_error {
            self.best_error = best_iteration_error;
            self.best_weights = improved_weights.clone();
            self.current_weights = improved_weights;

            // Adaptive learning rate
            self.learning_rate = (self.learning_rate * 1.05).min(1.0);
        } else {
            // Reduce learning rate if no improvement
            self.learning_rate *= 0.95;
        }

        best_iteration_error
    }

    fn optimize_material_values(
        &self,
        weights: &mut EvalWeights,
        best_error: &mut f64,
        move_gen: &MoveGen,
    ) {
        // Skip optimizing piece values as they remain const in our structure
        // Focus on mobility weights instead which affect material evaluation

        // Optimize mobility weights for each piece type
        let deltas = [1, 1, 1, 1]; // Small adjustments for mobility

        for piece_idx in 0..4 {
            // N, B, R, Q
            let delta = deltas[piece_idx];

            // Try increasing MG mobility weight
            let original_mg = weights.mobility_weights_mg[piece_idx];
            let original_eg = weights.mobility_weights_eg[piece_idx];

            weights.mobility_weights_mg[piece_idx] += delta;

            let error = self.calculate_error(weights, move_gen);
            if error < *best_error {
                *best_error = error;
            } else {
                // Try decreasing instead
                weights.mobility_weights_mg[piece_idx] = original_mg - delta;

                let error = self.calculate_error(weights, move_gen);
                if error < *best_error {
                    *best_error = error;
                } else {
                    // Try EG mobility weight
                    weights.mobility_weights_mg[piece_idx] = original_mg;
                    weights.mobility_weights_eg[piece_idx] += delta;

                    let error = self.calculate_error(weights, move_gen);
                    if error < *best_error {
                        *best_error = error;
                    } else {
                        // Revert
                        weights.mobility_weights_eg[piece_idx] = original_eg;
                    }
                }
            }
        }
    }

    fn optimize_pst_values(
        &self,
        _weights: &mut EvalWeights,
        _best_error: &mut f64,
        _move_gen: &MoveGen,
    ) {
        // Skip PST optimization as piece-square tables remain const
        // This function is kept for interface compatibility
        // Future implementations could optimize king attack zone weighting or similar
    }

    fn optimize_positional_bonuses(
        &self,
        weights: &mut EvalWeights,
        best_error: &mut f64,
        move_gen: &MoveGen,
    ) {
        // Optimize key positional evaluation weights

        // Two bishops bonus [mg, eg]
        let original = weights.two_bishops_bonus;
        for &delta in &[3, -3] {
            weights.two_bishops_bonus[0] = original[0] + delta; // MG
            weights.two_bishops_bonus[1] = original[1] + delta; // EG

            let error = self.calculate_error(weights, move_gen);
            if error < *best_error {
                *best_error = error;
                break;
            }
        }
        if self.calculate_error(weights, move_gen) >= *best_error {
            weights.two_bishops_bonus = original;
        }

        // King safety pawn shield bonus
        let original = weights.king_safety_pawn_shield_bonus;
        for &delta in &[2, -2] {
            weights.king_safety_pawn_shield_bonus[0] = original[0] + delta;
            weights.king_safety_pawn_shield_bonus[1] = original[1] + delta;

            let error = self.calculate_error(weights, move_gen);
            if error < *best_error {
                *best_error = error;
                break;
            }
        }
        if self.calculate_error(weights, move_gen) >= *best_error {
            weights.king_safety_pawn_shield_bonus = original;
        }

        // Rook open file bonus
        let original = weights.rook_open_file_bonus;
        for &delta in &[3, -3] {
            weights.rook_open_file_bonus[0] = original[0] + delta;
            weights.rook_open_file_bonus[1] = original[1] + delta;

            let error = self.calculate_error(weights, move_gen);
            if error < *best_error {
                *best_error = error;
                break;
            }
        }
        if self.calculate_error(weights, move_gen) >= *best_error {
            weights.rook_open_file_bonus = original;
        }

        // Isolated pawn penalty
        let original = weights.isolated_pawn_penalty;
        for &delta in &[2, -2] {
            weights.isolated_pawn_penalty[0] = original[0] + delta;
            weights.isolated_pawn_penalty[1] = original[1] + delta;

            let error = self.calculate_error(weights, move_gen);
            if error < *best_error {
                *best_error = error;
                break;
            }
        }
        if self.calculate_error(weights, move_gen) >= *best_error {
            weights.isolated_pawn_penalty = original;
        }
    }

    /// Run full optimization for specified iterations
    pub fn tune(&mut self, move_gen: &MoveGen, max_iterations: usize) -> EvalWeights {
        println!("🔧 Starting Texel Tuning");
        println!("========================");
        println!("Positions: {}", self.positions.len());
        println!(
            "Initial error: {:.6}",
            self.calculate_error(&self.current_weights, move_gen)
        );

        for iteration in 0..max_iterations {
            let error = self.optimize_iteration(move_gen);

            if iteration % 10 == 0 {
                println!(
                    "Iteration {}: Error = {:.6}, LR = {:.4}",
                    iteration, error, self.learning_rate
                );
            }

            // Early stopping if learning rate becomes too small
            if self.learning_rate < 0.001 {
                println!("Learning rate too small, stopping early");
                break;
            }
        }

        println!("\n🎯 Tuning Complete!");
        println!("Final error: {:.6}", self.best_error);
        println!(
            "Improvement: {:.6}",
            self.calculate_error(&EvalWeights::default(), move_gen) - self.best_error
        );

        self.best_weights.clone()
    }

    pub fn get_best_weights(&self) -> &EvalWeights {
        &self.best_weights
    }
}

/// Sigmoid function for converting evaluation to win probability
fn sigmoid(eval_cp: f64, k: f64) -> f64 {
    1.0 / (1.0 + (-eval_cp / k).exp())
}

/// Create a simple test dataset for tuning
pub fn create_test_dataset() -> Vec<TexelPosition> {
    vec![
        // Opening positions
        TexelPosition::new(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            0.5,
            "King's Pawn Opening".to_string(),
        )
        .unwrap(),
        TexelPosition::new(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
            0.5,
            "Open Game".to_string(),
        )
        .unwrap(),
        // Middlegame positions
        TexelPosition::new(
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
            0.6,
            "Italian Game".to_string(),
        )
        .unwrap(),
        TexelPosition::new(
            "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
            0.4,
            "Queen's Gambit Declined".to_string(),
        )
        .unwrap(),
        // Endgame positions
        TexelPosition::new(
            "8/8/8/4k3/4P3/4K3/8/8 w - - 0 1",
            0.7,
            "King and Pawn vs King".to_string(),
        )
        .unwrap(),
        TexelPosition::new(
            "8/8/8/8/8/4k3/4p3/4K3 b - - 0 1",
            0.3,
            "King and Pawn vs King (Black)".to_string(),
        )
        .unwrap(),
        // Tactical positions
        TexelPosition::new(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
            0.0,
            "Scholar's Mate".to_string(),
        )
        .unwrap(),
        TexelPosition::new(
            "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            1.0,
            "Légal's Mate".to_string(),
        )
        .unwrap(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::move_generation::MoveGen;

    #[test]
    fn test_sigmoid_function() {
        assert!((sigmoid(0.0, 400.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(400.0, 400.0) > 0.7);
        assert!(sigmoid(-400.0, 400.0) < 0.3);
    }

    #[test]
    fn test_texel_tuner_creation() {
        let positions = create_test_dataset();
        let weights = EvalWeights::default();
        let tuner = TexelTuner::new(positions, weights);

        assert!(tuner.positions.len() > 0);
        assert_eq!(tuner.learning_rate, 0.1);
    }

    #[test]
    fn test_error_calculation() {
        let positions = create_test_dataset();
        let weights = EvalWeights::default();
        let tuner = TexelTuner::new(positions, weights);
        let move_gen = MoveGen::new();

        let error = tuner.calculate_error(&tuner.current_weights, &move_gen);
        assert!(error >= 0.0);
        assert!(error <= 1.0); // MSE should be between 0 and 1
    }
}
