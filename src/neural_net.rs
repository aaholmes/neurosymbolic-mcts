//! Neural Network Policy Implementation
//!
//! This module handles loading models and performing inference to guide the MCTS search.
//!
//! It provides two implementations:
//! - Real implementation using `tch-rs` (enabled via "neural" feature)
//! - Stub implementation (default) to allow compilation without LibTorch

// ==========================================
// 1. Real Implementation (LibTorch)
// ==========================================
#[cfg(feature = "neural")]
mod real {
    use crate::board::Board;
    use crate::move_types::Move;
    use crate::piece_types::{PieceType, Color};
    use tch::{CModule, Tensor, Device, Kind};
    use std::path::Path;

    pub struct NeuralNetPolicy {
        model: Option<CModule>,
        device: Device,
    }

    impl NeuralNetPolicy {
        /// Creates a new, empty policy (no model loaded)
        pub fn new() -> Self {
            NeuralNetPolicy {
                model: None,
                // Auto-detect CUDA, fallback to CPU
                device: if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu },
            }
        }

        /// Loads a model from the specified path
        pub fn load(&mut self, path: &str) -> Result<(), String> {
            if !Path::new(path).exists() {
                return Err(format!("Model file not found: {}", path));
            }

            match CModule::load_on_device(path, self.device) {
                Ok(m) => {
                    self.model = Some(m);
                    println!("✅ Neural network loaded successfully on {:?}", self.device);
                    Ok(())
                }
                Err(e) => Err(format!("Failed to load model: {}", e)),
            }
        }

        /// Helper for demo/testing
        pub fn new_demo_enabled() -> Self {
            let mut nn = Self::new();
            let _ = nn.load("models/model.pt");
            nn
        }

        pub fn is_available(&self) -> bool {
            self.model.is_some()
        }

        /// Converts the board state to a 12x8x8 input tensor
        pub fn board_to_tensor(&self, board: &Board) -> Tensor {
            let mut planes = Vec::with_capacity(12);
            
            for &color in &[Color::White, Color::Black] {
                for &pt in &[
                    PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
                    PieceType::Rook, PieceType::Queen, PieceType::King
                ] {
                    let mut plane = vec![0.0f32; 64];
                    let bitboard = board.get_piece_bitboard(pt, color);
                    
                    for i in 0..64 {
                        if (bitboard >> i) & 1 == 1 {
                            let rank = i / 8;
                            let file = i % 8;
                            let tensor_row = 7 - rank; 
                            let tensor_col = file;
                            plane[tensor_row as usize * 8 + tensor_col as usize] = 1.0;
                        }
                    }
                    planes.extend(plane);
                }
            }

            Tensor::from_slice(&planes)
                .view([12, 8, 8])
                .to_device(self.device)
                .to_kind(Kind::Float)
        }

        /// Runs inference on the board
        pub fn predict(&mut self, board: &Board) -> Option<(Vec<f32>, f32)> {
            let model = self.model.as_ref()?;
            let input = self.board_to_tensor(board).unsqueeze(0);
            
            // Calculate material scalar [B, 1]
            let mat_imb = board.material_imbalance() as f32;
            let mat_tensor = Tensor::from_slice(&[mat_imb])
                .to_device(self.device)
                .to_kind(Kind::Float)
                .unsqueeze(0);

            // Forward pass with TWO inputs: [board_tensor, material_scalar]
            let ivalue = model.method_is("forward", &[tch::IValue::Tensor(input), tch::IValue::Tensor(mat_tensor)]).ok()?;
            
            if let tch::IValue::Tuple(elements) = ivalue {
                if elements.len() != 2 { return None; }
                
                let policy_tensor = match &elements[0] {
                    tch::IValue::Tensor(t) => t,
                    _ => return None,
                };
                
                let value_tensor = match &elements[1] {
                    tch::IValue::Tensor(t) => t,
                    _ => return None,
                };

                let policy_probs = policy_tensor.exp().view([-1]).to_vec::<f32>().ok()?;
                let value = value_tensor.double_value(&[0, 0]) as f32;

                Some((policy_probs, value))
            } else {
                None
            }
        }

        /// Maps the raw policy vector to legal moves
        pub fn policy_to_move_priors(&self, policy: &[f32], moves: &[Move]) -> Vec<(Move, f32)> {
            let mut result = Vec::with_capacity(moves.len());
            let mut total_prob = 0.0;

            for &mv in moves {
                let idx = (mv.from as usize * 64) + (mv.to as usize);
                
                if idx < policy.len() {
                    let prob = policy[idx];
                    result.push((mv, prob));
                    total_prob += prob;
                } else {
                    result.push((mv, 0.0));
                }
            }

            if total_prob > 0.0 {
                for (_, prob) in result.iter_mut() {
                    *prob /= total_prob;
                }
            } else {
                let uniform = 1.0 / moves.len() as f32;
                for (_, prob) in result.iter_mut() {
                    *prob = uniform;
                }
            }

            result
        }

        pub fn get_position_value(&mut self, board: &Board) -> Option<i32> {
            let (_, value) = self.predict(board)?;
            Some((value * 1000.0) as i32)
        }

        pub fn cache_stats(&self) -> (usize, usize) {
            (0, 0)
        }
    }
}

// ==========================================
// 2. Stub Implementation (No LibTorch)
// ==========================================
#[cfg(not(feature = "neural"))]
mod stub {
    use crate::board::Board;
    use crate::move_types::Move;

    #[derive(Debug, Clone)]
    pub struct NeuralNetPolicy {
        _dummy: u8,
    }

    impl NeuralNetPolicy {
        pub fn new() -> Self {
            NeuralNetPolicy { _dummy: 0 }
        }

        pub fn load(&mut self, _path: &str) -> Result<(), String> {
            Err("Neural network feature not enabled (compile with --features neural)".to_string())
        }

        pub fn new_demo_enabled() -> Self {
            NeuralNetPolicy::new()
        }

        pub fn is_available(&self) -> bool {
            false
        }
        
        // Stub for board_to_tensor to fix compilation of neural_test even in stub mode
        // Note: Returns a dummy unit type, callers must cfg check before using result
        pub fn board_to_tensor(&self, _board: &Board) -> () {
            ()
        }

        pub fn predict(&mut self, _board: &Board) -> Option<(Vec<f32>, f32)> {
            None
        }

        pub fn policy_to_move_priors(&self, _policy: &[f32], _moves: &[Move]) -> Vec<(Move, f32)> {
            Vec::new()
        }

        pub fn get_position_value(&mut self, _board: &Board) -> Option<i32> {
            None
        }
        
        pub fn cache_stats(&self) -> (usize, usize) {
            (0, 0)
        }
    }
}

// Re-export the active implementation
#[cfg(feature = "neural")]
pub use self::real::NeuralNetPolicy;

#[cfg(not(feature = "neural"))]
pub use self::stub::NeuralNetPolicy;