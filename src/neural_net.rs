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
    use crate::piece_types::{PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK};
    use crate::tensor::move_to_index;
    use tch::{CModule, Tensor, Device, Kind};
    use std::path::Path;

    pub struct NeuralNetPolicy {
        model: Option<CModule>,
        device: Device,
    }

    impl NeuralNetPolicy {
        pub fn new() -> Self {
            NeuralNetPolicy {
                model: None,
                device: if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu },
            }
        }

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

        pub fn new_demo_enabled() -> Self {
            let mut nn = Self::new();
            let _ = nn.load("models/model.pt");
            nn
        }

        pub fn is_available(&self) -> bool {
            self.model.is_some()
        }

        pub fn board_to_tensor(&self, board: &Board) -> Tensor {
            let mut planes = vec![0.0f32; 17 * 8 * 8];
            
            // 0-11: Pieces (P1 then P2)
            for color in 0..2 {
                for pt in 0..6 {
                    let offset = (color * 6 + pt) * 64;
                    let bb = board.get_piece_bitboard(color, pt);
                    for i in 0..64 {
                        if (bb >> i) & 1 == 1 {
                            let rank = i / 8;
                            let file = i % 8;
                            planes[offset + (7 - rank) * 8 + file] = 1.0;
                        }
                    }
                }
            }

            // 12: En Passant (Target square only)
            if let Some(sq) = board.en_passant {
                let offset = 12 * 64;
                let rank = sq / 8;
                let file = sq % 8;
                planes[offset + (usize::from(7 - rank)) * 8 + usize::from(file)] = 1.0;
            }

            // 13-16: Castling (Full planes of 1s)
            let rights = [
                board.castling_rights.white_kingside,
                board.castling_rights.white_queenside,
                board.castling_rights.black_kingside,
                board.castling_rights.black_queenside,
            ];
            for (i, &allowed) in rights.iter().enumerate() {
                if allowed {
                    let offset = (13 + i) * 64;
                    for j in 0..64 {
                        planes[offset + j] = 1.0;
                    }
                }
            }

            Tensor::from_slice(&planes).view([17, 8, 8]).to_device(self.device).to_kind(Kind::Float)
        }

        /// Runs inference on the board. Returns (policy_probs, value, k).
        pub fn predict(&mut self, board: &Board) -> Option<(Vec<f32>, f32, f32)> {
            let model = self.model.as_ref()?;
            let input = self.board_to_tensor(board).unsqueeze(0);
            
            let mat_imb = board.material_imbalance() as f32;
            let mat_tensor = Tensor::from_slice(&[mat_imb]).to_device(self.device).to_kind(Kind::Float).unsqueeze(0);

            let ivalue = model.method_is("forward", &[tch::IValue::Tensor(input), tch::IValue::Tensor(mat_tensor)]).ok()?;
            
            if let tch::IValue::Tuple(elements) = ivalue {
                if elements.len() != 3 { return None; }
                
                let policy_tensor = match &elements[0] {
                    tch::IValue::Tensor(t) => t,
                    _ => return None,
                };
                let value_tensor = match &elements[1] {
                    tch::IValue::Tensor(t) => t,
                    _ => return None,
                };
                let k_tensor = match &elements[2] {
                    tch::IValue::Tensor(t) => t,
                    _ => return None,
                };

                // Use copy_data to convert Tensor to Vec<f32>
                let mut policy_probs = vec![0.0f32; 4672];
                policy_tensor.exp().view([-1]).to_device(Device::Cpu).copy_data(&mut policy_probs, 4672);
                
                let value = value_tensor.double_value(&[0, 0]) as f32;
                let k_val = k_tensor.double_value(&[0, 0]) as f32;

                Some((policy_probs, value, k_val))
            } else {
                None
            }
        }

        /// Runs inference on a batch of boards. Returns Vec<(policy_probs, value, k)>.
        pub fn predict_batch(&mut self, boards: &[Board]) -> Vec<Option<(Vec<f32>, f32, f32)>> {
            let model = match self.model.as_ref() {
                Some(m) => m,
                None => return vec![None; boards.len()],
            };

            let mut input_tensors = Vec::with_capacity(boards.len());
            let mut mat_scalars = Vec::with_capacity(boards.len());

            for board in boards {
                input_tensors.push(self.board_to_tensor(board));
                mat_scalars.push(board.material_imbalance() as f32);
            }

            // Stack inputs into [B, 17, 8, 8] and [B, 1]
            let input_batch = Tensor::stack(&input_tensors, 0);
            let mat_batch = Tensor::from_slice(&mat_scalars)
                .to_device(self.device)
                .to_kind(Kind::Float)
                .unsqueeze(1);

            let ivalue = model.method_is("forward", &[tch::IValue::Tensor(input_batch), tch::IValue::Tensor(mat_batch)]);
            
            match ivalue {
                Ok(tch::IValue::Tuple(elements)) if elements.len() == 3 => {
                    let policy_batch = match &elements[0] {
                        tch::IValue::Tensor(t) => t,
                        _ => return vec![None; boards.len()],
                    };
                    let value_batch = match &elements[1] {
                        tch::IValue::Tensor(t) => t,
                        _ => return vec![None; boards.len()],
                    };
                    let k_batch = match &elements[2] {
                        tch::IValue::Tensor(t) => t,
                        _ => return vec![None; boards.len()],
                    };

                    let batch_size = boards.len();
                    let mut results = Vec::with_capacity(batch_size);

                    // Extract data from batch tensors
                    // policy_batch: [B, 4672], value_batch: [B, 1], k_batch: [B, 1]
                    
                    for i in 0..batch_size {
                        let mut policy_probs = vec![0.0f32; 4672];
                        policy_batch.get(i as i64).exp().to_device(Device::Cpu).copy_data(&mut policy_probs, 4672);
                        
                        let value = value_batch.double_value(&[i as i64, 0]) as f32;
                        let k_val = k_batch.double_value(&[i as i64, 0]) as f32;
                        
                        results.push(Some((policy_probs, value, k_val)));
                    }
                    results
                }
                _ => vec![None; boards.len()],
            }
        }

        pub fn policy_to_move_priors(&self, policy: &[f32], moves: &[Move]) -> Vec<(Move, f32)> {
            let mut result = Vec::with_capacity(moves.len());
            let mut total_prob = 0.0;
            for &mv in moves {
                let idx = move_to_index(mv);
                if idx < policy.len() {
                    let prob = policy[idx];
                    result.push((mv, prob));
                    total_prob += prob;
                } else {
                    result.push((mv, 0.0));
                }
            }
            if total_prob > 0.0 {
                for (_, prob) in result.iter_mut() { *prob /= total_prob; }
            } else {
                let uniform = 1.0 / moves.len() as f32;
                for (_, prob) in result.iter_mut() { *prob = uniform; }
            }
            result
        }

        pub fn get_position_value(&mut self, board: &Board) -> Option<i32> {
            let (_, value, _) = self.predict(board)?;
            Some((value * 1000.0) as i32)
        }

        pub fn cache_stats(&self) -> (usize, usize) { (0, 0) }
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
    pub struct NeuralNetPolicy { _dummy: u8 }

    impl NeuralNetPolicy {
        pub fn new() -> Self { NeuralNetPolicy { _dummy: 0 } }
        pub fn load(&mut self, _path: &str) -> Result<(), String> {
            Err("Neural network feature not enabled (compile with --features neural)".to_string())
        }
        pub fn new_demo_enabled() -> Self { NeuralNetPolicy::new() }
        pub fn is_available(&self) -> bool { false }
        pub fn board_to_tensor(&self, _board: &Board) -> () { () }
        pub fn predict(&mut self, _board: &Board) -> Option<(Vec<f32>, f32, f32)> { None }
        pub fn predict_batch(&mut self, boards: &[Board]) -> Vec<Option<(Vec<f32>, f32, f32)>> { vec![None; boards.len()] }
        pub fn policy_to_move_priors(&self, _policy: &[f32], _moves: &[Move]) -> Vec<(Move, f32)> { Vec::new() }
        pub fn get_position_value(&mut self, _board: &Board) -> Option<i32> { None }
        pub fn cache_stats(&self) -> (usize, usize) { (0, 0) }
    }
}

#[cfg(feature = "neural")]
pub use self::real::NeuralNetPolicy;

#[cfg(not(feature = "neural"))]
pub use self::stub::NeuralNetPolicy;
