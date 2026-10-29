//! Neural Network Policy Implementation
//!
//! This module handles loading models and performing inference to guide the MCTS search.
//!
//! Implementations:
//! - candle (feature "neural"; "cuda" or "cudnn" for the GPU)
//! - stub (default), which never has a model

// ==========================================
// 1. candle implementation (pure Rust; CPU, or CUDA with the "cuda" feature)
// ==========================================
#[cfg(feature = "neural")]
mod candle_impl {
    use crate::board::Board;
    use crate::mcts::tactical_mcts::TimingAccumulator;
    use crate::move_types::Move;
    use crate::nn::oracle_net::OracleNet;
    use crate::nn::weights::load_state_dict;
    use crate::nn::PolicyValueNet;
    use crate::tensor::{board_to_planes, policy_to_move_priors};
    use candle_core::{Device, Tensor};
    use std::path::Path;
    use std::time::Instant;

    const PLANES: usize = 17 * 8 * 8;

    type Prediction = Option<(Vec<f32>, f32, f32)>;

    pub struct NeuralNetPolicy {
        model: Option<OracleNet>,
        device: Device,
        pub transfer_to_gpu_timing: TimingAccumulator,
        pub forward_timing: TimingAccumulator,
        pub transfer_from_gpu_timing: TimingAccumulator,
    }

    impl Default for NeuralNetPolicy {
        fn default() -> Self {
            Self::new()
        }
    }

    impl NeuralNetPolicy {
        pub fn new() -> Self {
            Self::with_device(Device::cuda_if_available(0).unwrap_or(Device::Cpu))
        }

        pub fn with_device(device: Device) -> Self {
            NeuralNetPolicy {
                model: None,
                device,
                transfer_to_gpu_timing: TimingAccumulator::default(),
                forward_timing: TimingAccumulator::default(),
                transfer_from_gpu_timing: TimingAccumulator::default(),
            }
        }

        /// Load weights. `path` may be a `.safetensors` or `.pth` state dict, or a
        /// TorchScript `.pt` with one of those next to it.
        pub fn load(&mut self, path: &str) -> Result<(), String> {
            let state_dict = load_state_dict(Path::new(path), &self.device)?;
            let model = OracleNet::from_state_dict(state_dict, &self.device)
                .map_err(|e| format!("Failed to build OracleNet from {}: {}", path, e))?;
            let cfg = model.config();
            println!(
                "✅ Neural network loaded ({} blocks x {} channels) on {:?}",
                cfg.num_blocks, cfg.hidden_dim, self.device
            );
            self.model = Some(model);
            Ok(())
        }

        pub fn new_demo_enabled() -> Self {
            let mut nn = Self::new();
            let _ = nn.load("models/model.pt");
            nn
        }

        pub fn is_available(&self) -> bool {
            self.model.is_some()
        }

        /// The 17x8x8 input planes for `board`, on the inference device.
        pub fn board_to_tensor(&self, board: &Board) -> Tensor {
            Tensor::from_vec(board_to_planes(board), (17, 8, 8), &self.device)
                .expect("board planes have a fixed size")
        }

        /// Runs inference on one board. Returns (policy_probs, v_logit, k).
        pub fn predict(
            &mut self,
            board: &Board,
            qsearch_completed: bool,
            q_result: f32,
        ) -> Prediction {
            self.predict_batch(
                std::slice::from_ref(board),
                &[qsearch_completed],
                &[q_result],
            )
            .pop()
            .flatten()
        }

        /// Runs inference on a batch of boards. Returns Vec<(policy_probs, v_logit, k)>.
        pub fn predict_batch(
            &mut self,
            boards: &[Board],
            qsearch_flags: &[bool],
            q_results: &[f32],
        ) -> Vec<Prediction> {
            if self.model.is_none() || boards.is_empty() {
                return vec![None; boards.len()];
            }
            match self.try_predict_batch(boards, qsearch_flags, q_results) {
                Ok(results) => results.into_iter().map(Some).collect(),
                Err(e) => {
                    use std::sync::Once;
                    static ONCE: Once = Once::new();
                    ONCE.call_once(|| eprintln!("[NN-ERROR] predict_batch failed: {}", e));
                    vec![None; boards.len()]
                }
            }
        }

        fn try_predict_batch(
            &mut self,
            boards: &[Board],
            qsearch_flags: &[bool],
            q_results: &[f32],
        ) -> candle_core::Result<Vec<(Vec<f32>, f32, f32)>> {
            let model = self.model.as_ref().expect("checked by caller");
            let b = boards.len();

            // Phase 1: build inputs and copy them to the device
            let t0 = Instant::now();
            let mut planes = Vec::with_capacity(b * PLANES);
            for board in boards {
                planes.extend(board_to_planes(board));
            }
            let scalars: Vec<f32> = (0..b)
                .flat_map(|i| [q_results[i], if qsearch_flags[i] { 1.0 } else { 0.0 }])
                .collect();
            let planes = Tensor::from_vec(planes, (b, 17, 8, 8), &self.device)?;
            let scalars = Tensor::from_vec(scalars, (b, 2), &self.device)?;
            self.transfer_to_gpu_timing.record(t0.elapsed());

            // Phase 2: forward pass (synchronized so the timing covers the kernels)
            let t1 = Instant::now();
            let out = model.forward(&planes, &scalars)?;
            let policy = out.log_policy.exp()?;
            self.device.synchronize()?;
            self.forward_timing.record(t1.elapsed());

            // Phase 3: copy results back to the CPU
            let t2 = Instant::now();
            let policy: Vec<Vec<f32>> = policy.to_vec2()?;
            let v_logit: Vec<f32> = out.v_logit.flatten_all()?.to_vec1()?;
            self.transfer_from_gpu_timing.record(t2.elapsed());

            Ok(policy
                .into_iter()
                .zip(v_logit)
                .map(|(p, v)| (p, v, out.k))
                .collect())
        }

        pub fn policy_to_move_priors(
            &self,
            policy: &[f32],
            moves: &[Move],
            board: &Board,
        ) -> Vec<(Move, f32)> {
            policy_to_move_priors(policy, moves, board)
        }

        pub fn get_position_value(&mut self, board: &Board) -> Option<i32> {
            let (_, value, _) = self.predict(board, true, 0.0)?;
            Some((value * 1000.0) as i32)
        }

        pub fn cache_stats(&self) -> (usize, usize) {
            (0, 0)
        }

        pub fn print_inference_timing(&self) {
            println!();
            println!("=== NN Inference Phase Breakdown ===");
            println!(
                "{:<24} {:>10} {:>12} {:>12} {:>12}",
                "Phase", "Count", "Total(ms)", "Mean(us)", "Std(us)"
            );
            println!("{}", "-".repeat(72));
            Self::print_timing_row("CPU→device transfer", &self.transfer_to_gpu_timing);
            Self::print_timing_row("Forward pass", &self.forward_timing);
            Self::print_timing_row("device→CPU transfer", &self.transfer_from_gpu_timing);
        }

        fn print_timing_row(name: &str, acc: &TimingAccumulator) {
            if acc.count == 0 {
                println!("{:<24} {:>10} {:>12} {:>12} {:>12}", name, 0, "-", "-", "-");
            } else {
                println!(
                    "{:<24} {:>10} {:>12.1} {:>12.1} {:>12.1}",
                    name,
                    acc.count,
                    acc.total.as_secs_f64() * 1000.0,
                    acc.mean_us(),
                    acc.std_us()
                );
            }
        }
    }
}

// ==========================================
// 2. Stub Implementation (no "neural" feature)
// ==========================================
#[cfg(not(feature = "neural"))]
mod stub {
    use crate::board::Board;
    use crate::move_types::Move;

    use crate::mcts::tactical_mcts::TimingAccumulator;

    #[derive(Debug, Clone)]
    pub struct NeuralNetPolicy {
        _dummy: u8,
        pub transfer_to_gpu_timing: TimingAccumulator,
        pub forward_timing: TimingAccumulator,
        pub transfer_from_gpu_timing: TimingAccumulator,
    }

    impl Default for NeuralNetPolicy {
        fn default() -> Self {
            Self::new()
        }
    }

    impl NeuralNetPolicy {
        pub fn new() -> Self {
            NeuralNetPolicy {
                _dummy: 0,
                transfer_to_gpu_timing: TimingAccumulator::default(),
                forward_timing: TimingAccumulator::default(),
                transfer_from_gpu_timing: TimingAccumulator::default(),
            }
        }
        pub fn load(&mut self, _path: &str) -> Result<(), String> {
            Err(
                "Neural network feature not enabled (compile with --features neural or cuda)"
                    .to_string(),
            )
        }
        pub fn new_demo_enabled() -> Self {
            NeuralNetPolicy::new()
        }
        pub fn is_available(&self) -> bool {
            false
        }
        pub fn board_to_tensor(&self, _board: &Board) {}
        pub fn predict(
            &mut self,
            _board: &Board,
            _qsearch_completed: bool,
            _q_result: f32,
        ) -> Option<(Vec<f32>, f32, f32)> {
            None
        }
        pub fn predict_batch(
            &mut self,
            boards: &[Board],
            _qsearch_flags: &[bool],
            _q_results: &[f32],
        ) -> Vec<Option<(Vec<f32>, f32, f32)>> {
            vec![None; boards.len()]
        }
        pub fn policy_to_move_priors(
            &self,
            _policy: &[f32],
            _moves: &[Move],
            _board: &Board,
        ) -> Vec<(Move, f32)> {
            Vec::new()
        }
        pub fn get_position_value(&mut self, _board: &Board) -> Option<i32> {
            None
        }
        pub fn cache_stats(&self) -> (usize, usize) {
            (0, 0)
        }
        pub fn print_inference_timing(&self) {}
    }
}

#[cfg(feature = "neural")]
pub use self::candle_impl::NeuralNetPolicy;

#[cfg(not(feature = "neural"))]
pub use self::stub::NeuralNetPolicy;
