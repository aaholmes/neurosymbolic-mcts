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
    use crate::mcts::tactical_mcts::{StatsAccumulator, TimingAccumulator};
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

    /// A batch whose forward pass has been queued (see `launch_batch`).
    pub struct InFlight {
        inner: Option<Launched>,
        len: usize,
    }

    struct Launched {
        policy: Tensor,
        v_logit: Tensor,
        k: f32,
        /// Recorded after the forward pass; complete once the GPU has run it.
        #[cfg(feature = "cuda")]
        done: Option<candle_core::cuda::cudarc::driver::CudaEvent>,
    }

    impl Launched {
        fn is_complete(&self) -> bool {
            #[cfg(feature = "cuda")]
            if let Some(event) = &self.done {
                return event.is_complete();
            }
            true
        }
    }

    fn report_error(e: &candle_core::Error) {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| eprintln!("[NN-ERROR] predict_batch failed: {}", e));
    }

    pub struct NeuralNetPolicy {
        model: Option<OracleNet>,
        device: Device,
        pub transfer_to_gpu_timing: TimingAccumulator,
        pub forward_timing: TimingAccumulator,
        pub transfer_from_gpu_timing: TimingAccumulator,
        /// Number of positions in each forward pass.
        pub batch_sizes: StatsAccumulator,
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

        /// A policy on its own CUDA stream (or the CPU without a GPU), so that
        /// several inference workers can overlap their GPU work.
        pub fn new_on_own_stream() -> Self {
            let device = if candle_core::utils::cuda_is_available() {
                Device::new_cuda_with_stream(0).unwrap_or(Device::Cpu)
            } else {
                Device::Cpu
            };
            #[cfg(feature = "cuda")]
            if let Device::Cuda(dev) = &device {
                // A second stream switches candle to recording and waiting on CUDA
                // events around every tensor access (hundreds of calls per batch).
                // Each worker keeps all its tensors on its own stream, so none are
                // needed. The setting is per CUDA context, i.e. process-wide.
                // SAFETY: no tensor is shared between streams.
                unsafe { dev.disable_event_tracking() };
            }
            Self::with_device(device)
        }

        pub fn device(&self) -> &Device {
            &self.device
        }

        pub fn with_device(device: Device) -> Self {
            NeuralNetPolicy {
                model: None,
                device,
                transfer_to_gpu_timing: TimingAccumulator::default(),
                forward_timing: TimingAccumulator::default(),
                transfer_from_gpu_timing: TimingAccumulator::default(),
                batch_sizes: StatsAccumulator::default(),
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
            let pending = self.launch_batch(boards, qsearch_flags, q_results);
            self.finish_batch(pending)
        }

        /// Upload a batch and queue its forward pass without waiting for the GPU.
        /// Read the results with `finish_batch`; several batches may be in flight.
        pub fn launch_batch(
            &mut self,
            boards: &[Board],
            qsearch_flags: &[bool],
            q_results: &[f32],
        ) -> InFlight {
            let len = boards.len();
            if self.model.is_none() || len == 0 {
                return InFlight { inner: None, len };
            }
            match self.try_launch(boards, qsearch_flags, q_results) {
                Ok(inner) => InFlight {
                    inner: Some(inner),
                    len,
                },
                Err(e) => {
                    report_error(&e);
                    InFlight { inner: None, len }
                }
            }
        }

        /// Whether the GPU has finished a launched batch (always true on the CPU).
        pub fn is_finished(&self, pending: &InFlight) -> bool {
            pending.inner.as_ref().is_none_or(|i| i.is_complete())
        }

        /// Wait for a launched batch and copy its results to the CPU.
        pub fn finish_batch(&mut self, pending: InFlight) -> Vec<Prediction> {
            let Some(inner) = pending.inner else {
                return vec![None; pending.len];
            };
            match self.try_finish(inner) {
                Ok(results) => results.into_iter().map(Some).collect(),
                Err(e) => {
                    report_error(&e);
                    vec![None; pending.len]
                }
            }
        }

        fn try_launch(
            &mut self,
            boards: &[Board],
            qsearch_flags: &[bool],
            q_results: &[f32],
        ) -> candle_core::Result<Launched> {
            let model = self.model.as_ref().expect("checked by caller");
            let b = boards.len();
            self.batch_sizes.record(b as f64);

            // Build inputs and copy them to the device
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

            // Queue the forward pass (timed up to the last kernel launch, not its completion)
            let t1 = Instant::now();
            let out = model.forward(&planes, &scalars)?;
            let policy = out.log_policy.exp()?;
            let launched = Launched {
                policy,
                v_logit: out.v_logit,
                k: out.k,
                #[cfg(feature = "cuda")]
                done: match &self.device {
                    Device::Cuda(dev) => Some(
                        dev.cuda_stream()
                            .record_event(None)
                            .map_err(candle_core::Error::wrap)?,
                    ),
                    _ => None,
                },
            };
            self.forward_timing.record(t1.elapsed());
            Ok(launched)
        }

        fn try_finish(
            &mut self,
            launched: Launched,
        ) -> candle_core::Result<Vec<(Vec<f32>, f32, f32)>> {
            // Waits for the GPU, then copies the results back
            let t2 = Instant::now();
            let policy: Vec<Vec<f32>> = launched.policy.to_vec2()?;
            let v_logit: Vec<f32> = launched.v_logit.flatten_all()?.to_vec1()?;
            self.transfer_from_gpu_timing.record(t2.elapsed());
            Ok(policy
                .into_iter()
                .zip(v_logit)
                .map(|(p, v)| (p, v, launched.k))
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
            if self.batch_sizes.count > 0 {
                println!(
                    "Batch size: mean {:.1}, s.d. {:.1}, over {} forward passes",
                    self.batch_sizes.mean_val(),
                    self.batch_sizes.std_val(),
                    self.batch_sizes.count
                );
            }
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

    /// A queued batch; the stub has no network, so it only remembers the size.
    pub struct InFlight {
        len: usize,
    }

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
        pub fn new_on_own_stream() -> Self {
            Self::new()
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
        pub fn launch_batch(
            &mut self,
            boards: &[Board],
            _qsearch_flags: &[bool],
            _q_results: &[f32],
        ) -> InFlight {
            InFlight { len: boards.len() }
        }
        pub fn is_finished(&self, _pending: &InFlight) -> bool {
            true
        }
        pub fn finish_batch(&mut self, pending: InFlight) -> Vec<Option<(Vec<f32>, f32, f32)>> {
            vec![None; pending.len]
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
pub use self::candle_impl::{InFlight, NeuralNetPolicy};

#[cfg(not(feature = "neural"))]
pub use self::stub::{InFlight, NeuralNetPolicy};
