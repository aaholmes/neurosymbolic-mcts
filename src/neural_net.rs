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
    use crate::tensor::{board_to_planes, move_to_index, policy_to_move_priors};
    use std::path::Path;
    use tch::{CModule, Device, IValue, Kind, Tensor};

    /// Describe an IValue type for debugging (without printing full tensor data)
    fn describe_ivalue(iv: &IValue) -> String {
        match iv {
            IValue::Tensor(t) => format!("Tensor(shape={:?}, kind={:?})", t.size(), t.kind()),
            IValue::Tuple(elems) => {
                let descs: Vec<String> = elems.iter().map(describe_ivalue).collect();
                format!("Tuple(len={}, [{}])", elems.len(), descs.join(", "))
            }
            IValue::GenericList(elems) => format!("GenericList(len={})", elems.len()),
            IValue::None => "None".to_string(),
            IValue::Bool(b) => format!("Bool({})", b),
            IValue::Int(i) => format!("Int({})", i),
            IValue::Double(d) => format!("Double({})", d),
            IValue::String(s) => format!("String({:?})", s),
            _ => "Unknown".to_string(),
        }
    }

    /// Force-load libtorch_cuda.so so that PyTorch's CUDA hooks register.
    /// Without this, the dynamic linker only loads libtorch_cpu.so (since all
    /// symbols resolve through it), and torch::cuda::is_available() returns false.
    fn ensure_cuda_loaded() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            // Find libtorch_cuda.so via LD_LIBRARY_PATH (same dirs where libtorch_cpu.so lives)
            let search_paths: Vec<String> = std::env::var("LD_LIBRARY_PATH")
                .unwrap_or_default()
                .split(':')
                .map(|s| s.to_string())
                .collect();

            for dir in &search_paths {
                let cuda_lib = format!("{}/libtorch_cuda.so", dir);
                if std::path::Path::new(&cuda_lib).exists() {
                    // dlopen with RTLD_NOW | RTLD_GLOBAL to register CUDA hooks
                    extern "C" {
                        fn dlopen(
                            filename: *const std::ffi::c_char,
                            flags: i32,
                        ) -> *mut std::ffi::c_void;
                        fn dlerror() -> *const std::ffi::c_char;
                    }
                    const RTLD_NOW: i32 = 0x2;
                    const RTLD_GLOBAL: i32 = 0x100;
                    unsafe {
                        let cstr = std::ffi::CString::new(cuda_lib.clone()).unwrap();
                        let handle = dlopen(cstr.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
                        if handle.is_null() {
                            let err = dlerror();
                            if !err.is_null() {
                                let msg = std::ffi::CStr::from_ptr(err);
                                eprintln!("Warning: failed to load {}: {:?}", cuda_lib, msg);
                            }
                        } else {
                            eprintln!("Loaded libtorch_cuda.so from {}", dir);
                        }
                    }
                    return;
                }
            }
        });
    }

    pub struct NeuralNetPolicy {
        model: Option<CModule>,
        device: Device,
    }

    impl NeuralNetPolicy {
        pub fn new() -> Self {
            ensure_cuda_loaded();
            NeuralNetPolicy {
                model: None,
                device: if tch::Cuda::is_available() {
                    Device::Cuda(0)
                } else {
                    Device::Cpu
                },
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
            let planes = board_to_planes(board);
            Tensor::from_slice(&planes)
                .view([17, 8, 8])
                .to_device(self.device)
                .to_kind(Kind::Float)
        }

        /// Runs inference on the board. Returns (policy_probs, value, k).
        pub fn predict(&mut self, board: &Board, qsearch_completed: bool) -> Option<(Vec<f32>, f32, f32)> {
            let model = self.model.as_ref()?;
            let input = self.board_to_tensor(board).unsqueeze(0);

            // Build [1, 2] scalars tensor: [material=0.0, qsearch_flag]
            // Material scalar is unused by the traced eval-mode model (it returns
            // raw v_logit; Rust combines with k * delta_M separately). Pass zero.
            let scalars_data = [0.0f32, if qsearch_completed { 1.0 } else { 0.0 }];
            let scalars_tensor = Tensor::from_slice(&scalars_data)
                .view([1, 2])
                .to_device(self.device)
                .to_kind(Kind::Float);

            let ivalue = model
                .method_is(
                    "forward",
                    &[tch::IValue::Tensor(input), tch::IValue::Tensor(scalars_tensor)],
                )
                .ok()?;

            if let tch::IValue::Tuple(elements) = ivalue {
                if elements.len() != 3 {
                    return None;
                }

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
                policy_tensor
                    .exp()
                    .view([-1])
                    .to_device(Device::Cpu)
                    .copy_data(&mut policy_probs, 4672);

                let value = value_tensor.double_value(&[0, 0]) as f32;
                let k_val = k_tensor.double_value(&[0, 0]) as f32;

                Some((policy_probs, value, k_val))
            } else {
                None
            }
        }

        /// Runs inference on a batch of boards. Returns Vec<(policy_probs, value, k)>.
        pub fn predict_batch(
            &mut self,
            boards: &[Board],
            qsearch_flags: &[bool],
        ) -> Vec<Option<(Vec<f32>, f32, f32)>> {
            let model = match self.model.as_ref() {
                Some(m) => m,
                None => return vec![None; boards.len()],
            };

            let mut input_tensors = Vec::with_capacity(boards.len());

            for board in boards {
                input_tensors.push(self.board_to_tensor(board));
            }

            // Stack inputs into [B, 17, 8, 8]
            let input_batch = Tensor::stack(&input_tensors, 0);
            // Build [B, 2] scalars tensor: [material=0.0, qsearch_flag]
            // Material scalar is unused by the traced eval-mode model (it returns
            // raw v_logit; Rust combines with k * delta_M separately). Pass zeros.
            let b = boards.len();
            let mut scalars_data = vec![0.0f32; b * 2];
            for i in 0..b {
                // scalars_data[i*2 + 0] = 0.0 (material, already zero)
                scalars_data[i * 2 + 1] = if qsearch_flags[i] { 1.0 } else { 0.0 };
            }
            let scalars_batch = Tensor::from_slice(&scalars_data)
                .view([b as i64, 2])
                .to_device(self.device)
                .to_kind(Kind::Float);

            let ivalue = model.method_is(
                "forward",
                &[
                    tch::IValue::Tensor(input_batch),
                    tch::IValue::Tensor(scalars_batch),
                ],
            );

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
                        policy_batch
                            .get(i as i64)
                            .exp()
                            .to_device(Device::Cpu)
                            .copy_data(&mut policy_probs, 4672);

                        let value = value_batch.double_value(&[i as i64, 0]) as f32;
                        let k_val = k_batch.double_value(&[i as i64, 0]) as f32;

                        results.push(Some((policy_probs, value, k_val)));
                    }
                    results
                }
                Ok(ref other) => {
                    use std::sync::Once;
                    static ONCE: Once = Once::new();
                    ONCE.call_once(|| {
                        eprintln!(
                            "[NN-ERROR] predict_batch: unexpected output type: {:?}",
                            describe_ivalue(other)
                        );
                    });
                    vec![None; boards.len()]
                }
                Err(ref e) => {
                    use std::sync::Once;
                    static ONCE_ERR: Once = Once::new();
                    ONCE_ERR.call_once(|| {
                        eprintln!("[NN-ERROR] predict_batch forward failed: {:?}", e);
                    });
                    vec![None; boards.len()]
                }
            }
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
            let (_, value, _) = self.predict(board, true)?;
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
        pub fn board_to_tensor(&self, _board: &Board) -> () {
            ()
        }
        pub fn predict(&mut self, _board: &Board, _qsearch_completed: bool) -> Option<(Vec<f32>, f32, f32)> {
            None
        }
        pub fn predict_batch(&mut self, boards: &[Board], _qsearch_flags: &[bool]) -> Vec<Option<(Vec<f32>, f32, f32)>> {
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
    }
}

#[cfg(feature = "neural")]
pub use self::real::NeuralNetPolicy;

#[cfg(not(feature = "neural"))]
pub use self::stub::NeuralNetPolicy;
