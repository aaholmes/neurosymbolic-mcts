//! Batching Inference Server for MCTS
//!
//! This module provides a way to batch neural network inference requests from multiple
//! MCTS threads to improve GPU/CPU throughput.

use crate::board::Board;
use crate::mcts::tactical_mcts::TimingAccumulator;
use crate::move_types::Move;
use crate::neural_net::NeuralNetPolicy;
use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Request for a single board evaluation
pub struct PredictRequest {
    pub board: Board,
    /// The moves to return priors for, in this order; `None` for the full
    /// 4672-slot policy.
    pub moves: Option<Vec<Move>>,
    pub qsearch_completed: bool,
    pub q_result: f32,
    pub response_sender: Sender<Option<(Vec<f32>, f32, f32)>>,
}

/// Per-worker statistics, updated after every batch.
#[derive(Debug, Clone, Default)]
pub struct WorkerStats {
    /// Number of forward passes.
    pub batches: u64,
    /// Number of positions evaluated.
    pub positions: u64,
    /// For batches collected while the GPU was idle: time from the batch's first
    /// request until it was closed. Batches collected while the previous one ran
    /// on the GPU are not counted here.
    pub collect: TimingAccumulator,
    /// Time from launching a batch to having its results on the CPU.
    pub process: TimingAccumulator,
    /// Time from one batch's results reaching the CPU until the next launch,
    /// during which this worker has nothing queued on the GPU.
    pub gpu_idle: TimingAccumulator,
}

/// How inference workers batch and read back requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerOptions {
    /// How long a worker whose GPU is idle waits for more requests after the
    /// first one of a batch.
    pub idle_window: Duration,
    /// Collect the next batch while the current one runs on the GPU, and launch
    /// it as soon as the current one is read back. Off: collect, run, repeat.
    pub pipeline: bool,
    /// Read back the full 4672-slot policy and compute priors for the requested
    /// moves on the CPU, instead of gathering just those logits on the GPU.
    pub full_policy: bool,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            idle_window: DEFAULT_IDLE_WINDOW,
            pipeline: true,
            full_policy: false,
        }
    }
}

impl ServerOptions {
    /// Defaults, overridden by KF_PIPELINE=0|1 and KF_FULL_POLICY=0|1.
    pub fn from_env() -> Self {
        let flag = |name: &str, default: bool| match std::env::var(name).as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            Ok(v) => panic!("{name} must be 0 or 1, not {v:?}"),
            Err(_) => default,
        };
        let d = Self::default();
        Self {
            pipeline: flag("KF_PIPELINE", d.pipeline),
            full_policy: flag("KF_FULL_POLICY", d.full_policy),
            ..d
        }
    }
}

/// The Shared Inference Server
#[derive(Debug)]
pub struct InferenceServer {
    request_sender: Sender<PredictRequest>,
    stats: Vec<Arc<Mutex<WorkerStats>>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl InferenceServer {
    /// Creates a new inference server and starts the background worker thread.
    pub fn new(nn: NeuralNetPolicy, batch_size: usize) -> Self {
        Self::with_workers(vec![nn], batch_size)
    }

    /// Starts one worker thread per network. The workers share one request
    /// queue, so while one runs a forward pass another can collect the next
    /// batch. Give each network its own CUDA stream
    /// (`Device::new_cuda_with_stream`) so their GPU work can overlap.
    pub fn with_workers(nns: Vec<NeuralNetPolicy>, batch_size: usize) -> Self {
        Self::with_workers_and_window(nns, batch_size, DEFAULT_IDLE_WINDOW)
    }

    /// As `with_workers`, with the time a worker whose GPU is idle waits for more
    /// requests after the first one of a batch.
    pub fn with_workers_and_window(
        nns: Vec<NeuralNetPolicy>,
        batch_size: usize,
        idle_window: Duration,
    ) -> Self {
        Self::with_options(
            nns,
            batch_size,
            ServerOptions {
                idle_window,
                ..ServerOptions::default()
            },
        )
    }

    /// Starts one worker thread per network, batching as `options` says.
    pub fn with_options(
        nns: Vec<NeuralNetPolicy>,
        batch_size: usize,
        options: ServerOptions,
    ) -> Self {
        assert!(
            !nns.is_empty(),
            "an inference server needs at least one worker"
        );
        let (request_sender, request_receiver) = unbounded::<PredictRequest>();
        let mut stats = Vec::with_capacity(nns.len());
        let mut workers = Vec::with_capacity(nns.len());
        for nn in nns {
            let s = Arc::new(Mutex::new(WorkerStats::default()));
            stats.push(s.clone());
            let rx = request_receiver.clone();
            workers.push(thread::spawn(move || {
                run_worker(nn, rx, batch_size, options, s)
            }));
        }
        InferenceServer {
            request_sender,
            stats,
            workers,
        }
    }

    /// Load the model at `path` once per worker, each on its own CUDA stream,
    /// and start a server with `workers` workers.
    pub fn load(path: &str, workers: usize, batch_size: usize) -> Result<Self, String> {
        Self::load_with_window(path, workers, batch_size, DEFAULT_IDLE_WINDOW)
    }

    /// As `load`, with the idle-GPU collection window of `with_workers_and_window`,
    /// and pipelining and readback set by KF_PIPELINE and KF_FULL_POLICY.
    pub fn load_with_window(
        path: &str,
        workers: usize,
        batch_size: usize,
        idle_window: Duration,
    ) -> Result<Self, String> {
        let nns = (0..workers.max(1))
            .map(|_| {
                let mut nn = NeuralNetPolicy::new_on_own_stream();
                nn.load(path).map(|_| nn)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let options = ServerOptions {
            idle_window,
            ..ServerOptions::from_env()
        };
        Ok(Self::with_options(nns, batch_size, options))
    }

    /// Stop accepting requests, wait for every worker to finish, and return
    /// their final statistics.
    pub fn shutdown(self) -> Vec<WorkerStats> {
        let InferenceServer {
            request_sender,
            stats,
            workers,
        } = self;
        drop(request_sender);
        for w in workers {
            let _ = w.join();
        }
        stats
            .iter()
            .map(|s| s.lock().expect("stats lock").clone())
            .collect()
    }

    /// A snapshot of each worker's statistics (empty for mock servers).
    pub fn worker_stats(&self) -> Vec<WorkerStats> {
        self.stats
            .iter()
            .map(|s| s.lock().expect("stats lock").clone())
            .collect()
    }

    /// Creates a mock server that returns random v_logit values (for testing).
    /// Returns `(policy, v_logit, k)` where v_logit is unbounded (not tanh'd).
    pub fn new_mock() -> Self {
        let (request_sender, request_receiver) = unbounded::<PredictRequest>();
        thread::spawn(move || {
            loop {
                match request_receiver.recv() {
                    Ok(req) => {
                        let mut rng = rand::thread_rng();
                        // Random policy
                        let slots = req.moves.as_ref().map_or(4672, |m| m.len());
                        let mut policy = vec![0.0; slots];
                        // Set a few random moves to non-zero
                        for _ in 0..5.min(slots) {
                            let idx = rng.gen_range(0..slots);
                            policy[idx] = rng.gen_range(0.0..1.0);
                        }
                        let v_logit: f32 = rng.gen_range(-2.0..2.0);
                        let k: f32 = rng.gen_range(0.0..1.0);
                        let _ = req.response_sender.send(Some((policy, v_logit, k)));
                    }
                    Err(_) => break,
                }
            }
        });
        InferenceServer {
            request_sender,
            stats: Vec::new(),
            workers: Vec::new(),
        }
    }

    /// Creates a mock server that always returns a fixed v_logit (for testing).
    /// `v_logit` is the raw positional logit (unbounded, before material + tanh).
    pub fn new_mock_biased(v_logit: f32) -> Self {
        let (request_sender, request_receiver) = unbounded::<PredictRequest>();
        thread::spawn(move || loop {
            match request_receiver.recv() {
                Ok(req) => {
                    let policy = match &req.moves {
                        Some(m) => vec![1.0 / m.len().max(1) as f32; m.len()],
                        None => vec![0.0; 4672],
                    };
                    let k: f32 = 0.5;
                    let _ = req.response_sender.send(Some((policy, v_logit, k)));
                }
                Err(_) => break,
            }
        });
        InferenceServer {
            request_sender,
            stats: Vec::new(),
            workers: Vec::new(),
        }
    }

    /// Creates a mock server that returns uniform policy and 0.0 v_logit.
    pub fn new_mock_uniform() -> Self {
        Self::new_mock_biased(0.0)
    }

    /// Asynchronously requests evaluation for a board.
    /// Returns a receiver that will yield the result.
    pub fn predict_async(
        &self,
        board: Board,
        qsearch_completed: bool,
        q_result: f32,
    ) -> Receiver<Option<(Vec<f32>, f32, f32)>> {
        let (response_sender, response_receiver) = crossbeam_channel::bounded(1);
        let _ = self.request_sender.send(PredictRequest {
            board,
            moves: None,
            qsearch_completed,
            q_result,
            response_sender,
        });
        response_receiver
    }

    /// As `predict_async`, but the returned policy holds one prior per move in
    /// `moves` (a softmax over just those moves, in the same order).
    pub fn predict_legal_async(
        &self,
        board: Board,
        moves: Vec<Move>,
        qsearch_completed: bool,
        q_result: f32,
    ) -> Receiver<Option<(Vec<f32>, f32, f32)>> {
        let (response_sender, response_receiver) = crossbeam_channel::bounded(1);
        let _ = self.request_sender.send(PredictRequest {
            board,
            moves: Some(moves),
            qsearch_completed,
            q_result,
            response_sender,
        });
        response_receiver
    }
}

/// How long a worker whose GPU is idle waits for more requests after the first
/// one of a batch.
pub const DEFAULT_IDLE_WINDOW: Duration = Duration::from_micros(500);

/// Collect requests into batches, run them through the network and send back
/// the results, until every request sender has been dropped.
///
/// With `options.pipeline`, while one batch runs on the GPU the worker collects
/// the next, and launches it as soon as the first batch's results are back. Only
/// when nothing is in flight does it wait up to `idle_window` for a batch to fill.
fn run_worker(
    mut nn: NeuralNetPolicy,
    request_receiver: Receiver<PredictRequest>,
    batch_size: usize,
    options: ServerOptions,
    stats: Arc<Mutex<WorkerStats>>,
) {
    let idle_window = options.idle_window;
    let mut next: Vec<PredictRequest> = Vec::with_capacity(batch_size);
    let mut closed = false;
    let mut last_finish: Option<Instant> = None;

    loop {
        // 1. Nothing collected while the GPU ran: wait for a first request, then
        // spin for more. Spinning on try_recv picks requests up sooner than
        // recv_deadline, whose wake-ups added ~70 us per batch.
        let mut collect_time = None;
        if next.is_empty() {
            if closed {
                break;
            }
            match request_receiver.recv() {
                Ok(req) => next.push(req),
                Err(_) => break,
            }
            let start_wait = Instant::now();
            while next.len() < batch_size && start_wait.elapsed() < idle_window {
                match request_receiver.try_recv() {
                    Ok(req) => next.push(req),
                    Err(TryRecvError::Empty) => thread::yield_now(),
                    Err(TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
            collect_time = Some(start_wait.elapsed());
        }

        // 2. Launch the batch without waiting for the GPU
        let gpu_idle = last_finish.map(|t| t.elapsed());
        let mut requests = std::mem::replace(&mut next, Vec::with_capacity(batch_size));
        let t_process = Instant::now();
        let boards: Vec<Board> = requests.iter().map(|r| r.board.clone()).collect();
        let qsearch_flags: Vec<bool> = requests.iter().map(|r| r.qsearch_completed).collect();
        let q_results: Vec<f32> = requests.iter().map(|r| r.q_result).collect();
        let moves: Vec<Option<Vec<Move>>> = requests.iter_mut().map(|r| r.moves.take()).collect();
        let pending = if options.full_policy {
            nn.launch_batch(&boards, &qsearch_flags, &q_results)
        } else {
            nn.launch_batch_with_moves(&boards, &qsearch_flags, &q_results, &moves)
        };

        // 3. Collect the next batch while this one runs
        while options.pipeline && !closed && next.len() < batch_size && !nn.is_finished(&pending) {
            match request_receiver.try_recv() {
                Ok(req) => next.push(req),
                Err(TryRecvError::Empty) => thread::yield_now(),
                Err(TryRecvError::Disconnected) => closed = true,
            }
        }

        // 4. Read the results back
        let mut results = nn.finish_batch(pending);
        if options.full_policy {
            // Priors for the requested moves, from the full policy
            for ((res, mv), board) in results.iter_mut().zip(&moves).zip(&boards) {
                if let (Some((policy, _, _)), Some(mv)) = (res.as_mut(), mv) {
                    *policy = nn
                        .policy_to_move_priors(policy, mv, board)
                        .into_iter()
                        .map(|(_, p)| p)
                        .collect();
                }
            }
        }
        let process_time = t_process.elapsed();
        last_finish = Some(Instant::now());

        // 5. Record stats before answering, so a caller that has its answer sees them
        {
            let mut s = stats.lock().expect("stats lock");
            s.batches += 1;
            s.positions += requests.len() as u64;
            if let Some(t) = collect_time {
                s.collect.record(t);
            }
            if let Some(t) = gpu_idle {
                s.gpu_idle.record(t);
            }
            s.process.record(process_time);
        }

        // 6. Dispatch results
        for (req, res) in requests.into_iter().zip(results) {
            let _ = req.response_sender.send(res);
        }
    }
    nn.print_inference_timing();
    let s = stats.lock().expect("stats lock");
    if s.batches > 0 {
        println!(
            "Server: {} batches, collect mean {:.1} us, process mean {:.1} us, gpu idle mean {:.1} us, idle-window batches {}",
            s.batches,
            s.collect.mean_us(),
            s.process.mean_us(),
            s.gpu_idle.mean_us(),
            s.collect.count
        );
    }
    println!("🛑 Inference Server Worker shutting down.");
}
