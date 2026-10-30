//! Batching Inference Server for MCTS
//!
//! This module provides a way to batch neural network inference requests from multiple
//! MCTS threads to improve GPU/CPU throughput.

use crate::board::Board;
use crate::mcts::tactical_mcts::TimingAccumulator;
use crate::neural_net::NeuralNetPolicy;
use crossbeam_channel::{unbounded, Receiver, Sender};
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Request for a single board evaluation
pub struct PredictRequest {
    pub board: Board,
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
    /// Time from a batch's first request until the batch was closed.
    pub collect: TimingAccumulator,
    /// Time spent in `predict_batch` (input copies, forward pass, readback).
    pub process: TimingAccumulator,
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
            workers.push(thread::spawn(move || run_worker(nn, rx, batch_size, s)));
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
        let nns = (0..workers.max(1))
            .map(|_| {
                let mut nn = NeuralNetPolicy::new_on_own_stream();
                nn.load(path).map(|_| nn)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self::with_workers(nns, batch_size))
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
                        let mut policy = vec![0.0; 4672];
                        // Set a few random moves to non-zero
                        for _ in 0..5 {
                            let idx = rng.gen_range(0..4672);
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
                    let policy = vec![0.0; 4672];
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
            qsearch_completed,
            q_result,
            response_sender,
        });
        response_receiver
    }
}

/// How long a worker waits for more requests after the first one of a batch.
const COLLECT_WINDOW: Duration = Duration::from_micros(500);

/// Collect requests into batches, run them through the network and send back
/// the results, until every request sender has been dropped.
fn run_worker(
    mut nn: NeuralNetPolicy,
    request_receiver: Receiver<PredictRequest>,
    batch_size: usize,
    stats: Arc<Mutex<WorkerStats>>,
) {
    let mut requests = Vec::with_capacity(batch_size);

    loop {
        // 1. Wait for the first request (blocking)
        let first_req = match request_receiver.recv() {
            Ok(req) => req,
            Err(_) => break, // Channel closed, shut down worker
        };
        requests.push(first_req);

        // 2. Collect more requests until the batch is full or the window closes.
        // Spinning on try_recv picks requests up sooner than recv_deadline, whose
        // wake-ups added ~70 us per batch and cost ~3% of self-play throughput.
        let start_wait = Instant::now();
        while requests.len() < batch_size && start_wait.elapsed() < COLLECT_WINDOW {
            match request_receiver.try_recv() {
                Ok(req) => requests.push(req),
                Err(_) => thread::yield_now(),
            }
        }
        let collect_time = start_wait.elapsed();

        // 3. Process Batch
        let t_process = Instant::now();
        let boards: Vec<Board> = requests.iter().map(|r| r.board.clone()).collect();
        let qsearch_flags: Vec<bool> = requests.iter().map(|r| r.qsearch_completed).collect();
        let q_results: Vec<f32> = requests.iter().map(|r| r.q_result).collect();
        let results = nn.predict_batch(&boards, &qsearch_flags, &q_results);
        let process_time = t_process.elapsed();

        // 4. Record stats before answering, so a caller that has its answer sees them
        {
            let mut s = stats.lock().expect("stats lock");
            s.batches += 1;
            s.positions += requests.len() as u64;
            s.collect.record(collect_time);
            s.process.record(process_time);
        }

        // 5. Dispatch Results
        for (req, res) in requests.drain(..).zip(results) {
            let _ = req.response_sender.send(res);
        }
    }
    nn.print_inference_timing();
    let s = stats.lock().expect("stats lock");
    if s.batches > 0 {
        println!(
            "Server: {} batches, collect mean {:.1} us, process mean {:.1} us",
            s.batches,
            s.collect.mean_us(),
            s.process.mean_us()
        );
    }
    println!("🛑 Inference Server Worker shutting down.");
}
