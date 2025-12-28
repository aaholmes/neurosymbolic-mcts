//! Batching Inference Server for MCTS
//!
//! This module provides a way to batch neural network inference requests from multiple
//! MCTS threads to improve GPU/CPU throughput.

use crate::board::Board;
use crate::neural_net::NeuralNetPolicy;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::thread;
use std::time::Duration;
use rand::Rng;

/// Request for a single board evaluation
pub struct PredictRequest {
    pub board: Board,
    pub response_sender: Sender<Option<(Vec<f32>, f32, f32)>>,
}

/// The Shared Inference Server
#[derive(Debug)]
pub struct InferenceServer {
    request_sender: Sender<PredictRequest>,
}

impl InferenceServer {
    /// Creates a new inference server and starts the background worker thread.
    pub fn new(mut nn: NeuralNetPolicy, batch_size: usize) -> Self {
        let (request_sender, request_receiver) = unbounded::<PredictRequest>();

        thread::spawn(move || {
            let mut requests = Vec::with_capacity(batch_size);
            
            loop {
                // 1. Wait for the first request (blocking)
                let first_req = match request_receiver.recv() {
                    Ok(req) => req,
                    Err(_) => break, // Channel closed, shut down worker
                };
                requests.push(first_req);

                // 2. Collect more requests without blocking (up to batch_size or timeout)
                let start_wait = std::time::Instant::now();
                while requests.len() < batch_size && start_wait.elapsed() < Duration::from_micros(500) {
                    if let Ok(req) = request_receiver.try_recv() {
                        requests.push(req);
                    } else {
                        // Small pause to allow other threads to push
                        thread::yield_now();
                    }
                }

                // 3. Process Batch
                let boards: Vec<Board> = requests.iter().map(|r| r.board.clone()).collect();
                let results = nn.predict_batch(&boards);

                // 4. Dispatch Results
                for (req, res) in requests.drain(..).zip(results) {
                    let _ = req.response_sender.send(res);
                }
            }
            println!("🛑 Inference Server Worker shutting down.");
        });

        InferenceServer { request_sender }
    }

    /// Creates a mock server that returns random values (for testing).
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
                        let value: f32 = rng.gen_range(-1.0..1.0);
                        let k: f32 = rng.gen_range(0.0..1.0);
                        let _ = req.response_sender.send(Some((policy, value, k)));
                    }
                    Err(_) => break,
                }
            }
        });
        InferenceServer { request_sender }
    }

    /// Creates a mock server that always returns a biased value (for testing).
    pub fn new_mock_biased(bias: f32) -> Self {
        let (request_sender, request_receiver) = unbounded::<PredictRequest>();
        thread::spawn(move || {
            loop {
                match request_receiver.recv() {
                    Ok(req) => {
                        let policy = vec![0.0; 4672];
                        let value: f32 = bias;
                        let k: f32 = 0.5;
                        let _ = req.response_sender.send(Some((policy, value, k)));
                    }
                    Err(_) => break,
                }
            }
        });
        InferenceServer { request_sender }
    }

    /// Creates a mock server that returns uniform policy and 0.0 value.
    pub fn new_mock_uniform() -> Self {
        Self::new_mock_biased(0.0)
    }

    /// Asynchronously requests evaluation for a board.
    /// Returns a receiver that will yield the result.
    pub fn predict_async(&self, board: Board) -> Receiver<Option<(Vec<f32>, f32, f32)>> {
        let (response_sender, response_receiver) = crossbeam_channel::bounded(1);
        let _ = self.request_sender.send(PredictRequest {
            board,
            response_sender,
        });
        response_receiver
    }
}
