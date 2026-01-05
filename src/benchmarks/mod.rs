use crate::boardstack::BoardStack;
use crate::move_types::Move;
use crate::move_generation::MoveGen;
use crate::eval::PestoEval;
use crate::agent::{Agent, SimpleAgent, HumanlikeAgent};
use std::time::{Duration, Instant};

pub mod tactical_suite;
pub mod performance;
pub mod strength_testing;
pub mod elo_estimation;
pub mod elo_tournament;

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub position_name: String,
    pub engine_name: String,
    pub time_taken: Duration,
    pub nodes_searched: u64,
    pub best_move: Option<Move>,
    pub found_mate: bool,
    pub mate_depth: Option<u8>,
}

#[derive(Debug)]
pub struct BenchmarkSummary {
    pub total_positions: usize,
    pub positions_solved: usize,
    pub average_time: Duration,
    pub total_nodes: u64,
    pub mate_accuracy: f64, // Percentage of mates found
}

impl BenchmarkSummary {
    pub fn from_results(results: &[BenchmarkResult]) -> Self {
        let total_positions = results.len();
        let positions_solved = results.iter().filter(|r| r.found_mate).count();
        let total_time: Duration = results.iter().map(|r| r.time_taken).sum();
        let average_time = if total_positions > 0 {
            total_time / total_positions as u32
        } else {
            Duration::from_millis(0)
        };
        let total_nodes: u64 = results.iter().map(|r| r.nodes_searched).sum();
        let mate_accuracy = if total_positions > 0 {
            (positions_solved as f64 / total_positions as f64) * 100.0
        } else {
            0.0
        };

        BenchmarkSummary {
            total_positions,
            positions_solved,
            average_time,
            total_nodes,
            mate_accuracy,
        }
    }

    pub fn print_summary(&self, engine_name: &str) {
        println!("\n=== {} Benchmark Results ===", engine_name);
        println!("Total Positions: {}", self.total_positions);
        println!("Positions Solved: {}", self.positions_solved);
        println!("Mate Accuracy: {:.1}%", self.mate_accuracy);
        println!("Average Time: {:.2}ms", self.average_time.as_millis());
        println!("Total Nodes: {}", self.total_nodes);
        if self.total_positions > 0 {
            println!("Nodes per Position: {:.0}", self.total_nodes as f64 / self.total_positions as f64);
        }
        println!("=======================================\n");
    }
}

lazy_static::lazy_static! {
    static ref BENCH_MOVE_GEN: MoveGen = MoveGen::new();
    static ref BENCH_PESTO_EVAL: PestoEval = PestoEval::new();
}

pub fn create_simple_agent() -> SimpleAgent<'static> {
    SimpleAgent::new(
        3,     // mate_search_depth
        8,     // ab_search_depth  
        16,    // q_search_max_depth
        false, // verbose
        &*BENCH_MOVE_GEN,
        &*BENCH_PESTO_EVAL,
    )
}

pub fn create_humanlike_agent() -> HumanlikeAgent<'static> {
    HumanlikeAgent::new(
        &*BENCH_MOVE_GEN,
        &*BENCH_PESTO_EVAL,
        None, // No EGTB for benchmarking
        3,    // mate_search_depth
        1000, // mcts_iterations
        5000, // mcts_time_limit_ms
        8,    // placeholder_ab_depth
        16,   // placeholder_q_depth
    )
}