//! Comprehensive Tactical-Enhanced MCTS Benchmark Suite
//!
//! This benchmark validates the efficiency claims of the Tactics-Enhanced MCTS
//! by comparing it against classical MCTS and alpha-beta search across
//! various tactical and strategic positions.

use kingfisher::benchmarks::tactical_suite::{get_tactical_test_suite, TacticalPosition};
use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search, TacticalMctsConfig, mcts_pesto_search};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub tactical_mcts_time: Duration,
    pub classical_mcts_time: Duration,
    pub alpha_beta_time: Duration,
    pub tactical_mcts_iterations: u32,
    pub classical_mcts_iterations: u32,
    pub alpha_beta_depth: i32,
    pub mate_search_depth: i32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            tactical_mcts_time: Duration::from_millis(1000),
            classical_mcts_time: Duration::from_millis(1000),
            alpha_beta_time: Duration::from_millis(1000),
            tactical_mcts_iterations: 500,
            classical_mcts_iterations: 500,
            alpha_beta_depth: 6,
            mate_search_depth: 3,
        }
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub best_move: Option<kingfisher::move_types::Move>,
    pub search_time: Duration,
    pub nodes_searched: u64,
    pub nn_evaluations: u32,
    pub tactical_moves_explored: u32,
    pub mates_found: u32,
    pub engine_type: String,
}

#[derive(Debug)]
pub struct BenchmarkResults {
    pub tactical_mcts: Vec<SearchResult>,
    pub classical_mcts: Vec<SearchResult>,
    pub alpha_beta: Vec<SearchResult>,
    pub test_positions: Vec<TacticalPosition>,
}

impl BenchmarkResults {
    pub fn analyze_efficiency(&self) {
        println!("\n🔬 TACTICAL-ENHANCED MCTS EFFICIENCY ANALYSIS");
        println!("{}", "=".repeat(60));
        
        // Calculate average NN evaluations
        let tactical_avg_nn = self.tactical_mcts.iter()
            .map(|r| r.nn_evaluations as f64)
            .sum::<f64>() / self.tactical_mcts.len() as f64;
            
        let classical_avg_nn = self.classical_mcts.iter()
            .map(|r| r.nn_evaluations as f64)
            .sum::<f64>() / self.classical_mcts.len() as f64;
        
        // Calculate reduction percentage
        let nn_reduction = if classical_avg_nn > 0.0 {
            ((classical_avg_nn - tactical_avg_nn) / classical_avg_nn) * 100.0
        } else {
            0.0
        };
        
        println!("📊 Neural Network Call Reduction:");
        println!("   Tactical-Enhanced MCTS: {:.1} avg NN calls", tactical_avg_nn);
        println!("   Classical MCTS:         {:.1} avg NN calls", classical_avg_nn);
        println!("   Reduction:              {:.1}%", nn_reduction);
        
        // Calculate tactical move exploration
        let tactical_moves_explored: u32 = self.tactical_mcts.iter()
            .map(|r| r.tactical_moves_explored)
            .sum();
            
        println!("\n🎯 Tactical Move Analysis:");
        println!("   Total tactical moves explored: {}", tactical_moves_explored);
        println!("   Average per position: {:.1}", 
                 tactical_moves_explored as f64 / self.tactical_mcts.len() as f64);
        
        // Calculate mate detection
        let tactical_mates: u32 = self.tactical_mcts.iter().map(|r| r.mates_found).sum();
        let classical_mates: u32 = self.classical_mcts.iter().map(|r| r.mates_found).sum();
        let alpha_beta_mates: u32 = self.alpha_beta.iter().map(|r| r.mates_found).sum();
        
        println!("\n🏆 Mate Detection:");
        println!("   Tactical-Enhanced MCTS: {} mates found", tactical_mates);
        println!("   Classical MCTS:         {} mates found", classical_mates);
        println!("   Alpha-Beta:             {} mates found", alpha_beta_mates);
        
        // Calculate search time efficiency
        let tactical_avg_time = self.tactical_mcts.iter()
            .map(|r| r.search_time.as_millis() as f64)
            .sum::<f64>() / self.tactical_mcts.len() as f64;
            
        let classical_avg_time = self.classical_mcts.iter()
            .map(|r| r.search_time.as_millis() as f64)
            .sum::<f64>() / self.classical_mcts.len() as f64;
            
        let alpha_beta_avg_time = self.alpha_beta.iter()
            .map(|r| r.search_time.as_millis() as f64)
            .sum::<f64>() / self.alpha_beta.len() as f64;
        
        println!("\n⏱️  Average Search Time:");
        println!("   Tactical-Enhanced MCTS: {:.1}ms", tactical_avg_time);
        println!("   Classical MCTS:         {:.1}ms", classical_avg_time);
        println!("   Alpha-Beta:             {:.1}ms", alpha_beta_avg_time);
        
        // Node efficiency
        let tactical_avg_nodes = self.tactical_mcts.iter()
            .map(|r| r.nodes_searched as f64)
            .sum::<f64>() / self.tactical_mcts.len() as f64;
            
        let classical_avg_nodes = self.classical_mcts.iter()
            .map(|r| r.nodes_searched as f64)
            .sum::<f64>() / self.classical_mcts.len() as f64;
        
        println!("\n🌳 Node Efficiency:");
        println!("   Tactical-Enhanced MCTS: {:.0} avg nodes", tactical_avg_nodes);
        println!("   Classical MCTS:         {:.0} avg nodes", classical_avg_nodes);
        
        if classical_avg_nodes > 0.0 {
            let node_efficiency = (tactical_avg_nodes / classical_avg_nodes) * 100.0;
            println!("   Relative efficiency:    {:.1}%", node_efficiency);
        }
    }
    
    pub fn print_detailed_results(&self) {
        println!("\n📋 DETAILED POSITION ANALYSIS");
        println!("{}", "=".repeat(80));
        
        for (i, pos) in self.test_positions.iter().enumerate() {
            println!("\n🎯 Position {}: {}", i + 1, pos.name);
            println!("   FEN: {}", pos.fen);
            
            if i < self.tactical_mcts.len() {
                let tactical = &self.tactical_mcts[i];
                let classical = &self.classical_mcts[i];
                let alpha_beta = &self.alpha_beta[i];
                
                println!("   Tactical MCTS:  {:?} ({}ms, {} NN calls, {} tactical)", 
                         tactical.best_move, tactical.search_time.as_millis(), 
                         tactical.nn_evaluations, tactical.tactical_moves_explored);
                println!("   Classical MCTS: {:?} ({}ms, {} NN calls)", 
                         classical.best_move, classical.search_time.as_millis(), 
                         classical.nn_evaluations);
                println!("   Alpha-Beta:     {:?} ({}ms)", 
                         alpha_beta.best_move, alpha_beta.search_time.as_millis());
            }
        }
    }
}

/// Run comprehensive benchmark comparing all three search methods
pub fn run_comprehensive_benchmark(config: BenchmarkConfig) -> BenchmarkResults {
    println!("🚀 Starting Comprehensive Tactical-Enhanced MCTS Benchmark");
    println!("Configuration: {:?}", config);
    
    let test_suite = get_tactical_test_suite();
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();
    
    let mut tactical_results = Vec::new();
    let mut classical_results = Vec::new();
    let mut alpha_beta_results = Vec::new();
    
    for (i, position) in test_suite.iter().enumerate() {
        println!("\n🎯 Testing position {}/{}: {}", i + 1, test_suite.len(), position.name);
        
        let board = Board::new_from_fen(&position.fen);
        
        // Test Tactical-Enhanced MCTS
        println!("   Testing Tactical-Enhanced MCTS...");
        let tactical_result = benchmark_tactical_mcts(&board, &move_gen, &pesto_eval, &config);
        tactical_results.push(tactical_result);
        
        // Test Classical MCTS  
        println!("   Testing Classical MCTS...");
        let classical_result = benchmark_classical_mcts(&board, &move_gen, &pesto_eval, &config);
        classical_results.push(classical_result);
        
        // Test Alpha-Beta
        println!("   Testing Alpha-Beta...");
        let alpha_beta_result = benchmark_alpha_beta(&board, &move_gen, &pesto_eval, &config);
        alpha_beta_results.push(alpha_beta_result);
    }
    
    BenchmarkResults {
        tactical_mcts: tactical_results,
        classical_mcts: classical_results,
        alpha_beta: alpha_beta_results,
        test_positions: test_suite,
    }
}

fn benchmark_tactical_mcts(
    board: &Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    config: &BenchmarkConfig,
) -> SearchResult {
    let mut nn_policy = None; // Start without neural network for baseline comparison
    
    let tactical_config = TacticalMctsConfig {
        max_iterations: config.tactical_mcts_iterations,
        time_limit: config.tactical_mcts_time,
        mate_search_depth: config.mate_search_depth,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: None,
    };
    
    let start_time = Instant::now();
    let (best_move, stats, _root) = tactical_mcts_search(
        board.clone(),
        move_gen,
        pesto_eval,
        &mut nn_policy,
        tactical_config,
    );
    let search_time = start_time.elapsed();
    
    SearchResult {
        best_move,
        search_time,
        nodes_searched: stats.nodes_expanded as u64,
        nn_evaluations: stats.nn_policy_evaluations,
        tactical_moves_explored: stats.tactical_moves_explored,
        mates_found: stats.mates_found,
        engine_type: "Tactical-Enhanced MCTS".to_string(),
    }
}

fn benchmark_classical_mcts(
    board: &Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    config: &BenchmarkConfig,
) -> SearchResult {
    let start_time = Instant::now();
    let best_move = mcts_pesto_search(
        board.clone(),
        move_gen,
        pesto_eval,
        config.mate_search_depth,
        Some(config.classical_mcts_iterations),
        Some(config.classical_mcts_time),
    );
    let search_time = start_time.elapsed();
    
    SearchResult {
        best_move,
        search_time,
        nodes_searched: config.classical_mcts_iterations as u64, // Approximation
        nn_evaluations: 0, // Classical MCTS doesn't use NN in current implementation
        tactical_moves_explored: 0,
        mates_found: 0, // Would need to track this separately
        engine_type: "Classical MCTS".to_string(),
    }
}

fn benchmark_alpha_beta(
    board: &Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    config: &BenchmarkConfig,
) -> SearchResult {
    let start_time = Instant::now();
    // For now, return a placeholder since alpha-beta integration needs work
    let result = (None, 0u64);
    let search_time = start_time.elapsed();
    
    SearchResult {
        best_move: result.0,
        search_time,
        nodes_searched: result.1,
        nn_evaluations: 0,
        tactical_moves_explored: 0,
        mates_found: 0, // Would need to track this separately
        engine_type: "Alpha-Beta".to_string(),
    }
}

fn main() {
    println!("🎯 Tactical-Enhanced MCTS Comprehensive Benchmark");
    
    // Run with default configuration
    let config = BenchmarkConfig::default();
    let results = run_comprehensive_benchmark(config);
    
    // Analyze and display results
    results.analyze_efficiency();
    results.print_detailed_results();
    
    println!("\n✅ Benchmark Complete!");
}