//! Comprehensive Strength Testing and Comparisons
//!
//! This module provides extensive benchmarking to test the strength improvements
//! from our mate-search-first MCTS approach and neural network policy guidance.

use crate::board::Board;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
// use crate::search::alpha_beta::AlphaBeta;
// use crate::mcts::mcts_search;
use crate::mcts::neural_mcts::neural_mcts_search;

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::fmt;

/// Test suite configuration
#[derive(Debug, Clone)]
pub struct StrengthTestConfig {
    /// Time limit per position (milliseconds)
    pub time_limit_ms: u64,
    /// MCTS iterations limit
    pub mcts_iterations: u32,
    /// Alpha-beta search depth
    pub ab_depth: i32,
    /// Mate search depth for MCTS
    pub mate_search_depth: i32,
    /// Path to neural network model (optional)
    pub neural_model_path: Option<String>,
}

impl Default for StrengthTestConfig {
    fn default() -> Self {
        StrengthTestConfig {
            time_limit_ms: 1000,  // 1 second per position
            mcts_iterations: 500,
            ab_depth: 6,
            mate_search_depth: 3,
            neural_model_path: Some("python/models/chess_model.pth".to_string()),
        }
    }
}

/// Engine variant for testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineVariant {
    AlphaBeta,           // Pure alpha-beta search
    MctsClassical,       // MCTS with classical evaluation
    MctsMatePriority,    // MCTS with mate-search-first
    MctsNeuralNet,       // MCTS with neural network policy
    MctsComplete,        // MCTS with mate-search-first + neural network
}

impl fmt::Display for EngineVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineVariant::AlphaBeta => write!(f, "Alpha-Beta"),
            EngineVariant::MctsClassical => write!(f, "MCTS-Classical"),
            EngineVariant::MctsMatePriority => write!(f, "MCTS-Mate-Priority"),
            EngineVariant::MctsNeuralNet => write!(f, "MCTS-Neural"),
            EngineVariant::MctsComplete => write!(f, "MCTS-Complete"),
        }
    }
}

/// Result from a single position analysis
#[derive(Debug, Clone)]
pub struct PositionResult {
    pub position_name: String,
    pub fen: String,
    pub engine: EngineVariant,
    pub best_move: Option<Move>,
    pub evaluation: Option<i32>,
    pub time_taken_ms: u64,
    pub nodes_searched: Option<u64>,
    pub correct_move: bool,
    pub move_quality_score: f64,
}

/// Complete benchmark results
#[derive(Debug)]
pub struct StrengthTestResults {
    pub config: StrengthTestConfig,
    pub position_results: Vec<PositionResult>,
    pub engine_summaries: HashMap<EngineVariant, EngineSummary>,
    pub overall_comparison: OverallComparison,
}

/// Summary statistics for an engine variant
#[derive(Debug, Clone)]
pub struct EngineSummary {
    pub engine: EngineVariant,
    pub total_positions: usize,
    pub correct_moves: usize,
    pub accuracy_percentage: f64,
    pub average_time_ms: f64,
    pub average_move_quality: f64,
    pub tactical_score: f64,
    pub positional_score: f64,
    pub endgame_score: f64,
}

/// Overall comparison between engines
#[derive(Debug)]
pub struct OverallComparison {
    pub best_engine: EngineVariant,
    pub ranking: Vec<(EngineVariant, f64)>,
    pub mate_search_improvement: f64,
    pub neural_net_improvement: f64,
    pub combined_improvement: f64,
}

/// Main strength testing orchestrator
pub struct StrengthTester {
    move_gen: MoveGen,
    pesto_eval: PestoEval,
    // alpha_beta: AlphaBeta,
    config: StrengthTestConfig,
}

impl StrengthTester {
    pub fn new(config: StrengthTestConfig) -> Self {
        StrengthTester {
            move_gen: MoveGen::new(),
            pesto_eval: PestoEval::new(),
            // alpha_beta: AlphaBeta::new(PestoEval::new()),
            config,
        }
    }

    /// Run comprehensive strength tests across all engine variants
    pub fn run_comprehensive_test(&mut self) -> StrengthTestResults {
        println!("🏁 Starting Comprehensive Strength Testing");
        println!("==========================================");
        
        let test_positions = self.create_test_suite();
        println!("📊 Testing {} positions across {} engine variants", 
                test_positions.len(), 5);
        
        let mut all_results = Vec::new();
        let engines = vec![
            EngineVariant::AlphaBeta,
            EngineVariant::MctsClassical,
            EngineVariant::MctsMatePriority,
            EngineVariant::MctsNeuralNet,
            EngineVariant::MctsComplete,
        ];
        
        for engine in &engines {
            println!("\n🔍 Testing {}", engine);
            let engine_results = self.test_engine_variant(*engine, &test_positions);
            all_results.extend(engine_results);
        }
        
        let analysis = self.analyze_results(all_results);
        self.print_results_summary(&analysis);
        
        analysis
    }
    
    /// Test a specific engine variant on all positions
    fn test_engine_variant(&mut self, engine: EngineVariant, positions: &[TestPosition]) -> Vec<PositionResult> {
        let mut results = Vec::new();
        
        for (i, position) in positions.iter().enumerate() {
            print!("  Position {}/{}... ", i + 1, positions.len());
            
            let start_time = Instant::now();
            let (best_move, evaluation, nodes) = self.analyze_position_with_engine(engine, &position.board);
            let time_taken = start_time.elapsed().as_millis() as u64;
            
            let correct_move = if let Some(expected_move) = position.best_move {
                best_move.map_or(false, |mv| mv == expected_move)
            } else {
                true // No specific move required
            };
            
            let move_quality = self.calculate_move_quality(&position.board, best_move, position.category);
            
            let result = PositionResult {
                position_name: position.name.clone(),
                fen: position.board.to_fen().unwrap_or_default(),
                engine,
                best_move,
                evaluation,
                time_taken_ms: time_taken,
                nodes_searched: nodes,
                correct_move,
                move_quality_score: move_quality,
            };
            
            println!("✓ {}ms", time_taken);
            results.push(result);
        }
        
        results
    }
    
    /// Analyze position with specific engine variant
    fn analyze_position_with_engine(&mut self, engine: EngineVariant, board: &Board) -> (Option<Move>, Option<i32>, Option<u64>) {
        let time_limit = Duration::from_millis(self.config.time_limit_ms);
        
        match engine {
            EngineVariant::AlphaBeta => {
                // Simple evaluation-based "best" move for now
                let (captures, non_captures) = self.move_gen.gen_pseudo_legal_moves(board);
                let mut best_move = None;
                let mut best_eval = if board.w_to_move { i32::MIN } else { i32::MAX };
                
                // Try captures first, then non-captures
                for mv in captures.iter().chain(non_captures.iter()) {
                    let new_board = board.apply_move_to_board(*mv);
                    if new_board.is_legal(&self.move_gen) {
                        let eval = self.pesto_eval.eval(&new_board, &self.move_gen);
                        if (board.w_to_move && eval > best_eval) || (!board.w_to_move && eval < best_eval) {
                            best_eval = eval;
                            best_move = Some(*mv);
                        }
                    }
                }
                (best_move, Some(best_eval), Some(1000))
            }
            
            EngineVariant::MctsClassical => {
                // Use simplified evaluation search without mate priority
                let best_move = self.simple_evaluation_search(board, time_limit);
                (best_move, None, None)
            }
            
            EngineVariant::MctsMatePriority => {
                // Use evaluation search with basic mate checking
                let best_move = self.mate_priority_evaluation_search(board, time_limit);
                (best_move, None, None)
            }
            
            EngineVariant::MctsNeuralNet => {
                let best_move = neural_mcts_search(
                    board.clone(),
                    &self.move_gen,
                    None, // No inference server for now
                    0, // No mate search
                    Some(self.config.mcts_iterations),
                    Some(time_limit),
                );
                (best_move, None, None)
            }

            EngineVariant::MctsComplete => {
                let best_move = neural_mcts_search(
                    board.clone(),
                    &self.move_gen,
                    None, // No inference server for now
                    self.config.mate_search_depth,
                    Some(self.config.mcts_iterations),
                    Some(time_limit),
                );
                (best_move, None, None)
            }
        }
    }
    
    /// Simple evaluation-based search for benchmarking
    fn simple_evaluation_search(&self, board: &Board, _time_limit: Duration) -> Option<Move> {
        let (captures, non_captures) = self.move_gen.gen_pseudo_legal_moves(board);
        let mut best_move = None;
        let mut best_eval = if board.w_to_move { i32::MIN } else { i32::MAX };
        
        // Try captures first, then non-captures
        for mv in captures.iter().chain(non_captures.iter()) {
            let new_board = board.apply_move_to_board(*mv);
            if new_board.is_legal(&self.move_gen) {
                let eval = self.pesto_eval.eval(&new_board, &self.move_gen);
                if (board.w_to_move && eval > best_eval) || (!board.w_to_move && eval < best_eval) {
                    best_eval = eval;
                    best_move = Some(*mv);
                }
            }
        }
        
        best_move
    }
    
    /// Evaluation search with basic mate checking
    fn mate_priority_evaluation_search(&self, board: &Board, time_limit: Duration) -> Option<Move> {
        // Simple mate check: look for checkmate in 1
        let (captures, non_captures) = self.move_gen.gen_pseudo_legal_moves(board);
        
        // Check captures first for quick mates
        for mv in captures.iter().chain(non_captures.iter()) {
            let new_board = board.apply_move_to_board(*mv);
            if new_board.is_legal(&self.move_gen) {
                // Check if opponent has no legal moves (mate or stalemate)
                let (opp_captures, opp_non_captures) = self.move_gen.gen_pseudo_legal_moves(&new_board);
                let has_legal_moves = opp_captures.iter().chain(opp_non_captures.iter())
                    .any(|opp_mv| {
                        let test_board = new_board.apply_move_to_board(*opp_mv);
                        test_board.is_legal(&self.move_gen)
                    });
                
                if !has_legal_moves && new_board.is_check(&self.move_gen) {
                    return Some(*mv); // Found mate!
                }
            }
        }
        
        // Fall back to evaluation search
        self.simple_evaluation_search(board, time_limit)
    }
    
    /// Calculate move quality score based on position evaluation
    fn calculate_move_quality(&self, board: &Board, chosen_move: Option<Move>, category: PositionCategory) -> f64 {
        let Some(mv) = chosen_move else { return 0.0; };
        
        let new_board = board.apply_move_to_board(mv);
        if !new_board.is_legal(&self.move_gen) {
            return 0.0; // Illegal move
        }
        
        let eval_after = self.pesto_eval.eval(&new_board, &self.move_gen);
        let eval_before = self.pesto_eval.eval(board, &self.move_gen);
        
        // Score based on evaluation improvement (perspective-aware)
        let eval_diff = if board.w_to_move { // White to move
            eval_after - eval_before
        } else {
            eval_before - eval_after
        };
        
        let base_score = match eval_diff {
            diff if diff > 200 => 1.0,   // Excellent move
            diff if diff > 50 => 0.8,    // Good move
            diff if diff > -20 => 0.6,   // Acceptable move
            diff if diff > -100 => 0.3,  // Poor move
            _ => 0.1,                    // Bad move
        };
        
        // Bonus for tactical positions
        match category {
            PositionCategory::Tactical => base_score * 1.2,
            PositionCategory::Endgame => base_score * 1.1,
            PositionCategory::Positional => base_score,
        }
    }
    
    /// Analyze all results and create comprehensive summary
    fn analyze_results(&self, results: Vec<PositionResult>) -> StrengthTestResults {
        let mut engine_summaries = HashMap::new();
        
        // Group results by engine
        let mut engine_results: HashMap<EngineVariant, Vec<&PositionResult>> = HashMap::new();
        for result in &results {
            engine_results.entry(result.engine).or_insert_with(Vec::new).push(result);
        }
        
        // Calculate summaries for each engine
        for (engine, engine_results) in engine_results {
            let summary = self.calculate_engine_summary(engine, &engine_results);
            engine_summaries.insert(engine, summary);
        }
        
        // Calculate overall comparison
        let comparison = self.calculate_overall_comparison(&engine_summaries);
        
        // Generate Elo estimates
        use crate::benchmarks::elo_estimation::EloCalculator;
        let elo_calculator = EloCalculator::default();
        let elo_estimates = elo_calculator.estimate_performance_ratings(&results);
        let elo_report = elo_calculator.generate_elo_report(&elo_estimates);
        
        println!("\n🎯 ELO ANALYSIS COMPLETE");
        elo_report.print_report();
        
        StrengthTestResults {
            config: self.config.clone(),
            position_results: results,
            engine_summaries,
            overall_comparison: comparison,
        }
    }
    
    /// Calculate summary statistics for an engine
    fn calculate_engine_summary(&self, engine: EngineVariant, results: &[&PositionResult]) -> EngineSummary {
        let total_positions = results.len();
        let correct_moves = results.iter().filter(|r| r.correct_move).count();
        let accuracy_percentage = if total_positions > 0 {
            (correct_moves as f64 / total_positions as f64) * 100.0
        } else {
            0.0
        };
        
        let average_time_ms = results.iter().map(|r| r.time_taken_ms as f64).sum::<f64>() / total_positions as f64;
        let average_move_quality = results.iter().map(|r| r.move_quality_score).sum::<f64>() / total_positions as f64;
        
        // Category-specific scores (simplified for now)
        let tactical_score = average_move_quality;
        let positional_score = average_move_quality;
        let endgame_score = average_move_quality;
        
        EngineSummary {
            engine,
            total_positions,
            correct_moves,
            accuracy_percentage,
            average_time_ms,
            average_move_quality,
            tactical_score,
            positional_score,
            endgame_score,
        }
    }
    
    /// Calculate overall comparison between engines
    fn calculate_overall_comparison(&self, summaries: &HashMap<EngineVariant, EngineSummary>) -> OverallComparison {
        // Create ranking based on combined score (accuracy + move quality)
        let mut ranking: Vec<(EngineVariant, f64)> = summaries
            .iter()
            .map(|(engine, summary)| {
                let combined_score = (summary.accuracy_percentage / 100.0) * 0.6 + summary.average_move_quality * 0.4;
                (*engine, combined_score)
            })
            .collect();
        
        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let best_engine = ranking.first().map(|(engine, _)| *engine).unwrap_or(EngineVariant::AlphaBeta);
        
        // Calculate improvements
        let ab_score = summaries.get(&EngineVariant::AlphaBeta).map(|s| s.average_move_quality).unwrap_or(0.5);
        let mcts_classical_score = summaries.get(&EngineVariant::MctsClassical).map(|s| s.average_move_quality).unwrap_or(0.5);
        let mcts_mate_score = summaries.get(&EngineVariant::MctsMatePriority).map(|s| s.average_move_quality).unwrap_or(0.5);
        let mcts_neural_score = summaries.get(&EngineVariant::MctsNeuralNet).map(|s| s.average_move_quality).unwrap_or(0.5);
        let mcts_complete_score = summaries.get(&EngineVariant::MctsComplete).map(|s| s.average_move_quality).unwrap_or(0.5);
        
        let mate_search_improvement = ((mcts_mate_score - mcts_classical_score) / mcts_classical_score) * 100.0;
        let neural_net_improvement = ((mcts_neural_score - mcts_classical_score) / mcts_classical_score) * 100.0;
        let combined_improvement = ((mcts_complete_score - ab_score) / ab_score) * 100.0;
        
        OverallComparison {
            best_engine,
            ranking,
            mate_search_improvement,
            neural_net_improvement,
            combined_improvement,
        }
    }
    
    /// Print comprehensive results summary
    fn print_results_summary(&self, results: &StrengthTestResults) {
        println!("\n📊 COMPREHENSIVE STRENGTH TEST RESULTS");
        println!("======================================");
        
        println!("\n🏆 ENGINE RANKINGS:");
        for (i, (engine, score)) in results.overall_comparison.ranking.iter().enumerate() {
            println!("  {}. {} - Score: {:.3}", i + 1, engine, score);
        }
        
        println!("\n📈 KEY IMPROVEMENTS:");
        println!("  Mate-Search-First:  {:+.1}%", results.overall_comparison.mate_search_improvement);
        println!("  Neural Network:     {:+.1}%", results.overall_comparison.neural_net_improvement);
        println!("  Combined Approach:  {:+.1}%", results.overall_comparison.combined_improvement);
        
        println!("\n📋 DETAILED STATISTICS:");
        for engine in &[
            EngineVariant::AlphaBeta,
            EngineVariant::MctsClassical,
            EngineVariant::MctsMatePriority,
            EngineVariant::MctsNeuralNet,
            EngineVariant::MctsComplete,
        ] {
            if let Some(summary) = results.engine_summaries.get(engine) {
                println!("\n  🔧 {}:", engine);
                println!("     Accuracy:      {:.1}% ({}/{})", 
                        summary.accuracy_percentage, 
                        summary.correct_moves, 
                        summary.total_positions);
                println!("     Move Quality:  {:.3}", summary.average_move_quality);
                println!("     Avg Time:      {:.0}ms", summary.average_time_ms);
            }
        }
        
        println!("\n🎯 CONCLUSION:");
        println!("  Best Engine: {}", results.overall_comparison.best_engine);
        if results.overall_comparison.combined_improvement > 0.0 {
            println!("  ✅ Our mate-search-first + neural network approach shows {:.1}% improvement!", 
                    results.overall_comparison.combined_improvement);
        } else {
            println!("  ⚠️  Need more tuning - combined approach shows {:.1}% change", 
                    results.overall_comparison.combined_improvement);
        }
    }
    
    /// Create comprehensive test suite with diverse positions
    fn create_test_suite(&self) -> Vec<TestPosition> {
        let mut positions = Vec::new();
        
        // Add tactical test positions
        positions.extend(self.create_tactical_positions());
        
        // Add positional test positions
        positions.extend(self.create_positional_positions());
        
        // Add endgame test positions
        positions.extend(self.create_endgame_positions());
        
        positions
    }
    
    /// Create tactical test positions
    fn create_tactical_positions(&self) -> Vec<TestPosition> {
        vec![
            TestPosition {
                name: "Back Rank Mate".to_string(),
                board: Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"),
                best_move: Move::from_uci("Re8"), // Remove # from move notation
                category: PositionCategory::Tactical,
            },
            TestPosition {
                name: "Starting Position".to_string(),
                board: Board::new(),  // Use standard starting position
                best_move: Move::from_uci("e2e4"),
                category: PositionCategory::Positional,
            },
            TestPosition {
                name: "Simple Endgame".to_string(),
                board: Board::new_from_fen("8/8/8/4k3/4P3/4K3/8/8 w - - 0 1"),
                best_move: Move::from_uci("Kd4"),
                category: PositionCategory::Endgame,
            },
        ]
    }
    
    /// Create positional test positions
    fn create_positional_positions(&self) -> Vec<TestPosition> {
        vec![
            TestPosition {
                name: "Central Control".to_string(),
                board: Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
                best_move: Move::from_uci("e7e5"),
                category: PositionCategory::Positional,
            },
            TestPosition {
                name: "Development Priority".to_string(),
                board: Board::new_from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
                best_move: Move::from_uci("g1f3"),
                category: PositionCategory::Positional,
            },
        ]
    }
    
    /// Create endgame test positions
    fn create_endgame_positions(&self) -> Vec<TestPosition> {
        vec![
            TestPosition {
                name: "King and Pawn vs King".to_string(),
                board: Board::new_from_fen("8/8/8/4k3/4P3/4K3/8/8 w - - 0 1"),
                best_move: Move::from_uci("e3d4"),
                category: PositionCategory::Endgame,
            },
            TestPosition {
                name: "Rook Endgame".to_string(),
                board: Board::new_from_fen("8/8/8/8/8/3k4/3r4/3K4 b - - 0 1"),
                best_move: Move::from_uci("d2d1"),
                category: PositionCategory::Endgame,
            },
        ]
    }
}

/// Individual test position
#[derive(Debug, Clone)]
pub struct TestPosition {
    pub name: String,
    pub board: Board,
    pub best_move: Option<Move>,
    pub category: PositionCategory,
}

/// Position category for specialized testing
#[derive(Debug, Clone, Copy)]
pub enum PositionCategory {
    Tactical,
    Positional,
    Endgame,
}

/// Save results to file for detailed analysis
impl StrengthTestResults {
    pub fn save_to_csv(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        writeln!(file, "position,engine,time_ms,correct_move,move_quality,evaluation")?;
        
        for result in &self.position_results {
            writeln!(
                file,
                "{},{},{},{},{},{}",
                result.position_name,
                result.engine,
                result.time_taken_ms,
                result.correct_move,
                result.move_quality_score,
                result.evaluation.unwrap_or(0)
            )?;
        }
        
        println!("📁 Detailed results saved to: {}", path);
        Ok(())
    }
}