//! Elo Tournament Framework for Statistical Validation
//! 
//! Runs statistically significant head-to-head matches between engine variants
//! to produce confidence intervals on Elo differences.

use crate::board::Board;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use crate::search::iterative_deepening::iterative_deepening_ab_search;
use crate::transposition::TranspositionTable;
use crate::mcts::inference_server::InferenceServer;
use crate::boardstack::BoardStack;
use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;
use rand::prelude::*;
use rand::rngs::StdRng;

/// Engine variant for tournament play
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TournamentEngine {
    /// Pure alpha-beta search (baseline)
    AlphaBeta { depth: i32 },
    /// MCTS without mate search or tactical priority (ablation)
    PureMcts { iterations: u32 },
    /// MCTS with only Tier 1 (mate search gate)
    MctsTier1Only { iterations: u32, mate_depth: i32 },
    /// MCTS with Tier 1 + Tier 2 (tactical grafting)
    MctsTier1And2 { iterations: u32, mate_depth: i32 },
    /// Full Tactical-First MCTS (all 3 tiers)
    TacticalMctsFull { iterations: u32, mate_depth: i32 },
}

impl TournamentEngine {
    pub fn name(&self) -> &'static str {
        match self {
            TournamentEngine::AlphaBeta { .. } => "AlphaBeta",
            TournamentEngine::PureMcts { .. } => "PureMCTS",
            TournamentEngine::MctsTier1Only { .. } => "MCTS+T1",
            TournamentEngine::MctsTier1And2 { .. } => "MCTS+T1+T2",
            TournamentEngine::TacticalMctsFull { .. } => "TacticalMCTS",
        }
    }
}

/// Result of a single game
#[derive(Debug, Clone, Copy)]
pub enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

impl GameResult {
    pub fn score_for_white(&self) -> f64 {
        match self {
            GameResult::WhiteWin => 1.0,
            GameResult::BlackWin => 0.0,
            GameResult::Draw => 0.5,
        }
    }
}

/// Match result between two engines
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub engine_a: TournamentEngine,
    pub engine_b: TournamentEngine,
    pub games: Vec<(GameResult, u32)>, // (result, move count)
    pub a_wins: u32,
    pub b_wins: u32,
    pub draws: u32,
}

impl MatchResult {
    pub fn total_games(&self) -> u32 {
        self.a_wins + self.b_wins + self.draws
    }
    
    pub fn a_score(&self) -> f64 {
        let total = self.total_games();
        if total == 0 { return 0.5; }
        (self.a_wins as f64 + 0.5 * self.draws as f64) / total as f64
    }
    
    /// Calculate Elo difference using the standard formula
    /// Elo_diff = -400 * log10(1/score - 1)
    pub fn elo_difference(&self) -> f64 {
        let score = self.a_score();
        if score <= 0.0 { return -800.0; }
        if score >= 1.0 { return 800.0; }
        -400.0 * (1.0 / score - 1.0).log10()
    }
    
    /// Calculate 95% confidence interval using Wilson score interval
    pub fn elo_confidence_interval(&self) -> (f64, f64) {
        let n = self.total_games() as f64;
        if n == 0.0 { return (-800.0, 800.0); }
        let p = self.a_score();
        let z = 1.96; // 95% CI
        
        // Wilson score interval
        let denominator = 1.0 + z * z / n;
        let center = (p + z * z / (2.0 * n)) / denominator;
        let margin = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
        
        let low_score = (center - margin).max(0.001);
        let high_score = (center + margin).min(0.999);
        
        let elo_low = -400.0 * (1.0 / low_score - 1.0).log10();
        let elo_high = -400.0 * (1.0 / high_score - 1.0).log10();
        
        (elo_low, elo_high)
    }
}

/// Tournament configuration
#[derive(Debug, Clone)]
pub struct TournamentConfig {
    /// Number of games per engine pair
    pub games_per_pair: u32,
    /// Maximum moves per game before adjudication
    pub max_moves: u32,
    /// Time limit per move (ms)
    pub time_per_move_ms: u64,
    /// Opening book positions (FENs)
    pub opening_book: Vec<String>,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        TournamentConfig {
            games_per_pair: 100,
            max_moves: 200,
            time_per_move_ms: 1000,
            opening_book: vec![
                // Standard opening positions
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                // Italian Game
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3".to_string(),
                // Sicilian Defense
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2".to_string(),
                // Queen's Gambit
                "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2".to_string(),
                // French Defense
                "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".to_string(),
            ],
            seed: 42,
        }
    }
}

/// Main tournament runner
pub struct EloTournament {
    config: TournamentConfig,
    move_gen: MoveGen,
    pesto_eval: PestoEval,
    rng: StdRng,
}

impl EloTournament {
    pub fn new(config: TournamentConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        EloTournament {
            config,
            move_gen: MoveGen::new(),
            pesto_eval: PestoEval::new(),
            rng,
        }
    }
    
    /// Run a complete tournament between all engine pairs
    pub fn run_full_tournament(&mut self, engines: &[TournamentEngine]) -> TournamentResults {
        println!("🏆 Starting Elo Tournament");
        println!("   Engines: {}", engines.len());
        println!("   Games per pair: {}", self.config.games_per_pair);
        println!("   Total games: {}", engines.len() * (engines.len() - 1) * self.config.games_per_pair as usize / 2);
        println!();
        
        let mut results = TournamentResults::new();
        
        for i in 0..engines.len() {
            for j in (i + 1)..engines.len() {
                let match_result = self.run_match(engines[i], engines[j]);
                println!(
                    "  {} vs {}: +{}-{}={} (Elo: {:.0} [{:.0}, {:.0}])",
                    engines[i].name(),
                    engines[j].name(),
                    match_result.a_wins,
                    match_result.b_wins,
                    match_result.draws,
                    match_result.elo_difference(),
                    match_result.elo_confidence_interval().0,
                    match_result.elo_confidence_interval().1,
                );
                results.matches.push(match_result);
            }
        }
        
        results.calculate_ratings();
        results
    }
    
    /// Run a match between two engines
    fn run_match(&mut self, engine_a: TournamentEngine, engine_b: TournamentEngine) -> MatchResult {
        let mut result = MatchResult {
            engine_a,
            engine_b,
            games: Vec::new(),
            a_wins: 0,
            b_wins: 0,
            draws: 0,
        };
        
        for game_num in 0..self.config.games_per_pair {
            // Alternate colors
            let (white, black) = if game_num % 2 == 0 {
                (engine_a, engine_b)
            } else {
                (engine_b, engine_a)
            };
            
            // Select opening position
            let opening_idx = self.rng.gen_range(0..self.config.opening_book.len());
            let opening_fen = self.config.opening_book[opening_idx].clone();
            
            let (game_result, move_count) = self.play_game(white, black, &opening_fen);
            
            // Translate result to engine_a's perspective
            let a_result = if game_num % 2 == 0 {
                game_result
            } else {
                match game_result {
                    GameResult::WhiteWin => GameResult::BlackWin,
                    GameResult::BlackWin => GameResult::WhiteWin,
                    GameResult::Draw => GameResult::Draw,
                }
            };
            
            match a_result {
                GameResult::WhiteWin => result.a_wins += 1,
                GameResult::BlackWin => result.b_wins += 1,
                GameResult::Draw => result.draws += 1,
            }
            
            result.games.push((a_result, move_count));
        }
        
        result
    }
    
    /// Play a single game between two engines
    fn play_game(
        &mut self,
        white: TournamentEngine,
        black: TournamentEngine,
        opening_fen: &str,
    ) -> (GameResult, u32) {
        let mut board_stack = BoardStack::with_board(Board::new_from_fen(opening_fen));
        let mut move_count = 0;
        
        loop {
            // Check for game termination
            let (is_checkmate, is_stalemate) = board_stack.current_state().is_checkmate_or_stalemate(&self.move_gen);
            
            if is_checkmate {
                // Side to move is checkmated
                return if board_stack.current_state().w_to_move {
                    (GameResult::BlackWin, move_count)
                } else {
                    (GameResult::WhiteWin, move_count)
                };
            }
            
            if is_stalemate || board_stack.is_draw_by_repetition() || move_count >= self.config.max_moves {
                return (GameResult::Draw, move_count);
            }
            
            // Get move from appropriate engine
            let engine = if board_stack.current_state().w_to_move { white } else { black };
            let best_move = self.get_engine_move(engine, &board_stack);
            
            if let Some(mv) = best_move {
                board_stack.make_move(mv);
                move_count += 1;
            } else {
                // No legal move - shouldn't happen if checkmate/stalemate check works
                return (GameResult::Draw, move_count);
            }
        }
    }
    
    /// Get a move from an engine variant
    fn get_engine_move(&self, engine: TournamentEngine, board_stack: &BoardStack) -> Option<Move> {
        match engine {
            TournamentEngine::AlphaBeta { depth } => {
                let mut stack = board_stack.clone();
                let mut tt = TranspositionTable::new();
                let (_, _, best_move, _) = iterative_deepening_ab_search(
                    &mut stack,
                    &self.move_gen,
                    &self.pesto_eval,
                    &mut tt,
                    depth,
                    16, // q_search_max_depth
                    Some(Duration::from_millis(self.config.time_per_move_ms)),
                    false, // verbose
                );
                Some(best_move)
            }
            TournamentEngine::PureMcts { iterations } => {
                let config = TacticalMctsConfig {
                    max_iterations: iterations,
                    time_limit: Duration::from_millis(self.config.time_per_move_ms),
                    mate_search_depth: 0, // Disabled
                    exploration_constant: 1.414,
                    use_neural_policy: false,
                    inference_server: None,
                    logger: None,
                    enable_tier1_gate: false,
                    enable_tier2_graft: false,
                    enable_tier3_neural: false,
                    enable_q_init: false,
                    ..Default::default()
                };
                let (best, _, _) = tactical_mcts_search(
                    board_stack.current_state().clone(),
                    &self.move_gen,
                    &self.pesto_eval,
                    &mut None,
                    config,
                );
                best
            }
            TournamentEngine::MctsTier1Only { iterations, mate_depth } => {
                let config = TacticalMctsConfig {
                    max_iterations: iterations,
                    time_limit: Duration::from_millis(self.config.time_per_move_ms),
                    mate_search_depth: mate_depth,
                    exploration_constant: 1.414,
                    use_neural_policy: false,
                    inference_server: None,
                    logger: None,
                    enable_tier1_gate: true,
                    enable_tier2_graft: false,
                    enable_tier3_neural: false,
                    enable_q_init: false,
                    ..Default::default()
                };
                let (best, _, _) = tactical_mcts_search(
                    board_stack.current_state().clone(),
                    &self.move_gen,
                    &self.pesto_eval,
                    &mut None,
                    config,
                );
                best
            }
            TournamentEngine::MctsTier1And2 { iterations, mate_depth } => {
                let config = TacticalMctsConfig {
                    max_iterations: iterations,
                    time_limit: Duration::from_millis(self.config.time_per_move_ms),
                    mate_search_depth: mate_depth,
                    exploration_constant: 1.414,
                    use_neural_policy: false, // No NN = Tier 3 disabled effectively
                    inference_server: None,
                    logger: None,
                    enable_tier1_gate: true,
                    enable_tier2_graft: true,
                    enable_tier3_neural: false,
                    enable_q_init: true,
                    ..Default::default()
                };
                let (best, _, _) = tactical_mcts_search(
                    board_stack.current_state().clone(),
                    &self.move_gen,
                    &self.pesto_eval,
                    &mut None,
                    config,
                );
                best
            }
            TournamentEngine::TacticalMctsFull { iterations, mate_depth } => {
                let server = InferenceServer::new_mock(); // Or real NN
                let config = TacticalMctsConfig {
                    max_iterations: iterations,
                    time_limit: Duration::from_millis(self.config.time_per_move_ms),
                    mate_search_depth: mate_depth,
                    exploration_constant: 1.414,
                    use_neural_policy: true,
                    inference_server: Some(Arc::new(server)),
                    logger: None,
                    enable_tier1_gate: true,
                    enable_tier2_graft: true,
                    enable_tier3_neural: true,
                    enable_q_init: true,
                    ..Default::default()
                };
                let (best, _, _) = tactical_mcts_search(
                    board_stack.current_state().clone(),
                    &self.move_gen,
                    &self.pesto_eval,
                    &mut None,
                    config,
                );
                best
            }
        }
    }
}

/// Aggregated tournament results
#[derive(Debug)]
pub struct TournamentResults {
    pub matches: Vec<MatchResult>,
    pub ratings: HashMap<String, f64>,
}

impl TournamentResults {
    pub fn new() -> Self {
        TournamentResults {
            matches: Vec::new(),
            ratings: HashMap::new(),
        }
    }
    
    /// Calculate Elo ratings using iterative anchoring
    pub fn calculate_ratings(&mut self) {
        // Simple approach: anchor AlphaBeta at 1500, derive others
        let base_rating = 1500.0;
        
        // Find all unique engines
        let mut engines: Vec<String> = Vec::new();
        for m in &self.matches {
            let a_name = m.engine_a.name().to_string();
            let b_name = m.engine_b.name().to_string();
            if !engines.contains(&a_name) { engines.push(a_name); }
            if !engines.contains(&b_name) { engines.push(b_name); }
        }
        
        // Initialize ratings
        for engine in &engines {
            self.ratings.insert(engine.clone(), base_rating);
        }
        
        // Iteratively adjust ratings based on match results
        for _ in 0..10 {
            for m in &self.matches {
                let a_name = m.engine_a.name().to_string();
                let b_name = m.engine_b.name().to_string();
                
                let elo_diff = m.elo_difference();
                let current_diff = self.ratings[&a_name] - self.ratings[&b_name];
                let adjustment = (elo_diff - current_diff) * 0.1;
                
                if let Some(rating) = self.ratings.get_mut(&a_name) {
                    *rating += adjustment;
                }
                if let Some(rating) = self.ratings.get_mut(&b_name) {
                    *rating -= adjustment;
                }
            }
        }
    }
    
    /// Generate LaTeX table for paper
    pub fn to_latex_table(&self) -> String {
        let mut output = String::new();
        output.push_str(r"\begin{table}[h]");
        output.push_str("\n");
        output.push_str(r"\centering");
        output.push_str("\n");
        output.push_str(r"\begin{tabular}{lrrrrr}");
        output.push_str("\n");
        output.push_str(r"\toprule");
        output.push_str("\n");
        output.push_str(r"Engine A & Engine B & W & D & L & Elo Diff (95\% CI) \\");
        output.push_str("\n");
        output.push_str(r"\midrule");
        output.push_str("\n");
        
        for m in &self.matches {
            let (lo, hi) = m.elo_confidence_interval();
            output.push_str(&format!(
                r"{} & {} & {} & {} & {} & {:.0} [{:.0}, {:.0}] \\",
                m.engine_a.name(),
                m.engine_b.name(),
                m.a_wins,
                m.draws,
                m.b_wins,
                m.elo_difference(),
                lo,
                hi,
            ));
            output.push_str("\n");
        }
        
        output.push_str(r"\bottomrule");
        output.push_str("\n");
        output.push_str(r"\end{tabular}");
        output.push_str("\n");
        output.push_str(r"\caption{Head-to-head results between engine variants}");
        output.push_str("\n");
        output.push_str(r"\label{tab:elo_results}");
        output.push_str("\n");
        output.push_str(r"\end{table}");
        output.push_str("\n");
        
        output
    }
    
    /// Generate CSV for analysis
    pub fn to_csv(&self) -> String {
        let mut output = String::from("engine_a,engine_b,wins_a,draws,wins_b,elo_diff,ci_low,ci_high\n");
        
        for m in &self.matches {
            let (lo, hi) = m.elo_confidence_interval();
            output.push_str(&format!(
                "{},{},{},{},{},{:.1},{:.1},{:.1}\n",
                m.engine_a.name(),
                m.engine_b.name(),
                m.a_wins,
                m.draws,
                m.b_wins,
                m.elo_difference(),
                lo,
                hi,
            ));
        }
        
        output
    }
}
