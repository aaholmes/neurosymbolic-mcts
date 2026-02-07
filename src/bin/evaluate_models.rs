//! Model Evaluator Binary
//!
//! Plays N games between a candidate model and the current best model,
//! reporting win rate and whether the candidate should be accepted.
//!
//! Usage: evaluate_models <candidate.pt> <current.pt> <num_games> <simulations> [--threshold 0.55]

use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search_with_tt, TacticalMctsConfig};
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::transposition::TranspositionTable;
use std::time::Duration;

/// Result of a single evaluation game.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameResult {
    CandidateWin,
    CurrentWin,
    Draw,
}

/// Aggregate evaluation results.
#[derive(Debug)]
pub struct EvalResults {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl EvalResults {
    pub fn win_rate(&self) -> f64 {
        let total = self.wins + self.losses + self.draws;
        if total == 0 {
            return 0.0;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / total as f64
    }

    pub fn accepted(&self, threshold: f64) -> bool {
        self.win_rate() >= threshold
    }
}

/// Play a single evaluation game between two models.
/// When `candidate_is_white` is true, the candidate plays White.
pub fn play_evaluation_game(
    candidate_nn: &mut Option<NeuralNetPolicy>,
    current_nn: &mut Option<NeuralNetPolicy>,
    candidate_is_white: bool,
    simulations: u32,
) -> GameResult {
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();

    let config = TacticalMctsConfig {
        max_iterations: simulations,
        time_limit: Duration::from_secs(120),
        mate_search_depth: 5,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: None,
        logger: None,
        // No noise for evaluation
        dirichlet_alpha: 0.0,
        dirichlet_epsilon: 0.0,
        ..Default::default()
    };

    let mut board_stack = BoardStack::new();
    let mut move_count = 0;
    let mut tt_candidate = TranspositionTable::new();
    let mut tt_current = TranspositionTable::new();

    loop {
        let board = board_stack.current_state().clone();
        let white_to_move = board.w_to_move;

        // Determine which model plays this turn
        let candidate_turn = (white_to_move && candidate_is_white)
            || (!white_to_move && !candidate_is_white);

        let (best_move, _stats, _root) = if candidate_turn {
            tactical_mcts_search_with_tt(
                board.clone(),
                &move_gen,
                &pesto_eval,
                candidate_nn,
                config.clone(),
                &mut tt_candidate,
            )
        } else {
            tactical_mcts_search_with_tt(
                board.clone(),
                &move_gen,
                &pesto_eval,
                current_nn,
                config.clone(),
                &mut tt_current,
            )
        };

        match best_move {
            None => break,
            Some(mv) => {
                board_stack.make_move(mv);
                move_count += 1;
            }
        }

        // Check termination conditions
        if board_stack.is_draw_by_repetition() {
            break;
        }
        if board_stack.current_state().halfmove_clock() >= 100 {
            break;
        }
        if move_count > 200 {
            break;
        }

        let (mate, stalemate) = board_stack.current_state().is_checkmate_or_stalemate(&move_gen);
        if mate || stalemate {
            break;
        }
    }

    // Determine outcome
    let final_board = board_stack.current_state();
    let (mate, stalemate) = final_board.is_checkmate_or_stalemate(&move_gen);
    let is_repetition = board_stack.is_draw_by_repetition();
    let is_50_move = final_board.halfmove_clock() >= 100;

    if mate {
        // Side to move is in checkmate, so the other side won
        let white_wins = !final_board.w_to_move;
        if (white_wins && candidate_is_white) || (!white_wins && !candidate_is_white) {
            GameResult::CandidateWin
        } else {
            GameResult::CurrentWin
        }
    } else if stalemate || is_repetition || is_50_move || move_count > 200 {
        GameResult::Draw
    } else {
        // No legal moves but not mate/stalemate — treat as draw
        GameResult::Draw
    }
}

/// Run a full evaluation match between two models.
pub fn evaluate_models(
    candidate_path: &str,
    current_path: &str,
    num_games: u32,
    simulations: u32,
) -> EvalResults {
    let mut results = EvalResults {
        wins: 0,
        losses: 0,
        draws: 0,
    };

    for game_idx in 0..num_games {
        // Alternate colors each game
        let candidate_is_white = game_idx % 2 == 0;

        // Load fresh models for each game (avoids stale cache)
        let mut candidate_nn = {
            let mut nn = NeuralNetPolicy::new();
            if let Err(e) = nn.load(candidate_path) {
                eprintln!("Failed to load candidate model: {}", e);
                return results;
            }
            Some(nn)
        };

        let mut current_nn = {
            let mut nn = NeuralNetPolicy::new();
            if let Err(e) = nn.load(current_path) {
                eprintln!("Failed to load current model: {}", e);
                return results;
            }
            Some(nn)
        };

        let result = play_evaluation_game(
            &mut candidate_nn,
            &mut current_nn,
            candidate_is_white,
            simulations,
        );

        match result {
            GameResult::CandidateWin => results.wins += 1,
            GameResult::CurrentWin => results.losses += 1,
            GameResult::Draw => results.draws += 1,
        }

        let color_str = if candidate_is_white { "White" } else { "Black" };
        eprintln!(
            "Game {}/{}: Candidate ({}) -> {:?} [W:{} L:{} D:{}]",
            game_idx + 1,
            num_games,
            color_str,
            result,
            results.wins,
            results.losses,
            results.draws
        );
    }

    results
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 5 {
        eprintln!("Usage: evaluate_models <candidate.pt> <current.pt> <num_games> <simulations> [--threshold 0.55]");
        std::process::exit(2);
    }

    let candidate_path = &args[1];
    let current_path = &args[2];
    let num_games: u32 = args[3].parse().expect("num_games must be a number");
    let simulations: u32 = args[4].parse().expect("simulations must be a number");

    let threshold: f64 = args
        .iter()
        .position(|a| a == "--threshold")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.55);

    eprintln!("Evaluating: {} vs {}", candidate_path, current_path);
    eprintln!("Games: {}, Sims: {}, Threshold: {}", num_games, simulations, threshold);

    let results = evaluate_models(candidate_path, current_path, num_games, simulations);
    let win_rate = results.win_rate();
    let accepted = results.accepted(threshold);

    // Machine-readable output on stdout
    println!(
        "WINS={} LOSSES={} DRAWS={} WINRATE={:.4} ACCEPTED={}",
        results.wins,
        results.losses,
        results.draws,
        win_rate,
        accepted
    );

    // Exit code: 0 = accepted, 1 = rejected
    std::process::exit(if accepted { 0 } else { 1 });
}
