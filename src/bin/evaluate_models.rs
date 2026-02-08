//! Model Evaluator Binary
//!
//! Plays N games between a candidate model and the current best model,
//! reporting win rate and whether the candidate should be accepted.
//!
//! Usage: evaluate_models <candidate.pt> <current.pt> <num_games> <simulations> [--threshold 0.55]

use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::mcts::{tactical_mcts_search_with_tt, MctsNode, TacticalMctsConfig, InferenceServer};
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::transposition::TranspositionTable;
use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
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

/// Select a move for evaluation: deterministic for forced wins, (counts-1) sampling otherwise.
fn select_eval_move(root: &Rc<RefCell<MctsNode>>) -> Option<Move> {
    let root_ref = root.borrow();

    // 1. If any child is a forced win (terminal_or_mate_value < -0.5 from child's STM = we win),
    //    pick it deterministically. If multiple, prefer the one with the strongest signal.
    let mut best_win: Option<(Move, f64)> = None;
    for child in &root_ref.children {
        let cr = child.borrow();
        if let Some(v) = cr.terminal_or_mate_value {
            if v < -0.5 {
                // This is a forced win for us; lower value = more decisive
                if let Some(mv) = cr.action {
                    if best_win.is_none() || v < best_win.unwrap().1 {
                        best_win = Some((mv, v));
                    }
                }
            }
        }
    }
    if let Some((mv, _)) = best_win {
        return Some(mv);
    }

    // 2. Check root mate_move (from mate search gate)
    if let Some(mate_mv) = root_ref.mate_move {
        return Some(mate_mv);
    }

    // 3. Otherwise, sample proportionally from (visits - 1)
    let visit_pairs: Vec<(Move, u32)> = root_ref.children.iter()
        .filter_map(|c| {
            let cr = c.borrow();
            cr.action.map(|mv| (mv, cr.visits))
        })
        .collect();

    sample_proportional(&visit_pairs)
}

/// Sample a move proportionally from visit counts (temperature = 1, counts-1).
fn sample_proportional(policy: &[(Move, u32)]) -> Option<Move> {
    let total: u32 = policy.iter().map(|(_, v)| v.saturating_sub(1)).sum();
    if total == 0 {
        // All moves have 0 or 1 visit — fall back to most-visited
        return policy.iter().max_by_key(|(_, v)| *v).map(|(mv, _)| *mv);
    }
    let threshold = rand::thread_rng().gen_range(0..total);
    let mut cumulative = 0u32;
    for (mv, visits) in policy {
        cumulative += visits.saturating_sub(1);
        if cumulative > threshold {
            return Some(*mv);
        }
    }
    policy.last().map(|(mv, _)| *mv)
}

/// Play a single evaluation game between two models.
/// When `candidate_is_white` is true, the candidate plays White.
pub fn play_evaluation_game(
    candidate_nn: &mut Option<NeuralNetPolicy>,
    current_nn: &mut Option<NeuralNetPolicy>,
    candidate_is_white: bool,
    simulations: u32,
) -> GameResult {
    play_evaluation_game_koth(candidate_nn, current_nn, candidate_is_white, simulations, false, true, true)
}

pub fn play_evaluation_game_koth(
    candidate_nn: &mut Option<NeuralNetPolicy>,
    current_nn: &mut Option<NeuralNetPolicy>,
    candidate_is_white: bool,
    simulations: u32,
    enable_koth: bool,
    enable_tier1: bool,
    enable_material: bool,
) -> GameResult {
    let move_gen = MoveGen::new();

    // Create InferenceServers from NNs (takes ownership via .take())
    let candidate_server: Option<Arc<InferenceServer>> = candidate_nn.take()
        .map(|nn| Arc::new(InferenceServer::new(nn, 1)));
    let current_server: Option<Arc<InferenceServer>> = current_nn.take()
        .map(|nn| Arc::new(InferenceServer::new(nn, 1)));

    let candidate_has_nn = candidate_server.is_some();
    let current_has_nn = current_server.is_some();

    let config_candidate = TacticalMctsConfig {
        max_iterations: simulations,
        time_limit: Duration::from_secs(120),
        mate_search_depth: if enable_tier1 { 5 } else { 0 },
        exploration_constant: 1.414,
        use_neural_policy: candidate_has_nn,
        inference_server: candidate_server,
        logger: None,
        dirichlet_alpha: 0.0,
        dirichlet_epsilon: 0.0,
        enable_koth,
        enable_tier1_gate: enable_tier1,
        enable_material_value: enable_material,
        enable_tier3_neural: candidate_has_nn,
        ..Default::default()
    };

    let config_current = TacticalMctsConfig {
        max_iterations: simulations,
        time_limit: Duration::from_secs(120),
        mate_search_depth: if enable_tier1 { 5 } else { 0 },
        exploration_constant: 1.414,
        use_neural_policy: current_has_nn,
        inference_server: current_server,
        logger: None,
        dirichlet_alpha: 0.0,
        dirichlet_epsilon: 0.0,
        enable_koth,
        enable_tier1_gate: enable_tier1,
        enable_material_value: enable_material,
        enable_tier3_neural: current_has_nn,
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

        let (_best_move, _stats, root) = if candidate_turn {
            tactical_mcts_search_with_tt(
                board.clone(),
                &move_gen,
                &mut None,
                config_candidate.clone(),
                &mut tt_candidate,
            )
        } else {
            tactical_mcts_search_with_tt(
                board.clone(),
                &move_gen,
                &mut None,
                config_current.clone(),
                &mut tt_current,
            )
        };

        // Use (counts-1) sampling for variety, but play forced wins deterministically
        let selected_move = select_eval_move(&root);
        match selected_move {
            None => break,
            Some(mv) => {
                board_stack.make_move(mv);
                move_count += 1;
            }
        }

        // Check KOTH win after each move
        if enable_koth {
            let (white_won, black_won) = board_stack.current_state().is_koth_win();
            if white_won || black_won {
                break;
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
    let (koth_white, koth_black) = if enable_koth {
        final_board.is_koth_win()
    } else {
        (false, false)
    };

    if koth_white || koth_black {
        // KOTH win: determine which side won
        let white_wins = koth_white;
        if (white_wins && candidate_is_white) || (!white_wins && !candidate_is_white) {
            GameResult::CandidateWin
        } else {
            GameResult::CurrentWin
        }
    } else if mate {
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
    evaluate_models_koth(candidate_path, current_path, num_games, simulations, false, true, true)
}

/// Run a full evaluation match between two models with optional KOTH mode.
pub fn evaluate_models_koth(
    candidate_path: &str,
    current_path: &str,
    num_games: u32,
    simulations: u32,
    enable_koth: bool,
    enable_tier1: bool,
    enable_material: bool,
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

        let result = play_evaluation_game_koth(
            &mut candidate_nn,
            &mut current_nn,
            candidate_is_white,
            simulations,
            enable_koth,
            enable_tier1,
            enable_material,
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

    let enable_koth = args.iter().any(|a| a == "--enable-koth");
    let enable_tier1 = !args.iter().any(|a| a == "--disable-tier1");
    let enable_material = !args.iter().any(|a| a == "--disable-material");

    eprintln!("Evaluating: {} vs {}", candidate_path, current_path);
    eprintln!("Games: {}, Sims: {}, Threshold: {}, KOTH: {}, Tier1: {}, Material: {}",
              num_games, simulations, threshold, enable_koth, enable_tier1, enable_material);

    let results = evaluate_models_koth(candidate_path, current_path, num_games, simulations, enable_koth, enable_tier1, enable_material);
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
