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
use kingfisher::mcts::sprt::{SprtConfig, SprtState, SprtResult};
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::search::quiescence::forced_material_balance;
use kingfisher::tensor::move_to_index;
use kingfisher::training_data::{TrainingSample, save_binary_data};
use kingfisher::transposition::TranspositionTable;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;

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

/// Result of a single evaluation game with optional training data.
pub struct EvalGameData {
    pub result: GameResult,
    pub candidate_samples: Vec<TrainingSample>,
    pub current_samples: Vec<TrainingSample>,
}

/// Number of half-moves using proportional sampling for opening diversity.
/// After this, switch to greedy (most-visited) for strength measurement.
const EVAL_EXPLORATION_PLIES: u32 = 10;

/// Select a move for evaluation: deterministic for forced wins,
/// proportional sampling for the first few moves, then greedy.
fn select_eval_move(root: &Rc<RefCell<MctsNode>>, rng: &mut impl Rng, move_count: u32) -> Option<Move> {
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

    // 3. Collect visit counts
    let visit_pairs: Vec<(Move, u32)> = root_ref.children.iter()
        .filter_map(|c| {
            let cr = c.borrow();
            cr.action.map(|mv| (mv, cr.visits))
        })
        .collect();

    if move_count < EVAL_EXPLORATION_PLIES {
        // Early game: proportional sampling for opening diversity
        sample_proportional(&visit_pairs, rng)
    } else {
        // Rest of game: greedy (most-visited) for strength measurement
        visit_pairs.iter().max_by_key(|(_, v)| *v).map(|(mv, _)| *mv)
    }
}

/// Sample a move proportionally from visit counts (temperature = 1, counts-1).
fn sample_proportional(policy: &[(Move, u32)], rng: &mut impl Rng) -> Option<Move> {
    let total: u32 = policy.iter().map(|(_, v)| v.saturating_sub(1)).sum();
    if total == 0 {
        // All moves have 0 or 1 visit — fall back to most-visited
        return policy.iter().max_by_key(|(_, v)| *v).map(|(mv, _)| *mv);
    }
    let threshold = rng.gen_range(0..total);
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
    candidate_server: &Option<Arc<InferenceServer>>,
    current_server: &Option<Arc<InferenceServer>>,
    candidate_is_white: bool,
    simulations: u32,
) -> GameResult {
    play_evaluation_game_koth(candidate_server, current_server, candidate_is_white, simulations, false, true, true, 0)
}

pub fn play_evaluation_game_koth(
    candidate_server: &Option<Arc<InferenceServer>>,
    current_server: &Option<Arc<InferenceServer>>,
    candidate_is_white: bool,
    simulations: u32,
    enable_koth: bool,
    enable_tier1: bool,
    enable_material: bool,
    game_seed: u64,
) -> GameResult {
    play_evaluation_game_with_servers(
        candidate_server, current_server, candidate_is_white, simulations,
        enable_koth, enable_tier1, enable_material, game_seed, false,
    ).result
}

/// Play a single evaluation game using shared InferenceServers (no ownership transfer).
pub fn play_evaluation_game_with_servers(
    candidate_server: &Option<Arc<InferenceServer>>,
    current_server: &Option<Arc<InferenceServer>>,
    candidate_is_white: bool,
    simulations: u32,
    enable_koth: bool,
    enable_tier1: bool,
    enable_material: bool,
    game_seed: u64,
    collect_training_data: bool,
) -> EvalGameData {
    let mut rng = StdRng::seed_from_u64(game_seed);
    let move_gen = MoveGen::new();

    let candidate_has_nn = candidate_server.is_some();
    let current_has_nn = current_server.is_some();

    let config_candidate = TacticalMctsConfig {
        max_iterations: simulations,
        time_limit: Duration::from_secs(120),
        mate_search_depth: if enable_tier1 { 5 } else { 0 },
        exploration_constant: 1.414,
        use_neural_policy: candidate_has_nn,
        inference_server: candidate_server.clone(),
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
        inference_server: current_server.clone(),
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
    let mut game_moves: Vec<String> = Vec::new();
    let mut tt_candidate = TranspositionTable::new();
    let mut tt_current = TranspositionTable::new();

    let mut candidate_samples: Vec<TrainingSample> = Vec::new();
    let mut current_samples: Vec<TrainingSample> = Vec::new();

    loop {
        let board = board_stack.current_state().clone();
        let white_to_move = board.w_to_move;

        let candidate_turn = (white_to_move && candidate_is_white)
            || (!white_to_move && !candidate_is_white);

        let (_best_move, _stats, root) = if candidate_turn {
            tactical_mcts_search_with_tt(
                board.clone(),
                &move_gen,
                config_candidate.clone(),
                &mut tt_candidate,
            )
        } else {
            tactical_mcts_search_with_tt(
                board.clone(),
                &move_gen,
                config_current.clone(),
                &mut tt_current,
            )
        };

        // Collect training data from MCTS root visit counts
        if collect_training_data {
            let root_ref = root.borrow();
            let total_visits: u32 = root_ref.children.iter()
                .map(|c| c.borrow().visits)
                .sum();

            if total_visits > 0 {
                let mut policy_dist = Vec::new();
                let mut policy_moves = Vec::new();
                for child in &root_ref.children {
                    let cr = child.borrow();
                    if let Some(mv) = cr.action {
                        let relative_mv = if board.w_to_move { mv } else { mv.flip_vertical() };
                        let idx = move_to_index(relative_mv) as u16;
                        let prob = cr.visits as f32 / total_visits as f32;
                        policy_dist.push((idx, prob));
                        policy_moves.push((mv, prob));
                    }
                }

                let mut temp_stack = BoardStack::with_board(board.clone());
                let material_scalar = forced_material_balance(&mut temp_stack, &move_gen) as f32;

                let sample = TrainingSample {
                    board: board.clone(),
                    policy: policy_dist,
                    policy_moves,
                    value_target: 0.0, // Backfilled after game ends
                    material_scalar,
                    w_to_move: board.w_to_move,
                };

                if candidate_turn {
                    candidate_samples.push(sample);
                } else {
                    current_samples.push(sample);
                }
            }
            drop(root_ref);
        }

        let selected_move = select_eval_move(&root, &mut rng, move_count);
        match selected_move {
            None => break,
            Some(mv) => {
                game_moves.push(mv.to_uci());
                board_stack.make_move(mv);
                move_count += 1;
            }
        }

        if enable_koth {
            let (white_won, black_won) = board_stack.current_state().is_koth_win();
            if white_won || black_won {
                break;
            }
        }

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

    let final_board = board_stack.current_state();
    let (mate, stalemate) = final_board.is_checkmate_or_stalemate(&move_gen);
    let is_repetition = board_stack.is_draw_by_repetition();
    let is_50_move = final_board.halfmove_clock() >= 100;
    let (koth_white, koth_black) = if enable_koth {
        final_board.is_koth_win()
    } else {
        (false, false)
    };

    let (result, result_str) = if koth_white || koth_black {
        let white_wins = koth_white;
        let r = if (white_wins && candidate_is_white) || (!white_wins && !candidate_is_white) {
            GameResult::CandidateWin
        } else {
            GameResult::CurrentWin
        };
        let s = if white_wins { "White wins (KOTH)" } else { "Black wins (KOTH)" };
        (r, s)
    } else if mate {
        let white_wins = !final_board.w_to_move;
        let r = if (white_wins && candidate_is_white) || (!white_wins && !candidate_is_white) {
            GameResult::CandidateWin
        } else {
            GameResult::CurrentWin
        };
        let s = if white_wins { "White wins (checkmate)" } else { "Black wins (checkmate)" };
        (r, s)
    } else if stalemate {
        (GameResult::Draw, "Draw (stalemate)")
    } else if is_repetition {
        (GameResult::Draw, "Draw (repetition)")
    } else if is_50_move {
        (GameResult::Draw, "Draw (50-move rule)")
    } else if move_count > 200 {
        (GameResult::Draw, "Draw (move limit)")
    } else {
        (GameResult::Draw, "Draw")
    };

    // Backfill value targets: +1 for white win, -1 for black win, 0 for draw
    if collect_training_data {
        let final_score_white = if koth_white {
            1.0f32
        } else if koth_black {
            -1.0
        } else if mate {
            if final_board.w_to_move { -1.0 } else { 1.0 }
        } else {
            0.0
        };

        for sample in candidate_samples.iter_mut().chain(current_samples.iter_mut()) {
            sample.value_target = if sample.w_to_move {
                final_score_white
            } else {
                -final_score_white
            };
        }
    }

    let mut move_str = String::new();
    for (i, uci) in game_moves.iter().enumerate() {
        if i % 2 == 0 {
            if !move_str.is_empty() { move_str.push(' '); }
            move_str.push_str(&format!("{}.", i / 2 + 1));
        }
        move_str.push(' ');
        move_str.push_str(uci);
    }
    let cand_color = if candidate_is_white { "White" } else { "Black" };
    eprintln!("  [seed={}] Candidate={} | {} moves | {} -> {:?}",
              game_seed, cand_color, game_moves.len(), result_str, result);
    eprintln!("  {}", move_str);

    EvalGameData {
        result,
        candidate_samples,
        current_samples,
    }
}

/// Run a full evaluation match between two models.
pub fn evaluate_models(
    candidate_path: &str,
    current_path: &str,
    num_games: u32,
    simulations: u32,
) -> EvalResults {
    evaluate_models_koth(candidate_path, current_path, num_games, simulations, false, true, true, 8, 0)
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
    inference_batch_size: usize,
    seed_offset: u64,
) -> EvalResults {
    // Load each model once, create shared InferenceServers
    let candidate_server: Option<Arc<InferenceServer>> = {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(candidate_path) {
            eprintln!("Failed to load candidate model: {}", e);
            return EvalResults { wins: 0, losses: 0, draws: 0 };
        }
        Some(Arc::new(InferenceServer::new(nn, inference_batch_size)))
    };

    let current_server: Option<Arc<InferenceServer>> = {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(current_path) {
            eprintln!("Failed to load current model: {}", e);
            return EvalResults { wins: 0, losses: 0, draws: 0 };
        }
        Some(Arc::new(InferenceServer::new(nn, inference_batch_size)))
    };

    let results = Mutex::new(EvalResults {
        wins: 0,
        losses: 0,
        draws: 0,
    });

    (0..num_games).into_par_iter().for_each(|game_idx| {
        let candidate_is_white = game_idx % 2 == 0;

        let game_data = play_evaluation_game_with_servers(
            &candidate_server,
            &current_server,
            candidate_is_white,
            simulations,
            enable_koth,
            enable_tier1,
            enable_material,
            seed_offset + game_idx as u64,
            false,
        );

        let mut r = results.lock().unwrap();
        match game_data.result {
            GameResult::CandidateWin => r.wins += 1,
            GameResult::CurrentWin => r.losses += 1,
            GameResult::Draw => r.draws += 1,
        }

        let color_str = if candidate_is_white { "White" } else { "Black" };
        eprintln!(
            "Game {}/{}: Candidate ({}) -> {:?} [W:{} L:{} D:{}]",
            game_idx + 1,
            num_games,
            color_str,
            game_data.result,
            r.wins,
            r.losses,
            r.draws
        );
    });

    results.into_inner().unwrap()
}

/// Run an SPRT-based evaluation match with early stopping.
///
/// Returns (results, final_llr, sprt_decision).
/// When `save_training_data` is Some, saves candidate/current training samples
/// to `{base_dir}/candidate/` and `{base_dir}/current/` respectively.
pub fn evaluate_models_koth_sprt(
    candidate_path: &str,
    current_path: &str,
    max_games: u32,
    simulations: u32,
    enable_koth: bool,
    enable_tier1: bool,
    enable_material: bool,
    inference_batch_size: usize,
    sprt_config: &SprtConfig,
    seed_offset: u64,
    save_training_data: Option<&str>,
) -> (EvalResults, Option<f64>, SprtResult) {
    let collect = save_training_data.is_some();

    // Load each model once, create shared InferenceServers
    let candidate_server: Option<Arc<InferenceServer>> = {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(candidate_path) {
            eprintln!("Failed to load candidate model: {}", e);
            return (EvalResults { wins: 0, losses: 0, draws: 0 }, None, SprtResult::Inconclusive);
        }
        Some(Arc::new(InferenceServer::new(nn, inference_batch_size)))
    };

    let current_server: Option<Arc<InferenceServer>> = {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(current_path) {
            eprintln!("Failed to load current model: {}", e);
            return (EvalResults { wins: 0, losses: 0, draws: 0 }, None, SprtResult::Inconclusive);
        }
        Some(Arc::new(InferenceServer::new(nn, inference_batch_size)))
    };

    let sprt_state = Arc::new(Mutex::new(SprtState::new()));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let games_completed = Arc::new(Mutex::new(0u32));

    // Thread-safe accumulators for training samples
    let all_candidate_samples: Arc<Mutex<Vec<TrainingSample>>> = Arc::new(Mutex::new(Vec::new()));
    let all_current_samples: Arc<Mutex<Vec<TrainingSample>>> = Arc::new(Mutex::new(Vec::new()));

    (0..max_games).into_par_iter().for_each(|game_idx| {
        if stop_flag.load(Ordering::Relaxed) {
            return;
        }

        let candidate_is_white = game_idx % 2 == 0;

        let game_data = play_evaluation_game_with_servers(
            &candidate_server,
            &current_server,
            candidate_is_white,
            simulations,
            enable_koth,
            enable_tier1,
            enable_material,
            seed_offset + game_idx as u64,
            collect,
        );

        // Collect training samples
        if collect {
            if !game_data.candidate_samples.is_empty() {
                all_candidate_samples.lock().unwrap().extend(game_data.candidate_samples);
            }
            if !game_data.current_samples.is_empty() {
                all_current_samples.lock().unwrap().extend(game_data.current_samples);
            }
        }

        let mut state = sprt_state.lock().unwrap();
        match game_data.result {
            GameResult::CandidateWin => state.wins += 1,
            GameResult::CurrentWin => state.losses += 1,
            GameResult::Draw => state.draws += 1,
        }

        let mut completed = games_completed.lock().unwrap();
        *completed += 1;

        let color_str = if candidate_is_white { "White" } else { "Black" };
        let decision = state.check_decision(sprt_config);
        let llr_str = state.compute_llr(sprt_config)
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "N/A".to_string());
        eprintln!(
            "Game {}/{}: Candidate ({}) -> {:?} [W:{} L:{} D:{}] LLR={} {:?}",
            *completed,
            max_games,
            color_str,
            game_data.result,
            state.wins,
            state.losses,
            state.draws,
            llr_str,
            decision,
        );

        match decision {
            SprtResult::AcceptH1 | SprtResult::AcceptH0 => {
                stop_flag.store(true, Ordering::Relaxed);
            }
            SprtResult::Inconclusive => {}
        }
    });

    // Save training data to disk
    if let Some(base_dir) = save_training_data {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        let candidate_dir = format!("{}/candidate", base_dir);
        let current_dir = format!("{}/current", base_dir);
        std::fs::create_dir_all(&candidate_dir).ok();
        std::fs::create_dir_all(&current_dir).ok();

        let candidate_samples: Vec<TrainingSample> = {
            let mut guard = all_candidate_samples.lock().unwrap();
            std::mem::take(&mut *guard)
        };
        let current_samples: Vec<TrainingSample> = {
            let mut guard = all_current_samples.lock().unwrap();
            std::mem::take(&mut *guard)
        };

        if !candidate_samples.is_empty() {
            let path = format!("{}/eval_{}.bin", candidate_dir, timestamp);
            if let Err(e) = save_binary_data(&path, &candidate_samples) {
                eprintln!("Failed to save candidate training data: {}", e);
            } else {
                eprintln!("Saved {} candidate training samples to {}", candidate_samples.len(), path);
            }
        }

        if !current_samples.is_empty() {
            let path = format!("{}/eval_{}.bin", current_dir, timestamp);
            if let Err(e) = save_binary_data(&path, &current_samples) {
                eprintln!("Failed to save current training data: {}", e);
            } else {
                eprintln!("Saved {} current training samples to {}", current_samples.len(), path);
            }
        }
    }

    let final_state = sprt_state.lock().unwrap();
    let llr = final_state.compute_llr(sprt_config);
    let decision = final_state.check_decision(sprt_config);
    let results = EvalResults {
        wins: final_state.wins,
        losses: final_state.losses,
        draws: final_state.draws,
    };

    (results, llr, decision)
}

fn parse_arg_f64(args: &[String], flag: &str, default: f64) -> f64 {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 5 {
        eprintln!("Usage: evaluate_models <candidate.pt> <current.pt> <num_games> <simulations> [--sprt --elo0 0 --elo1 10 --sprt-alpha 0.05 --sprt-beta 0.05] [--threshold 0.55]");
        std::process::exit(2);
    }

    let candidate_path = &args[1];
    let current_path = &args[2];
    let num_games: u32 = args[3].parse().expect("num_games must be a number");
    let simulations: u32 = args[4].parse().expect("simulations must be a number");

    let use_sprt = args.iter().any(|a| a == "--sprt");

    let threshold: f64 = parse_arg_f64(&args, "--threshold", 0.55);

    let enable_koth = args.iter().any(|a| a == "--enable-koth");
    let enable_tier1 = !args.iter().any(|a| a == "--disable-tier1");
    let enable_material = !args.iter().any(|a| a == "--disable-material");

    let inference_batch_size: usize = args.iter()
        .position(|a| a == "--batch-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let num_threads: Option<usize> = args.iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());

    let seed_offset: u64 = args.iter()
        .position(|a| a == "--seed-offset")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let no_save_training_data = args.iter().any(|a| a == "--no-save-training-data");
    let save_training_data: Option<String> = if no_save_training_data {
        None
    } else {
        // Default: save alongside candidate model; --save-training-data <dir> overrides
        let explicit = args.iter()
            .position(|a| a == "--save-training-data")
            .and_then(|i| args.get(i + 1))
            .cloned();
        Some(explicit.unwrap_or_else(|| {
            let parent = std::path::Path::new(candidate_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            parent.join("eval_training_data").to_string_lossy().into_owned()
        }))
    };

    if let Some(threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    eprintln!("Evaluating: {} vs {}", candidate_path, current_path);

    if use_sprt {
        let sprt_config = SprtConfig {
            elo0: parse_arg_f64(&args, "--elo0", 0.0),
            elo1: parse_arg_f64(&args, "--elo1", 10.0),
            alpha: parse_arg_f64(&args, "--sprt-alpha", 0.05),
            beta: parse_arg_f64(&args, "--sprt-beta", 0.05),
        };

        let (lower, upper) = sprt_config.bounds();
        eprintln!("SPRT: elo0={}, elo1={}, alpha={}, beta={}, bounds=[{:.3}, {:.3}]",
                  sprt_config.elo0, sprt_config.elo1, sprt_config.alpha, sprt_config.beta, lower, upper);
        eprintln!("Max games: {}, Sims: {}, KOTH: {}, Tier1: {}, Material: {}",
                  num_games, simulations, enable_koth, enable_tier1, enable_material);

        let (results, llr, decision) = evaluate_models_koth_sprt(
            candidate_path, current_path, num_games, simulations,
            enable_koth, enable_tier1, enable_material, inference_batch_size,
            &sprt_config, seed_offset,
            save_training_data.as_deref(),
        );

        let win_rate = results.win_rate();
        let games_played = results.wins + results.losses + results.draws;
        let accepted = decision == SprtResult::AcceptH1;
        let llr_str = llr.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "N/A".to_string());

        // Machine-readable output on stdout (backward-compatible + SPRT fields)
        println!(
            "WINS={} LOSSES={} DRAWS={} WINRATE={:.4} ACCEPTED={} GAMES_PLAYED={} LLR={} SPRT_RESULT={}",
            results.wins,
            results.losses,
            results.draws,
            win_rate,
            accepted,
            games_played,
            llr_str,
            decision,
        );

        std::process::exit(if accepted { 0 } else { 1 });
    } else {
        // Legacy fixed-threshold mode
        eprintln!("Games: {}, Sims: {}, Threshold: {}, KOTH: {}, Tier1: {}, Material: {}",
                  num_games, simulations, threshold, enable_koth, enable_tier1, enable_material);

        let results = evaluate_models_koth(
            candidate_path, current_path, num_games, simulations,
            enable_koth, enable_tier1, enable_material, inference_batch_size,
            seed_offset,
        );
        let win_rate = results.win_rate();
        let games_played = results.wins + results.losses + results.draws;
        let accepted = results.accepted(threshold);

        println!(
            "WINS={} LOSSES={} DRAWS={} WINRATE={:.4} ACCEPTED={} GAMES_PLAYED={} LLR=N/A SPRT_RESULT=inconclusive",
            results.wins,
            results.losses,
            results.draws,
            win_rate,
            accepted,
            games_played,
        );

        std::process::exit(if accepted { 0 } else { 1 });
    }
}
