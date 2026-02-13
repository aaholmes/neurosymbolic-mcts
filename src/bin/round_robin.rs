//! Round-Robin Tournament Binary
//!
//! Plays a full round-robin tournament between multiple neural network models,
//! each with per-player tier configuration (tiered vs vanilla).
//!
//! Usage:
//!   round_robin --games-per-pair 100 --simulations 200 \
//!     --model "tiered_gen0:path/to/gen_0.pt:tiered" \
//!     --model "vanilla_gen0:path/to/gen_0.pt:vanilla" \
//!     [--output results.csv] [--batch-size 8] [--seed 42]

use kingfisher::boardstack::BoardStack;
use kingfisher::mcts::{
    tactical_mcts_search_with_tt, InferenceServer, MctsNode, TacticalMctsConfig,
};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::transposition::TranspositionTable;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

/// Number of half-moves using proportional sampling for opening diversity.
const EVAL_EXPLORATION_PLIES: u32 = 10;

struct ModelEntry {
    name: String,
    server: Arc<InferenceServer>,
    enable_tier1: bool,
    enable_material: bool,
    enable_koth: bool,
}

struct PairResult {
    model_a: String,
    model_b: String,
    a_wins: u32,
    b_wins: u32,
    draws: u32,
}

impl PairResult {
    fn total(&self) -> u32 {
        self.a_wins + self.b_wins + self.draws
    }

    fn a_score(&self) -> f64 {
        let total = self.total() as f64;
        if total == 0.0 {
            return 0.5;
        }
        (self.a_wins as f64 + 0.5 * self.draws as f64) / total
    }

    fn elo_difference(&self) -> f64 {
        let score = self.a_score();
        if score <= 0.001 {
            return -800.0;
        }
        if score >= 0.999 {
            return 800.0;
        }
        -400.0 * (1.0 / score - 1.0).log10()
    }

    fn elo_confidence_interval(&self) -> (f64, f64) {
        let n = self.total() as f64;
        if n == 0.0 {
            return (-800.0, 800.0);
        }
        let p = self.a_score();
        let z = 1.96; // 95% CI
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum GameOutcome {
    WhiteWin,
    BlackWin,
    Draw,
}

/// Select a move for evaluation: deterministic for forced wins,
/// proportional sampling for the first few moves, then greedy.
fn select_eval_move(
    root: &Rc<RefCell<MctsNode>>,
    rng: &mut impl Rng,
    move_count: u32,
) -> Option<Move> {
    let root_ref = root.borrow();

    // 1. Forced win detection
    let mut best_win: Option<(Move, f64)> = None;
    for child in &root_ref.children {
        let cr = child.borrow();
        if let Some(v) = cr.terminal_or_mate_value {
            if v < -0.5 {
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

    // 2. Mate move from gate
    if let Some(mate_mv) = root_ref.mate_move {
        return Some(mate_mv);
    }

    // 3. Visit-count based selection
    let visit_pairs: Vec<(Move, u32)> = root_ref
        .children
        .iter()
        .filter_map(|c| {
            let cr = c.borrow();
            cr.action.map(|mv| (mv, cr.visits))
        })
        .collect();

    if move_count < EVAL_EXPLORATION_PLIES {
        sample_proportional(&visit_pairs, rng)
    } else {
        visit_pairs
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(mv, _)| *mv)
    }
}

fn sample_proportional(policy: &[(Move, u32)], rng: &mut impl Rng) -> Option<Move> {
    let total: u32 = policy.iter().map(|(_, v)| v.saturating_sub(1)).sum();
    if total == 0 {
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

fn build_config(model: &ModelEntry, simulations: u32) -> TacticalMctsConfig {
    TacticalMctsConfig {
        max_iterations: simulations,
        time_limit: Duration::from_secs(120),
        mate_search_depth: if model.enable_tier1 { 5 } else { 0 },
        exploration_constant: 1.414,
        use_neural_policy: true,
        inference_server: Some(model.server.clone()),
        logger: None,
        dirichlet_alpha: 0.0,
        dirichlet_epsilon: 0.0,
        enable_koth: model.enable_koth,
        enable_tier1_gate: model.enable_tier1,
        enable_material_value: model.enable_material,
        enable_tier3_neural: true,
        randomize_move_order: true,
        ..Default::default()
    }
}

/// Play a single game between two models. Returns outcome from white's perspective.
fn play_game(white: &ModelEntry, black: &ModelEntry, simulations: u32, seed: u64) -> GameOutcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let move_gen = MoveGen::new();

    let config_white = build_config(white, simulations);
    let config_black = build_config(black, simulations);

    let mut board_stack = BoardStack::new();
    let mut move_count = 0u32;
    let mut tt_white = TranspositionTable::new();
    let mut tt_black = TranspositionTable::new();

    let enable_koth = white.enable_koth || black.enable_koth;

    loop {
        let board = board_stack.current_state().clone();

        let (_best_move, _stats, root) = if board.w_to_move {
            tactical_mcts_search_with_tt(
                board.clone(),
                &move_gen,
                config_white.clone(),
                &mut tt_white,
            )
        } else {
            tactical_mcts_search_with_tt(
                board.clone(),
                &move_gen,
                config_black.clone(),
                &mut tt_black,
            )
        };

        let selected_move = select_eval_move(&root, &mut rng, move_count);
        match selected_move {
            None => break,
            Some(mv) => {
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

        let (mate, stalemate) = board_stack
            .current_state()
            .is_checkmate_or_stalemate(&move_gen);
        if mate || stalemate {
            break;
        }
    }

    // Determine result
    let final_board = board_stack.current_state();
    let (mate, stalemate) = final_board.is_checkmate_or_stalemate(&move_gen);
    let is_repetition = board_stack.is_draw_by_repetition();
    let is_50_move = final_board.halfmove_clock() >= 100;
    let (koth_white, koth_black) = if enable_koth {
        final_board.is_koth_win()
    } else {
        (false, false)
    };

    if koth_white {
        GameOutcome::WhiteWin
    } else if koth_black {
        GameOutcome::BlackWin
    } else if mate {
        if final_board.w_to_move {
            GameOutcome::BlackWin // White is in checkmate
        } else {
            GameOutcome::WhiteWin // Black is in checkmate
        }
    } else if stalemate || is_repetition || is_50_move || move_count > 200 {
        GameOutcome::Draw
    } else {
        GameOutcome::Draw
    }
}

/// Play all games for one pair of models.
fn play_pair(
    model_a: &ModelEntry,
    model_b: &ModelEntry,
    games_per_pair: u32,
    simulations: u32,
    base_seed: u64,
) -> PairResult {
    let mut a_wins = 0u32;
    let mut b_wins = 0u32;
    let mut draws = 0u32;

    for game_idx in 0..games_per_pair {
        // Alternate colors
        let (white, black, a_is_white) = if game_idx % 2 == 0 {
            (model_a, model_b, true)
        } else {
            (model_b, model_a, false)
        };

        let seed = base_seed + game_idx as u64;
        let outcome = play_game(white, black, simulations, seed);

        match (outcome, a_is_white) {
            (GameOutcome::WhiteWin, true) | (GameOutcome::BlackWin, false) => a_wins += 1,
            (GameOutcome::WhiteWin, false) | (GameOutcome::BlackWin, true) => b_wins += 1,
            (GameOutcome::Draw, _) => draws += 1,
        }

        eprint!(
            "\r  {}: game {}/{} [+{} ={} -{} for {}]",
            format!("{} vs {}", model_a.name, model_b.name),
            game_idx + 1,
            games_per_pair,
            a_wins,
            draws,
            b_wins,
            model_a.name,
        );
    }
    eprintln!();

    PairResult {
        model_a: model_a.name.clone(),
        model_b: model_b.name.clone(),
        a_wins,
        b_wins,
        draws,
    }
}

/// Compute Elo ratings via Maximum Likelihood Estimation (Bradley-Terry model).
///
/// Finds ratings that maximize the joint probability of all observed game outcomes.
/// Uses gradient ascent on the log-likelihood:
///   log L = sum over pairs { s_ij * log(E_i) + (n_ij - s_ij) * log(1 - E_i) }
/// where E_i = 1 / (1 + 10^((r_j - r_i) / 400)) is the expected score.
///
/// The mean rating is anchored at 1500 after each iteration (only differences matter).
fn calculate_ratings(results: &[PairResult], model_names: &[String]) -> HashMap<String, f64> {
    let n = model_names.len();
    let name_to_idx: HashMap<&String, usize> = model_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let mut ratings = vec![1500.0f64; n];

    let lr = 10.0; // learning rate (Elo units per step)
    let iterations = 1000;

    for _ in 0..iterations {
        let mut grad = vec![0.0f64; n];

        for r in results {
            let i = name_to_idx[&r.model_a];
            let j = name_to_idx[&r.model_b];
            let n_games = r.total() as f64;
            if n_games == 0.0 {
                continue;
            }
            let s_ij = r.a_score() * n_games; // score for model_a

            // Expected score: E_i = 1 / (1 + 10^((r_j - r_i) / 400))
            let expected = 1.0 / (1.0 + 10.0f64.powf((ratings[j] - ratings[i]) / 400.0));

            // Gradient: d(log L)/d(r_i) = (ln10 / 400) * (s_ij - n_games * E_i)
            let g = (10.0f64.ln() / 400.0) * (s_ij - n_games * expected);
            grad[i] += g;
            grad[j] -= g;
        }

        for k in 0..n {
            ratings[k] += lr * grad[k];
        }

        // Re-anchor mean to 1500
        let mean: f64 = ratings.iter().sum::<f64>() / n as f64;
        for k in 0..n {
            ratings[k] += 1500.0 - mean;
        }
    }

    let mut result = HashMap::new();
    for (i, name) in model_names.iter().enumerate() {
        result.insert(name.clone(), ratings[i]);
    }
    result
}

struct CliArgs {
    models: Vec<(String, String, String)>,
    games_per_pair: u32,
    simulations: u32,
    output: Option<String>,
    batch_size: usize,
    seed: u64,
    only_involving: Vec<String>,
    import_csv: Option<String>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();

    let mut models: Vec<(String, String, String)> = Vec::new();
    let mut games_per_pair = 100u32;
    let mut simulations = 200u32;
    let mut output: Option<String> = None;
    let mut batch_size = 8usize;
    let mut seed = 42u64;
    let mut only_involving: Vec<String> = Vec::new();
    let mut import_csv: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                let spec = &args[i];
                let parts: Vec<&str> = spec.splitn(3, ':').collect();
                if parts.len() != 3 {
                    eprintln!(
                        "Error: --model format is 'name:path:preset' (got '{}')",
                        spec
                    );
                    std::process::exit(1);
                }
                models.push((
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                ));
            }
            "--games-per-pair" => {
                i += 1;
                games_per_pair = args[i].parse().expect("Invalid --games-per-pair");
            }
            "--simulations" => {
                i += 1;
                simulations = args[i].parse().expect("Invalid --simulations");
            }
            "--output" => {
                i += 1;
                output = Some(args[i].clone());
            }
            "--batch-size" => {
                i += 1;
                batch_size = args[i].parse().expect("Invalid --batch-size");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid --seed");
            }
            "--only-involving" => {
                i += 1;
                only_involving.push(args[i].clone());
            }
            "--import-csv" => {
                i += 1;
                import_csv = Some(args[i].clone());
            }
            "--help" | "-h" => {
                eprintln!("Usage: round_robin [OPTIONS]");
                eprintln!("  --model NAME:PATH:PRESET  Add a model (preset: tiered or vanilla)");
                eprintln!("  --games-per-pair N        Games per pair (default: 100)");
                eprintln!("  --simulations N           MCTS simulations per move (default: 200)");
                eprintln!("  --output FILE             CSV output file");
                eprintln!("  --batch-size N            Inference batch size (default: 8)");
                eprintln!("  --seed N                  Base random seed (default: 42)");
                eprintln!(
                    "  --only-involving NAME     Only play pairs involving this model (repeatable)"
                );
                eprintln!(
                    "  --import-csv FILE         Import prior results CSV to merge with new games"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if models.len() < 2 {
        eprintln!("Error: need at least 2 models for a tournament");
        std::process::exit(1);
    }

    CliArgs {
        models,
        games_per_pair,
        simulations,
        output,
        batch_size,
        seed,
        only_involving,
        import_csv,
    }
}

/// Import pairwise results from a prior CSV file.
fn import_results(path: &str) -> Vec<PairResult> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading CSV {}: {}", path, e);
        std::process::exit(1);
    });
    let mut results = Vec::new();
    for line in content.lines().skip(1) {
        // Stop at blank line (ratings section follows)
        if line.trim().is_empty() {
            break;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 {
            continue;
        }
        results.push(PairResult {
            model_a: fields[0].to_string(),
            model_b: fields[1].to_string(),
            a_wins: fields[2].parse().unwrap_or(0),
            b_wins: fields[4].parse().unwrap_or(0),
            draws: fields[3].parse().unwrap_or(0),
        });
    }
    results
}

fn main() {
    let cli = parse_args();
    let model_specs = cli.models;
    let games_per_pair = cli.games_per_pair;
    let simulations = cli.simulations;
    let output_path = cli.output;
    let batch_size = cli.batch_size;
    let base_seed = cli.seed;
    let only_involving = cli.only_involving;
    let import_csv = cli.import_csv;

    eprintln!("=== Round-Robin Tournament ===");
    eprintln!("Models: {}", model_specs.len());
    eprintln!("Games per pair: {}", games_per_pair);
    eprintln!("Simulations per move: {}", simulations);
    eprintln!("Batch size: {}", batch_size);
    eprintln!("Base seed: {}", base_seed);
    eprintln!();

    // Load all models
    let mut models: Vec<ModelEntry> = Vec::new();
    for (name, path, preset) in &model_specs {
        eprint!("Loading {} from {} (preset={})... ", name, path, preset);
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(path) {
            eprintln!("FAILED: {}", e);
            std::process::exit(1);
        }
        let server = Arc::new(InferenceServer::new(nn, batch_size));

        let (enable_tier1, enable_material, enable_koth) = match preset.as_str() {
            "tiered" => (true, true, true),
            "vanilla" => (false, false, true),
            other => {
                eprintln!("Unknown preset '{}'. Use 'tiered' or 'vanilla'.", other);
                std::process::exit(1);
            }
        };

        models.push(ModelEntry {
            name: name.clone(),
            server,
            enable_tier1,
            enable_material,
            enable_koth,
        });
        eprintln!("OK");
    }

    // Import prior results if specified
    let mut results: Vec<PairResult> = Vec::new();
    if let Some(ref csv_path) = import_csv {
        let imported = import_results(csv_path);
        eprintln!(
            "Imported {} prior pair results from {}",
            imported.len(),
            csv_path
        );
        results.extend(imported);
    }

    let n = models.len();

    // Determine which pairs to play
    let mut pairs_to_play: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            // Skip if --only-involving is set and neither model matches
            if !only_involving.is_empty() {
                let a_match = only_involving.iter().any(|name| *name == models[i].name);
                let b_match = only_involving.iter().any(|name| *name == models[j].name);
                if !a_match && !b_match {
                    continue;
                }
            }
            pairs_to_play.push((i, j));
        }
    }

    let total_pairs = pairs_to_play.len();
    eprintln!(
        "\nPlaying {} pairs x {} games = {} total games\n",
        total_pairs,
        games_per_pair,
        total_pairs as u32 * games_per_pair
    );

    // Play pairs
    let mut pair_idx = 0u32;
    for &(i, j) in &pairs_to_play {
        pair_idx += 1;
        eprintln!(
            "[Pair {}/{}] {} vs {}",
            pair_idx, total_pairs, models[i].name, models[j].name
        );

        // Use distinct seed per pair based on model names for reproducibility
        let pair_seed = base_seed
            .wrapping_mul(1000)
            .wrapping_add(pair_idx as u64 * 10000);
        let result = play_pair(
            &models[i],
            &models[j],
            games_per_pair,
            simulations,
            pair_seed,
        );

        eprintln!(
            "  Result: {} +{} ={} -{} (score {:.1}%, Elo {:+.0})",
            models[i].name,
            result.a_wins,
            result.draws,
            result.b_wins,
            result.a_score() * 100.0,
            result.elo_difference(),
        );
        eprintln!();

        results.push(result);
    }

    // Compute Elo ratings — include all models from both loaded and imported results
    let mut all_names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    for r in &results {
        if !all_names.contains(&r.model_a) {
            all_names.push(r.model_a.clone());
        }
        if !all_names.contains(&r.model_b) {
            all_names.push(r.model_b.clone());
        }
    }
    let ratings = calculate_ratings(&results, &all_names);

    // Sort by Elo descending
    let mut ranked: Vec<(&String, f64)> = ratings.iter().map(|(k, v)| (k, *v)).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Print final ranking
    eprintln!("=== Final Elo Ratings ===");
    eprintln!("{:<20} {:>6}", "Model", "Elo");
    eprintln!("{:-<27}", "");
    for (name, elo) in &ranked {
        eprintln!("{:<20} {:>+6.0}", name, elo);
    }
    eprintln!();

    // Print pairwise results table
    eprintln!("=== Pairwise Results ===");
    eprintln!(
        "{:<20} {:<20} {:>5} {:>5} {:>5} {:>7} {:>16}",
        "Model A", "Model B", "+", "=", "-", "Elo", "95% CI"
    );
    eprintln!("{:-<80}", "");
    for r in &results {
        let (ci_low, ci_high) = r.elo_confidence_interval();
        eprintln!(
            "{:<20} {:<20} {:>5} {:>5} {:>5} {:>+7.0} [{:>+.0}, {:>+.0}]",
            r.model_a,
            r.model_b,
            r.a_wins,
            r.draws,
            r.b_wins,
            r.elo_difference(),
            ci_low,
            ci_high,
        );
    }

    // Write CSV
    if let Some(ref path) = output_path {
        let mut csv = String::new();
        csv.push_str("model_a,model_b,wins_a,draws,wins_b,elo_diff,ci_low,ci_high\n");
        for r in &results {
            let (ci_low, ci_high) = r.elo_confidence_interval();
            csv.push_str(&format!(
                "{},{},{},{},{},{:.1},{:.1},{:.1}\n",
                r.model_a,
                r.model_b,
                r.a_wins,
                r.draws,
                r.b_wins,
                r.elo_difference(),
                ci_low,
                ci_high,
            ));
        }

        // Append ratings section
        csv.push_str("\nmodel,elo\n");
        for (name, elo) in &ranked {
            csv.push_str(&format!("{},{:.1}\n", name, elo));
        }

        if let Err(e) = std::fs::write(path, &csv) {
            eprintln!("Error writing CSV to {}: {}", path, e);
        } else {
            eprintln!("\nResults written to {}", path);
        }
    }
}
