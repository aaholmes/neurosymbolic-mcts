// src/bin/run_experiments.rs
//! Main experiment runner for ablation studies.
//!
//! Usage:
//!   cargo run --release --bin run_experiments -- --config ablation
//!   cargo run --release --bin run_experiments -- --config single --name full_system

use kingfisher::board::Board;
use kingfisher::experiments::config::{
    ExperimentConfig,
    generate_ablation_configs,
    TestSuite,
};
use kingfisher::experiments::metrics::{
    AggregatedMetrics,
    ExperimentResults,
    PositionResult,
    SafetyMetrics,
    SearchMetrics,
    TierUsed,
};
use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;
use std::fs;
use std::io::Write;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let mode = args.iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("ablation");
    
    match mode {
        "ablation" => run_ablation_study(),
        "single" => {
            let name = args.iter()
                .position(|a| a == "--name")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str())
                .unwrap_or("full_system");
            run_single_config(name);
        }
        _ => {
            println!("Usage: run_experiments --config [ablation|single] [--name config_name]");
        }
    }
}

fn run_ablation_study() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     CAISSAWARY ABLATION STUDY - Safe & Efficient RL       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    let configs = generate_ablation_configs();
    let test_positions = load_test_suite(&TestSuite::TacticalSuite);
    
    println!("Running {} configurations on {} positions\n", configs.len(), test_positions.len());
    
    let mut all_results: Vec<ExperimentResults> = Vec::new();
    
    for config in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Configuration: {}", config.name);
        println!("Description: {}", config.description);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        let results = run_experiment(config, &test_positions);
        
        print_summary(&results);
        all_results.push(results);
    }
    
    // Generate comparison table
    generate_latex_table(&all_results);
    save_results_json(&all_results);
    
    println!("\n✅ Ablation study complete! Results saved to results/");
}

fn run_single_config(name: &str) {
    let configs = generate_ablation_configs();
    let config = configs.iter()
        .find(|c| c.name == name)
        .expect(&format!("Config '{}' not found", name));
    
    let test_positions = load_test_suite(&TestSuite::TacticalSuite);
    let results = run_experiment(config, &test_positions);
    
    print_summary(&results);
    
    // Save individual result
    let json = serde_json::to_string_pretty(&results).unwrap();
    fs::create_dir_all("results").ok();
    fs::write(format!("results/{}.json", name), json).unwrap();
}

fn run_experiment(
    config: &ExperimentConfig,
    test_positions: &[(String, Board, Option<String>, String)],
) -> ExperimentResults {
    let move_gen = MoveGen::new();

    let mut search_metrics: Vec<SearchMetrics> = Vec::new();
    let mut position_results: Vec<PositionResult> = Vec::new();
    let mut safety = SafetyMetrics::default();
    
    for (i, (name, board, expected_move, pos_type)) in test_positions.iter().enumerate() {
        print!("  [{}/{}] {}... ", i + 1, test_positions.len(), name); 
        std::io::stdout().flush().ok();
        
        // Build search config from experiment config
        let mcts_config = TacticalMctsConfig {
            max_iterations: config.search_config.max_iterations,
            time_limit: Duration::from_millis(config.search_config.time_limit_ms),
            mate_search_depth: config.search_config.mate_search_depth,
            exploration_constant: config.search_config.exploration_constant,
            enable_tier1_gate: config.ablation.enable_tier1_gate,
            enable_tier3_neural: config.ablation.enable_tier3_neural,
            use_neural_policy: config.ablation.enable_tier3_neural,
            inference_server: None,
            logger: None,
            ..Default::default()
        };
        
        let start = Instant::now();
        let (best_move, stats, _root) = tactical_mcts_search(
            board.clone(),
            &move_gen,
            mcts_config,
        );
        let elapsed = start.elapsed();
        
        // Convert stats to SearchMetrics
        let metrics = SearchMetrics {
            total_iterations: stats.iterations,
            nodes_expanded: stats.nodes_expanded,
            search_time: elapsed,
            tier1_activations: stats.tier1_solutions,
            mates_found_by_gate: stats.mates_found,
            koth_wins_detected: 0, // TODO: track separately
            nn_calls_saved_tier1: stats.nn_saved_by_tier1,
            tier2_grafts: 0,
            positions_with_tactical_moves: 0,
            avg_graft_depth: 0.0,
            nn_calls_saved_tier2: 0,
            nn_evaluations: stats.nn_evaluations,
            nn_policy_queries: stats.nn_policy_evaluations,
            avg_nn_inference_time_us: 0.0, // TODO
            best_move_visits: 0, // TODO
            second_best_visits: 0,
            root_value: 0.0,
            value_confidence: 0.0,
        };
        
        // Determine which tier was used
        let tier_used = if stats.tier1_solutions > 0 {
            TierUsed::Tier1Gate
        } else if stats.nn_evaluations > 0 {
            TierUsed::Tier3Neural
        } else {
            TierUsed::ClassicalEval
        };
        
        // Check correctness
        let engine_move_str = best_move.map(|m| m.to_uci()).unwrap_or_default();
        let correct = expected_move.as_ref()
            .map(|exp| engine_move_str == *exp)
            .unwrap_or(true);
        
        // Update safety metrics
        safety.positions_analyzed += 1;
        if pos_type == "mate" {
            safety.positions_with_forced_mate += 1;
            if correct {
                safety.forced_mates_found += 1;
            }
        }
        
        let pos_result = PositionResult {
            fen: board.to_fen().unwrap_or_default(),
            position_type: pos_type.clone(),
            expected_move: expected_move.clone(),
            engine_move: engine_move_str,
            correct,
            time_ms: elapsed.as_millis() as u64,
            tier_used,
        };
        
        let status = if correct { "✓" } else { "✗" };
        println!("{} ({}ms, NN reduction: {:.1}%)", 
                 status, 
                 elapsed.as_millis(),
                 metrics.nn_call_reduction_percent());
        
        search_metrics.push(metrics);
        position_results.push(pos_result);
    }
    
    let aggregated = AggregatedMetrics::from_search_metrics(&config.name, &search_metrics);
    
    ExperimentResults {
        config_name: config.name.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        search_metrics,
        aggregated,
        safety,
        position_results,
    }
}

fn load_test_suite(suite: &TestSuite) -> Vec<(String, Board, Option<String>, String)> {
    // Return (name, board, expected_move, position_type)
    match suite {
        TestSuite::TacticalSuite => {
            vec![
                // Mate in 1 positions
                ("Back Rank Mate".to_string(),
                 Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"),
                 Some("e1e8".to_string()),
                 "mate".to_string()),
                
                ("Queen Mate".to_string(),
                 Board::new_from_fen("6k1/5ppp/8/8/8/5Q2/8/6K1 w - - 0 1"),
                 Some("f3f8".to_string()),
                 "mate".to_string()),
                
                // Mate in 2
                ("Anastasia's Mate Setup".to_string(),
                 Board::new_from_fen("5rk1/4Nppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),
                 Some("d1d8".to_string()),
                 "mate".to_string()),
                
                // Tactical - winning material
                ("Fork Knight".to_string(),
                 Board::new_from_fen("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"),
                 Some("d2d4".to_string()),
                 "tactical".to_string()),
                
                ("Pin Win Material".to_string(),
                 Board::new_from_fen("r2qkbnr/ppp2ppp/2np4/4p3/2B1P1b1/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"),
                 Some("f1e2".to_string()), // Or Qb3 type moves
                 "tactical".to_string()),
                
                // Positional
                ("Development".to_string(),
                 Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
                 None, // Multiple good moves
                 "positional".to_string()),
                
                // Endgame
                ("K+P vs K Opposition".to_string(),
                 Board::new_from_fen("8/8/8/3k4/8/3K4/3P4/8 w - - 0 1"),
                 Some("d3c3".to_string()), // Or d3e3 - gain opposition
                 "endgame".to_string()),
                
                // Safety test - don't blunder
                ("Don't Hang Queen".to_string(),
                 Board::new_from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"),
                 None, // Many good moves, but not Qe2 or similar
                 "safety".to_string()),
            ]
        }
        _ => Vec::new(),
    }
}

fn print_summary(results: &ExperimentResults) {
    println!("\n📊 Summary for '{}':", results.config_name);
    println!("   Positions: {}", results.aggregated.num_positions);
    println!("   Mean iterations: {:.0}", results.aggregated.mean_iterations);
    println!("   Mean nodes: {:.0}", results.aggregated.mean_nodes);
    println!("   Mean time: {:.0}ms", results.aggregated.mean_time_ms);
    println!("   NN call reduction: {:.1}% ± {:.1}%", 
             results.aggregated.mean_nn_reduction_percent,
             1.96 * results.aggregated.std_nn_reduction / (results.aggregated.num_positions as f64).sqrt());
    println!("   Tier 1 activations: {}", results.aggregated.total_tier1_activations);
    println!("   Tier 2 grafts: {}", results.aggregated.total_tier2_grafts);
    println!("   NN evaluations: {}", results.aggregated.total_nn_evaluations);
    println!("   Safety - Tactical accuracy: {:.1}%", results.safety.tactical_safety_rate() * 100.0);
    println!();
}

fn generate_latex_table(results: &[ExperimentResults]) {
    let mut latex = String::new();
    latex.push_str(r"\begin{table}[h]");
    latex.push_str("\n");
    latex.push_str(r"\centering");
    latex.push_str("\n");
    latex.push_str(r"\caption{Ablation Study Results}");
    latex.push_str("\n");
    latex.push_str(r"\label{tab:ablation}");
    latex.push_str("\n");
    latex.push_str(r"\begin{tabular}{lrrrrrrr}");
    latex.push_str("\n");
    latex.push_str(r"\toprule");
    latex.push_str("\n");
    latex.push_str("Config & Iter & Nodes & NN Reduction & T1 & T2 & NN Calls \\\\
");
    latex.push_str(r"\midrule");
    latex.push_str("\n");
    
    for result in results {
        latex.push_str(&result.aggregated.to_latex_row());
        latex.push_str("\n");
    }
    
    latex.push_str(r"\bottomrule");
    latex.push_str("\n");
    latex.push_str(r"\end{tabular}");
    latex.push_str("\n");
    latex.push_str(r"\end{table}");
    latex.push_str("\n");
    
    fs::create_dir_all("results").ok();
    fs::write("results/ablation_table.tex", latex).unwrap();
    println!("📄 LaTeX table saved to results/ablation_table.tex");
}

fn save_results_json(results: &[ExperimentResults]) {
    fs::create_dir_all("results").ok();
    let json = serde_json::to_string_pretty(results).unwrap();
    fs::write("results/ablation_results.json", json).unwrap();
    println!("💾 Full results saved to results/ablation_results.json");
}
