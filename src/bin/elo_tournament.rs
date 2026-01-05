//! Elo Tournament Runner for Paper Experiments
//!
//! Usage: elo_tournament [--games N] [--seed S] [--output FILE]

use kingfisher::benchmarks::elo_tournament::{
    EloTournament,
    TournamentConfig,
    TournamentEngine,
};
use std::env;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🏆 Caissawary Elo Tournament");
    println!("============================");
    println!();
    
    let args: Vec<String> = env::args().collect();
    let games = parse_arg(&args, "--games").unwrap_or(100);
    let seed = parse_arg(&args, "--seed").unwrap_or(42) as u64;
    let output_csv = parse_string_arg(&args, "--output")
        .unwrap_or_else(|| "tournament_results.csv".to_string());
    let output_latex = output_csv.replace(".csv", ".tex");
    
    println!("Configuration:");
    println!("  Games per pair: {}", games);
    println!("  Random seed: {}", seed);
    println!("  Output: {}", output_csv);
    println!();
    
    let config = TournamentConfig {
        games_per_pair: games,
        seed,
        ..Default::default()
    };
    
    let mut tournament = EloTournament::new(config);
    
    // Define engine variants for ablation study
    let engines = vec![
        TournamentEngine::AlphaBeta { depth: 6 },
        TournamentEngine::PureMcts { iterations: 800 },
        TournamentEngine::MctsTier1Only { iterations: 800, mate_depth: 5 },
        TournamentEngine::MctsTier1And2 { iterations: 800, mate_depth: 5 },
        TournamentEngine::TacticalMctsFull { iterations: 800, mate_depth: 5 },
    ];
    
    let results = tournament.run_full_tournament(&engines);
    
    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(&output_csv).parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }
    
    // Print summary
    println!("\n📊 Final Ratings:");
    let mut ratings: Vec<_> = results.ratings.iter().collect();
    ratings.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (engine, rating) in ratings {
        println!("  {}: {:.0}", engine, rating);
    }
    
    // Save CSV
    let csv = results.to_csv();
    let mut file = File::create(&output_csv).expect("Failed to create CSV file");
    file.write_all(csv.as_bytes()).expect("Failed to write CSV");
    println!("\n✅ CSV saved to: {}", output_csv);
    
    // Save LaTeX
    let latex = results.to_latex_table();
    let mut file = File::create(&output_latex).expect("Failed to create LaTeX file");
    file.write_all(latex.as_bytes()).expect("Failed to write LaTeX");
    println!("✅ LaTeX saved to: {}", output_latex);
}

fn parse_arg(args: &[String], flag: &str) -> Option<u32> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
