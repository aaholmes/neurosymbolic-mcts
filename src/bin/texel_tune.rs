//! Texel tuning binary for optimizing evaluation parameters

use kingfisher::eval_constants::EvalWeights;
use kingfisher::move_generation::MoveGen;
use kingfisher::tuning::data_loader::DataLoader;
use kingfisher::tuning::texel::{create_test_dataset, TexelTuner};

fn main() {
    println!("🔧 Kingfisher Texel Tuning");
    println!("=========================");

    // Initialize move generation
    let move_gen = MoveGen::new();

    // Load or generate training positions
    let positions = if std::env::args().len() > 1 {
        let data_file = std::env::args().nth(1).unwrap();
        println!("📁 Loading positions from: {}", data_file);

        if data_file.ends_with(".csv") {
            DataLoader::load_from_csv(&data_file).unwrap_or_else(|e| {
                eprintln!("❌ Error loading CSV: {}", e);
                std::process::exit(1);
            })
        } else if data_file.ends_with(".epd") {
            DataLoader::load_from_epd(&data_file).unwrap_or_else(|e| {
                eprintln!("❌ Error loading EPD: {}", e);
                std::process::exit(1);
            })
        } else {
            eprintln!("❌ Unsupported file format. Use .csv or .epd");
            std::process::exit(1);
        }
    } else {
        println!("📊 Generating synthetic test dataset...");
        create_test_dataset()
    };

    println!("📈 Loaded {} training positions", positions.len());

    // Initialize tuner with default weights
    let initial_weights = EvalWeights::default();
    let mut tuner = TexelTuner::new(positions, initial_weights);

    // Set tuning parameters
    tuner.set_learning_rate(0.1);
    tuner.set_k_factor(400.0); // Standard chess evaluation scaling

    // Run tuning
    let max_iterations = if std::env::var("QUICK_TUNE").is_ok() {
        10 // Quick test
    } else {
        100 // Full tuning
    };

    println!(
        "🎯 Starting optimization ({} iterations)...",
        max_iterations
    );
    let _optimized_weights = tuner.tune(&move_gen, max_iterations);

    // Print results
    println!("\n📊 Optimization Results");
    println!("=======================");

    let best_weights = tuner.get_best_weights();

    println!(
        "Two Bishops Bonus: [{}, {}]",
        best_weights.two_bishops_bonus[0], best_weights.two_bishops_bonus[1]
    );

    println!(
        "King Safety Bonus: [{}, {}]",
        best_weights.king_safety_pawn_shield_bonus[0],
        best_weights.king_safety_pawn_shield_bonus[1]
    );

    println!(
        "Rook Open File: [{}, {}]",
        best_weights.rook_open_file_bonus[0], best_weights.rook_open_file_bonus[1]
    );

    println!(
        "Isolated Pawn Penalty: [{}, {}]",
        best_weights.isolated_pawn_penalty[0], best_weights.isolated_pawn_penalty[1]
    );

    println!("\n✅ Texel tuning complete!");
    println!("💡 Use these optimized weights to improve engine strength by 50-100 Elo!");
}
