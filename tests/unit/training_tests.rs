//! Tests for training data generation module

use kingfisher::training::{TrainingDataGenerator, TrainingPosition, ParsedGame};
use kingfisher::board::Board;
use kingfisher::move_types::Move;
use std::collections::HashMap;

// === TrainingPosition Tests ===

#[test]
fn test_training_position_new() {
    let board = Board::new();
    let mv = Move::new(12, 28, None); // e2e4
    let pos = TrainingPosition::new(&board, 1.0, Some(mv), "test".to_string());

    assert_eq!(pos.game_result, 1.0);
    assert!(pos.best_move.is_some());
    assert_eq!(pos.description, "test");
    assert!(pos.engine_eval.is_none());
    assert!(!pos.fen.is_empty());
}

#[test]
fn test_training_position_new_no_move() {
    let board = Board::new();
    let pos = TrainingPosition::new(&board, 0.5, None, "draw".to_string());

    assert_eq!(pos.game_result, 0.5);
    assert!(pos.best_move.is_none());
}

#[test]
fn test_training_position_csv_roundtrip() {
    let board = Board::new();
    let mv = Move::new(12, 28, None);
    let pos = TrainingPosition::new(&board, 0.6, Some(mv), "Test position".to_string());

    let csv_line = pos.to_csv_line();
    assert!(csv_line.contains("0.6"), "CSV should contain game result");
    assert!(csv_line.contains("Test position"), "CSV should contain description");
    // The FEN should be present
    assert!(csv_line.contains("rnbqkbnr"), "CSV should contain the FEN");
}

#[test]
fn test_training_position_csv_no_move() {
    let board = Board::new();
    let pos = TrainingPosition::new(&board, 0.5, None, "draw".to_string());

    let csv_line = pos.to_csv_line();
    // Should have empty best_move field
    assert!(csv_line.contains("0.5"));
}

// === TrainingDataGenerator Tests ===

#[test]
fn test_generator_new_defaults() {
    let gen = TrainingDataGenerator::new();
    // Just verify it creates without panic
    assert!(true, "TrainingDataGenerator::new() should not panic");
}

#[test]
fn test_generator_set_search_depth() {
    let mut gen = TrainingDataGenerator::new();
    gen.set_search_depth(12);
    // No direct way to verify private field, but shouldn't panic
}

#[test]
fn test_generator_default_trait() {
    let gen = TrainingDataGenerator::default();
    // Verify Default trait works
    let positions = gen.generate_tactical_positions();
    assert!(!positions.is_empty());
}

#[test]
fn test_generate_tactical_positions() {
    let gen = TrainingDataGenerator::new();
    let positions = gen.generate_tactical_positions();

    assert!(positions.len() >= 5, "Should generate at least 5 tactical positions");

    // Verify each position has required fields
    for pos in &positions {
        assert!(!pos.fen.is_empty(), "FEN should not be empty");
        assert!(!pos.description.is_empty(), "Description should not be empty");
        assert!(pos.game_result >= 0.0 && pos.game_result <= 1.0, "Result should be in [0, 1]");
    }
}

#[test]
fn test_generate_tactical_positions_has_back_rank_mate() {
    let gen = TrainingDataGenerator::new();
    let positions = gen.generate_tactical_positions();
    assert!(positions.iter().any(|p| p.description.contains("Back rank")));
}

#[test]
fn test_generate_tactical_positions_have_evals() {
    let gen = TrainingDataGenerator::new();
    let positions = gen.generate_tactical_positions();

    // All tactical positions should have engine evals
    for pos in &positions {
        assert!(pos.engine_eval.is_some(), "Tactical positions should have engine eval");
    }
}

// === ParsedGame Tests ===

#[test]
fn test_parsed_game_new() {
    let game = ParsedGame::new();
    assert!(game.moves.is_empty());
    assert_eq!(game.result, 0.5);
    assert!(game.white_elo.is_none());
    assert!(game.black_elo.is_none());
    assert!(game.metadata.is_empty());
}

#[test]
fn test_parsed_game_default() {
    let game = ParsedGame::default();
    assert!(game.moves.is_empty());
    assert_eq!(game.result, 0.5);
}

#[test]
fn test_parsed_game_with_data() {
    let mut game = ParsedGame::new();
    game.moves.push(Move::new(12, 28, None)); // e2e4
    game.moves.push(Move::new(52, 36, None)); // e7e5
    game.result = 1.0;
    game.white_elo = Some(2000);
    game.black_elo = Some(1900);
    game.metadata.insert("Event".to_string(), "Test".to_string());

    assert_eq!(game.moves.len(), 2);
    assert_eq!(game.result, 1.0);
    assert_eq!(game.white_elo, Some(2000));
}

// === Generate from Games Tests ===

#[test]
fn test_generate_from_empty_games() {
    let gen = TrainingDataGenerator::new();
    let positions = gen.generate_from_games(&[]);
    assert!(positions.is_empty(), "No games should produce no positions");
}

#[test]
fn test_generate_from_short_game() {
    let gen = TrainingDataGenerator::new();
    // A very short game (< 21 moves) won't produce positions due to filtering
    let mut game = ParsedGame::new();
    game.moves = vec![
        Move::new(12, 28, None), // e2e4
        Move::new(52, 36, None), // e7e5
    ];
    game.result = 0.5;

    let positions = gen.generate_from_games(&[game]);
    // Short games are filtered out (need > 10 moves to start sampling)
    assert_eq!(positions.len(), 0, "Very short games should produce no positions");
}

// === CSV Save/Load Tests ===

#[test]
fn test_save_and_load_csv() {
    let gen = TrainingDataGenerator::new();
    let positions = gen.generate_tactical_positions();

    let path = "/tmp/kingfisher_test_training.csv";
    gen.save_to_csv(&positions, path).expect("Should save CSV");

    let loaded = TrainingDataGenerator::load_from_csv(path).expect("Should load CSV");
    assert_eq!(loaded.len(), positions.len(), "Should load same number of positions");

    // Verify fields roundtrip correctly
    for (orig, loaded) in positions.iter().zip(loaded.iter()) {
        assert_eq!(orig.game_result, loaded.game_result, "Game result should match");
    }

    // Clean up
    std::fs::remove_file(path).ok();
}

#[test]
fn test_load_csv_nonexistent_file() {
    let result = TrainingDataGenerator::load_from_csv("/tmp/nonexistent_kingfisher_test.csv");
    assert!(result.is_err(), "Should error on nonexistent file");
}

#[test]
fn test_save_empty_positions() {
    let gen = TrainingDataGenerator::new();
    let path = "/tmp/kingfisher_test_empty.csv";
    gen.save_to_csv(&[], path).expect("Should save empty CSV");

    let loaded = TrainingDataGenerator::load_from_csv(path).expect("Should load empty CSV");
    assert_eq!(loaded.len(), 0, "Should load 0 positions from empty file");

    std::fs::remove_file(path).ok();
}
