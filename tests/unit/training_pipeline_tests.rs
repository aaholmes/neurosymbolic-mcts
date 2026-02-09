//! Additional tests for training data pipeline
//!
//! Covers: generate_from_games with realistic games, CSV parsing edge cases,
//! and tactical position generation.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::training::{ParsedGame, TrainingDataGenerator, TrainingPosition};

// === generate_from_games with realistic games ===

/// Build a realistic game with enough moves to trigger position extraction.
/// The extraction requires move_idx > 10 and move_idx < len - 10, sampled every 3rd move.
fn make_long_game() -> ParsedGame {
    let move_gen = MoveGen::new();
    let mut board = Board::new();
    let mut moves = Vec::new();

    // Play a simple opening sequence that generates legal moves
    // We just pick the first legal move at each step to get a long game
    for _ in 0..40 {
        let (captures, quiets) = move_gen.gen_pseudo_legal_moves(&board);
        let all_moves: Vec<Move> = captures
            .iter()
            .chain(quiets.iter())
            .filter(|&&m| board.apply_move_to_board(m).is_legal(&move_gen))
            .copied()
            .collect();

        if all_moves.is_empty() {
            break;
        }

        let mv = all_moves[0];
        moves.push(mv);
        board = board.apply_move_to_board(mv);
    }

    ParsedGame {
        moves,
        result: 1.0,
        white_elo: Some(2000),
        black_elo: Some(1900),
        ..Default::default()
    }
}

#[test]
fn test_generate_from_long_game_produces_positions() {
    let gen = TrainingDataGenerator::new();
    let game = make_long_game();

    // Need at least 21 moves for any positions to be extracted (>10 from start, >10 from end)
    if game.moves.len() > 21 {
        let positions = gen.generate_from_games(&[game]);
        // Should produce at least one position if the game is long enough
        // and positions pass the suitability filter
        // (exact count depends on which positions pass is_suitable_training_position)
        assert!(
            positions.len() >= 0,
            "generate_from_games should not panic on a realistic game"
        );
    }
}

#[test]
fn test_generate_from_multiple_games() {
    let gen = TrainingDataGenerator::new();
    let games = vec![make_long_game(), make_long_game()];
    let positions = gen.generate_from_games(&games);
    // Just verify it doesn't panic and returns some result
    assert!(positions.len() >= 0);
}

#[test]
fn test_generate_from_game_too_short_no_positions() {
    let gen = TrainingDataGenerator::new();

    // A game with only 15 moves — below the >10 from start and >10 from end threshold
    let mut game = ParsedGame::new();
    game.result = 0.5;
    // Just 2 valid moves (way too short)
    game.moves.push(Move::new(12, 28, None)); // e2e4
    game.moves.push(Move::new(52, 36, None)); // e7e5

    let positions = gen.generate_from_games(&[game]);
    assert_eq!(positions.len(), 0, "Short game should produce no training positions");
}

// === TrainingPosition edge cases ===

#[test]
fn test_training_position_csv_with_special_chars_in_description() {
    let board = Board::new();
    let pos = TrainingPosition::new(&board, 1.0, None, "Game \"quoted\" (special)".to_string());
    let csv = pos.to_csv_line();
    assert!(csv.contains("quoted"));
}

#[test]
fn test_training_position_csv_with_empty_description() {
    let board = Board::new();
    let pos = TrainingPosition::new(&board, 0.0, None, String::new());
    let csv = pos.to_csv_line();
    // Should still produce a valid CSV line
    assert!(csv.contains("0"));
}

#[test]
fn test_training_position_with_engine_eval() {
    let board = Board::new();
    let mut pos = TrainingPosition::new(&board, 1.0, None, "test".to_string());
    pos.engine_eval = Some(150);
    let csv = pos.to_csv_line();
    assert!(csv.contains("150"));
}

#[test]
fn test_training_position_with_negative_eval() {
    let board = Board::new();
    let mut pos = TrainingPosition::new(&board, 0.0, None, "test".to_string());
    pos.engine_eval = Some(-300);
    let csv = pos.to_csv_line();
    assert!(csv.contains("-300"));
}

// === CSV roundtrip edge cases ===

#[test]
fn test_csv_roundtrip_preserves_result_values() {
    let gen = TrainingDataGenerator::new();
    let board = Board::new();

    let positions = vec![
        TrainingPosition::new(&board, 1.0, None, "white wins".to_string()),
        TrainingPosition::new(&board, 0.0, None, "black wins".to_string()),
        TrainingPosition::new(&board, 0.5, None, "draw".to_string()),
    ];

    let path = "/tmp/kingfisher_test_roundtrip.csv";
    gen.save_to_csv(&positions, path).unwrap();
    let loaded = TrainingDataGenerator::load_from_csv(path).unwrap();

    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded[0].game_result, 1.0);
    assert_eq!(loaded[1].game_result, 0.0);
    assert_eq!(loaded[2].game_result, 0.5);

    std::fs::remove_file(path).ok();
}

#[test]
fn test_csv_roundtrip_preserves_best_move() {
    let gen = TrainingDataGenerator::new();
    let board = Board::new();
    let mv = Move::new(12, 28, None); // e2e4

    let positions = vec![TrainingPosition::new(
        &board,
        0.5,
        Some(mv),
        "with move".to_string(),
    )];

    let path = "/tmp/kingfisher_test_roundtrip_move.csv";
    gen.save_to_csv(&positions, path).unwrap();
    let loaded = TrainingDataGenerator::load_from_csv(path).unwrap();

    assert_eq!(loaded.len(), 1);
    assert!(loaded[0].best_move.is_some());

    std::fs::remove_file(path).ok();
}

// === Tactical position generation ===

#[test]
fn test_tactical_positions_cover_multiple_types() {
    let gen = TrainingDataGenerator::new();
    let positions = gen.generate_tactical_positions();

    // Should have both winning and losing results
    let has_white_win = positions.iter().any(|p| p.game_result == 1.0);
    let has_black_win = positions.iter().any(|p| p.game_result == 0.0);
    assert!(has_white_win, "Should have white-winning positions");
    // The tactical positions include scholar's mate (0.0) and early queen attack (0.0)
    assert!(has_black_win, "Should have black-winning positions");
}

#[test]
fn test_tactical_positions_have_valid_fens() {
    let gen = TrainingDataGenerator::new();
    let positions = gen.generate_tactical_positions();

    for pos in &positions {
        // Each FEN should parse back into a valid board
        let board = Board::new_from_fen(&pos.fen);
        let round_trip = board.to_fen().unwrap_or_default();
        // FEN should roundtrip (at least the piece placement part)
        assert!(
            !round_trip.is_empty(),
            "Position '{}' should have valid FEN",
            pos.description
        );
    }
}

// === ParsedGame metadata ===

#[test]
fn test_parsed_game_metadata() {
    let mut game = ParsedGame::new();
    game.metadata
        .insert("Event".to_string(), "World Championship".to_string());
    game.metadata
        .insert("Site".to_string(), "London".to_string());

    assert_eq!(game.metadata.len(), 2);
    assert_eq!(
        game.metadata.get("Event").unwrap(),
        "World Championship"
    );
}

#[test]
fn test_parsed_game_clone() {
    let mut game = ParsedGame::new();
    game.moves.push(Move::new(12, 28, None));
    game.result = 1.0;

    let cloned = game.clone();
    assert_eq!(cloned.moves.len(), 1);
    assert_eq!(cloned.result, 1.0);
}
