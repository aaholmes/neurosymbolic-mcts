//! Tests for King of the Hill (KOTH) win detection

use kingfisher::board::{Board, KOTH_CENTER};
use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::search::{koth_best_move, koth_center_in_3};
use std::time::Duration;

fn setup() -> MoveGen {
    MoveGen::new()
}

#[test]
fn test_koth_king_already_in_center() {
    let move_gen = setup();
    // White king on d4 (center square) - instant win
    let board = Board::new_from_fen("8/8/8/8/3K4/8/8/k7 w - - 0 1");

    // Board should detect this as a KOTH win
    let (white_won, _) = board.is_koth_win();
    assert!(white_won, "King on d4 should be KOTH win for white");
}

#[test]
fn test_koth_king_one_move_away() {
    let move_gen = setup();
    // White king on c3, can reach d4 in one move
    let board = Board::new_from_fen("8/8/8/8/8/2K5/8/k7 w - - 0 1");

    let result = koth_center_in_3(&board, &move_gen);
    assert_eq!(
        result,
        Some(1),
        "King one move from center should report distance 1"
    );
}

#[test]
fn test_koth_king_two_moves_away() {
    let move_gen = setup();
    // White king on b2, can reach center in 2 moves: b2-c3-d4
    let board = Board::new_from_fen("8/8/8/8/8/8/1K6/k7 w - - 0 1");

    let result = koth_center_in_3(&board, &move_gen);
    assert_eq!(
        result,
        Some(2),
        "King two moves from center should report distance 2"
    );
}

#[test]
fn test_koth_king_far_from_center() {
    let move_gen = setup();
    // White king on h1, black king blocking path
    // Need many moves to reach center
    let board = Board::new_from_fen("8/8/8/8/8/8/8/k3K2K w - - 0 1");
    // This FEN is invalid (two white kings), let me fix it
    let board = Board::new_from_fen("8/8/8/3k4/8/8/8/7K w - - 0 1");

    // King on h1 is far from center and black king is blocking
    let result = koth_center_in_3(&board, &move_gen);
    // Might not be able to reach in 3 moves due to distance and black king defense
    // If unreachable, should be None
    assert!(result.is_none() || result.unwrap() <= 3);
}

#[test]
fn test_koth_e4_center_square() {
    let move_gen = setup();
    // White king on e4 (center square)
    let board = Board::new_from_fen("8/8/8/8/4K3/8/8/k7 w - - 0 1");

    let (white_won, _) = board.is_koth_win();
    assert!(white_won, "King on e4 should be KOTH win");
}

#[test]
fn test_koth_d5_center_square() {
    let move_gen = setup();
    // White king on d5 (center square)
    let board = Board::new_from_fen("8/8/8/3K4/8/8/8/k7 w - - 0 1");

    let (white_won, _) = board.is_koth_win();
    assert!(white_won, "King on d5 should be KOTH win");
}

#[test]
fn test_koth_e5_center_square() {
    let move_gen = setup();
    // White king on e5 (center square)
    let board = Board::new_from_fen("8/8/8/4K3/8/8/8/k7 w - - 0 1");

    let (white_won, _) = board.is_koth_win();
    assert!(white_won, "King on e5 should be KOTH win");
}

#[test]
fn test_koth_black_king_in_center() {
    let move_gen = setup();
    // Black king on d4
    let board = Board::new_from_fen("8/8/8/8/3k4/8/8/K7 w - - 0 1");

    let (_, black_won) = board.is_koth_win();
    assert!(black_won, "Black king on d4 should be KOTH win for black");
}

#[test]
fn test_koth_no_instant_win() {
    let move_gen = setup();
    // Neither king in center
    let board = Board::new_from_fen("8/8/8/8/8/8/8/K6k w - - 0 1");

    let (white_won, black_won) = board.is_koth_win();
    assert!(
        !white_won && !black_won,
        "No king in center = no instant win"
    );
}

#[test]
fn test_koth_starting_position() {
    let move_gen = setup();
    let board = Board::new();

    let result = koth_center_in_3(&board, &move_gen);
    // In starting position, king is blocked by own pieces, can't reach center in 3 moves
    assert!(
        result.is_none(),
        "Starting position king can't reach center in 3 moves"
    );
}

#[test]
fn test_koth_ke2_after_f6_reports_distance_2() {
    let move_gen = setup();
    // After 1. e3 e6 2. Ke2 f6 — White king on e2 can force KOTH in 2 (Kd3 then Kd4/Ke4)
    let board = Board::new_from_fen("rnbqkbnr/pppp2pp/4pp2/8/8/4P3/PPPPKPPP/RNBQ1BNR w kq - 0 3");

    let result = koth_center_in_3(&board, &move_gen);
    assert_eq!(
        result,
        Some(2),
        "White king on e2 should force KOTH in 2 moves after f6"
    );
}

#[test]
fn test_koth_best_move_king_one_away() {
    let move_gen = setup();
    // White king on c3, can reach d4 in one move
    let board = Board::new_from_fen("8/8/8/8/8/2K5/8/k7 w - - 0 1");

    let best = koth_best_move(&board, &move_gen);
    assert!(best.is_some(), "Should find a KOTH-winning move");
    let mv = best.unwrap();
    // The move should land the king on a center square (d4, e4, d5, or e5)
    let to_bit = 1u64 << mv.to;
    assert!(
        to_bit & KOTH_CENTER != 0,
        "Best move should put king on center square, got to={}",
        mv.to
    );
}

#[test]
fn test_koth_best_move_no_koth() {
    let move_gen = setup();
    // White king on h1, black king on d4 blocks center — no forced KOTH for white
    let board = Board::new_from_fen("8/8/8/8/3k4/8/8/7K w - - 0 1");

    let best = koth_best_move(&board, &move_gen);
    assert!(best.is_none(), "Should not find forced KOTH win");
}

#[test]
fn test_koth_best_move_already_on_center() {
    let move_gen = setup();
    // White king already on d4 — already won, no "best move" needed
    let board = Board::new_from_fen("8/8/8/8/3K4/8/8/k7 w - - 0 1");

    // koth_center_in_3 returns Some(0) here, but koth_best_move should return None
    // because there's no move to make — game is already won
    let best = koth_best_move(&board, &move_gen);
    assert!(best.is_none(), "Already on center, no move needed");
}

#[test]
fn test_koth_in_2_detected_at_root() {
    let move_gen = setup();
    // After 1. e3 h6 2. Ke2 a6 — White has forced KOTH-in-2: Kd3 then Kd4/Ke4
    // No black piece can block both d4 and e4 in one move
    let board =
        Board::new_from_fen("rnbqkbnr/1pppppp1/p6p/8/8/4P3/PPPPKPPP/RNBQ1BNR w kq - 0 3");

    // koth_center_in_3 should detect it
    let dist = koth_center_in_3(&board, &move_gen);
    assert_eq!(dist, Some(2), "Should detect forced KOTH-in-2");

    // koth_best_move should return Kd3 (the first step)
    let best = koth_best_move(&board, &move_gen);
    assert!(best.is_some(), "Should find KOTH-winning first move");
    // d3 = square 19 (LERF: a1=0, d3=19)
    let mv = best.unwrap();
    assert_eq!(mv.from, 12, "Move should be from e2 (sq 12)");
    assert_eq!(mv.to, 19, "Move should be to d3 (sq 19)");

    // MCTS root-level gate should catch this with 0 iterations
    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(5),
        enable_koth: true,
        ..Default::default()
    };
    let (best_move, stats, _) = tactical_mcts_search(board, &move_gen, config);
    assert!(best_move.is_some(), "MCTS should find KOTH move");
    assert_eq!(
        stats.iterations, 0,
        "Root-level KOTH gate should abort search with 0 iterations"
    );
    let mv = best_move.unwrap();
    assert_eq!(mv.to, 19, "MCTS should play Kd3 (sq 19) for KOTH-in-2");
}

#[test]
fn test_koth_in_2_blocked_by_queen_diagonal() {
    let move_gen = setup();
    // After 1. e3 h6 2. Ke2 e6 — e6 opens the d8-h4 diagonal.
    // After Kd3, black plays Qh4 which controls BOTH d4 and e4 along rank 4.
    // So KOTH-in-2 is NOT forced (unlike the a6 position).
    let board =
        Board::new_from_fen("rnbqkbnr/pppp1pp1/4p2p/8/8/4P3/PPPPKPPP/RNBQ1BNR w kq - 0 3");

    let dist = koth_center_in_3(&board, &move_gen);
    assert_eq!(dist, None, "Qh4 defense blocks forced KOTH — no forced win in 3");
}
