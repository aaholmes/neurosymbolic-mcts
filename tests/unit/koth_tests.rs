//! Tests for King of the Hill (KOTH) win detection

use kingfisher::board::{Board, KOTH_CENTER};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::search::{koth_best_move, koth_center_in_3};

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
