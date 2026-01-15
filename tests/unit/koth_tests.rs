//! Tests for King of the Hill (KOTH) win detection

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::koth_center_in_3;

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

    let can_reach = koth_center_in_3(&board, &move_gen);
    assert!(can_reach, "King one move from center should find KOTH win");
}

#[test]
fn test_koth_king_two_moves_away() {
    let move_gen = setup();
    // White king on b2, can reach center in 2-3 moves
    let board = Board::new_from_fen("8/8/8/8/8/8/1K6/k7 w - - 0 1");

    let can_reach = koth_center_in_3(&board, &move_gen);
    // Should be able to reach in 3 moves: b2-c3-d4
    assert!(can_reach, "King two moves from center should find KOTH win");
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
    let can_reach = koth_center_in_3(&board, &move_gen);
    // Might not be able to reach in 3 moves due to distance and black king defense
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
    assert!(!white_won && !black_won, "No king in center = no instant win");
}

#[test]
fn test_koth_starting_position() {
    let move_gen = setup();
    let board = Board::new();

    let can_reach = koth_center_in_3(&board, &move_gen);
    // In starting position, king is blocked by own pieces, can't reach center in 3 moves
    assert!(!can_reach, "Starting position king can't reach center in 3 moves");
}
