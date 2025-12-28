//! Unit tests for move generation correctness

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use crate::common::{board_from_fen, legal_moves_set, positions, generate_legal_moves};

#[test]
fn test_en_passant_generation() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::EN_PASSANT);
    let moves = legal_moves_set(&board, &move_gen);
    
    // bxa6 e.p. should be legal
    let ep_capture = Move::new(33, 40, None); // b5 to a6
    assert!(
        moves.contains(&ep_capture),
        "En passant capture should be generated"
    );
}

#[test]
fn test_castling_generation() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::CASTLING_BOTH);
    let moves = legal_moves_set(&board, &move_gen);
    
    let kingside = Move::new(4, 6, None); // e1 to g1
    let queenside = Move::new(4, 2, None); // e1 to c1
    
    assert!(moves.contains(&kingside), "Kingside castling should be legal");
    assert!(moves.contains(&queenside), "Queenside castling should be legal");
}

#[test]
fn test_castling_blocked_by_check() {
    let move_gen = MoveGen::new();
    // King in check, castling should be illegal
    // Removed pawn on e2 so Queen on e5 checks King on e1
    let board = board_from_fen("r3k2r/pppp1ppp/8/4q3/8/8/PPPP1PPP/R3K2R w KQkq - 0 1");
    let moves = legal_moves_set(&board, &move_gen);
    
    let kingside = Move::new(4, 6, None);
    let queenside = Move::new(4, 2, None);
    
    assert!(!moves.contains(&kingside), "Cannot castle out of check (kingside)");
    assert!(!moves.contains(&queenside), "Cannot castle out of check (queenside)");
}

#[test]
fn test_promotion_moves() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::PROMOTION);
    let moves: Vec<Move> = generate_legal_moves(&board, &move_gen);
    
    // a7-a8 should generate 4 promotion moves
    let promotions: Vec<_> = moves.iter()
        .filter(|m| m.from == 48 && m.to == 56)
        .collect();
    
    assert_eq!(promotions.len(), 4, "Should have 4 promotion options (Q, R, B, N)");
}