//! Unit tests for apply_move_to_board (making moves on the board)

use kingfisher::board::Board;
use kingfisher::board_utils::sq_ind_to_bit;
use kingfisher::move_types::Move;
use kingfisher::piece_types::{BLACK, BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use crate::common::{board_from_fen, positions};

#[test]
fn test_standard_pawn_push() {
    let board = Board::new();
    // e2e3 (single push)
    let mv = Move::new(12, 20, None);
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(20), Some((WHITE, PAWN)), "Pawn should be on e3");
    assert_eq!(new_board.get_piece(12), None, "e2 should be empty");
    assert!(!new_board.w_to_move, "Should be black to move");
    assert_eq!(new_board.en_passant(), None, "Single push should not set en passant");
}

#[test]
fn test_double_pawn_push_sets_en_passant() {
    let board = Board::new();
    // e2e4 (double push)
    let mv = Move::new(12, 28, None);
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(28), Some((WHITE, PAWN)), "Pawn should be on e4");
    assert_eq!(new_board.get_piece(12), None, "e2 should be empty");
    assert_eq!(new_board.en_passant(), Some(20), "En passant should be set to e3 (20)");
}

#[test]
fn test_en_passant_capture() {
    // Position with white pawn on b5 and black pawn on a5 after double push, en passant on a6
    let board = board_from_fen(positions::EN_PASSANT);
    // b5xa6 en passant
    let mv = Move::new(33, 40, None); // b5 -> a6

    let new_board = board.apply_move_to_board(mv);
    assert_eq!(new_board.get_piece(40), Some((WHITE, PAWN)), "White pawn should be on a6");
    assert_eq!(new_board.get_piece(33), None, "b5 should be empty");
    // The captured pawn on a5 (32) should be removed
    assert_eq!(new_board.get_piece(32), None, "Captured pawn on a5 should be removed");
}

#[test]
fn test_kingside_castling() {
    let board = board_from_fen(positions::CASTLING_BOTH);
    // White kingside castle: e1g1
    let mv = Move::new(4, 6, None);
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(6), Some((WHITE, KING)), "King should be on g1");
    assert_eq!(new_board.get_piece(5), Some((WHITE, ROOK)), "Rook should be on f1");
    assert_eq!(new_board.get_piece(4), None, "e1 should be empty");
    assert_eq!(new_board.get_piece(7), None, "h1 should be empty");
    assert!(!new_board.castling_rights.white_kingside);
    assert!(!new_board.castling_rights.white_queenside);
}

#[test]
fn test_queenside_castling() {
    let board = board_from_fen(positions::CASTLING_BOTH);
    // White queenside castle: e1c1
    let mv = Move::new(4, 2, None);
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(2), Some((WHITE, KING)), "King should be on c1");
    assert_eq!(new_board.get_piece(3), Some((WHITE, ROOK)), "Rook should be on d1");
    assert_eq!(new_board.get_piece(4), None, "e1 should be empty");
    assert_eq!(new_board.get_piece(0), None, "a1 should be empty");
    assert!(!new_board.castling_rights.white_kingside);
    assert!(!new_board.castling_rights.white_queenside);
}

#[test]
fn test_pawn_promotion() {
    let board = board_from_fen(positions::PROMOTION);
    // a7a8=Q
    let mv = Move::new(48, 56, Some(QUEEN));
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(56), Some((WHITE, QUEEN)), "Should be a queen on a8");
    assert_eq!(new_board.get_piece(48), None, "a7 should be empty");
    // Verify the pawn bitboard no longer has the pawn
    assert_eq!(
        new_board.get_piece_bitboard(WHITE, PAWN) & sq_ind_to_bit(56),
        0,
        "No pawn should be on a8"
    );
    assert_ne!(
        new_board.get_piece_bitboard(WHITE, QUEEN) & sq_ind_to_bit(56),
        0,
        "Queen should be on a8 in queen bitboard"
    );
}

// === Additional make_move tests ===

#[test]
fn test_promotion_to_knight() {
    let board = board_from_fen(positions::PROMOTION);
    let mv = Move::new(48, 56, Some(KNIGHT));
    let new_board = board.apply_move_to_board(mv);
    assert_eq!(new_board.get_piece(56), Some((WHITE, KNIGHT)));
}

#[test]
fn test_promotion_to_rook() {
    let board = board_from_fen(positions::PROMOTION);
    let mv = Move::new(48, 56, Some(ROOK));
    let new_board = board.apply_move_to_board(mv);
    assert_eq!(new_board.get_piece(56), Some((WHITE, ROOK)));
}

#[test]
fn test_promotion_to_bishop() {
    let board = board_from_fen(positions::PROMOTION);
    let mv = Move::new(48, 56, Some(BISHOP));
    let new_board = board.apply_move_to_board(mv);
    assert_eq!(new_board.get_piece(56), Some((WHITE, BISHOP)));
}

#[test]
fn test_black_kingside_castling() {
    let board = board_from_fen(positions::CASTLING_BOTH);
    // First make a white move (e.g., a2a3) to give black the turn
    let white_move = Move::new(8, 16, None); // a2a3
    let board_after_white = board.apply_move_to_board(white_move);

    // Black kingside castle: e8g8
    let mv = Move::new(60, 62, None);
    let new_board = board_after_white.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(62), Some((BLACK, KING)), "King should be on g8");
    assert_eq!(new_board.get_piece(61), Some((BLACK, ROOK)), "Rook should be on f8");
    assert_eq!(new_board.get_piece(60), None, "e8 should be empty");
    assert_eq!(new_board.get_piece(63), None, "h8 should be empty");
    assert!(!new_board.castling_rights.black_kingside);
    assert!(!new_board.castling_rights.black_queenside);
}

#[test]
fn test_black_queenside_castling() {
    let board = board_from_fen(positions::CASTLING_BOTH);
    let white_move = Move::new(8, 16, None);
    let board_after_white = board.apply_move_to_board(white_move);

    // Black queenside castle: e8c8
    let mv = Move::new(60, 58, None);
    let new_board = board_after_white.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(58), Some((BLACK, KING)), "King should be on c8");
    assert_eq!(new_board.get_piece(59), Some((BLACK, ROOK)), "Rook should be on d8");
    assert_eq!(new_board.get_piece(60), None, "e8 should be empty");
    assert_eq!(new_board.get_piece(56), None, "a8 should be empty");
}

#[test]
fn test_rook_move_removes_castling_rights() {
    let board = board_from_fen(positions::CASTLING_BOTH);
    // Move the h1 rook: h1h2
    let mv = Move::new(7, 15, None);
    let new_board = board.apply_move_to_board(mv);

    assert!(!new_board.castling_rights.white_kingside, "Moving h1 rook should remove kingside rights");
    assert!(new_board.castling_rights.white_queenside, "Queenside rights should remain");
}

#[test]
fn test_a1_rook_move_removes_queenside_rights() {
    let board = board_from_fen(positions::CASTLING_BOTH);
    // Move the a1 rook: a1a2
    let mv = Move::new(0, 8, None);
    let new_board = board.apply_move_to_board(mv);

    assert!(new_board.castling_rights.white_kingside, "Kingside rights should remain");
    assert!(!new_board.castling_rights.white_queenside, "Moving a1 rook should remove queenside rights");
}

#[test]
fn test_capturing_h1_rook_removes_castling() {
    // Position where black can capture the h1 rook
    let board = board_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPq/R3K2R b KQkq - 0 1");
    // Qxh1 (sq 15 -> sq 7) -- actually Qh2 is sq 15, let me use the correct position
    // Black queen on h2 captures rook on h1: h2h1
    let mv = Move::new(15, 7, None);
    let new_board = board.apply_move_to_board(mv);

    assert!(!new_board.castling_rights.white_kingside,
        "Capturing rook on h1 should remove white kingside rights");
}

#[test]
fn test_capturing_a1_rook_removes_castling() {
    // Position where black can capture the a1 rook
    let board = board_from_fen("r3k2r/pppppppp/8/8/8/8/qPPPPPPP/R3K2R b KQkq - 0 1");
    // Qxa1: a2 (sq 8) captures a1 (sq 0)
    let mv = Move::new(8, 0, None);
    let new_board = board.apply_move_to_board(mv);

    assert!(!new_board.castling_rights.white_queenside,
        "Capturing rook on a1 should remove white queenside rights");
}

#[test]
fn test_halfmove_clock_increments_on_normal_move() {
    // Position with knight that can move
    let board = Board::new();
    // Nb1-c3 (knight move, not a pawn move or capture)
    let mv = Move::new(1, 18, None);
    let new_board = board.apply_move_to_board(mv);
    assert_eq!(new_board.halfmove_clock(), 1, "Halfmove clock should be 1 after knight move");
}

#[test]
fn test_halfmove_clock_resets_on_pawn_move() {
    // Start with halfmove > 0
    let board = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 3");
    let mv = Move::new(12, 28, None); // e2e4
    let new_board = board.apply_move_to_board(mv);
    assert_eq!(new_board.halfmove_clock(), 0, "Halfmove clock should reset on pawn move");
}

#[test]
fn test_halfmove_clock_resets_on_capture() {
    // White knight captures black queen on d5
    let board = board_from_fen("8/8/8/3q4/4N3/8/8/K6k w - - 10 20");
    let mv = Move::new(28, 35, None); // Ne4xd5
    let new_board = board.apply_move_to_board(mv);
    assert_eq!(new_board.halfmove_clock(), 0, "Halfmove clock should reset on capture");
}

#[test]
fn test_fullmove_number_increments_after_black_move() {
    let board = Board::new();
    // White move
    let mv1 = Move::new(12, 28, None); // e2e4
    let board2 = board.apply_move_to_board(mv1);
    // Check via FEN that fullmove is still 1 after white's move
    let fen2 = board2.to_fen().unwrap();
    assert!(fen2.ends_with(" 0 1"), "FEN should end with '0 1' after white move, got: {}", fen2);

    // Black move
    let mv2 = Move::new(52, 36, None); // e7e5
    let board3 = board2.apply_move_to_board(mv2);
    let fen3 = board3.to_fen().unwrap();
    assert!(fen3.ends_with(" 0 2"), "FEN should end with '0 2' after black move, got: {}", fen3);
}

#[test]
fn test_side_to_move_toggles() {
    let board = Board::new();
    assert!(board.w_to_move);

    let mv = Move::new(12, 28, None);
    let board2 = board.apply_move_to_board(mv);
    assert!(!board2.w_to_move);

    let mv2 = Move::new(52, 36, None);
    let board3 = board2.apply_move_to_board(mv2);
    assert!(board3.w_to_move);
}

#[test]
fn test_en_passant_clears_after_non_double_push() {
    let board = Board::new();
    // Double pawn push: e2e4
    let mv1 = Move::new(12, 28, None);
    let board2 = board.apply_move_to_board(mv1);
    assert!(board2.en_passant().is_some());

    // Non-double-push black move: e7e6
    let mv2 = Move::new(52, 44, None);
    let board3 = board2.apply_move_to_board(mv2);
    assert_eq!(board3.en_passant(), None, "En passant should clear after non-double push");
}

#[test]
fn test_zobrist_hash_updated_after_move() {
    let board = Board::new();
    let hash_before = board.compute_zobrist_hash();

    let mv = Move::new(12, 28, None); // e2e4
    let new_board = board.apply_move_to_board(mv);
    let hash_after = new_board.compute_zobrist_hash();

    assert_ne!(hash_before, hash_after, "Zobrist hash should change after a move");
}

#[test]
fn test_pieces_occupancy_updated() {
    let board = Board::new();
    let mv = Move::new(12, 28, None); // e2e4
    let new_board = board.apply_move_to_board(mv);

    // e4 should be in white occupancy
    assert_ne!(new_board.get_color_occupancy(WHITE) & sq_ind_to_bit(28), 0);
    // e2 should not be in white occupancy
    assert_eq!(new_board.get_color_occupancy(WHITE) & sq_ind_to_bit(12), 0);
    // e4 should be in all occupancy
    assert_ne!(new_board.get_all_occupancy() & sq_ind_to_bit(28), 0);
}
