//! Unit tests for Board representation and FEN parsing

use kingfisher::board::{Board, KOTH_CENTER};
use kingfisher::move_generation::MoveGen;
use kingfisher::piece_types::{WHITE, BLACK, PAWN, KING};
use crate::common::{board_from_fen, positions};

#[test]
fn test_starting_position_parsing() {
    let board = board_from_fen(positions::STARTING);
    
    // Verify side to move
    assert!(board.w_to_move, "Starting position should be White to move");
    
    // Verify castling rights
    assert!(board.castling_rights.white_kingside);
    assert!(board.castling_rights.white_queenside);
    assert!(board.castling_rights.black_kingside);
    assert!(board.castling_rights.black_queenside);
    
    // Verify piece counts
    assert_eq!(board.get_piece_bitboard(WHITE, PAWN).count_ones(), 8);
    assert_eq!(board.get_piece_bitboard(BLACK, PAWN).count_ones(), 8);
    assert_eq!(board.get_piece_bitboard(WHITE, KING).count_ones(), 1);
    assert_eq!(board.get_piece_bitboard(BLACK, KING).count_ones(), 1);
}

#[test]
fn test_fen_roundtrip() {
    let test_fens = [
        positions::STARTING,
        positions::EN_PASSANT,
        positions::CASTLING_BOTH,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    ];
    
    for fen in test_fens {
        let board = board_from_fen(fen);
        let roundtrip = board.to_fen().expect("Failed to generate FEN");
        // Note: FEN might normalize differently, so parse and compare boards
        let board2 = board_from_fen(&roundtrip);
        // Compare FENs again as Board doesn't implement PartialEq
        assert_eq!(board.to_fen().unwrap(), board2.to_fen().unwrap(), "FEN roundtrip failed for: {}", fen);
    }
}

#[test]
fn test_en_passant_square_parsing() {
    let board = board_from_fen(positions::EN_PASSANT);
    assert_eq!(board.en_passant(), Some(40)); // a6 = index 40 (rank 5, file 0)
}

#[test]
fn test_checkmate_detection() {
    let move_gen = MoveGen::new();
    
    // Back rank mate position
    // Actually need a real checkmate position
    let mated = board_from_fen("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1"); // Qb7#
    
    let (is_checkmate, is_stalemate) = mated.is_checkmate_or_stalemate(&move_gen);
    assert!(is_checkmate, "Position should be checkmate");
    assert!(!is_stalemate);
}

#[test]
fn test_stalemate_detection() {
    let move_gen = MoveGen::new();
    let stalemate = board_from_fen(positions::STALEMATE);
    
    let (is_checkmate, is_stalemate_result) = stalemate.is_checkmate_or_stalemate(&move_gen);
    assert!(!is_checkmate, "Position should not be checkmate");
    assert!(is_stalemate_result, "Position should be stalemate");
}

#[test]
fn test_koth_center_squares() {
    // Verify KOTH_CENTER constant covers d4, e4, d5, e5
    let center_squares = [27, 28, 35, 36]; // d4, e4, d5, e5
    for sq in center_squares {
        assert!(
            (KOTH_CENTER >> sq) & 1 == 1,
            "Square {} should be in KOTH center", sq
        );
    }
    assert_eq!(KOTH_CENTER.count_ones(), 4, "KOTH center should have exactly 4 squares");
}
