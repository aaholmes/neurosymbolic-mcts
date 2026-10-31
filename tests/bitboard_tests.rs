use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::piece_types::{BLACK, KING, KNIGHT, PAWN, ROOK, WHITE};

#[test]
fn test_initial_position() {
    let board = Board::new();
    assert_eq!(board.get_piece(0), Some((WHITE, ROOK))); // White Rook
    assert_eq!(board.get_piece(4), Some((WHITE, KING))); // White King
    assert_eq!(board.get_piece(63), Some((BLACK, ROOK))); // Black Rook
    assert!(board.w_to_move);
    assert!(
        board.castling_rights.white_kingside
            && board.castling_rights.white_queenside
            && board.castling_rights.black_kingside
            && board.castling_rights.black_queenside
    );
}

#[test]
fn test_get_piece_bitboard() {
    let board = Board::new(); // Assume this creates a standard starting position

    // Test white pawns
    assert_eq!(board.get_piece_bitboard(WHITE, PAWN), 0x000000000000FF00);

    // Test black pawns
    assert_eq!(board.get_piece_bitboard(BLACK, PAWN), 0x00FF000000000000);

    // Test white knights (should be on b1 and g1)
    assert_eq!(board.get_piece_bitboard(WHITE, KNIGHT), 0x0000000000000042);

    // Test black knights (should be on b8 and g8)
    assert_eq!(board.get_piece_bitboard(BLACK, KNIGHT), 0x4200000000000000);

    // Test white king (should be on e1)
    assert_eq!(board.get_piece_bitboard(WHITE, KING), 0x0000000000000010);

    // Test black king (should be on e8)
    assert_eq!(board.get_piece_bitboard(BLACK, KING), 0x1000000000000000);

    // Make a move and test again
    let e2e4 = Move::from_uci("e2e4").unwrap();
    let new_board = board.apply_move_to_board(e2e4);

    // White pawns should have changed
    assert_eq!(
        new_board.get_piece_bitboard(WHITE, PAWN),
        0x000000001000EF00
    );
}

#[test]
fn test_make_move() {
    let mut board = BoardStack::new();
    let move_gen = MoveGen::new();
    let moves = move_gen.gen_pseudo_legal_moves(&board.current_state());
    let e4_move = moves
        .1
        .iter()
        .find(|&m| m.from() == 12 && m.to() == 28)
        .unwrap();

    board.make_move(*e4_move);
    board.current_state().print();
    assert_eq!(board.current_state().get_piece(28), Some((WHITE, PAWN))); // White Pawn on e4
    assert_eq!(board.current_state().get_piece(12), None); // e2 is empty
    assert!(!board.current_state().w_to_move);
}

#[test]
fn test_is_legal() {
    // Legal position
    let board =
        Board::new_from_fen("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4");
    let move_gen = MoveGen::new();
    assert!(board.is_legal(&move_gen));
}

#[test]
fn test_is_illegal() {
    // Illegal position (black king in check)
    let illegal_board =
        Board::new_from_fen("r1bq1bnr/pppp1kpp/2n2p2/4p3/2BPP3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 5");
    let move_gen = MoveGen::new();
    assert!(!illegal_board.is_legal(&move_gen));
}

#[test]
fn test_checkmate_detection() {
    let board =
        Board::new_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    let move_gen = MoveGen::new();
    let (is_checkmate, is_stalemate) = board.is_checkmate_or_stalemate(&move_gen);
    assert!(is_checkmate);
    assert!(!is_stalemate);
}

#[test]
fn test_stalemate_detection() {
    let board = Board::new_from_fen("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1");
    board.print();
    let move_gen = MoveGen::new();
    let (is_checkmate, is_stalemate) = board.is_checkmate_or_stalemate(&move_gen);
    assert!(!is_checkmate);
    assert!(is_stalemate);
}
