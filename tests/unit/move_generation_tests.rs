//! Unit tests for move generation correctness

use crate::common::{board_from_fen, generate_legal_moves, legal_moves_set, positions};
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;

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

    assert!(
        moves.contains(&kingside),
        "Kingside castling should be legal"
    );
    assert!(
        moves.contains(&queenside),
        "Queenside castling should be legal"
    );
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

    assert!(
        !moves.contains(&kingside),
        "Cannot castle out of check (kingside)"
    );
    assert!(
        !moves.contains(&queenside),
        "Cannot castle out of check (queenside)"
    );
}

#[test]
fn test_promotion_moves() {
    let move_gen = MoveGen::new();
    let board = board_from_fen(positions::PROMOTION);
    let moves: Vec<Move> = generate_legal_moves(&board, &move_gen);

    // a7-a8 should generate 4 promotion moves
    let promotions: Vec<_> = moves
        .iter()
        .filter(|m| m.from() == 48 && m.to() == 56)
        .collect();

    assert_eq!(
        promotions.len(),
        4,
        "Should have 4 promotion options (Q, R, B, N)"
    );
}

// Checking moves generated directly must be exactly the pseudo-legal moves
// that give check.

fn checking_moves_by_filtering(board: &Board, mg: &MoveGen) -> Vec<Move> {
    let (captures, quiet) = mg.gen_pseudo_legal_moves(board);
    let mut v: Vec<Move> = captures
        .into_iter()
        .chain(quiet)
        .filter(|m| board.gives_check(*m, mg))
        .collect();
    v.sort_by_key(|m| (m.from(), m.to(), m.promotion()));
    v
}

fn checking_moves_generated(board: &Board, mg: &MoveGen) -> Vec<Move> {
    let mut v = Vec::new();
    mg.gen_checking_moves_into(board, &mut v);
    v.sort_by_key(|m| (m.from(), m.to(), m.promotion()));
    v
}

#[test]
fn checking_moves_are_generated_directly_in_special_cases() {
    let mg = MoveGen::new();
    let fens = [
        "3k4/P7/8/8/8/8/8/K7 w - - 0 1",     // a8=Q+ and a8=R+
        "8/1P6/2k5/8/8/8/8/K7 w - - 0 1",    // b8=N+
        "8/8/8/k2pP2R/8/8/8/7K w - d6 0 1",  // exd6 e.p. uncovers the rook
        "5k2/8/8/8/8/8/8/4K2R w K - 0 1",    // O-O+
        "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1",    // O-O-O+
        "4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", // knight uncovers the rook
        "4k3/8/8/8/8/8/4K3/4R3 w - - 0 1",   // king uncovers the rook
        "7k/8/8/8/8/8/1P6/B6K w - - 0 1",    // pawn push uncovers the bishop
        "8/8/8/3k4/8/8/2P5/K7 w - - 0 1",    // c4+ by a double push
        "4k3/8/8/8/8/8/8/1Q2K3 w - - 0 1",   // queen checks on file, rank, diagonal
        "4k3/8/3P4/8/8/8/8/4K3 w - - 0 1",   // d7+
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "8/8/8/2k5/3Pp3/8/8/4K2B b - d3 0 1", // exd3 e.p. by Black
    ];
    for fen in fens {
        let board = Board::new_from_fen(fen);
        assert_eq!(
            checking_moves_generated(&board, &mg),
            checking_moves_by_filtering(&board, &mg),
            "{fen}"
        );
    }
}

#[test]
fn checking_moves_are_generated_directly_on_game_positions_and_their_children() {
    let mg = MoveGen::new();
    let fens = include_str!("../fixtures/mate_positions.fen");
    let mut with_checks = 0;
    for fen in fens.lines().filter(|l| !l.trim().is_empty()) {
        let board = Board::new_from_fen(fen);
        let (captures, quiet) = mg.gen_pseudo_legal_moves(&board);
        let children: Vec<Board> = captures
            .into_iter()
            .chain(quiet)
            .filter(|m| board.is_legal_after_move(*m, &mg))
            .map(|m| board.apply_move_to_board(m))
            .collect();
        for b in std::iter::once(board).chain(children) {
            let expected = checking_moves_by_filtering(&b, &mg);
            with_checks += !expected.is_empty() as usize;
            assert_eq!(
                checking_moves_generated(&b, &mg),
                expected,
                "{}",
                b.to_fen().unwrap()
            );
        }
    }
    assert!(with_checks > 1000, "{with_checks}");
}

// When the side to move is in check, the generated evasions must contain
// exactly the legal moves, once filtered for legality.

fn legal_sorted(board: &Board, mg: &MoveGen, moves: Vec<Move>) -> Vec<Move> {
    let mut v: Vec<Move> = moves
        .into_iter()
        .filter(|m| board.is_legal_after_move(*m, mg))
        .collect();
    v.sort_by_key(|m| (m.from(), m.to(), m.promotion()));
    v
}

fn assert_evasions_complete(board: &Board, mg: &MoveGen) {
    let (captures, quiet) = mg.gen_pseudo_legal_moves(board);
    let all = legal_sorted(board, mg, captures.into_iter().chain(quiet).collect());
    let mut evasions = Vec::new();
    mg.gen_evasions_into(board, &mut evasions);
    let mut unique = evasions.clone();
    unique.sort_by_key(|m| (m.from(), m.to(), m.promotion()));
    unique.dedup();
    assert_eq!(
        unique.len(),
        evasions.len(),
        "duplicates: {}",
        board.to_fen().unwrap()
    );
    assert_eq!(
        legal_sorted(board, mg, evasions),
        all,
        "{}",
        board.to_fen().unwrap()
    );
}

#[test]
fn evasions_are_the_legal_moves_in_special_cases() {
    let mg = MoveGen::new();
    let fens = [
        "4k3/8/8/8/8/8/3PPP2/r3K3 w - - 0 1", // block on the first rank or step aside
        "4k3/8/8/8/8/5n2/8/R3K2R w KQ - 0 1", // knight check: no castling out of check
        "4k3/8/8/8/1b6/8/8/1N2K3 w - - 0 1",  // block with the knight
        "4k3/8/8/8/1b6/8/8/3QK1r1 w - - 0 1", // double check: king moves only
        "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",  // exd3 e.p. captures the checking pawn
        "8/6p1/R6k/8/8/8/8/K7 b - - 0 1",     // g6 blocks by a single push
        "1r2k3/P7/8/8/8/8/8/1K6 w - - 0 1",   // axb8=Q captures the checker by promotion
        "4k3/8/8/8/8/8/1p6/R1K5 w - - 0 1",   // pawn check on the king
        "4k3/8/8/8/3b4/8/5B2/r5K1 w - - 0 1", // pinned bishop cannot block
        "3k4/8/8/8/8/8/2P5/1K1R4 b - - 0 1",  // black in check on the d-file
        "8/6p1/8/R6k/8/8/8/K7 b - - 0 1",     // g5 blocks by a double push
    ];
    for fen in fens {
        let board = Board::new_from_fen(fen);
        assert!(board.is_check(&mg), "{fen}");
        assert_evasions_complete(&board, &mg);
    }
}

#[test]
fn evasions_are_the_legal_moves_on_game_positions_and_their_children() {
    let mg = MoveGen::new();
    let fens = include_str!("../fixtures/mate_positions.fen");
    let mut in_check = 0;
    for fen in fens.lines().filter(|l| !l.trim().is_empty()) {
        let board = Board::new_from_fen(fen);
        let (captures, quiet) = mg.gen_pseudo_legal_moves(&board);
        let children: Vec<Board> = captures
            .into_iter()
            .chain(quiet)
            .filter(|m| board.is_legal_after_move(*m, &mg))
            .map(|m| board.apply_move_to_board(m))
            .collect();
        for b in std::iter::once(board).chain(children) {
            if b.is_check(&mg) {
                in_check += 1;
                assert_evasions_complete(&b, &mg);
            }
        }
    }
    assert!(in_check > 500, "{in_check}");
}
