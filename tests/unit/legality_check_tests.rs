//! Tests for is_legal_after_move (clone-free legality checking)
//!
//! Verifies that is_legal_after_move produces the same result as
//! apply_move_to_board followed by is_legal for all pseudo-legal moves.

use crate::common::{board_from_fen, positions};
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;

fn setup() -> MoveGen {
    MoveGen::new()
}

#[test]
fn test_is_legal_after_move_starting_position() {
    let move_gen = setup();
    let board = Board::new();
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);

    for mv in captures.iter().chain(moves.iter()) {
        let expected = board.apply_move_to_board(*mv).is_legal(&move_gen);
        let actual = board.is_legal_after_move(*mv, &move_gen);
        assert_eq!(
            expected,
            actual,
            "is_legal_after_move mismatch for move {} in starting position",
            mv.print_algebraic()
        );
    }
}

#[test]
fn test_is_legal_after_move_pinned_piece() {
    let move_gen = setup();
    // White king on e1, white bishop on d2, black rook on a5
    // The bishop is pinned by the rook on a5 diagonally... no.
    // Better: White king on e1, white knight on e2, black rook on e8
    // The knight on e2 is pinned by the rook on e8 on the e-file
    let board = board_from_fen("4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1");
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);

    for mv in captures.iter().chain(moves.iter()) {
        let expected = board.apply_move_to_board(*mv).is_legal(&move_gen);
        let actual = board.is_legal_after_move(*mv, &move_gen);
        assert_eq!(
            expected,
            actual,
            "is_legal_after_move mismatch for move {} in pinned piece position",
            mv.print_algebraic()
        );
    }
}

#[test]
fn test_is_legal_after_move_en_passant_discovered_check() {
    let move_gen = setup();
    // The notorious en passant discovered check case:
    // White king on a5, white pawn on b5, black pawn on c5 (just double-pushed),
    // black rook on h5. If white plays bxc6 e.p., the b5 pawn leaves and the c5 pawn
    // leaves, discovering a check from the rook on h5.
    let board = board_from_fen("8/8/8/K1pP3r/8/8/8/7k w - c6 0 1");
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);

    let mut found_ep = false;
    for mv in captures.iter().chain(moves.iter()) {
        let expected = board.apply_move_to_board(*mv).is_legal(&move_gen);
        let actual = board.is_legal_after_move(*mv, &move_gen);
        assert_eq!(
            expected,
            actual,
            "is_legal_after_move mismatch for move {} in EP discovered check position",
            mv.print_algebraic()
        );
        if mv.to() == 42 {
            // c6
            found_ep = true;
            // This particular EP should be illegal due to discovered check
            assert!(
                !actual,
                "dxc6 e.p. should be illegal due to discovered check"
            );
        }
    }
    assert!(found_ep, "Should have found the en passant move");
}

#[test]
fn test_is_legal_after_move_castling() {
    let move_gen = setup();
    let board = board_from_fen(positions::CASTLING_BOTH);
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);

    for mv in captures.iter().chain(moves.iter()) {
        let expected = board.apply_move_to_board(*mv).is_legal(&move_gen);
        let actual = board.is_legal_after_move(*mv, &move_gen);
        assert_eq!(
            expected,
            actual,
            "is_legal_after_move mismatch for move {} in castling position",
            mv.print_algebraic()
        );
    }
}

#[test]
fn test_is_legal_after_move_in_check() {
    let move_gen = setup();
    // White king in check from black rook
    let board = board_from_fen("4k3/8/8/8/8/8/8/r3K3 w - - 0 1");
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);

    for mv in captures.iter().chain(moves.iter()) {
        let expected = board.apply_move_to_board(*mv).is_legal(&move_gen);
        let actual = board.is_legal_after_move(*mv, &move_gen);
        assert_eq!(
            expected,
            actual,
            "is_legal_after_move mismatch for move {} when in check",
            mv.print_algebraic()
        );
    }
}

#[test]
fn test_is_legal_after_move_perft2() {
    // Walk perft depth 2 and verify all results match
    let move_gen = setup();
    let board = Board::new();
    let mut mismatches = 0;
    let mut total = 0;

    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
    for mv in captures.iter().chain(moves.iter()) {
        let expected = board.apply_move_to_board(*mv).is_legal(&move_gen);
        let actual = board.is_legal_after_move(*mv, &move_gen);
        total += 1;
        if expected != actual {
            mismatches += 1;
        }
        if !expected {
            continue;
        }

        let next_board = board.apply_move_to_board(*mv);
        let (caps2, moves2) = move_gen.gen_pseudo_legal_moves(&next_board);
        for mv2 in caps2.iter().chain(moves2.iter()) {
            let expected2 = next_board.apply_move_to_board(*mv2).is_legal(&move_gen);
            let actual2 = next_board.is_legal_after_move(*mv2, &move_gen);
            total += 1;
            if expected2 != actual2 {
                mismatches += 1;
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "All {} perft-2 legality checks should match",
        total
    );
    assert!(total > 0, "Should have checked some moves");
}

#[test]
fn test_is_legal_after_move_complex_positions() {
    let move_gen = setup();

    let positions = vec![
        // Kiwipete - complex position with many tactical possibilities
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        // Position 3 from perft suite
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        // Position with many pins and discovered checks
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ];

    for fen in positions {
        let board = board_from_fen(fen);
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);

        for mv in captures.iter().chain(moves.iter()) {
            let expected = board.apply_move_to_board(*mv).is_legal(&move_gen);
            let actual = board.is_legal_after_move(*mv, &move_gen);
            assert_eq!(
                expected,
                actual,
                "is_legal_after_move mismatch for move {} in position {}",
                mv.print_algebraic(),
                fen
            );
        }
    }
}

#[test]
fn test_is_legal_after_move_promotion_positions() {
    let move_gen = setup();

    // Position with promotion possibilities
    let board = board_from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1");
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);

    for mv in captures.iter().chain(moves.iter()) {
        let expected = board.apply_move_to_board(*mv).is_legal(&move_gen);
        let actual = board.is_legal_after_move(*mv, &move_gen);
        assert_eq!(
            expected,
            actual,
            "is_legal_after_move mismatch for promotion move {}",
            mv.print_algebraic()
        );
    }
}
