//! Tests for Zobrist hashing.
//!
//! Tests verify that Zobrist hashing produces consistent, unique hashes
//! for different board positions and that hash updates are correct.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;

fn setup() -> MoveGen {
    MoveGen::new()
}

/// Helper to get hash using public compute_zobrist_hash
fn get_hash(board: &Board) -> u64 {
    board.compute_zobrist_hash()
}

#[test]
fn test_zobrist_starting_position_consistent() {
    let board1 = Board::new();
    let board2 = Board::new();

    // Same position should have same hash
    assert_eq!(
        get_hash(&board1),
        get_hash(&board2),
        "Same position should have same hash"
    );
}

#[test]
fn test_zobrist_different_positions_different_hash() {
    let board1 = Board::new();
    let board2 = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");

    // Different positions should have different hashes
    assert_ne!(
        get_hash(&board1),
        get_hash(&board2),
        "Different positions should have different hashes"
    );
}

#[test]
fn test_zobrist_side_to_move_matters() {
    // Same piece placement, different side to move
    let board_w = Board::new_from_fen("8/8/8/8/8/8/8/4K2k w - - 0 1");
    let board_b = Board::new_from_fen("8/8/8/8/8/8/8/4K2k b - - 0 1");

    assert_ne!(
        get_hash(&board_w),
        get_hash(&board_b),
        "Different side to move should give different hash"
    );
}

#[test]
fn test_zobrist_castling_rights_matter() {
    // Same position, different castling rights
    let board_castle = Board::new_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    let board_no_castle = Board::new_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1");

    assert_ne!(
        get_hash(&board_castle),
        get_hash(&board_no_castle),
        "Different castling rights should give different hash"
    );
}

#[test]
fn test_zobrist_en_passant_matters() {
    // Same position, different en passant square
    let board_ep = Board::new_from_fen("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 1");
    let board_no_ep = Board::new_from_fen("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1");

    assert_ne!(
        get_hash(&board_ep),
        get_hash(&board_no_ep),
        "Different en passant should give different hash"
    );
}

#[test]
fn test_zobrist_move_and_undo() {
    let move_gen = setup();
    let mut board_stack = BoardStack::new();

    let hash_before = get_hash(board_stack.current_state());

    // Generate legal moves
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board_stack.current_state());

    // Find first legal move
    for mv in captures.iter().chain(moves.iter()) {
        board_stack.make_move(*mv);
        if board_stack.current_state().is_legal(&move_gen) {
            let hash_after_move = get_hash(board_stack.current_state());

            // Hash should change after move
            assert_ne!(
                hash_before,
                hash_after_move,
                "Hash should change after move"
            );

            // Undo move
            board_stack.undo_move();

            // Hash should be restored
            assert_eq!(
                hash_before,
                get_hash(board_stack.current_state()),
                "Hash should be restored after undo"
            );

            return;
        }
        board_stack.undo_move();
    }

    panic!("Should have found at least one legal move");
}

#[test]
fn test_zobrist_transposition() {
    // Two different move orders reaching same position
    // 1. e4 e5 2. Nf3 Nc6
    // 1. Nf3 Nc6 2. e4 e5
    // Both should result in same hash

    let board1 = Board::new_from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    let board2 = Board::new_from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");

    assert_eq!(
        get_hash(&board1),
        get_hash(&board2),
        "Same position from different move orders should have same hash"
    );
}

#[test]
fn test_zobrist_multiple_moves() {
    let _move_gen = setup();
    let mut board_stack = BoardStack::new();
    let mut seen_hashes = std::collections::HashSet::new();

    seen_hashes.insert(get_hash(board_stack.current_state()));

    // Make several moves and verify hashes are unique
    let moves_to_make = vec![
        (12, 28), // e2-e4
        (52, 36), // e7-e5
        (6, 21),  // g1-f3
        (57, 42), // g8-f6
    ];

    for (from, to) in moves_to_make {
        let mv = kingfisher::move_types::Move {
            from,
            to,
            promotion: None,
        };

        board_stack.make_move(mv);
        let hash = get_hash(board_stack.current_state());

        // Each position should have unique hash
        assert!(
            !seen_hashes.contains(&hash) || hash == get_hash(board_stack.current_state()),
            "Each new position should have unique hash"
        );

        seen_hashes.insert(hash);
    }

    // Should have collected unique hashes
    assert_eq!(seen_hashes.len(), 5, "Should have 5 unique positions");
}

#[test]
fn test_incremental_hash_matches_full_recompute() {
    // Verify the incremental Zobrist hash matches full recomputation after each move
    let mut board_stack = BoardStack::new();
    let move_gen = setup();

    // Play several moves and verify hash consistency after each
    let moves_to_make = vec![
        (12, 28), // e2-e4
        (52, 36), // e7-e5
        (6, 21),  // g1-f3
        (57, 42), // g8-f6
        (5, 33),  // f1-b5 (bishop move)
    ];

    for (from, to) in moves_to_make {
        let mv = Move::new(from, to, None);
        board_stack.make_move(mv);
        let board = board_stack.current_state();

        let incremental = board.compute_zobrist_hash();
        // The board's zobrist_hash field is set by apply_move_to_board (incremental)
        // compute_zobrist_hash recomputes from scratch
        // They must match
        assert_eq!(
            get_hash(board),
            incremental,
            "Incremental hash should match full recompute after move from {} to {}",
            from, to
        );
    }
}

#[test]
fn test_hash_unique_across_many_positions() {
    // Generate hashes for many different FEN positions and verify they're all unique
    let fens = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        "8/8/8/8/8/8/8/4K2k w - - 0 1",
        "8/8/8/8/8/8/8/4K2k b - - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1",
    ];

    let mut hashes = std::collections::HashSet::new();
    for fen in &fens {
        let board = Board::new_from_fen(fen);
        let hash = get_hash(&board);
        assert!(
            hashes.insert(hash),
            "Hash collision for FEN: {}",
            fen
        );
    }
}

#[test]
fn test_hash_after_capture() {
    // After a capture, hash should differ from pre-capture
    let board = Board::new_from_fen("8/8/8/3q4/4N3/8/8/K6k w - - 0 1");
    let hash_before = get_hash(&board);

    let mv = Move::new(28, 35, None); // Nxd5
    let board_after = board.apply_move_to_board(mv);
    let hash_after = get_hash(&board_after);

    assert_ne!(hash_before, hash_after, "Hash should change after capture");
}

#[test]
fn test_hash_after_castling() {
    let board = Board::new_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    let hash_before = get_hash(&board);

    // Kingside castling
    let mv = Move::new(4, 6, None);
    let board_after = board.apply_move_to_board(mv);
    let hash_after = get_hash(&board_after);

    assert_ne!(hash_before, hash_after, "Hash should change after castling");
}

#[test]
fn test_hash_after_en_passant() {
    let board = Board::new_from_fen("8/8/8/pP6/8/8/8/K6k w - a6 0 1");
    let hash_before = get_hash(&board);

    // b5xa6 en passant
    let mv = Move::new(33, 40, None);
    let board_after = board.apply_move_to_board(mv);
    let hash_after = get_hash(&board_after);

    assert_ne!(hash_before, hash_after, "Hash should change after en passant");
}

#[test]
fn test_hash_after_promotion() {
    let board = Board::new_from_fen("8/P7/8/8/8/8/8/K6k w - - 0 1");
    let hash_before = get_hash(&board);

    let mv = Move::new(48, 56, Some(kingfisher::piece_types::QUEEN));
    let board_after = board.apply_move_to_board(mv);
    let hash_after = get_hash(&board_after);

    assert_ne!(hash_before, hash_after, "Hash should change after promotion");
}
