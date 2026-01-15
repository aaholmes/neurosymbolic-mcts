//! Unit tests for mate search algorithm

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::mate_search;
use kingfisher::move_types::Move;

/// Helper to run mate search and return results
fn run_mate_search(fen: &str, depth: i32) -> (i32, Move, i32) {
    let move_gen = MoveGen::new();
    let board = Board::new_from_fen(fen);
    let mut stack = BoardStack::with_board(board);
    mate_search(&mut stack, &move_gen, depth, false)
}

#[test]
fn test_mate_in_1_back_rank() {
    // White to move: Re8 is mate
    let (score, best_move, nodes) = run_mate_search(
        "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
        2,
    );

    assert!(score >= 1_000_000, "Should find mate, got score {}", score);
    assert_eq!(best_move.to, 60, "Should find Re8# (to e8 = square 60)");
    assert!(nodes > 0, "Should search some nodes");
}

#[test]
fn test_mate_in_1_queen() {
    // White to move: Qh7 is mate (queen on f5, black king on h8, pawns block escape)
    // Position: Queen on f5, King on h8 with pawns on g7/h7
    let (score, best_move, _) = run_mate_search(
        "7k/5ppp/8/5Q2/8/8/8/6K1 w - - 0 1",
        3,
    );

    // This should find Qf8# or similar
    assert!(score >= 1_000_000, "Should find mate, got score {}", score);
}

#[test]
fn test_no_mate_starting_position() {
    // Starting position - no mate at shallow depth
    let (score, _, _) = run_mate_search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        4,
    );

    assert!(score.abs() < 1_000_000, "Should not find mate in starting position");
}

#[test]
fn test_already_mated_position() {
    // Position where white is already checkmated (fool's mate)
    let (score, _, _) = run_mate_search(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        2,
    );

    // When side to move is mated, score should be very negative
    // The function should return 0 or a first legal move since it's terminal
}

#[test]
fn test_mate_in_2_scholars_mate() {
    // Position leading to Scholar's mate: Qxf7#
    let (score, best_move, _) = run_mate_search(
        "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        3,
    );

    assert!(score >= 1_000_000, "Should find mate");
    // Qxf7# is h5 (39) to f7 (53)
    assert_eq!(best_move.to, 53, "Should find Qxf7#");
}

#[test]
fn test_mate_for_black() {
    // Black to move: Re1 is mate
    let (score, best_move, _) = run_mate_search(
        "4r1k1/8/8/8/8/8/5PPP/6K1 b - - 0 1",
        2,
    );

    assert!(score >= 1_000_000, "Should find mate for black");
    // Re1# is e8 (60) to e1 (4)
    assert_eq!(best_move.from, 60, "Should move from e8");
    assert_eq!(best_move.to, 4, "Should move to e1");
}

#[test]
fn test_stalemate_not_mate() {
    // Stalemate position - black to move, no legal moves but not in check
    let (score, _, _) = run_mate_search(
        "k7/1R6/K7/8/8/8/8/8 b - - 0 1",
        2,
    );

    // Should not be a mate score
    assert!(score.abs() < 1_000_000, "Stalemate should not be reported as mate");
}

#[test]
fn test_mate_search_returns_legal_move() {
    let move_gen = MoveGen::new();

    // Various positions to test
    let positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/8/8/8/8/5k2/8/4K2R w - - 0 1",
    ];

    for fen in &positions {
        let board = Board::new_from_fen(fen);
        let mut stack = BoardStack::with_board(board.clone());

        let (_, best_move, _) = mate_search(&mut stack, &move_gen, 3, false);

        if best_move != Move::null() {
            // Verify the returned move is legal
            let next = board.apply_move_to_board(best_move);
            assert!(next.is_legal(&move_gen),
                "Mate search returned illegal move for FEN: {}", fen);
        }
    }
}

#[test]
fn test_depth_affects_search() {
    // Position with mate in 2 - shallow search might miss it
    let (score_d1, _, nodes_d1) = run_mate_search(
        "7k/R7/5K2/8/8/8/8/8 w - - 0 1", // Rook mate in 2
        1,
    );

    let (score_d3, _, nodes_d3) = run_mate_search(
        "7k/R7/5K2/8/8/8/8/8 w - - 0 1",
        4,
    );

    // Deeper search should find the mate
    assert!(nodes_d3 >= nodes_d1, "Deeper search should examine at least as many nodes");
    // At depth 4, should find the mate in 2
    assert!(score_d3 >= 1_000_000, "Depth 4 should find mate in 2");
}

#[test]
fn test_node_budget_limited() {
    // Even with high depth, should respect node budget
    let (_, _, nodes) = run_mate_search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        10, // High depth
    );

    // The mate search uses a 1,000,000 node budget
    assert!(nodes <= 1_000_001, "Should respect node budget");
}
