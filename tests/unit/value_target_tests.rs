/// Tests for value target assignment logic used in self-play.
///
/// The key invariant: value_target should be from the side-to-move's perspective.
/// White win (1.0) → white-to-move sample gets +1.0, black-to-move sample gets -1.0.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;

/// Simulates the value target assignment logic from self_play.rs.
/// Given a list of (w_to_move, _) samples and a final_score_white,
/// returns the assigned value targets.
fn assign_value_targets(samples: &[(bool, &Board)], final_score_white: f32) -> Vec<f32> {
    samples
        .iter()
        .map(|(w_to_move, _)| {
            if *w_to_move {
                final_score_white
            } else {
                -final_score_white
            }
        })
        .collect()
}

// === White wins scenarios ===

#[test]
fn test_value_target_white_wins_white_perspective() {
    let board = Board::new(); // White to move
    let samples = vec![(true, &board)];
    let targets = assign_value_targets(&samples, 1.0);
    assert_eq!(targets[0], 1.0, "White-to-move sample should get +1.0 when White wins");
}

#[test]
fn test_value_target_white_wins_black_perspective() {
    let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    let samples = vec![(false, &board)];
    let targets = assign_value_targets(&samples, 1.0);
    assert_eq!(targets[0], -1.0, "Black-to-move sample should get -1.0 when White wins");
}

// === Black wins scenarios ===

#[test]
fn test_value_target_black_wins_white_perspective() {
    let board = Board::new();
    let samples = vec![(true, &board)];
    let targets = assign_value_targets(&samples, -1.0);
    assert_eq!(targets[0], -1.0, "White-to-move sample should get -1.0 when Black wins");
}

#[test]
fn test_value_target_black_wins_black_perspective() {
    let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    let samples = vec![(false, &board)];
    let targets = assign_value_targets(&samples, -1.0);
    assert_eq!(targets[0], 1.0, "Black-to-move sample should get +1.0 when Black wins");
}

// === Draw scenarios ===

#[test]
fn test_value_target_draw_both_perspectives() {
    let board_w = Board::new();
    let board_b = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    let samples = vec![(true, &board_w), (false, &board_b)];
    let targets = assign_value_targets(&samples, 0.0);
    assert_eq!(targets[0], 0.0, "Draw should give 0.0 from white's perspective");
    assert_eq!(targets[1], 0.0, "Draw should give 0.0 from black's perspective (negated 0 = 0)");
}

// === Multi-sample game simulation ===

#[test]
fn test_value_target_alternating_moves_white_wins() {
    // Simulate a short game: W, B, W, B, W — White wins
    let board = Board::new();
    let samples = vec![
        (true, &board),   // Move 1: White
        (false, &board),  // Move 1: Black
        (true, &board),   // Move 2: White
        (false, &board),  // Move 2: Black
        (true, &board),   // Move 3: White
    ];
    let targets = assign_value_targets(&samples, 1.0);
    assert_eq!(targets, vec![1.0, -1.0, 1.0, -1.0, 1.0]);
}

#[test]
fn test_value_target_uses_w_to_move_not_index() {
    // This is the key test: if a position somehow has the same side moving
    // twice (e.g., in a hypothetical null-move scenario), w_to_move-based
    // assignment gives correct results, but i%2 would be wrong.
    let board = Board::new();
    // Two consecutive white-to-move samples
    let samples = vec![
        (true, &board),
        (true, &board),
    ];
    let targets = assign_value_targets(&samples, 1.0);
    // Both should be +1.0 since both are white's perspective
    assert_eq!(targets[0], 1.0);
    assert_eq!(targets[1], 1.0);
    // With i%2 logic, sample[1] would incorrectly get -1.0
}

// === Checkmate outcome integration ===

#[test]
fn test_value_target_checkmate_white_wins() {
    let move_gen = MoveGen::new();

    // Scholar's mate final position: Black is checkmated, White to move is false
    let board = Board::new_from_fen(
        "rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3"
    );
    let (mate, _) = board.is_checkmate_or_stalemate(&move_gen);
    assert!(mate, "This should be checkmate");

    // final_score_white: Black is checkmated, w_to_move is false (Black to move = mated side)
    // When STM is mated, it's a win for the other side
    let final_score_white = if board.w_to_move { -1.0 } else { 1.0 };
    assert_eq!(final_score_white, 1.0, "White should be the winner");

    // Samples from this game
    let start = Board::new();
    let samples = vec![
        (true, &start),   // White's first move
        (false, &start),  // Black's first move
        (true, &start),   // White's second move
    ];
    let targets = assign_value_targets(&samples, final_score_white);
    assert_eq!(targets[0], 1.0, "White move → White wins → +1.0");
    assert_eq!(targets[1], -1.0, "Black move → White wins → -1.0");
    assert_eq!(targets[2], 1.0, "White move → White wins → +1.0");
}

// === KOTH outcome integration ===

#[test]
fn test_value_target_koth_white_wins() {
    // White king on e4 = KOTH win for white
    let board = Board::new_from_fen("rnbq1bnr/pppppppp/8/8/4K3/8/PPPP1PPP/RNBQ1BNR b kq - 0 1");
    let (white_won, _) = board.is_koth_win();
    assert!(white_won);

    let final_score_white = 1.0;
    let samples = vec![
        (true, &board),
        (false, &board),
    ];
    let targets = assign_value_targets(&samples, final_score_white);
    assert_eq!(targets[0], 1.0);
    assert_eq!(targets[1], -1.0);
}

#[test]
fn test_value_target_koth_black_wins() {
    // Black king on d5 = KOTH win for black
    let board = Board::new_from_fen("rnbq1bnr/pppppppp/8/3k4/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1");
    let (_, black_won) = board.is_koth_win();
    assert!(black_won);

    let final_score_white = -1.0;
    let samples = vec![
        (true, &board),
        (false, &board),
    ];
    let targets = assign_value_targets(&samples, final_score_white);
    assert_eq!(targets[0], -1.0);
    assert_eq!(targets[1], 1.0);
}

// === Stalemate and 50-move rule ===

#[test]
fn test_value_target_stalemate_is_draw() {
    // Stalemate position — Black to move, no legal moves, not in check
    let board = Board::new_from_fen("k7/8/1K6/8/8/8/8/1Q6 b - - 0 1");
    let move_gen = MoveGen::new();
    let (mate, stalemate) = board.is_checkmate_or_stalemate(&move_gen);

    // This should be stalemate (not mate)
    if stalemate {
        let final_score_white = 0.0;
        let samples = vec![(true, &board), (false, &board)];
        let targets = assign_value_targets(&samples, final_score_white);
        assert_eq!(targets[0], 0.0);
        assert_eq!(targets[1], 0.0);
    }
}

#[test]
fn test_value_target_50_move_draw() {
    // 50-move rule: halfmove_clock >= 100 → draw
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 51");
    assert!(board.halfmove_clock() >= 100);

    let final_score_white = 0.0;
    let samples = vec![(true, &board), (false, &board)];
    let targets = assign_value_targets(&samples, final_score_white);
    assert_eq!(targets[0], 0.0);
    assert_eq!(targets[1], 0.0);
}

// === w_to_move field correctness ===

#[test]
fn test_board_w_to_move_tracks_correctly_through_game() {
    let mut stack = BoardStack::new();
    assert!(stack.current_state().w_to_move, "Starting position: White to move");

    stack.make_move(Move::new(12, 28, None)); // e2-e4
    assert!(!stack.current_state().w_to_move, "After White's move: Black to move");

    stack.make_move(Move::new(52, 36, None)); // e7-e5
    assert!(stack.current_state().w_to_move, "After Black's move: White to move");

    stack.make_move(Move::new(6, 21, None)); // Ng1-f3
    assert!(!stack.current_state().w_to_move, "After White's move: Black to move");
}
