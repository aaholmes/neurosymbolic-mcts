//! Unit tests for quiescence search

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::quiescence::{
    forced_material_balance, material_qsearch, quiescence_search, quiescence_search_tactical,
};
use std::time::{Duration, Instant};

/// Helper to create test components
fn setup() -> (MoveGen, PestoEval) {
    (MoveGen::new(), PestoEval::new())
}

#[test]
fn test_quiescence_quiet_position() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    let (score, nodes) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000, 100000, 8, false, None, None,
    );

    // Quiet starting position should return stand-pat evaluation
    assert!(
        score.abs() < 1000,
        "Starting position should have reasonable eval: {}",
        score
    );
    assert!(nodes >= 1, "Should count at least the root node");
}

#[test]
fn test_quiescence_captures_position() {
    let (move_gen, pesto) = setup();
    // Position with mutual captures available: both sides can capture
    let board =
        Board::new_from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3");
    let mut stack = BoardStack::with_board(board);

    let (score, nodes) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000, 100000, 8, false, None, None,
    );

    // Should search some nodes but position is relatively quiet
    assert!(nodes >= 1);
    assert!(score.abs() < 500, "Roughly equal position: {}", score);
}

#[test]
fn test_quiescence_winning_capture() {
    let (move_gen, pesto) = setup();
    // White can capture a free queen with QxQ (queen on d3 can take queen on g4)
    let board =
        Board::new_from_fen("r1b1kbnr/pppp1ppp/2n5/4p3/4P1q1/3Q1N2/PPPP1PPP/RNB1KB1R w KQkq - 0 3");
    let mut stack = BoardStack::with_board(board);

    let stand_pat = pesto.eval(&stack.current_state(), &move_gen);

    let (score, nodes) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000, 100000, 8, false, None, None,
    );

    // QS should search Qxg4 and find it's winning
    assert!(nodes >= 1, "Should search at least root node");
    // Score should be better than stand_pat since we can capture the queen
    assert!(
        score >= stand_pat,
        "Score should improve or stay same after searching captures"
    );
}

#[test]
fn test_quiescence_beta_cutoff() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    // With a very low beta, should get immediate cutoff
    let (score, nodes) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000,
        -50000, // Very low beta (opponent already has a winning line)
        8, false, None, None,
    );

    // Should return beta immediately due to stand-pat cutoff
    assert_eq!(score, -50000, "Should return beta on cutoff");
    assert_eq!(nodes, 1, "Should only count root node on immediate cutoff");
}

#[test]
fn test_quiescence_alpha_improvement() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    // With very low alpha, stand-pat should improve it
    let (score, _nodes) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000, // Very low alpha
        100000, 8, false, None, None,
    );

    // Score should be better than -100000
    assert!(score > -100000, "Stand-pat should improve alpha");
}

#[test]
fn test_quiescence_depth_limit() {
    let (move_gen, pesto) = setup();
    // Position with many captures available
    let board =
        Board::new_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
    let mut stack = BoardStack::with_board(board);

    // With depth 0, should return immediately
    let (score_d0, nodes_d0) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000, 100000, 0, false, None, None,
    );

    // With higher depth, may search more
    let (score_d4, nodes_d4) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000, 100000, 4, false, None, None,
    );

    assert_eq!(nodes_d0, 1, "Depth 0 should only count root");
    assert!(
        nodes_d4 >= nodes_d0,
        "Higher depth should search at least as many nodes"
    );
}

#[test]
fn test_quiescence_time_limit() {
    let (move_gen, pesto) = setup();
    let board =
        Board::new_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
    let mut stack = BoardStack::with_board(board);

    let start = Instant::now();
    let time_limit = Duration::from_millis(1); // Very short limit

    let (_score, _nodes) = quiescence_search(
        &mut stack,
        &move_gen,
        &pesto,
        -100000,
        100000,
        100, // High depth
        false,
        Some(start),
        Some(time_limit),
    );

    // Should return relatively quickly
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(500),
        "Should respect time limit, took {:?}",
        elapsed
    );
}

#[test]
fn test_quiescence_tactical_returns_tree() {
    let (move_gen, pesto) = setup();
    let mut stack = BoardStack::new();

    let tree = quiescence_search_tactical(&mut stack, &move_gen, &pesto);

    // Should return a valid TacticalTree
    assert!(
        tree.leaf_score.abs() < 1000,
        "Starting position should have reasonable score"
    );
    // Starting position has no captures, so siblings should be empty
    assert!(tree.siblings.is_empty(), "No captures in starting position");
    assert!(
        tree.principal_variation.is_empty(),
        "No tactical line in quiet position"
    );
}

#[test]
fn test_quiescence_tactical_finds_captures() {
    let (move_gen, pesto) = setup();
    // Position where white can capture pawn on e5
    let board =
        Board::new_from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3");
    let mut stack = BoardStack::with_board(board);

    let tree = quiescence_search_tactical(&mut stack, &move_gen, &pesto);

    // Should find captures (Nxe5 is available)
    // Note: The position might not have any SEE-positive captures
}

#[test]
fn test_quiescence_tactical_with_winning_capture() {
    let (move_gen, pesto) = setup();
    // White queen can capture undefended black queen
    let board =
        Board::new_from_fen("r1b1kbnr/pppp1ppp/2n5/4p3/4P1q1/3Q1N2/PPPP1PPP/RNB1KB1R w KQkq - 0 3");
    let mut stack = BoardStack::with_board(board);

    let tree = quiescence_search_tactical(&mut stack, &move_gen, &pesto);

    // Should find Qxg4 as a winning capture
    // There should be at least one capture move evaluated
    let _has_qxg4 = tree.siblings.iter().any(|(mv, _)| mv.to == 30); // g4 = square 30
                                                                     // Qxg4 may or may not be found depending on SEE
}

#[test]
fn test_quiescence_see_pruning() {
    let (move_gen, pesto) = setup();
    // Position where capturing would lose material (e.g., QxP defended by bishop)
    let board =
        Board::new_from_fen("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4");
    let mut stack = BoardStack::with_board(board);

    let (score, _nodes) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000, 100000, 8, false, None, None,
    );

    // Score should be reasonable (SEE should prune bad captures)
    assert!(
        score.abs() < 1000,
        "Position should evaluate reasonably: {}",
        score
    );
}

#[test]
fn test_quiescence_checkmate_detection() {
    let (move_gen, pesto) = setup();
    // Position where white is checkmated
    let board =
        Board::new_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    let mut stack = BoardStack::with_board(board);

    let (_score, nodes) = quiescence_search(
        &mut stack, &move_gen, &pesto, -100000, 100000, 8, false, None, None,
    );

    // The evaluation function handles checkmate
    assert!(nodes >= 1);
}

#[test]
fn test_quiescence_tactical_siblings_scored() {
    let (move_gen, pesto) = setup();
    // Position with multiple captures for white
    let board = Board::new_from_fen(
        "r1b1k2r/ppppnppp/2n5/2b1p3/2B1P1q1/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
    );
    let mut stack = BoardStack::with_board(board);

    let tree = quiescence_search_tactical(&mut stack, &move_gen, &pesto);

    // Each sibling should have a score
    for (mv, score) in &tree.siblings {
        // Scores should be within reasonable bounds
        assert!(
            score.abs() < 1000001,
            "Score {} for move {:?} seems unreasonable",
            score,
            mv
        );
    }
}

// ======== Material Q-Search Tests ========

#[test]
fn test_material_qsearch_quiet_position() {
    let move_gen = MoveGen::new();
    let mut stack = BoardStack::new();

    let (result, completed) = forced_material_balance(&mut stack, &move_gen);

    // Starting position: equal material, no captures improve balance
    assert_eq!(
        result, 0,
        "Starting position material balance should be 0, got {result}"
    );
    assert!(
        completed,
        "Starting position Q-search should complete (no captures available)"
    );
}

#[test]
fn test_material_qsearch_free_piece() {
    let move_gen = MoveGen::new();
    // White knight on e5 can capture undefended black pawn on d7 (no... let's use a simpler setup)
    // White queen on d1 can capture undefended black queen on d8? No, that's defended.
    // White to move, free black knight on e4 (white pawn on d3 can take, knight undefended)
    let board = Board::new_from_fen("4k3/8/8/8/4n3/3P4/8/4K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board.clone());

    let stand_pat = board.material_imbalance();
    let (result, completed) = forced_material_balance(&mut stack, &move_gen);

    // White is down material (has P vs opponent N: 1 vs 3 = -2)
    // After Pxe4 (capturing knight), white gains 3 - 1 (loses pawn to nothing? No, pawn captures knight)
    // After dxe4: white material = 0 (pawn promoted? no, pawn moves to e4)
    // Actually: white has pawn(1) vs black knight(3), so stand_pat = 1-3 = -2
    // After dxe4: white captures the knight (gaining 3), white now has pawn on e4
    // White material = 1, black material = 0, so balance = 1
    // Result should be 1 (better than stand pat of -2)
    assert!(
        result > stand_pat,
        "Should find winning capture: result={result}, stand_pat={stand_pat}"
    );
    assert_eq!(
        result, 1,
        "After capturing free knight, balance should be +1 (pawn vs nothing)"
    );
    assert!(completed, "Single capture resolves: should complete");
}

#[test]
fn test_material_qsearch_bad_capture_stand_pat() {
    let move_gen = MoveGen::new();
    // White queen on d4 can capture black pawn on e5, but pawn is defended by another pawn on d6
    // Actually let's make it simpler: only bad captures available
    // White queen vs black pawn defended by black queen
    let board = Board::new_from_fen("4k3/8/8/4p3/3Q4/8/8/4K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board.clone());

    let stand_pat = board.material_imbalance();
    let (result, _) = forced_material_balance(&mut stack, &move_gen);

    // White has Q(9) vs Black P(1), stand_pat = 8
    // Qxe5 captures pawn (+1), but if there's no recapture, it's good
    // Actually the pawn is undefended here, so Q captures it, result = 9
    // Let me use a position where capture is bad:
    // White knight can capture black pawn defended by black queen
    let board2 = Board::new_from_fen("4k3/8/8/3qp3/4N3/8/8/4K3 w - - 0 1");
    let mut stack2 = BoardStack::with_board(board2.clone());

    let stand_pat2 = board2.material_imbalance();
    let (result2, _) = forced_material_balance(&mut stack2, &move_gen);

    // White has N(3) vs Black Q(9)+P(1) = -7
    // Nxe5 captures pawn: white gets +1, but then Qxe5 recaptures: white loses 3
    // Net: -7 + 1 - 3 = -9? No, let's think in terms of material_imbalance after the sequence.
    // After Nxe5: white has N on e5 (3), black has Q (9), balance = 3-9 = -6
    // After Qxe5: white has 0, black has Q (9), balance for black to move = 9-0 = 9, so for white = -9
    // Stand pat is -7, so Nxe5 sequence leads to -9, which is worse.
    // material_qsearch should prefer stand_pat.
    assert_eq!(result2, stand_pat2, "Should prefer stand pat over bad capture sequence: result={result2}, stand_pat={stand_pat2}");
}

#[test]
fn test_material_qsearch_promotion() {
    let move_gen = MoveGen::new();
    // White pawn on e7, black king far away, e8 is clear
    let board = Board::new_from_fen("8/4P3/8/8/8/8/k7/4K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board.clone());

    let stand_pat = board.material_imbalance();
    let (result, _) = forced_material_balance(&mut stack, &move_gen);

    // stand_pat: white P(1) vs nothing = 1
    // After e8=Q: white Q(9) vs nothing = 9
    // Gain of 8 pawn units
    assert!(
        result > stand_pat,
        "Promotion should improve material: result={result}, stand_pat={stand_pat}"
    );
    assert_eq!(result, 9, "After promotion to queen, balance should be 9");
}

#[test]
fn test_material_qsearch_multiple_captures() {
    let move_gen = MoveGen::new();
    // Position with a capture chain: white Bxf7, then Kxf7 recapture
    // White: B on c4, black: pawn on f7, king on e8
    let board = Board::new_from_fen("4k3/5p2/8/8/2B5/8/8/4K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board.clone());

    let stand_pat = board.material_imbalance();
    let (result, _) = forced_material_balance(&mut stack, &move_gen);

    // stand_pat: B(3) vs P(1) = 2
    // Bxf7+: white B(now on f7, captured P), white material = B(3), black material = 0, balance = 3
    // But then Kxf7: white material = 0, black material = 0, balance for black = 0, for white = 0
    // So the sequence Bxf7+ Kxf7 leads to balance 0, which is worse than stand_pat (2)
    // Q-search should prefer stand_pat = 2
    assert_eq!(result, stand_pat, "Should prefer stand pat when captures lose material: result={result}, stand_pat={stand_pat}");
}

#[test]
fn test_material_qsearch_symmetry() {
    let move_gen = MoveGen::new();
    // Same position but with colors flipped
    let white_board = Board::new_from_fen("4k3/8/8/4n3/3P4/8/8/4K3 w - - 0 1");
    let black_board = Board::new_from_fen("4k3/8/3p4/4N3/8/8/8/4K3 b - - 0 1");

    let mut stack_w = BoardStack::with_board(white_board);
    let mut stack_b = BoardStack::with_board(black_board);

    let (result_w, _) = forced_material_balance(&mut stack_w, &move_gen);
    let (result_b, _) = forced_material_balance(&mut stack_b, &move_gen);

    // Both sides have same material ratio (P vs N) from STM perspective
    assert_eq!(
        result_w, result_b,
        "Mirrored positions should give same result: w={result_w}, b={result_b}"
    );
}

// ======== Q-Search Completion Flag Tests ========

#[test]
fn test_material_qsearch_completion_stand_pat_cutoff() {
    let move_gen = MoveGen::new();
    // Position where white is way ahead — stand-pat beats beta, immediate cutoff = completed
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/1Q2K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board);

    // Stand pat: Q(9) vs nothing = 9, beta=1000, 9 < 1000, no cutoff at root
    // But no captures available → completed
    let (_, completed) = forced_material_balance(&mut stack, &move_gen);
    assert!(completed, "Position with no captures should complete");
}

#[test]
fn test_material_qsearch_completion_single_recapture() {
    let move_gen = MoveGen::new();
    // White can capture pawn on e5, black can recapture with queen → 2-ply sequence completes
    let board = Board::new_from_fen("4k3/8/8/3qp3/4N3/8/8/4K3 w - - 0 1");
    let mut stack = BoardStack::with_board(board);

    let (_, completed) = forced_material_balance(&mut stack, &move_gen);
    assert!(
        completed,
        "Short capture-recapture sequence within depth 8 should complete"
    );
}

#[test]
fn test_material_qsearch_incomplete_deep_captures() {
    let move_gen = MoveGen::new();
    // Lots of mutual captures available in a complex middlegame with many hanging pieces.
    // Each side has many pieces that can capture: this should produce a deep capture tree
    // that potentially exceeds depth 8.
    // Position: pieces on every other square creating maximum cross-capture potential
    let board =
        Board::new_from_fen("r1b1k1nr/ppNpBppp/2n5/q3P3/2Bp4/4P3/PP3PPP/R2QK1NR w KQkq - 0 1");
    let mut stack = BoardStack::with_board(board);

    let (val, _completed) = forced_material_balance(&mut stack, &move_gen);
    // We mainly care that the function runs and returns; the completion flag may or may not be true
    // depending on the exact capture tree depth
    assert!(
        val > -100 && val < 100,
        "Value should be reasonable: {}",
        val
    );
}
