use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::{alpha_beta_search, iterative_deepening_ab_search};
use kingfisher::search::{mate_search, negamax_search};
use kingfisher::transposition::TranspositionTable;

#[test]
fn test_mate_in_one_detection() {
    let mut board = BoardStack::new_from_fen("3qk3/3ppp2/8/8/8/8/3PPP2/3QK2R w K - 0 1");
    let move_gen = MoveGen::new();
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 1, false);
    assert!(score > 900000); // Should detect mate
    assert_eq!(best_move.to, 63); // Rh8# (assuming 0-63 board representation)
}

#[test]
fn test_mate_in_one_rook() {
    let mut board = BoardStack::new_from_fen("8/8/8/8/8/8/k1K5/R7 w - - 0 1");
    let move_gen = MoveGen::new();
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 1, false);
    assert!(score > 900000); // Should detect mate
    assert_eq!(best_move.to_uci(), "a1a8"); // Ra8#
}

#[test]
fn test_mate_in_one_queen() {
    let mut board = BoardStack::new_from_fen("8/8/8/8/8/k1K5/8/Q7 w - - 0 1");
    let move_gen = MoveGen::new();
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 1, false);
    assert!(score > 900000); // Should detect mate
    assert_eq!(best_move.to_uci(), "a1a8"); // Qa8#
}

#[test]
fn test_mate_in_one_bishop() {
    let mut board = BoardStack::new_from_fen("8/8/8/8/8/k1K5/1B6/8 w - - 0 1");
    let move_gen = MoveGen::new();
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 1, false);
    assert!(score > 900000); // Should detect mate
    assert_eq!(best_move.to_uci(), "b2a3"); // Ba3#
}

#[test]
fn test_mate_in_one_knight() {
    let mut board = BoardStack::new_from_fen("8/8/8/8/8/k1K5/1N6/8 w - - 0 1");
    let move_gen = MoveGen::new();
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 1, false);
    assert!(score > 900000); // Should detect mate
    assert_eq!(best_move.to_uci(), "b2c4"); // Nc4#
}

#[test]
fn test_mate_in_one_pawn() {
    let mut board = BoardStack::new_from_fen("8/8/8/8/8/k1K5/P7/8 w - - 0 1");
    let move_gen = MoveGen::new();
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 1, false);
    assert!(score > 900000); // Should detect mate
    assert_eq!(best_move.to_uci(), "a2a3"); // a3#
}

#[test]
fn test_mate_in_two_detection() {
    let mut board = BoardStack::new_from_fen("3qk3/3ppp2/5n2/8/8/8/3PPP2/3QK2R w K - 0 1");
    let move_gen = MoveGen::new();
    let (score, _, _) = mate_search(&mut board, &move_gen, 1, false);
    assert!(score < 900000); // Should not detect mate in 1
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 2, false);
    assert!(score > 900000); // Should detect mate in 2
    assert_eq!(best_move.to_uci(), "h1h8"); // Rh8+
}

#[test]
fn test_mate_search_no_mate_in_depth() {
    let mut board = BoardStack::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); // Initial position
    let move_gen = MoveGen::new();
    let (score, _, _) = mate_search(&mut board, &move_gen, 3, false); // Search for mate in 3
    assert!(score < 900000); // Should not detect mate
}

#[test]
fn test_mate_search_position_close_to_mate() {
    let mut board = BoardStack::new_from_fen("8/8/8/8/8/k1K5/1Q6/8 w - - 0 1"); // Queen is one square away from mate
    let move_gen = MoveGen::new();
    let (score, _, _) = mate_search(&mut board, &move_gen, 1, false); // Search for mate in 1
    assert!(score < 900000); // Should not detect mate in 1
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 2, false); // Search for mate in 2
    assert!(score > 900000); // Should detect mate in 2
    assert_eq!(best_move.to_uci(), "b2a3"); // Qb2-a3#
}

#[test]
fn test_mate_in_three_detection() {
    let mut board = BoardStack::new_from_fen("3qk3/3pppr1/5n2/8/8/8/3PPP2/3QK1RR w K - 0 1");
    let move_gen = MoveGen::new();
    let (score, _, _) = mate_search(&mut board, &move_gen, 1, false);
    assert!(score < 900000); // Should not detect mate in 1
    let (score, _, _) = mate_search(&mut board, &move_gen, 2, false);
    assert!(score < 900000); // Should not detect mate in 2
    let (score, best_move, _) = mate_search(&mut board, &move_gen, 3, false);
    assert!(score > 900000); // Should detect mate in 3
    assert_eq!(best_move.to, 63); // Rh8+ (assuming 0-63 board representation)
}

#[test]
fn test_alpha_beta_pruning_effectiveness() {
    let mut board = BoardStack::new_from_fen(
        "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    );
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let mut tt = TranspositionTable::new();

    let depth = 4;
    let infinity = 1000000;
    let (score_full, _, nodes_full, _) = alpha_beta_search(
        &mut board, &move_gen, &pesto, &mut tt, depth, -infinity, infinity, 0, false, None, None,
    );

    // Now search with a narrow window
    let (score_narrow, _, nodes_narrow, _) = alpha_beta_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        depth,
        score_full - 50,
        score_full + 50,
        0,
        false,
        None,
        None,
    );

    println!(
        "Full window (White) - Score: {}, Nodes: {}",
        score_full, nodes_full
    );
    println!(
        "Narrow window (White) - Score: {}, Nodes: {}",
        score_narrow, nodes_narrow
    );

    assert_eq!(score_full, score_narrow, "Scores don't match for White");
    assert!(
        nodes_narrow < nodes_full * 2 / 3,
        "Not enough pruning for White. Full: {}, Narrow: {}",
        nodes_full,
        nodes_narrow
    );

    // Test for black
    board = BoardStack::new_from_fen(
        "r1bqkbnr/ppp2ppp/2np4/4p3/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
    );
    tt = TranspositionTable::new();
    let (score_full_black, _, nodes_full_black, _) = alpha_beta_search(
        &mut board, &move_gen, &pesto, &mut tt, depth, -infinity, infinity, 0, false, None, None,
    );
    let (score_narrow_black, _, nodes_narrow_black, _) = alpha_beta_search(
        &mut board,
        &move_gen,
        &pesto,
        &mut tt,
        depth,
        score_full_black - 50,
        score_full_black + 50,
        0,
        false,
        None,
        None,
    );

    println!(
        "Full window (Black) - Score: {}, Nodes: {}",
        score_full_black, nodes_full_black
    );
    println!(
        "Narrow window (Black) - Score: {}, Nodes: {}",
        score_narrow_black, nodes_narrow_black
    );

    assert_eq!(
        score_full_black, score_narrow_black,
        "Scores don't match for Black"
    );
    assert!(
        nodes_narrow_black < nodes_full_black * 2 / 3,
        "Not enough pruning for Black. Full: {}, Narrow: {}",
        nodes_full_black,
        nodes_narrow_black
    );
}

#[test]
fn test_search_stability() {
    let mut board = BoardStack::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();

    let max_depth = 6;
    let q_search_max_depth = 99;
    let (depth1, score1, best_move1, eval1) = iterative_deepening_ab_search(
        &mut board,
        &move_gen,
        &pesto,
        max_depth,
        q_search_max_depth,
        None,
        false,
    );
    let (depth2, score2, best_move2, eval2) = iterative_deepening_ab_search(
        &mut board,
        &move_gen,
        &pesto,
        max_depth,
        q_search_max_depth,
        None,
        false,
    );

    // The scores and best moves should be the same across multiple runs
    assert_eq!(depth1, depth2);
    assert_eq!(score1, score2);
    assert_eq!(best_move1, best_move2);
    assert_eq!(eval1, eval2);
}

// Count the number of nodes visited in the negamax search and the alpha-beeta search, for a depth of up to 4 ply.
// This is used to determine the effectiveness of the alpha-beta search.
#[test]
fn test_pruning() {
    let mut board = BoardStack::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let mut tt = TranspositionTable::new();
    for depth in 1..6 {
        let (negamax_eval, negamax_move, negamax_nodes) =
            negamax_search(&mut board, &move_gen, &pesto, depth);
        let (alpha_beta_eval, alpha_beta_move, alpha_beta_nodes, _) = alpha_beta_search(
            &mut board, &move_gen, &pesto, &mut tt, depth, -1000000, 1000000, 0, false, None, None,
        );
        assert!(
            negamax_eval == alpha_beta_eval,
            "Evals don't match for depth {}, negamax eval: {}, alpha-beta eval: {}",
            depth,
            negamax_eval,
            alpha_beta_eval
        );
        assert!(
            negamax_move == alpha_beta_move,
            "Moves don't match for depth {}, negamax move: {}, alpha-beta move: {}",
            depth,
            negamax_move.print_algebraic(),
            alpha_beta_move.print_algebraic()
        );
        println!(
            "Move, eval = {}, {}",
            &negamax_move.print_algebraic(),
            negamax_eval
        );
        println!(
            "Depth: {}, Negamax nodes: {}, Alpha-beta nodes: {}",
            depth, negamax_nodes, alpha_beta_nodes
        );
    }
}
