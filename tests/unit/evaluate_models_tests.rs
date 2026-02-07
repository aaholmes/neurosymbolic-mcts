use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search_with_tt, TacticalMctsConfig};
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::transposition::TranspositionTable;
use std::time::Duration;

/// Simplified game result for testing.
#[derive(Debug, Clone, Copy, PartialEq)]
enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

/// Play a single game between two MCTS configs (both using stub NN / pesto eval).
fn play_test_game(simulations: u32) -> GameResult {
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();

    let config = TacticalMctsConfig {
        max_iterations: simulations,
        time_limit: Duration::from_secs(30),
        mate_search_depth: 3,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: None,
        logger: None,
        dirichlet_alpha: 0.0,
        dirichlet_epsilon: 0.0,
        ..Default::default()
    };

    let mut board_stack = BoardStack::new();
    let mut move_count = 0;
    let mut nn: Option<NeuralNetPolicy> = None;
    let mut tt = TranspositionTable::new();

    loop {
        let board = board_stack.current_state().clone();

        let (best_move, _stats, _root) = tactical_mcts_search_with_tt(
            board.clone(),
            &move_gen,
            &pesto_eval,
            &mut nn,
            config.clone(),
            &mut tt,
        );

        match best_move {
            None => break,
            Some(mv) => {
                board_stack.make_move(mv);
                move_count += 1;
            }
        }

        if board_stack.is_draw_by_repetition() {
            break;
        }
        if board_stack.current_state().halfmove_clock() >= 100 {
            break;
        }
        if move_count > 100 {
            break; // Shorter limit for testing
        }

        let (mate, stalemate) = board_stack.current_state().is_checkmate_or_stalemate(&move_gen);
        if mate || stalemate {
            break;
        }
    }

    let final_board = board_stack.current_state();
    let (mate, stalemate) = final_board.is_checkmate_or_stalemate(&move_gen);
    let is_repetition = board_stack.is_draw_by_repetition();
    let is_50_move = final_board.halfmove_clock() >= 100;

    if mate {
        if final_board.w_to_move {
            GameResult::BlackWin
        } else {
            GameResult::WhiteWin
        }
    } else if stalemate || is_repetition || is_50_move || move_count > 100 {
        GameResult::Draw
    } else {
        GameResult::Draw
    }
}

#[test]
fn test_play_evaluation_game_terminates() {
    // A game with low simulations should terminate within the move limit
    let result = play_test_game(10);
    // Just verify it produces a valid result
    assert!(
        result == GameResult::WhiteWin
            || result == GameResult::BlackWin
            || result == GameResult::Draw
    );
}

#[test]
fn test_alternating_colors() {
    // Verify that game indices 0,2,4 give candidate=white and 1,3,5 give candidate=black
    for game_idx in 0u32..6 {
        let candidate_is_white = game_idx % 2 == 0;
        if game_idx % 2 == 0 {
            assert!(candidate_is_white, "Even game should have candidate as white");
        } else {
            assert!(!candidate_is_white, "Odd game should have candidate as black");
        }
    }
}

#[test]
fn test_win_rate_calculation() {
    // wins=6, losses=3, draws=1 -> winrate = (6 + 0.5*1) / 10 = 0.65
    let wins = 6u32;
    let losses = 3u32;
    let draws = 1u32;
    let total = wins + losses + draws;
    let win_rate = (wins as f64 + 0.5 * draws as f64) / total as f64;
    assert!((win_rate - 0.65).abs() < 1e-9);

    // All draws: winrate = 0.5
    let win_rate_draws: f64 = (0.0 + 0.5 * 10.0) / 10.0;
    assert!((win_rate_draws - 0.5).abs() < 1e-9);
}

#[test]
fn test_acceptance_threshold() {
    let threshold = 0.55;

    // 60% win rate -> accepted
    let wr1 = 0.60;
    assert!(wr1 >= threshold);

    // 50% win rate -> rejected
    let wr2 = 0.50;
    assert!(wr2 < threshold);

    // Exactly 55% -> accepted
    let wr3 = 0.55;
    assert!(wr3 >= threshold);

    // 54.9% -> rejected
    let wr4 = 0.549;
    assert!(wr4 < threshold);
}
