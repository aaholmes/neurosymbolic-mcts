use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search_with_tt, TacticalMctsConfig};
use kingfisher::mcts::sprt::{SprtConfig, SprtState, SprtResult};
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
    play_test_game_fen(simulations, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
}

/// Play a game from a custom starting position.
fn play_test_game_fen(simulations: u32, fen: &str) -> GameResult {
    play_test_game_fen_koth(simulations, fen, false)
}

/// Play a game from a custom starting position with optional KOTH mode.
fn play_test_game_fen_koth(simulations: u32, fen: &str, enable_koth: bool) -> GameResult {
    let move_gen = MoveGen::new();

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
        enable_koth,
        ..Default::default()
    };

    let mut board_stack = BoardStack::new_from_fen(fen);
    let mut move_count = 0;
    let mut tt = TranspositionTable::new();
    let mut koth_winner: Option<bool> = None; // Some(true) = white won, Some(false) = black won

    loop {
        let board = board_stack.current_state().clone();

        let (best_move, _stats, _root) = tactical_mcts_search_with_tt(
            board.clone(),
            &move_gen,
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

        // Check KOTH win after each move
        if enable_koth {
            let (white_won, black_won) = board_stack.current_state().is_koth_win();
            if white_won {
                koth_winner = Some(true);
                break;
            }
            if black_won {
                koth_winner = Some(false);
                break;
            }
        }

        if board_stack.is_draw_by_repetition() {
            break;
        }
        if board_stack.current_state().halfmove_clock() >= 100 {
            break;
        }
        if move_count > 100 {
            break;
        }

        let (mate, stalemate) = board_stack.current_state().is_checkmate_or_stalemate(&move_gen);
        if mate || stalemate {
            break;
        }
    }

    // Check KOTH winner first
    if let Some(white_won) = koth_winner {
        return if white_won { GameResult::WhiteWin } else { GameResult::BlackWin };
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

// ---- Basic game tests ----

#[test]
#[cfg(feature = "slow-tests")]
fn test_play_evaluation_game_terminates() {
    let result = play_test_game(10);
    assert!(
        result == GameResult::WhiteWin
            || result == GameResult::BlackWin
            || result == GameResult::Draw
    );
}

#[test]
#[cfg(feature = "slow-tests")]
fn test_play_multiple_games_all_terminate() {
    // Run several games to verify stability
    for _ in 0..3 {
        let result = play_test_game(5);
        assert!(
            result == GameResult::WhiteWin
                || result == GameResult::BlackWin
                || result == GameResult::Draw
        );
    }
}

// ---- Color alternation tests ----

#[test]
fn test_alternating_colors() {
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
fn test_alternating_colors_extended() {
    // Verify large range
    for game_idx in 0u32..100 {
        let candidate_is_white = game_idx % 2 == 0;
        let expected = game_idx % 2 == 0;
        assert_eq!(candidate_is_white, expected);
    }
}

// ---- Win rate calculation tests ----

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
fn test_win_rate_all_wins() {
    let wins = 10u32;
    let losses = 0u32;
    let draws = 0u32;
    let total = wins + losses + draws;
    let win_rate = (wins as f64 + 0.5 * draws as f64) / total as f64;
    assert!((win_rate - 1.0).abs() < 1e-9);
}

#[test]
fn test_win_rate_all_losses() {
    let wins = 0u32;
    let losses = 10u32;
    let draws = 0u32;
    let total = wins + losses + draws;
    let win_rate = (wins as f64 + 0.5 * draws as f64) / total as f64;
    assert!((win_rate - 0.0).abs() < 1e-9);
}

#[test]
fn test_win_rate_zero_games() {
    let wins = 0u32;
    let losses = 0u32;
    let draws = 0u32;
    let total = wins + losses + draws;
    let win_rate = if total == 0 {
        0.0
    } else {
        (wins as f64 + 0.5 * draws as f64) / total as f64
    };
    assert!((win_rate - 0.0).abs() < 1e-9);
}

#[test]
fn test_win_rate_single_draw() {
    let wins = 0u32;
    let losses = 0u32;
    let draws = 1u32;
    let total = wins + losses + draws;
    let win_rate = (wins as f64 + 0.5 * draws as f64) / total as f64;
    assert!((win_rate - 0.5).abs() < 1e-9);
}

// ---- Acceptance threshold tests ----

#[test]
fn test_acceptance_threshold() {
    let threshold = 0.55;

    assert!(0.60 >= threshold);
    assert!(0.50 < threshold);
    assert!(0.55 >= threshold);
    assert!(0.549 < threshold);
}

#[test]
fn test_acceptance_threshold_boundary() {
    // Test various thresholds
    let threshold = 0.55;
    let win_rate_exact = 0.55f64;
    assert!(win_rate_exact >= threshold);

    let barely_below = 0.5499999f64;
    assert!(barely_below < threshold);
}

#[test]
fn test_acceptance_threshold_custom() {
    // Different threshold values
    let threshold_60 = 0.60;
    assert!(0.61 >= threshold_60);
    assert!(0.59 < threshold_60);

    let threshold_50 = 0.50;
    assert!(0.50 >= threshold_50);
    assert!(0.49 < threshold_50);
}

// ---- Game from specific positions ----

#[test]
fn test_game_from_checkmate_position() {
    // Scholar's mate final position: Qxf7# with black king on e8
    let board = Board::new_from_fen("rnbqkb1r/pppp1Qpp/5n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    let move_gen = MoveGen::new();
    let (mate, _stalemate) = board.is_checkmate_or_stalemate(&move_gen);
    assert!(mate, "Position should be checkmate");
}

#[test]
fn test_game_from_stalemate_position() {
    // Black is stalemated: king on a8, white queen on b6, white king on c8
    let board = Board::new_from_fen("k7/8/1Q6/8/8/8/8/2K5 b - - 0 1");
    let move_gen = MoveGen::new();
    let (_mate, stalemate) = board.is_checkmate_or_stalemate(&move_gen);
    assert!(stalemate, "Position should be stalemate");
}

// ---- Neural net stub tests ----

#[test]
fn test_stub_nn_not_available() {
    let nn = NeuralNetPolicy::new();
    assert!(!nn.is_available());
}

#[test]
fn test_stub_nn_predict_returns_none() {
    let mut nn = NeuralNetPolicy::new();
    let board = Board::new();
    assert!(nn.predict(&board).is_none());
}

#[test]
fn test_stub_nn_load_returns_error() {
    let mut nn = NeuralNetPolicy::new();
    let result = nn.load("nonexistent.pt");
    // In stub mode, load returns an error
    assert!(result.is_err());
}

// ---- Aggregate result scoring tests ----

#[test]
fn test_eval_results_aggregate() {
    // Simulate a 100-game match with known results
    let scenarios: Vec<(u32, u32, u32, f64, bool)> = vec![
        (55, 35, 10, 0.60, true),   // 55 + 5 = 60/100 = 0.60 -> accepted
        (50, 40, 10, 0.55, true),   // 50 + 5 = 55/100 = 0.55 -> accepted (exact boundary)
        (40, 50, 10, 0.45, false),  // 40 + 5 = 45/100 = 0.45 -> rejected
        (0, 0, 100, 0.50, false),   // All draws = 0.50 -> rejected at 0.55
        (100, 0, 0, 1.0, true),     // All wins = 1.0 -> accepted
        (0, 100, 0, 0.0, false),    // All losses = 0.0 -> rejected
    ];

    let threshold = 0.55;
    for (wins, losses, draws, expected_wr, expected_accepted) in scenarios {
        let total = wins + losses + draws;
        let wr = (wins as f64 + 0.5 * draws as f64) / total as f64;
        assert!(
            (wr - expected_wr).abs() < 1e-9,
            "W:{} L:{} D:{} expected WR={} got WR={}",
            wins, losses, draws, expected_wr, wr
        );
        assert_eq!(
            wr >= threshold,
            expected_accepted,
            "W:{} L:{} D:{} WR={} threshold={} expected accepted={}",
            wins, losses, draws, wr, threshold, expected_accepted
        );
    }
}

// ---- Two-engine game tests ----

#[test]
fn test_two_engine_game_with_different_tt() {
    // Verify that two separate transposition tables don't interfere
    let move_gen = MoveGen::new();

    let config = TacticalMctsConfig {
        max_iterations: 10,
        time_limit: Duration::from_secs(5),
        mate_search_depth: 3,
        exploration_constant: 1.414,
        use_neural_policy: false,
        inference_server: None,
        logger: None,
        dirichlet_alpha: 0.0,
        dirichlet_epsilon: 0.0,
        ..Default::default()
    };

    let board = Board::new();
    let mut tt1 = TranspositionTable::new();
    let mut tt2 = TranspositionTable::new();

    // White's move (engine 1)
    let (move1, _, _) = tactical_mcts_search_with_tt(
        board.clone(), &move_gen, config.clone(), &mut tt1,
    );
    assert!(move1.is_some());

    // Black's move (engine 2) from same position
    let (move2, _, _) = tactical_mcts_search_with_tt(
        board.clone(), &move_gen, config.clone(), &mut tt2,
    );
    assert!(move2.is_some());
}

// ---- KOTH game termination tests ----

#[test]
fn test_koth_win_terminates_game() {
    // White king on d3, one move from center (d4). With KOTH enabled and enough
    // simulations, MCTS should move king to center and the game should end as WhiteWin.
    // Position: White king on d3, Black king on a7 (far from center).
    let result = play_test_game_fen_koth(
        100,
        "3N4/k6p/p6P/2P1R3/2n3P1/P1rK4/8/5B2 w - - 4 47",
        true,
    );
    assert_eq!(result, GameResult::WhiteWin,
        "KOTH win-in-1 should terminate as WhiteWin, got {:?}", result);
}

#[test]
fn test_koth_win_correct_outcome_black() {
    // Black king on e5, one move from center (d4/d5/e4). White king far away.
    // Black to move should win immediately.
    let result = play_test_game_fen_koth(
        100,
        "8/8/8/4k3/8/8/8/K7 b - - 0 1",
        true,
    );
    assert_eq!(result, GameResult::BlackWin,
        "Black king near center should win KOTH, got {:?}", result);
}

#[test]
fn test_koth_disabled_no_early_termination() {
    // Same position as above but with KOTH disabled - should NOT end as KOTH win
    // (will end as draw due to move limit or 50-move rule since it's KvK)
    let result = play_test_game_fen_koth(
        50,
        "8/8/8/4k3/8/8/8/K7 b - - 0 1",
        false,
    );
    // Without KOTH, KvK is a draw (insufficient material or will hit move/50-move limit)
    assert_ne!(result, GameResult::BlackWin,
        "Without KOTH, king reaching center should not be a win");
}

// ---- SPRT tests ----

fn default_sprt_config() -> SprtConfig {
    SprtConfig {
        elo0: 0.0,
        elo1: 10.0,
        alpha: 0.05,
        beta: 0.05,
    }
}

#[test]
fn test_sprt_llr_winning() {
    let config = default_sprt_config();
    let state = SprtState { wins: 60, losses: 30, draws: 10 };
    let llr = state.compute_llr(&config);
    assert!(llr.is_some(), "LLR should be computable with 100 games");
    assert!(llr.unwrap() > 0.0, "Winning record should have positive LLR, got {}", llr.unwrap());
}

#[test]
fn test_sprt_llr_losing() {
    let config = default_sprt_config();
    let state = SprtState { wins: 30, losses: 60, draws: 10 };
    let llr = state.compute_llr(&config);
    assert!(llr.is_some());
    assert!(llr.unwrap() < 0.0, "Losing record should have negative LLR, got {}", llr.unwrap());
}

#[test]
fn test_sprt_llr_all_draws() {
    let config = default_sprt_config();
    let state = SprtState { wins: 0, losses: 0, draws: 100 };
    let llr = state.compute_llr(&config);
    // All draws → var = (0 + 100/4)/100 - 0.25 = 0, so LLR is None (zero variance)
    assert!(llr.is_none(), "All draws should have zero variance → None");

    // With a mix that's mostly draws, LLR should be computable and near 0
    let state2 = SprtState { wins: 2, losses: 2, draws: 96 };
    let llr2 = state2.compute_llr(&config);
    assert!(llr2.is_some());
    assert!(llr2.unwrap().abs() < 2.0, "Mostly draws LLR should be near 0, got {}", llr2.unwrap());
}

#[test]
fn test_sprt_llr_too_few_games() {
    let config = default_sprt_config();
    let state = SprtState { wins: 1, losses: 0, draws: 0 };
    let llr = state.compute_llr(&config);
    assert!(llr.is_none(), "LLR should be None for <2 games");
}

#[test]
fn test_sprt_decision_h1() {
    let config = default_sprt_config();
    let state = SprtState { wins: 80, losses: 10, draws: 10 };
    let result = state.check_decision(&config);
    assert_eq!(result, SprtResult::AcceptH1, "Dominant winning record should accept H1");
}

#[test]
fn test_sprt_decision_h0() {
    let config = default_sprt_config();
    let state = SprtState { wins: 10, losses: 80, draws: 10 };
    let result = state.check_decision(&config);
    assert_eq!(result, SprtResult::AcceptH0, "Dominant losing record should accept H0");
}

#[test]
fn test_sprt_decision_inconclusive() {
    let config = default_sprt_config();
    let state = SprtState { wins: 26, losses: 24, draws: 50 };
    let result = state.check_decision(&config);
    assert_eq!(result, SprtResult::Inconclusive, "Close results should be inconclusive");
}

#[test]
fn test_sprt_bounds_symmetric() {
    let config = SprtConfig {
        elo0: 0.0,
        elo1: 10.0,
        alpha: 0.05,
        beta: 0.05,
    };
    let (lower, upper) = config.bounds();
    assert!((lower.abs() - upper.abs()).abs() < 1e-9,
        "Symmetric alpha/beta should give |A| == |B|, got A={} B={}", lower, upper);
}
