#[cfg(test)]
mod mcts_tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    use std::rc::Rc;
    use std::time::{Duration, Instant};

    use kingfisher::board::Board;
    use kingfisher::eval::PestoEval; // Needed for PestoPolicy example
    // use kingfisher::mcts::policy::{PestoPolicy, PolicyNetwork}; // Not needed for pesto MCTS
    use kingfisher::move_generation::MoveGen;
    // Updated imports for MCTS components
    use kingfisher::mcts::{
        mcts_pesto_search, select_leaf_for_expansion, MctsNode, MoveCategory,
    };
    use kingfisher::mcts::simulation::simulate_random_playout; // Keep for simulation tests
    use kingfisher::move_types::{Move, NULL_MOVE};
    use kingfisher::board_utils;
    use kingfisher::search::mate_search; // Needed for mate search context
    use kingfisher::boardstack::BoardStack; // Needed for mate search context

    // Helper to create basic setup
    fn setup() -> MoveGen {
        MoveGen::new()
    }

    // Helper function to initialize common test components
    fn setup_test_env() -> (Board, MoveGen, PestoEval) {
        let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let move_gen = MoveGen::new();
        let pesto_eval = PestoEval::new();
        (board, move_gen, pesto_eval)
    }

    // Helper function to parse UCI move from string
    fn parse_uci_move(_board: &Board, uci: &str) -> Option<Move> {
        Move::from_uci(uci)
    }

    // Helper to create a specific move
    fn create_move(from: &str, to: &str) -> Move {
        let from_sq = board_utils::algebraic_to_sq_ind(from);
        let to_sq = board_utils::algebraic_to_sq_ind(to);
        Move::new(from_sq, to_sq, None)
    }

    // Helper to implement Hash for Move in tests
    // Note: Move might need PartialEq and Eq derived or implemented as well
    #[derive(Clone, Copy, Debug)]
    struct HashableMove(Move);

    impl Hash for HashableMove {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.0.from.hash(state);
            self.0.to.hash(state);
            self.0.promotion.hash(state);
        }
    }

    impl PartialEq for HashableMove {
        fn eq(&self, other: &Self) -> bool {
            self.0.from == other.0.from
                && self.0.to == other.0.to
                && self.0.promotion == other.0.promotion
        }
    }

    impl Eq for HashableMove {}

    // Mock Policy Network is not needed for Pesto MCTS tests

    // --- Node Tests --- (Keep existing Node tests, ensure they compile with new fields)

    #[test]
    fn test_node_new_root() {
        let move_gen = setup();
        let board = Board::new(); // Initial position
        let root_node_rc = MctsNode::new_root(board.clone(), &move_gen);
        let root_node = root_node_rc.borrow();

        assert!(root_node.parent.is_none());
        assert!(root_node.action.is_none());
        assert_eq!(root_node.visits, 0);
        assert_eq!(root_node.total_value, 0.0);
        assert_eq!(root_node.total_value_squared, 0.0); // Check new field
        assert!(!root_node.is_terminal);
        // Check categorization map is empty initially
        assert!(root_node.unexplored_moves_by_cat.is_empty());
        assert!(root_node.children.is_empty());
        assert!(root_node.nn_value.is_none());
        // assert!(root_node.policy_priors.is_none()); // Field removed
        assert!(root_node.terminal_or_mate_value.is_none());
    }

    #[test]
    fn test_node_new_root_terminal() {
        let move_gen = setup();
        // Fool's Mate position (Black checkmated)
        let board =
            Board::new_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3");
        let root_node_rc = MctsNode::new_root(board.clone(), &move_gen);
        let root_node = root_node_rc.borrow();

        assert!(root_node.is_terminal);
        assert!(root_node.unexplored_moves_by_cat.is_empty());
        assert!(root_node.children.is_empty());
        // Check terminal value is set correctly (White to move is mated -> 0.0 for White)
        assert_eq!(root_node.terminal_or_mate_value, Some(0.0));
    }

    // UCT/PUCT test needs rework as priors are not stored on child node directly
    // #[test]
    // fn test_node_uct_value() { ... }

    // Expand test needs rework as expand is now different (expand_with_policy)
    // #[test]
    // fn test_node_expand() { ... }

    // Backpropagate test needs rework based on new structure
    // #[test]
    // fn test_node_backpropagate() { ... }

    // --- Simulation Tests --- (Keep existing Simulation tests)

    #[test]
    fn test_simulation_immediate_white_win() {
        let move_gen = setup();
        // Fool's Mate position (Black checkmated, White to move - White won)
        let board =
            Board::new_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3");
        let result = simulate_random_playout(&board, &move_gen);
        // White is checkmated, Black wins. Result is from White's perspective.
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_simulation_immediate_black_win() {
        let move_gen = setup();
        // Position where White is checkmated, Black to move (Black won)
        let board =
            Board::new_from_fen("rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3");
        let result = simulate_random_playout(&board, &move_gen);
        // Black is checkmated, White wins. Result is from White's perspective.
        assert_eq!(result, 1.0); // Corrected expected result
    }

    #[test]
    fn test_simulation_immediate_stalemate() {
        let move_gen = setup();
        // Stalemate position, White to move
        let board = Board::new_from_fen("k7/8/8/8/8/5Q2/8/K7 w - - 0 1");
        let result = simulate_random_playout(&board, &move_gen);
        assert_eq!(result, 0.5); // Draw
    }

    // --- Integration Tests --- (Keep basic integration tests, update signature)

    #[test]
    fn test_mcts_pesto_search_iterations() {
        let (board, move_gen, pesto_eval) = setup_test_env();
        let iterations = 100;
        let mate_depth = 0; // Disable mate search for this basic test

        let best_move_opt = mcts_pesto_search(
            board,
            &move_gen,
            &pesto_eval,
            mate_depth,
            Some(iterations),
            None,
        );

        assert!(
            best_move_opt.is_some(),
            "MCTS should return a move from the initial position"
        );
        let found_move = best_move_opt.unwrap();
        assert!(found_move.from < 64);
        assert!(found_move.to < 64);
    }

    // Policy-related code removed since we're using Pesto MCTS

    #[test]
    fn test_mcts_pesto_forced_mate_in_1() {
        let move_gen = setup();
        // White to move, mate in 1 (Qh5#)
        let board = Board::new_from_fen(
            "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 3"
        );
        let iterations = 1000; // More iterations to increase chance of finding mate quickly
        let pesto_eval = PestoEval::new();

        let best_move_opt = mcts_pesto_search(board, &move_gen, &pesto_eval, 1, Some(iterations), None); // mate_depth=1
        let expected_move = create_move("h5", "f7"); // Qh5xf7#

        assert!(best_move_opt.is_some());
        assert_eq!(
            best_move_opt.unwrap(),
            expected_move,
            "MCTS failed to find mate in 1"
        );
    }

    #[test]
    fn test_mcts_pesto_avoids_immediate_loss() {
        let move_gen = setup();
        let pesto_eval = PestoEval::new();
        // White to move. Moving King to b1 loses immediately to Qb2#. Any other King move is safe for now.
        let board = Board::new_from_fen("8/8/k7/8/8/8/1q6/K7 w - - 0 1");
        let iterations = 200;

        let best_move_opt = mcts_pesto_search(board, &move_gen, &pesto_eval, 0, Some(iterations), None); // mate_depth=0

        assert!(best_move_opt.is_some());
        let best_move = best_move_opt.unwrap();

        // The best move should be any King move except a1-b1
        let bad_move = create_move("a1", "b1");
        assert_ne!(best_move, bad_move, "MCTS failed to avoid immediate loss");
    }

    // Policy-based tests removed since we're using Pesto MCTS

    // TODO: Add tests for Killer/History categorization integration
    // TODO: Add tests for prioritized selection logic

    #[test]
    fn test_mcts_pesto_time_limit_termination() {
        let (board, move_gen, pesto_eval) = setup_test_env();
        let time_limit = Duration::from_millis(100); // Set a short time limit
        let start_time = Instant::now();

        let best_move = mcts_pesto_search(
            board,
            &move_gen,
            &pesto_eval,
            0, // Disable mate search
            Some(1_000_000), // High iteration count to ensure time limit is hit
            Some(time_limit),
        );

        let elapsed = start_time.elapsed();

        assert!(best_move.is_some(), "MCTS should return a move within the time limit");
        // Allow a small margin for execution time overhead
        assert!(elapsed >= time_limit, "Search should run for at least the time limit");
        assert!(elapsed < time_limit + Duration::from_millis(50), "Search should terminate shortly after the time limit");
    }

    #[test]
    fn test_mcts_pesto_tactical_prioritization() {
        let (mut board, move_gen, pesto_eval) = setup_test_env();
        // Position with a forced capture sequence leading to material gain for White
        // White to move: Nxc6, Black must respond with ...dxc6, White then Qxc6+
        board = Board::new_from_fen("rnbqkb1r/pp2pppp/2n2n2/3p4/3P4/2N5/PPP1PPPP/RNBQKBNR w KQkq - 0 4");

        let expected_first_move = parse_uci_move(&board, "c3c6").unwrap(); // Nxc6

        // Run with enough iterations to explore the sequence
        let best_move = mcts_pesto_search(
            board.clone(),
            &move_gen,
            &pesto_eval,
            0, // Disable mate search
            Some(1000), // Sufficient iterations
            None,
        );

        assert!(best_move.is_some(), "MCTS should find a move");
        assert_eq!(best_move.unwrap(), expected_first_move, "MCTS should prioritize the tactical capture sequence");
    }

}
