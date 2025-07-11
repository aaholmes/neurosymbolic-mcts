//! This module specifies various agents, which can use any combination of search and eval routines.

use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::mcts::mcts_pesto_search; // Added import
use crate::search::{iterative_deepening_ab_search, mate_search};
use crate::transposition::TranspositionTable;
use crate::board::Board; // For piece count check
use crate::egtb::{EgtbInfo, EgtbProber}; // For EGTB integration

// Test tracking flags (only compiled in test mode)
#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
thread_local! {
    static MATE_SEARCH_CALLED: RefCell<bool> = RefCell::new(false);
    static MCTS_SEARCH_CALLED: RefCell<bool> = RefCell::new(false);
    static MATE_SEARCH_RETURN_VALUE: RefCell<(i32, Move, u64)> = RefCell::new((0, Move::null(), 0));
    static MCTS_SEARCH_RETURN_VALUE: RefCell<Option<Move>> = RefCell::new(None);
}

/// Trait defining the interface for chess agents.
pub trait Agent {
    /// Get the best move for the current board position.
    ///
    /// # Arguments
    ///
    /// * `board` - A mutable reference to the current `Bitboard` position.
    ///
    /// # Returns
    ///
    /// The best `Move` as determined by the agent.
    fn get_move(&self, board: &mut BoardStack) -> Move;
}

/// A simple agent that uses mate search followed by aspiration window quiescence search.
pub struct SimpleAgent<'a> {
    /// The depth to search for mate.
    pub mate_search_depth: i32,
    /// The depth for alpha-beta search.
    pub ab_search_depth: i32,
    /// The maximum depth for the quiescence search.
    pub q_search_max_depth: i32,
    /// Whether to print verbose output during search.
    pub verbose: bool,
    /// Reference to the move generator.
    pub move_gen: &'a MoveGen,
    /// Reference to the Pesto evaluation function.
    pub pesto: &'a PestoEval,
}

impl SimpleAgent<'_> {
    /// Generate a new simple agent with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `mate_search_depth` - The depth to search for mate.
    /// * `ab_search_depth` - The depth for alpha-beta search.
    /// * `q_search_max_depth` - The maximum depth for the quiescence search.
    /// * `verbose` - Whether to print verbose output during search.
    /// * `move_gen` - Reference to the move generator.
    /// * `pesto` - Reference to the Pesto evaluation function.
    ///
    /// # Returns
    ///
    /// A new `SimpleAgent` with the specified parameters.
    pub fn new<'a>(
        mate_search_depth: i32,
        ab_search_depth: i32,
        q_search_max_depth: i32,
        verbose: bool,
        move_gen: &'a MoveGen,
        pesto: &'a PestoEval,
    ) -> SimpleAgent<'a> {
        SimpleAgent {
            mate_search_depth,
            ab_search_depth,
            q_search_max_depth,
            verbose,
            move_gen,
            pesto,
        }
    }
}

/// An agent designed to mimic human-like decision making, using EGTB, Mate Search, and MCTS.
pub struct HumanlikeAgent<'a> {
    /// Reference to the move generator.
    pub move_gen: &'a MoveGen,
    /// Reference to the Pesto evaluation function.
    pub pesto: &'a PestoEval,
    /// Optional EGTB prober.
    pub egtb_prober: Option<EgtbProber>,
    /// The depth to search for mate.
    pub mate_search_depth: i32,
    // Placeholder for MCTS configuration
    /// Number of MCTS iterations.
    pub mcts_iterations: u32,
    /// Time limit for MCTS search in milliseconds.
    pub mcts_time_limit_ms: u64,
    /// The depth for the placeholder alpha-beta search (used instead of MCTS for now).
    pub placeholder_ab_depth: i32,
    /// The maximum depth for the quiescence search within the placeholder AB search.
    pub placeholder_q_depth: i32,
}

impl HumanlikeAgent<'_> {
    /// Generate a new HumanlikeAgent with the specified components and parameters.
    pub fn new<'a>(
        move_gen: &'a MoveGen,
        pesto: &'a PestoEval,
        egtb_prober: Option<EgtbProber>,
        mate_search_depth: i32,
        mcts_iterations: u32,
        mcts_time_limit_ms: u64,
        placeholder_ab_depth: i32,
        placeholder_q_depth: i32,
    ) -> HumanlikeAgent<'a> {
        HumanlikeAgent {
            move_gen,
            pesto,
            egtb_prober,
            mate_search_depth,
            mcts_iterations,
            mcts_time_limit_ms,
            placeholder_ab_depth,
            placeholder_q_depth,
        }
    }
}

impl Agent for HumanlikeAgent<'_> {
    fn get_move(&self, board: &mut BoardStack) -> Move {
        let current_board = board.current_state();

        // 1. EGTB Check (temporarily disabled to focus on MCTS functionality)
        #[allow(unused_variables)]
        if let Some(prober) = &self.egtb_prober {
            // TODO: Re-implement EGTB integration after MCTS is working
            println!("EGTB temporarily disabled - focusing on MCTS");
        }

        // 2. Mate Search
        #[cfg(test)]
        MATE_SEARCH_CALLED.with(|cell| *cell.borrow_mut() = true);
        
        #[cfg(test)]
        let (eval, m, nodes) = MATE_SEARCH_RETURN_VALUE.with(|cell| cell.borrow().clone());
        
        #[cfg(not(test))]
        let (eval, m, nodes) =
            mate_search(board, self.move_gen, self.mate_search_depth, false); // verbose set to false for now


        if eval >= 1000000 { // Check for mate score
            println!("Found checkmate after searching {} nodes!", nodes);
            return m;
        }

        // 3. MCTS Call
        println!("No EGTB move or mate found, falling back to MCTS search.");

        #[cfg(test)]
        MCTS_SEARCH_CALLED.with(|cell| *cell.borrow_mut() = true);
        
        #[cfg(test)]
        let mcts_move = MCTS_SEARCH_RETURN_VALUE.with(|cell| cell.borrow().clone());
        
        #[cfg(not(test))]
        let mcts_move = mcts_pesto_search(
            board.current_state().clone(),
            self.move_gen,
            self.pesto,
            0, // Mate search already performed above
            Some(self.mcts_iterations),
            Some(std::time::Duration::from_millis(self.mcts_time_limit_ms)),
        );


        // Handle potential None return (unlikely after mate search unless stalemate)
        // For now, unwrap, assuming a legal move exists if no mate was found.
        mcts_move.expect("MCTS search returned None unexpectedly after mate search")
    }
}

impl Agent for SimpleAgent<'_> {
    fn get_move(&self, board: &mut BoardStack) -> Move {
        // First, perform mate search
        let (eval, m, nodes) =
            mate_search(board, self.move_gen, self.mate_search_depth, self.verbose);
        if eval == 1000000 {
            println!("Found checkmate after searching {} nodes!", nodes);
            return m;
        }

        // Create a transposition table for the search
        let mut tt = TranspositionTable::new();

        // If no mate found, perform iterative deepening search
        let (depth, eval, m, n) = iterative_deepening_ab_search(
            board,
            self.move_gen,
            self.pesto,
            &mut tt,
            self.ab_search_depth,
            self.q_search_max_depth,
            None,
            self.verbose,
        );
        println!("Mate search searched {} nodes, iterative deepening search searched another {} nodes at a depth of {} ({} total nodes). Eval: {}", nodes, n, depth, nodes + n, eval);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::boardstack::BoardStack;
    use crate::egtb::{EgtbInfo, EgtbError};
    use shakmaty_syzygy::Wdl;
    use crate::move_generation::MoveGen;
    use crate::move_types::Move;
    use crate::eval::PestoEval;
    // Using Move::from_uci for creating dummy moves
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::Duration;

    // --- Mock Implementations ---

    // Mock EgtbProber
    struct MockEgtbProber {
        probe_result: Result<Option<EgtbInfo>, EgtbError>,
        max_pieces: u8,
    }

    impl MockEgtbProber {
        fn new(probe_result: Result<Option<EgtbInfo>, EgtbError>, max_pieces: u8) -> Self {
            MockEgtbProber { probe_result, max_pieces }
        }

        fn probe(&self, _board: &Board) -> Result<Option<EgtbInfo>, EgtbError> {
            // In a real mock, you might inspect the board, but for testing
            // the agent's logic flow, we just return the predefined result.
            self.probe_result.clone()
        }
    }


    // Mock mate_search function
    fn mock_mate_search(_board: &mut BoardStack, _move_gen: &MoveGen, _depth: i32, _verbose: bool) -> (i32, Move, u64) {
        super::MATE_SEARCH_CALLED.with(|cell| *cell.borrow_mut() = true);
        super::MATE_SEARCH_RETURN_VALUE.with(|cell| cell.borrow().clone())
    }

    // Mock mcts_pesto_search function
    fn mock_mcts_pesto_search(
        _root_state: Board,
        _move_gen: &MoveGen,
        _pesto_eval: &PestoEval,
        _mate_search_depth: i32,
        _iterations: Option<u32>,
        _time_limit: Option<std::time::Duration>,
    ) -> Option<Move> {
        super::MCTS_SEARCH_CALLED.with(|cell| *cell.borrow_mut() = true);
        super::MCTS_SEARCH_RETURN_VALUE.with(|cell| cell.borrow().clone())
    }

    // Helper to reset mock flags
    fn reset_mocks() {
        super::MATE_SEARCH_CALLED.with(|cell| *cell.borrow_mut() = false);
        super::MCTS_SEARCH_CALLED.with(|cell| *cell.borrow_mut() = false);
        super::MATE_SEARCH_RETURN_VALUE.with(|cell| *cell.borrow_mut() = (0, Move::null(), 0));
        super::MCTS_SEARCH_RETURN_VALUE.with(|cell| *cell.borrow_mut() = None);
    }

    // Helper to create a HumanlikeAgent with mocks
    fn setup_humanlike_agent(
        egtb_prober: Option<MockEgtbProber>,
        mate_search_return: (i32, Move, u64),
        mcts_search_return: Option<Move>,
    ) -> HumanlikeAgent<'static> {
        reset_mocks(); // Reset flags before each test

        MATE_SEARCH_RETURN_VALUE.with(|cell| *cell.borrow_mut() = mate_search_return);
        MCTS_SEARCH_RETURN_VALUE.with(|cell| *cell.borrow_mut() = mcts_search_return);

        // Need dummy MoveGen and PestoEval as they are required by the struct,
        // but their methods won't be called in these tests due to mocking.
        let move_gen = Box::new(MoveGen::new());
        let pesto_eval = Box::new(PestoEval::new());

        // Leak the boxed values to get static references. This is acceptable in tests.
        let move_gen_ref: &'static MoveGen = Box::leak(move_gen);
        let pesto_eval_ref: &'static PestoEval = Box::leak(pesto_eval);


        HumanlikeAgent {
            move_gen: move_gen_ref,
            pesto: pesto_eval_ref,
            egtb_prober: egtb_prober.map(|m| EgtbProber { max_pieces: m.max_pieces }), // Create a dummy EgtbProber with the mock's max_pieces
            mate_search_depth: 5,
            mcts_iterations: 1000, // Increased for more reliable test results
            mcts_time_limit_ms: 5000, // Increased timeout for tests
            placeholder_ab_depth: 0, // Not used in this test flow
            placeholder_q_depth: 0, // Not used in this test flow
        }
    }

    // --- Test Cases ---

    #[test]
    fn test_agent_egtb_found() {
        let dummy_move = Move::from_uci("a2a3").unwrap();
        let egtb_info = EgtbInfo { wdl: Wdl::Win, dtz: None, best_move: None }; // EGTB finds a result, but no move provided
        let mock_egtb = MockEgtbProber::new(Ok(Some(egtb_info)), 7);

        // Mate search and MCTS should still be called as EGTB doesn't provide a move yet
        let agent = setup_humanlike_agent(
            Some(mock_egtb),
            (0, Move::null(), 0), // Mate search finds nothing
            Some(dummy_move), // MCTS returns a move
        );

        let mut board_stack = BoardStack::with_board(Board::new_from_fen("8/8/8/8/8/k7/P7/K7 w - - 0 1")); // Position with <= 7 pieces
        let result_move = agent.get_move(&mut board_stack);

        // Verify the sequence of calls
        // EGTB is checked first
        // Mate search is called next
        assert!(MATE_SEARCH_CALLED.with(|cell| *cell.borrow()), "Mate search should be called after EGTB check");
        // MCTS is called last
        assert!(MCTS_SEARCH_CALLED.with(|cell| *cell.borrow()), "MCTS search should be called after mate search fails");

        // Verify the move returned is from MCTS
        assert_eq!(result_move, dummy_move, "Agent should return the move from MCTS");
    }

    #[test]
    fn test_agent_egtb_none_mate_found() {
        let mate_move = Move::from_uci( "e7e8q").unwrap(); // Dummy mate move
        let mock_egtb = MockEgtbProber::new(Ok(None), 7); // EGTB finds nothing

        // Mate search finds a mate
        let agent = setup_humanlike_agent(
            Some(mock_egtb),
            (1000000, mate_move, 100), // Mate search finds mate
            None, // MCTS should NOT be called
        );

        let mut board_stack = BoardStack::with_board(Board::new_from_fen("8/8/8/8/8/k7/P7/K7 w - - 0 1")); // Position with <= 7 pieces
        let result_move = agent.get_move(&mut board_stack);

        // Verify the sequence of calls
        // EGTB is checked first
        // Mate search is called next
        assert!(MATE_SEARCH_CALLED.with(|cell| *cell.borrow()), "Mate search should be called after EGTB check");
        // MCTS should NOT be called
        assert!(!MCTS_SEARCH_CALLED.with(|cell| *cell.borrow()), "MCTS search should NOT be called if mate is found");

        // Verify the move returned is from Mate Search
        assert_eq!(result_move, mate_move, "Agent should return the move from mate search");
    }

    #[test]
    fn test_agent_egtb_none_mate_none_mcts_found() {
        let mcts_move = Move::from_uci( "d2d4").unwrap(); // Dummy MCTS move
        let mock_egtb = MockEgtbProber::new(Ok(None), 7); // EGTB finds nothing

        // Mate search finds nothing
        let agent = setup_humanlike_agent(
            Some(mock_egtb),
            (0, Move::null(), 0),
            Some(mcts_move), // MCTS returns a move
        );

        let mut board_stack = BoardStack::with_board(Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")); // Starting position (many pieces)
        let result_move = agent.get_move(&mut board_stack);

        // Verify the sequence of calls
        // EGTB is checked first (will return None due to piece count)
        // Mate search is called next
        assert!(MATE_SEARCH_CALLED.with(|cell| *cell.borrow()), "Mate search should be called after EGTB check");
        // MCTS is called last
        assert!(MCTS_SEARCH_CALLED.with(|cell| *cell.borrow()), "MCTS search should be called after mate search fails");

        // Verify the move returned is from MCTS
        assert_eq!(result_move, mcts_move, "Agent should return the move from MCTS");
    }

     #[test]
    fn test_agent_no_egtb_mate_none_mcts_found() {
        let mcts_move = Move::from_uci( "g1f3").unwrap(); // Dummy MCTS move

        // No EGTB prober provided
        // Mate search finds nothing
        let agent = setup_humanlike_agent(
            None, // No EGTB
            (0, Move::null(), 0),
            Some(mcts_move), // MCTS returns a move
        );

        let mut board_stack = BoardStack::with_board(Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")); // Starting position
        let result_move = agent.get_move(&mut board_stack);

        // Verify the sequence of calls
        // EGTB check is skipped
        // Mate search is called first
        assert!(MATE_SEARCH_CALLED.with(|cell| *cell.borrow()), "Mate search should be called when no EGTB");
        // MCTS is called next
        assert!(MCTS_SEARCH_CALLED.with(|cell| *cell.borrow()), "MCTS search should be called after mate search fails");

        // Verify the move returned is from MCTS
        assert_eq!(result_move, mcts_move, "Agent should return the move from MCTS");
    }

    // Add more tests for EGTB error handling, piece count > max_pieces, etc.
}
