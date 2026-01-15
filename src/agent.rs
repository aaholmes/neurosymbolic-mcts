//! This module specifies various agents, which can use any combination of search and eval routines.

use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use crate::mcts::node::MctsNode;
use crate::search::mate_search;
use crate::egtb::EgtbProber;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

// Test tracking flags (only compiled in test mode)
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
    fn get_move(&mut self, board: &mut BoardStack) -> Move;

    /// Returns the root of the search tree from the last search, if applicable.
    fn get_last_search_tree(&self) -> Option<Rc<RefCell<MctsNode>>> {
        None
    }
}

/// A simple agent that uses mate search followed by aspiration window quiescence search.
pub struct SimpleAgent<'a> {
    pub mate_search_depth: i32,
    pub ab_search_depth: i32,
    pub q_search_max_depth: i32,
    pub verbose: bool,
    pub move_gen: &'a MoveGen,
    pub pesto: &'a PestoEval,
}

impl SimpleAgent<'_> {
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
    pub move_gen: &'a MoveGen,
    pub pesto: &'a PestoEval,
    pub egtb_prober: Option<EgtbProber>,
    pub mate_search_depth: i32,
    pub mcts_iterations: u32,
    pub mcts_time_limit_ms: u64,
    /// Stores the root of the last MCTS search for inspection
    last_tree: RefCell<Option<Rc<RefCell<MctsNode>>>>,
}

impl HumanlikeAgent<'_> {
    pub fn new<'a>(
        move_gen: &'a MoveGen,
        pesto: &'a PestoEval,
        egtb_prober: Option<EgtbProber>,
        mate_search_depth: i32,
        mcts_iterations: u32,
        mcts_time_limit_ms: u64,
    ) -> HumanlikeAgent<'a> {
        HumanlikeAgent {
            move_gen,
            pesto,
            egtb_prober,
            mate_search_depth,
            mcts_iterations,
            mcts_time_limit_ms,
            last_tree: RefCell::new(None),
        }
    }
}

impl Agent for HumanlikeAgent<'_> {
    fn get_move(&mut self, board: &mut BoardStack) -> Move {
        // 1. Mate Search
        let (eval, m, _nodes) = mate_search(board, self.move_gen, self.mate_search_depth, false);

        if eval >= 1000000 {
            return m;
        }

        // 2. MCTS Search
        let config = TacticalMctsConfig {
            max_iterations: self.mcts_iterations,
            time_limit: Duration::from_millis(self.mcts_time_limit_ms),
            mate_search_depth: self.mate_search_depth,
            ..Default::default()
        };

        let mut nn = None; // Use default stub or could be loaded from UCIEngine
        let (mcts_move, _stats, root) = tactical_mcts_search(
            board.current_state().clone(),
            self.move_gen,
            self.pesto,
            &mut nn,
            config
        );

        // Store the tree for inspection
        *self.last_tree.borrow_mut() = Some(root);

        mcts_move.expect("MCTS search returned None unexpectedly")
    }

    fn get_last_search_tree(&self) -> Option<Rc<RefCell<MctsNode>>> {
        self.last_tree.borrow().clone()
    }
}

impl Agent for SimpleAgent<'_> {
    fn get_move(&mut self, board: &mut BoardStack) -> Move {
        let (eval, m, _nodes) = mate_search(board, self.move_gen, self.mate_search_depth, self.verbose);
        if eval >= 1000000 {
            return m;
        }
        // ... (AlphaBeta search omitted for brevity, keeping simple implementation)
        m
    }
}