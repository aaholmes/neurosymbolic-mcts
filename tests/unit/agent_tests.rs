//! Tests for Agent trait implementations (SimpleAgent and HumanlikeAgent)

use kingfisher::agent::{Agent, SimpleAgent, HumanlikeAgent};
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;

fn setup() -> (MoveGen, PestoEval) {
    (MoveGen::new(), PestoEval::new())
}

// === SimpleAgent Tests ===

#[test]
fn test_simple_agent_finds_mate_in_1() {
    let (move_gen, pesto) = setup();
    // Back rank mate: Re8#
    let mut agent = SimpleAgent::new(2, 2, 2, false, &move_gen, &pesto);
    let mut board = BoardStack::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");

    let mv = agent.get_move(&mut board);
    assert_eq!(mv.to, 60, "SimpleAgent should find Re8# (back rank mate)");
}

#[test]
fn test_simple_agent_no_last_search_tree() {
    let (move_gen, pesto) = setup();
    let agent = SimpleAgent::new(1, 1, 1, false, &move_gen, &pesto);
    assert!(agent.get_last_search_tree().is_none(), "SimpleAgent has no search tree");
}

#[test]
fn test_simple_agent_shallow_returns_legal_move() {
    let (move_gen, pesto) = setup();
    // Depth 1 search from a simple endgame position (fast)
    let mut agent = SimpleAgent::new(1, 1, 1, false, &move_gen, &pesto);
    let mut board = BoardStack::new_from_fen("8/8/8/8/8/4k3/8/4K3 w - - 0 1");

    let mv = agent.get_move(&mut board);
    let new_board = board.current_state().apply_move_to_board(mv);
    assert!(new_board.is_legal(&move_gen), "SimpleAgent should return a legal move");
}

// === HumanlikeAgent Tests ===

#[test]
fn test_humanlike_agent_finds_mate_in_1() {
    let (move_gen, pesto) = setup();
    // Mate search finds this immediately, no need for heavy MCTS
    let mut agent = HumanlikeAgent::new(&move_gen, &pesto, None, 3, 10, 1000);
    let mut board = BoardStack::new_from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");

    let mv = agent.get_move(&mut board);
    assert_eq!(mv.to, 60, "HumanlikeAgent should find Re8# (back rank mate)");
}

#[test]
fn test_humanlike_agent_stores_search_tree() {
    let (move_gen, pesto) = setup();
    // Use simple endgame + minimal iterations for speed
    let mut agent = HumanlikeAgent::new(&move_gen, &pesto, None, 1, 10, 1000);
    let mut board = BoardStack::new_from_fen("8/8/8/3q4/4N3/8/8/K6k w - - 0 1");

    agent.get_move(&mut board);
    let tree = agent.get_last_search_tree();
    assert!(tree.is_some(), "HumanlikeAgent should store search tree after get_move");
}

#[test]
fn test_humanlike_agent_without_egtb() {
    let (move_gen, pesto) = setup();
    let agent = HumanlikeAgent::new(&move_gen, &pesto, None, 1, 10, 1000);
    assert!(agent.egtb_prober.is_none());
}

#[test]
fn test_humanlike_agent_returns_valid_move_tactical() {
    let (move_gen, pesto) = setup();
    // Tactical position — just verify it returns a legal move (not flaky on move choice)
    let mut agent = HumanlikeAgent::new(&move_gen, &pesto, None, 1, 20, 1000);
    let mut board = BoardStack::new_from_fen("8/8/8/3q4/4N3/8/8/K6k w - - 0 1");

    let mv = agent.get_move(&mut board);
    let new_board = board.current_state().apply_move_to_board(mv);
    assert!(new_board.is_legal(&move_gen), "Should return a legal move in tactical position");
}

#[test]
fn test_humanlike_agent_returns_legal_move_endgame() {
    let (move_gen, pesto) = setup();
    // Simple endgame — fast to search
    let mut agent = HumanlikeAgent::new(&move_gen, &pesto, None, 1, 10, 1000);
    let mut board = BoardStack::new_from_fen("8/8/8/8/8/4k3/8/4K3 w - - 0 1");

    let mv = agent.get_move(&mut board);
    let new_board = board.current_state().apply_move_to_board(mv);
    assert!(new_board.is_legal(&move_gen), "HumanlikeAgent should return a legal move");
}
