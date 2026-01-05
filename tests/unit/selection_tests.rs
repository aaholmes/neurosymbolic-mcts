//! Tests for PUCT selection logic

use kingfisher::mcts::selection::calculate_ucb_value;
use kingfisher::mcts::node::MctsNode;
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use crate::common::assert_approx_eq;

#[test]
fn test_ucb_unvisited_uses_q_init() {
    // When a child has 0 visits, UCB should use q_init instead of 0
    
    let parent_visits = 100;
    let prior_prob = 0.1;
    let exploration = 1.5;
    
    // With q_init = 0.0 (default)
    let ucb_default = calculate_ucb_value_for_test(
        0,    // visits
        0.0,  // total_value
        parent_visits,
        prior_prob,
        0.0,  // q_init
        exploration,
    );
    
    // With q_init = 0.5 (tactical evaluation says this move is good)
    let ucb_tactical = calculate_ucb_value_for_test(
        0,
        0.0,
        parent_visits,
        prior_prob,
        0.5,  // q_init from tactical search
        exploration,
    );
    
    assert!(
        ucb_tactical > ucb_default,
        "Higher q_init should give higher UCB for unvisited nodes"
    );
    
    // The difference should be exactly 0.5 (the q component)
    assert_approx_eq(ucb_tactical - ucb_default, 0.5, 0.0001, "Q-init difference");
}

#[test]
fn test_ucb_visited_ignores_q_init() {
    // Once a node has visits, q_init should be ignored
    
    let parent_visits = 100;
    let prior_prob = 0.1;
    let exploration = 1.5;
    
    // Visited node with total_value = 5.0 over 10 visits = Q of 0.5
    let ucb_visited = calculate_ucb_value_for_test(
        10,   // visits
        5.0,  // total_value
        parent_visits,
        prior_prob,
        -0.9, // q_init (should be ignored)
        exploration,
    );
    
    // Same but with different q_init
    let ucb_visited_2 = calculate_ucb_value_for_test(
        10,
        5.0,
        parent_visits,
        prior_prob,
        0.9,  // different q_init
        exploration,
    );
    
    assert_approx_eq(ucb_visited, ucb_visited_2, 0.0001, 
        "q_init should not affect UCB for visited nodes");
}

#[test]
fn test_ucb_exploration_term() {
    // Verify the exploration term decreases as a node gets more visits
    
    let parent_visits = 100;
    let prior_prob = 0.2;
    let exploration = 1.5;
    
    let ucb_1_visit = calculate_ucb_value_for_test(1, 0.5, parent_visits, prior_prob, 0.0, exploration);
    let ucb_10_visits = calculate_ucb_value_for_test(10, 5.0, parent_visits, prior_prob, 0.0, exploration);
    let ucb_50_visits = calculate_ucb_value_for_test(50, 25.0, parent_visits, prior_prob, 0.0, exploration);
    
    // All have same average Q (0.5), so differences are purely from exploration term
    // Exploration should decrease with more visits
    assert!(ucb_1_visit > ucb_10_visits, "More visits should reduce exploration bonus");
    assert!(ucb_10_visits > ucb_50_visits, "More visits should reduce exploration bonus");
}

fn calculate_ucb_value_for_test(
    visits: u32,
    total_value: f64,
    parent_visits: u32,
    prior_prob: f64,
    q_init: f64,
    exploration_constant: f64,
) -> f64 {
    // Use the REAL MctsNode and REAL calculate_ucb_value function
    let move_gen = MoveGen::new();
    let board = Board::new(); // Dummy board
    let node = MctsNode::new_root(board, &move_gen);
    
    {
        let mut node_mut = node.borrow_mut();
        node_mut.visits = visits;
        node_mut.total_value = total_value;
    }
    
    let node_ref = node.borrow();
    calculate_ucb_value(
        &node_ref,
        parent_visits,
        prior_prob,
        q_init,
        exploration_constant,
        true
    )
}
