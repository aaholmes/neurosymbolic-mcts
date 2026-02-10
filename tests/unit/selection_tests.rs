//! Tests for PUCT selection logic

use kingfisher::mcts::selection::calculate_ucb_value;
use kingfisher::mcts::node::MctsNode;
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;

#[test]
fn test_ucb_unvisited_uses_zero_q() {
    // When a child has 0 visits, UCB should use Q=0
    let parent_visits = 100;
    let prior_prob = 0.1;
    let exploration = 1.5;

    let ucb = calculate_ucb_value_for_test(
        0,    // visits
        0.0,  // total_value
        parent_visits,
        prior_prob,
        exploration,
    );

    // UCB = Q(0) + U = 0 + c * p * sqrt(N) / (1 + 0)
    let expected_u = exploration * prior_prob * (parent_visits as f64).sqrt();
    let diff = (ucb - expected_u).abs();
    assert!(diff < 0.0001, "UCB for unvisited node should equal the exploration term, got {} vs {}", ucb, expected_u);
}

#[test]
fn test_ucb_visited_uses_average_q() {
    // Once a node has visits, Q = total_value / visits
    let parent_visits = 100;
    let prior_prob = 0.1;
    let exploration = 1.5;

    // Visited node with total_value = 5.0 over 10 visits = Q of 0.5
    let ucb = calculate_ucb_value_for_test(
        10,   // visits
        5.0,  // total_value
        parent_visits,
        prior_prob,
        exploration,
    );

    let expected_q = 5.0 / 10.0;
    let expected_u = exploration * prior_prob * (parent_visits as f64).sqrt() / (1.0 + 10.0);
    let expected = expected_q + expected_u;
    let diff = (ucb - expected).abs();
    assert!(diff < 0.0001, "UCB should be Q + U = {}, got {}", expected, ucb);
}

#[test]
fn test_ucb_exploration_term() {
    // Verify the exploration term decreases as a node gets more visits

    let parent_visits = 100;
    let prior_prob = 0.2;
    let exploration = 1.5;

    let ucb_1_visit = calculate_ucb_value_for_test(1, 0.5, parent_visits, prior_prob, exploration);
    let ucb_10_visits = calculate_ucb_value_for_test(10, 5.0, parent_visits, prior_prob, exploration);
    let ucb_50_visits = calculate_ucb_value_for_test(50, 25.0, parent_visits, prior_prob, exploration);

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
        exploration_constant,
    )
}
