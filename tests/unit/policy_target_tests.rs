//! The root's visit counts become the policy training target with one visit
//! removed from each move: the search gives every root move a first visit,
//! and a move whose only visit showed it was bad should get no target weight.
use kingfisher::board::Board;
use kingfisher::mcts::{tactical_mcts_search_for_training_with_reuse, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::training_data::policy_target;
use kingfisher::transposition::TranspositionTable;
use std::time::Duration;

fn mv(from: usize, to: usize) -> Move {
    Move::new(from, to, None)
}

#[test]
fn one_visit_is_removed_from_every_move() {
    let target = policy_target(&[(mv(12, 28), 7), (mv(11, 27), 3), (mv(6, 21), 1)]);
    assert_eq!(target, vec![(mv(12, 28), 0.75), (mv(11, 27), 0.25)]);
}

#[test]
fn moves_visited_once_get_no_weight() {
    let target = policy_target(&[(mv(12, 28), 1), (mv(11, 27), 2), (mv(6, 21), 0)]);
    assert_eq!(target, vec![(mv(11, 27), 1.0)]);
}

#[test]
fn falls_back_to_raw_counts_when_no_move_has_a_second_visit() {
    let target = policy_target(&[(mv(12, 28), 1), (mv(11, 27), 1), (mv(6, 21), 0)]);
    assert_eq!(target, vec![(mv(12, 28), 0.5), (mv(11, 27), 0.5)]);
}

#[test]
fn no_visits_gives_an_empty_target() {
    assert!(policy_target(&[]).is_empty());
    assert!(policy_target(&[(mv(12, 28), 0)]).is_empty());
}

#[test]
fn the_search_visits_every_root_move() {
    let move_gen = MoveGen::new();
    let config = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_secs(10),
        ..Default::default()
    };
    let result = tactical_mcts_search_for_training_with_reuse(
        Board::new(),
        &move_gen,
        config,
        None,
        &mut TranspositionTable::new(),
    );
    assert_eq!(result.root_policy.len(), 20);
    assert!(
        result.root_policy.iter().all(|(_, v)| *v >= 1),
        "every root move gets a first visit: {:?}",
        result.root_policy
    );
}
