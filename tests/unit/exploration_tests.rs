//! Exploration in self-play: the move played is sampled in proportion to
//! (visits - 1)^(1/tau), with tau falling over the game, and the search can
//! mix Dirichlet noise into the root's priors.
use kingfisher::board::Board;
use kingfisher::mcts::{tactical_mcts_search_for_training_with_reuse, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::training_data::{sample_with_temperature, temperature};
use kingfisher::transposition::TranspositionTable;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Duration;

fn mv(from: usize, to: usize) -> Move {
    Move::new(from, to, None)
}

fn frequency_of_first(policy: &[(Move, u32)], tau: f64, n: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(1);
    let hits = (0..n)
        .filter(|_| sample_with_temperature(policy, tau, &mut rng) == Some(policy[0].0))
        .count();
    hits as f64 / n as f64
}

#[test]
fn temperature_starts_at_tau0_and_decays_per_move() {
    assert_eq!(temperature(1, 1.0, 0.5, 0.0), 1.0);
    assert_eq!(temperature(3, 1.0, 0.5, 0.0), 0.25);
    assert_eq!(temperature(3, 0.8, 0.5, 0.0), 0.2);
}

#[test]
fn temperature_never_falls_below_tau_min() {
    assert_eq!(temperature(50, 1.0, 0.5, 0.1), 0.1);
}

#[test]
fn tau_one_samples_in_proportion_to_visits_minus_one() {
    // Weights 3, 1, 0: the first move is played 3/4 of the time.
    let policy = [(mv(12, 28), 4), (mv(11, 27), 2), (mv(6, 21), 1)];
    let f = frequency_of_first(&policy, 1.0, 8000);
    assert!((f - 0.75).abs() < 0.02, "frequency {f}");
}

#[test]
fn lower_tau_sharpens_the_distribution() {
    // Weights 3^2 and 1^2 at tau = 0.5: the first move is played 9/10 of the time.
    let policy = [(mv(12, 28), 4), (mv(11, 27), 2)];
    let f = frequency_of_first(&policy, 0.5, 8000);
    assert!((f - 0.9).abs() < 0.015, "frequency {f}");
}

#[test]
fn a_move_visited_once_is_never_played_when_another_has_more_visits() {
    let policy = [(mv(12, 28), 1), (mv(11, 27), 3), (mv(6, 21), 1)];
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..500 {
        assert_eq!(
            sample_with_temperature(&policy, 1.0, &mut rng),
            Some(mv(11, 27))
        );
    }
}

#[test]
fn tau_zero_plays_the_most_visited_move() {
    let policy = [(mv(12, 28), 5), (mv(11, 27), 9), (mv(6, 21), 7)];
    let mut rng = StdRng::seed_from_u64(3);
    assert_eq!(
        sample_with_temperature(&policy, 0.0, &mut rng),
        Some(mv(11, 27))
    );
}

#[test]
fn very_small_tau_does_not_overflow() {
    let policy = [(mv(12, 28), 200), (mv(11, 27), 199)];
    let mut rng = StdRng::seed_from_u64(4);
    assert_eq!(
        sample_with_temperature(&policy, 1e-4, &mut rng),
        Some(mv(12, 28))
    );
}

#[test]
fn falls_back_to_the_most_visited_move_when_none_has_a_second_visit() {
    let policy = [(mv(12, 28), 0), (mv(11, 27), 1)];
    let mut rng = StdRng::seed_from_u64(5);
    assert_eq!(
        sample_with_temperature(&policy, 1.0, &mut rng),
        Some(mv(11, 27))
    );
    assert_eq!(sample_with_temperature(&[], 1.0, &mut rng), None);
}

fn root_priors(epsilon: f64, seed: u64) -> Vec<(Move, f64)> {
    let move_gen = MoveGen::new();
    let config = TacticalMctsConfig {
        max_iterations: 60,
        time_limit: Duration::from_secs(10),
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: epsilon,
        noise_seed: seed,
        ..Default::default()
    };
    let result = tactical_mcts_search_for_training_with_reuse(
        Board::new(),
        &move_gen,
        config,
        None,
        &mut TranspositionTable::new(),
    );
    let root = result.root_node.borrow();
    let mut priors: Vec<(Move, f64)> = root.move_priorities.iter().map(|(m, p)| (*m, *p)).collect();
    priors.sort_by_key(|(m, _)| (m.from(), m.to()));
    priors
}

#[test]
fn search_mixes_noise_into_the_root_priors() {
    let clean = root_priors(0.0, 7);
    let noisy = root_priors(0.25, 7);
    assert_eq!(clean.len(), 20);
    assert_eq!(noisy.len(), 20);
    let sum: f64 = noisy.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-9, "noisy priors sum to {sum}");
    let largest_change = clean
        .iter()
        .zip(&noisy)
        .map(|((_, a), (_, b))| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        largest_change > 0.01,
        "noise changed priors by at most {largest_change}"
    );
    // No prior falls below its share of the clean prior.
    assert!(clean
        .iter()
        .zip(&noisy)
        .all(|((_, a), (_, b))| *b >= 0.75 * a - 1e-12));
}

#[test]
fn root_noise_is_reproducible_from_its_seed() {
    assert_eq!(root_priors(0.25, 7), root_priors(0.25, 7));
    assert_ne!(root_priors(0.25, 7), root_priors(0.25, 8));
}

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

#[test]
fn c_puct_defaults_to_the_constant_used_so_far() {
    use kingfisher::training_data::{c_puct_arg, DEFAULT_C_PUCT};
    assert_eq!(DEFAULT_C_PUCT, 1.414);
    assert_eq!(c_puct_arg(&args(&["sp", "10"]), "--c-puct"), Ok(1.414));
}

#[test]
fn c_puct_is_read_from_its_flag() {
    use kingfisher::training_data::c_puct_arg;
    let a = args(&["ev", "--candidate-c-puct", "2.5", "--current-c-puct", "1.1"]);
    assert_eq!(c_puct_arg(&a, "--candidate-c-puct"), Ok(2.5));
    assert_eq!(c_puct_arg(&a, "--current-c-puct"), Ok(1.1));
}

#[test]
fn c_puct_must_be_a_positive_number() {
    use kingfisher::training_data::c_puct_arg;
    for bad in ["abc", "0", "-1", "nan"] {
        assert!(
            c_puct_arg(&args(&["sp", "--c-puct", bad]), "--c-puct").is_err(),
            "{bad}"
        );
    }
    assert!(c_puct_arg(&args(&["sp", "--c-puct"]), "--c-puct").is_err());
}
