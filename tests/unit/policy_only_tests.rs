//! The policy-only engine: the value of a leaf comes from the classical search
//! alone, tanh(k·ΔM) with k fixed, and the network supplies only move priors.
//! A leaf's value is then known without the network, so its priors can wait
//! until a simulation needs to choose among its children.

use kingfisher::board::Board;
use kingfisher::mcts::node::MctsNode;
use kingfisher::mcts::tactical_mcts::{
    tactical_mcts_search, PolicyTiming, TacticalMctsConfig, TacticalMctsStats,
};
use kingfisher::mcts::InferenceServer;
use kingfisher::move_generation::MoveGen;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

const MIDDLEGAME: &str = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
const SHARP: &str = "r2qk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq - 0 7";

fn config(
    server: InferenceServer,
    policy_only: Option<PolicyTiming>,
    leaves: u32,
) -> TacticalMctsConfig {
    TacticalMctsConfig {
        max_iterations: 300,
        time_limit: Duration::from_secs(60),
        inference_server: Some(Arc::new(server)),
        enable_tier3_neural: true,
        use_neural_policy: true,
        randomize_move_order: false,
        leaves_per_iteration: leaves,
        policy_only,
        ..Default::default()
    }
}

fn search(fen: &str, cfg: TacticalMctsConfig) -> (TacticalMctsStats, Rc<RefCell<MctsNode>>) {
    let (_, stats, root) = tactical_mcts_search(Board::new_from_fen(fen), &MoveGen::new(), cfg);
    (stats, root)
}

/// Visit counts of the whole tree, depth first in child order.
fn tree_visits(node: &Rc<RefCell<MctsNode>>, out: &mut Vec<u32>) {
    let n = node.borrow();
    out.push(n.visits);
    for c in &n.children {
        tree_visits(c, out);
    }
}

fn visits(root: &Rc<RefCell<MctsNode>>) -> Vec<u32> {
    let mut v = Vec::new();
    tree_visits(root, &mut v);
    v
}

/// Nodes whose priors came from the network.
fn nodes_with_priors(node: &Rc<RefCell<MctsNode>>) -> u32 {
    let n = node.borrow();
    n.nn_priors.is_some() as u32 + n.children.iter().map(nodes_with_priors).sum::<u32>()
}

#[test]
fn the_default_is_the_full_engine() {
    assert_eq!(TacticalMctsConfig::default().policy_only, None);
}

#[test]
fn every_network_request_is_counted() {
    // The full engine sends one request per leaf the classical tiers do not solve.
    let (s, _) = search(
        MIDDLEGAME,
        config(InferenceServer::new_mock_hashed(), None, 1),
    );
    assert!(s.nn_requests > 0);
    assert!(s.nn_requests >= s.nn_evaluations, "{s:?}");
}

#[test]
fn policy_only_values_ignore_the_network_value() {
    for timing in [PolicyTiming::AtLeaf, PolicyTiming::OnRevisit] {
        let (_, up) = search(
            MIDDLEGAME,
            config(InferenceServer::new_mock_biased(3.0), Some(timing), 1),
        );
        let (_, down) = search(
            MIDDLEGAME,
            config(InferenceServer::new_mock_biased(-3.0), Some(timing), 1),
        );
        assert_eq!(visits(&up), visits(&down), "{timing:?}");
        let q = |r: &Rc<RefCell<MctsNode>>| r.borrow().total_value / r.borrow().visits as f64;
        assert!((q(&up) - q(&down)).abs() < 1e-12, "{timing:?}");
    }
    // With the network's value in use, the two networks search differently.
    let (_, up) = search(
        MIDDLEGAME,
        config(InferenceServer::new_mock_biased(3.0), None, 1),
    );
    let (_, down) = search(
        MIDDLEGAME,
        config(InferenceServer::new_mock_biased(-3.0), None, 1),
    );
    let q = |r: &Rc<RefCell<MctsNode>>| r.borrow().total_value / r.borrow().visits as f64;
    assert!((q(&up) - q(&down)).abs() > 1e-3);
}

#[test]
fn requesting_priors_on_revisit_gives_the_same_search() {
    for fen in [MIDDLEGAME, SHARP] {
        let (_, eager) = search(
            fen,
            config(
                InferenceServer::new_mock_hashed(),
                Some(PolicyTiming::AtLeaf),
                1,
            ),
        );
        let (_, lazy) = search(
            fen,
            config(
                InferenceServer::new_mock_hashed(),
                Some(PolicyTiming::OnRevisit),
                1,
            ),
        );
        assert_eq!(visits(&eager), visits(&lazy), "{fen}");
    }
}

#[test]
fn requesting_priors_on_revisit_sends_fewer_requests() {
    for fen in [MIDDLEGAME, SHARP] {
        let (se, _) = search(
            fen,
            config(
                InferenceServer::new_mock_hashed(),
                Some(PolicyTiming::AtLeaf),
                1,
            ),
        );
        let (sl, root) = search(
            fen,
            config(
                InferenceServer::new_mock_hashed(),
                Some(PolicyTiming::OnRevisit),
                1,
            ),
        );
        assert!(
            sl.nn_requests < se.nn_requests,
            "{fen}: {} vs {}",
            sl.nn_requests,
            se.nn_requests
        );
        // One request per node whose children were chosen among by priors, no more.
        assert_eq!(sl.nn_requests, nodes_with_priors(&root), "{fen}");
        // No leaf value comes from the network.
        assert_eq!(sl.nn_evaluations, 0, "{fen}");
    }
}

#[test]
fn several_leaves_per_iteration_work_with_priors_on_revisit() {
    let (s1, r1) = search(
        MIDDLEGAME,
        config(
            InferenceServer::new_mock_hashed(),
            Some(PolicyTiming::OnRevisit),
            1,
        ),
    );
    let (s4, r4) = search(
        MIDDLEGAME,
        config(
            InferenceServer::new_mock_hashed(),
            Some(PolicyTiming::OnRevisit),
            4,
        ),
    );
    assert_eq!(s4.iterations, s1.iterations);
    assert_eq!(r4.borrow().visits, r1.borrow().visits);
    assert_eq!(s4.nn_requests, nodes_with_priors(&r4));
}

#[test]
fn the_hashed_mock_policy_is_deterministic_and_not_uniform() {
    let board = Board::new();
    let moves = {
        let mg = MoveGen::new();
        let (c, q) = mg.gen_pseudo_legal_moves(&board);
        c.into_iter().chain(q).collect::<Vec<_>>()
    };
    let ask = |s: &InferenceServer| {
        s.predict_legal_async(board.clone(), moves.clone(), true, 0.0)
            .recv()
            .unwrap()
            .unwrap()
            .0
    };
    let (a, b) = (
        InferenceServer::new_mock_hashed(),
        InferenceServer::new_mock_hashed(),
    );
    let (pa, pb) = (ask(&a), ask(&b));
    assert_eq!(pa, pb);
    assert!(pa.iter().any(|&p| (p - pa[0]).abs() > 1e-3));
    assert!((pa.iter().sum::<f32>() - 1.0).abs() < 1e-4);
}

fn args(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

#[test]
fn the_policy_only_flag_parses() {
    assert_eq!(PolicyTiming::from_args(&args("self_play 8 200")), Ok(None));
    assert_eq!(
        PolicyTiming::from_args(&args("self_play --policy-only leaf")),
        Ok(Some(PolicyTiming::AtLeaf))
    );
    assert_eq!(
        PolicyTiming::from_args(&args("self_play --policy-only revisit --threads 4")),
        Ok(Some(PolicyTiming::OnRevisit))
    );
    assert!(PolicyTiming::from_args(&args("self_play --policy-only lazy")).is_err());
    assert!(PolicyTiming::from_args(&args("self_play --policy-only")).is_err());
}

/// White is a queen and two rooks ahead, with no short forced mate.
const BIG_LEAD: &str = "1nb1kbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1";

fn tier1_config(tier1: bool) -> TacticalMctsConfig {
    TacticalMctsConfig {
        enable_tier1_gate: tier1,
        ..config(InferenceServer::new_mock_uniform(), None, 1)
    }
}

#[test]
fn early_stops_are_counted_and_need_tier1() {
    // With Tier 1 on, the search for a move stops once a root move averages
    // above 0.99, which the classical score alone reaches with a large lead.
    let (on, root_on) = search(BIG_LEAD, tier1_config(true));
    assert_eq!(on.early_stops, 1, "{on:?}");
    assert!(root_on.borrow().visits < 300);
    let (off, root_off) = search(BIG_LEAD, tier1_config(false));
    assert_eq!(off.early_stops, 0);
    assert_eq!(root_off.borrow().visits, 300);
}

#[test]
fn no_early_stop_in_a_balanced_position() {
    let (s, root) = search(MIDDLEGAME, tier1_config(true));
    assert_eq!(s.early_stops, 0);
    assert_eq!(root.borrow().visits, 300);
}
