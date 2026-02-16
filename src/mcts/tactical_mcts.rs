//! Tactical-First MCTS Implementation
//!
//! This module implements the main tactical-first MCTS search algorithm that combines:
//! 1. Mate search for exact forced sequences
//! 2. Tactical move prioritization (captures, checks, forks)
//! 3. Lazy neural network policy evaluation
//! 4. Strategic move exploration using UCB

use crate::board::Board;
use crate::boardstack::BoardStack;
use crate::mcts::inference_server::InferenceServer;
use crate::mcts::node::{MctsNode, NodeOrigin};
use crate::mcts::search_logger::{GateReason, SearchLogger};
use crate::mcts::selection::select_child_with_tactical_priority;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::search::forced_material_balance;
use crate::search::mate_search;
use crate::search::{koth_best_move, koth_center_in_3};
use crate::transposition::TranspositionTable;
use rand_distr::{Distribution, Gamma};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for tactical-first MCTS search
#[derive(Debug, Clone)]
pub struct TacticalMctsConfig {
    pub max_iterations: u32,
    pub time_limit: Duration,
    pub mate_search_depth: i32,
    /// Max ply depth for exhaustive (all-moves) mate search. Depths above this use checks-only.
    /// Default 3 = exhaustive mate-in-1 and mate-in-2, checks-only mate-in-3.
    pub exhaustive_mate_depth: i32,
    pub exploration_constant: f64,
    pub use_neural_policy: bool,
    pub inference_server: Option<Arc<InferenceServer>>,
    /// Search logger for stream of consciousness output
    pub logger: Option<Arc<SearchLogger>>,

    // Ablation flags for paper experiments
    /// Enable Tier 1 (Safety Gates: Mate Search + KOTH)
    pub enable_tier1_gate: bool,
    /// Enable Tier 3 (Neural Network Policy)
    pub enable_tier3_neural: bool,
    /// Enable KOTH (King of the Hill) win detection — off for standard chess
    pub enable_koth: bool,
    /// Enable material integration in value: V = tanh(V_logit + k·ΔM)
    /// When false (pure AlphaZero): V = tanh(V_logit), no Q-search material
    pub enable_material_value: bool,
    /// Dirichlet noise alpha (0.0 = disabled, 0.3 for chess training)
    pub dirichlet_alpha: f64,
    /// Dirichlet noise epsilon (0.0 = disabled, 0.25 for chess training)
    pub dirichlet_epsilon: f64,
    /// Shuffle children after expansion to break move-generation-order bias
    /// Enable for training (self-play), disable for deterministic analysis/evaluation
    pub randomize_move_order: bool,
}

impl Default for TacticalMctsConfig {
    fn default() -> Self {
        TacticalMctsConfig {
            max_iterations: 1000,
            time_limit: Duration::from_secs(5),
            mate_search_depth: 5,
            exhaustive_mate_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: true,
            inference_server: None,
            logger: None,
            // All tiers enabled by default
            enable_tier1_gate: true,
            enable_tier3_neural: true,
            enable_koth: false,
            enable_material_value: true,
            dirichlet_alpha: 0.0,
            dirichlet_epsilon: 0.0,
            randomize_move_order: false,
        }
    }
}

/// Statistics collected during tactical-first MCTS search
#[derive(Debug, Default)]
pub struct TacticalMctsStats {
    pub iterations: u32,
    pub mates_found: u32,
    pub tactical_moves_explored: u32,
    pub nn_policy_evaluations: u32,
    pub search_time: Duration,
    pub nodes_expanded: u32,
    pub tt_mate_hits: u32,
    pub tt_mate_misses: u32,

    // Paper-ready metrics
    /// Total NN evaluations (expensive operations)
    pub nn_evaluations: u32,
    /// NN evaluations saved by Tier 1 gates
    pub nn_saved_by_tier1: u32,
    /// Positions where Tier 1 found forced win/loss
    pub tier1_solutions: u32,
}

impl TacticalMctsStats {
    /// Calculate NN call reduction percentage
    pub fn nn_reduction_percentage(&self) -> f64 {
        let total_potential = self.nn_evaluations + self.nn_saved_by_tier1;
        if total_potential == 0 {
            return 0.0;
        }
        100.0 * self.nn_saved_by_tier1 as f64 / total_potential as f64
    }

    /// Generate LaTeX metrics table
    pub fn to_latex(&self) -> String {
        let mut s = String::new();
        s.push_str(r"\begin{tabular}{lr}");
        s.push_str("\n");
        s.push_str(r"\toprule");
        s.push_str("\n");
        s.push_str(r"Metric & Value \\");
        s.push_str("\n");
        s.push_str(r"\midrule");
        s.push_str("\n");
        s.push_str(&format!(r"Iterations & {} \\", self.iterations));
        s.push_str("\n");
        s.push_str(&format!(r"Nodes Expanded & {} \\", self.nodes_expanded));
        s.push_str("\n");
        s.push_str(&format!(r"NN Evaluations & {} \\", self.nn_evaluations));
        s.push_str("\n");
        s.push_str(&format!(
            r"NN Saved (Tier 1) & {} \\",
            self.nn_saved_by_tier1
        ));
        s.push_str("\n");
        s.push_str(&format!(
            r"Reduction & {:.1}\% \\",
            self.nn_reduction_percentage()
        ));
        s.push_str("\n");
        s.push_str(r"\bottomrule");
        s.push_str("\n");
        s.push_str(r"\end{tabular}");
        s.push_str("\n");
        s
    }
}

pub fn tactical_mcts_search(
    board: Board,
    move_gen: &MoveGen,
    config: TacticalMctsConfig,
) -> (Option<Move>, TacticalMctsStats, Rc<RefCell<MctsNode>>) {
    let mut transposition_table = TranspositionTable::new();
    tactical_mcts_search_with_tt(board, move_gen, config, &mut transposition_table)
}

pub fn tactical_mcts_search_with_tt(
    board: Board,
    move_gen: &MoveGen,
    config: TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
) -> (Option<Move>, TacticalMctsStats, Rc<RefCell<MctsNode>>) {
    let start_time = Instant::now();
    let mut stats = TacticalMctsStats::default();

    // Get logger reference (or silent default)
    let logger = config.logger.as_ref();

    if config.enable_koth && config.enable_tier1_gate {
        if let Some(dist) = koth_center_in_3(&board, move_gen) {
            if let Some(best) = koth_best_move(&board, move_gen) {
                if let Some(log) = logger {
                    log.log_tier1_gate(&GateReason::KothWin { distance: dist }, Some(best));
                }
                stats.tier1_solutions += 1;
                stats.search_time = start_time.elapsed();
                let root_node = MctsNode::new_root(board, move_gen);
                root_node.borrow_mut().origin = NodeOrigin::Gate;
                return (Some(best), stats, root_node);
            }
        }
    }

    let mate_move_result = if config.enable_tier1_gate && config.mate_search_depth > 0 {
        if let Some((mate_depth, _mate_move)) =
            transposition_table.probe_mate(&board, config.mate_search_depth)
        {
            stats.tt_mate_hits += 1;
            if mate_depth != 0 && _mate_move != Move::null() {
                // Verify move is actually pseudo-legal in this position to guard against collisions
                let is_pseudo_legal = board.is_pseudo_legal(_mate_move, move_gen);

                if is_pseudo_legal {
                    let next = board.apply_move_to_board(_mate_move);
                    if next.is_legal(move_gen) {
                        if let Some(log) = logger {
                            log.log_tier1_gate(
                                &GateReason::TtMateHit { depth: mate_depth },
                                Some(_mate_move),
                            );
                        }
                        stats.tier1_solutions += 1;
                        let root_node = MctsNode::new_root(board, move_gen);
                        root_node.borrow_mut().origin = NodeOrigin::Gate;
                        root_node.borrow_mut().mate_move = Some(_mate_move);
                        stats.search_time = start_time.elapsed();
                        return (Some(_mate_move), stats, root_node);
                    }
                }
            }
        }

        if let Some(log) = logger {
            log.log_mate_search_start(config.mate_search_depth);
        }

        let mut mate_search_stack = BoardStack::with_board(board.clone());
        let (mate_score, mate_move, nodes) = mate_search(
            &mut mate_search_stack,
            move_gen,
            config.mate_search_depth,
            false,
            config.exhaustive_mate_depth,
        );

        if let Some(log) = logger {
            log.log_mate_search_result(mate_score, mate_move, nodes as u64);
        }

        if mate_score >= 1_000_000 {
            stats.mates_found += 1;
            stats.tier1_solutions += 1;
            if let Some(log) = logger {
                log.log_tier1_gate(
                    &GateReason::MateFound {
                        depth: config.mate_search_depth,
                        score: mate_score,
                    },
                    Some(mate_move),
                );
            }
            Some(mate_move)
        } else if mate_score <= -1_000_000 {
            Some(mate_move)
        } else {
            None
        }
    } else {
        None
    };

    if let Some(immediate_mate_move) = mate_move_result {
        stats.search_time = start_time.elapsed();
        let root_node = MctsNode::new_root(board, move_gen);
        root_node.borrow_mut().origin = NodeOrigin::Gate;
        root_node.borrow_mut().mate_move = Some(immediate_mate_move);
        return (Some(immediate_mate_move), stats, root_node);
    }

    let root_node = MctsNode::new_root(board, move_gen);

    if root_node.borrow().is_game_terminal() {
        return (None, stats, root_node);
    }

    evaluate_and_expand_node(root_node.clone(), move_gen, &mut stats, &config, logger);

    for iteration in 0..config.max_iterations {
        if start_time.elapsed() > config.time_limit {
            break;
        }

        if let Some(log) = logger {
            log.log_iteration_start(iteration + 1);
        }

        let leaf_node = select_leaf_node(root_node.clone(), move_gen, &config, &mut stats, logger);
        let value = evaluate_leaf_node(
            leaf_node.clone(),
            move_gen,
            &config,
            transposition_table,
            &mut stats,
            logger,
        );

        if !leaf_node.borrow().is_game_terminal()
            && leaf_node.borrow().visits == 0
            && leaf_node.borrow().terminal_or_mate_value.is_none()
        {
            evaluate_and_expand_node(leaf_node.clone(), move_gen, &mut stats, &config, logger);
        }

        MctsNode::backpropagate(leaf_node, value);
        stats.iterations = iteration + 1;

        if let Some(log) = logger {
            let best = select_best_move_from_root(root_node.clone(), move_gen);
            log.log_iteration_summary(iteration + 1, best, root_node.borrow().visits);
        }

        if iteration % 100 == 0 && start_time.elapsed() > config.time_limit {
            break;
        }

        // Early termination on forced KOTH win
        if config.enable_koth {
            let root_ref = root_node.borrow();
            let has_win_in_1 = root_ref.children.iter().any(|c| {
                c.borrow()
                    .terminal_or_mate_value
                    .map_or(false, |v| v < -0.5)
            });
            if has_win_in_1 {
                drop(root_ref);
                break;
            }
            let has_forced_win = root_ref.children.iter().any(|c| {
                let cr = c.borrow();
                cr.visits > 0 && (cr.total_value / cr.visits as f64) > 0.99
            });
            if has_forced_win {
                let all_visited = root_ref.children.iter().all(|c| c.borrow().visits > 0);
                if all_visited {
                    drop(root_ref);
                    break;
                }
            }
        }
    }

    stats.search_time = start_time.elapsed();
    let best_move = select_best_move_from_root(root_node.clone(), move_gen);

    if let Some(log) = logger {
        log.log_search_complete(
            best_move,
            stats.iterations,
            stats.nodes_expanded,
            stats.mates_found,
        );
    }

    (best_move, stats, root_node)
}

fn select_leaf_node(
    mut current: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    config: &TacticalMctsConfig,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) -> Rc<RefCell<MctsNode>> {
    let mut depth = 0;
    loop {
        if let Some(log) = logger {
            log.log_enter_node(&current.borrow(), depth);
        }

        let is_terminal = current.borrow().is_game_terminal()
            || current.borrow().terminal_or_mate_value.is_some();
        let has_children = !current.borrow().children.is_empty();

        if is_terminal || !has_children {
            return current;
        }

        if let Some(child) = select_child_with_tactical_priority(
            current.clone(),
            config,
            move_gen,
            stats,
            logger,
            depth,
        ) {
            if !current.borrow().policy_evaluated && config.use_neural_policy {
                stats.nn_policy_evaluations += 1;
            }
            current = child;
            depth += 1;
        } else {
            return current;
        }
    }
}

fn evaluate_leaf_node(
    node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    config: &TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) -> f64 {
    let mut node_ref = node.borrow_mut();
    let board = &node_ref.state;

    if let Some(cached_value) = node_ref.terminal_or_mate_value {
        if cached_value >= -1.0 {
            stats.nn_saved_by_tier1 += 1;
            return cached_value;
        }
    }

    if config.enable_koth && config.enable_tier1_gate {
        // Opponent already on center = STM loses (KOTH terminal)
        let (w_won, b_won) = board.is_koth_win();
        let opponent_won = (board.w_to_move && b_won) || (!board.w_to_move && w_won);
        if opponent_won {
            node_ref.terminal_or_mate_value = Some(-1.0);
            node_ref.is_terminal = true;
            stats.nn_saved_by_tier1 += 1;
            stats.tier1_solutions += 1;
            return -1.0;
        }

        if let Some(dist) = koth_center_in_3(board, move_gen) {
            let win_val = 1.0; // StM wins
            if let Some(log) = logger {
                log.log_koth_check(true, Some(dist as u32));
            }
            node_ref.terminal_or_mate_value = Some(win_val);
            stats.nn_saved_by_tier1 += 1;
            stats.tier1_solutions += 1;
            return win_val;
        }
    }

    if config.enable_tier1_gate && config.mate_search_depth > 0 {
        if let Some((mate_depth, _mate_move)) =
            transposition_table.probe_mate(board, config.mate_search_depth)
        {
            stats.tt_mate_hits += 1;
            if mate_depth != 0 {
                // Validate move legality if present to guard against collisions
                let is_valid = if _mate_move != Move::null() {
                    let (captures, quiet) = move_gen.gen_pseudo_legal_moves(board);
                    let is_pseudo_legal =
                        captures.contains(&_mate_move) || quiet.contains(&_mate_move);
                    is_pseudo_legal && board.is_legal_after_move(_mate_move, move_gen)
                } else {
                    true
                };

                if is_valid {
                    if let Some(log) = logger {
                        log.log_tier1_gate(&GateReason::TtMateHit { depth: mate_depth }, None);
                    }
                    let mate_value = if mate_depth > 0 { 1.0 } else { -1.0 };
                    node_ref.terminal_or_mate_value = Some(mate_value);
                    node_ref.origin = NodeOrigin::Gate;
                    stats.nn_saved_by_tier1 += 1;
                    return mate_value;
                }
            }
        } else {
            stats.tt_mate_misses += 1;
            let mut board_stack = BoardStack::with_board(board.clone());
            let mate_result = mate_search(
                &mut board_stack,
                move_gen,
                config.mate_search_depth,
                false,
                config.exhaustive_mate_depth,
            );
            transposition_table.store_mate_result(
                board,
                mate_result.0.abs(),
                mate_result.1,
                config.mate_search_depth,
            );

            if mate_result.0 != 0 {
                if let Some(log) = logger {
                    log.log_mate_search_result(mate_result.0, mate_result.1, 0);
                }
                let mate_value = if mate_result.0 > 0 { 1.0 } else { -1.0 };
                node_ref.terminal_or_mate_value = Some(mate_value);
                stats.nn_saved_by_tier1 += 1;
                stats.tier1_solutions += 1;
                return mate_value;
            }
        }
    }

    if node_ref.nn_value.is_none() {
        let mut k_val: f32 = 0.5;
        let mut v_logit: f64 = f64::NEG_INFINITY; // sentinel: no NN result yet

        // 1. Run Q-search FIRST so completion flag is available for NN inference
        let (delta_m, qsearch_completed) = if config.enable_material_value {
            let mut board_stack = BoardStack::with_board(node_ref.state.clone());
            forced_material_balance(&mut board_stack, move_gen)
        } else {
            (0, true)
        };

        // 2. Batched Inference — NN now returns raw v_logit (unbounded)
        if config.enable_tier3_neural {
            if let Some(server) = &config.inference_server {
                if config.use_neural_policy {
                    let receiver = server.predict_async(node_ref.state.clone(), qsearch_completed);
                    if let Ok(Some((policy, nn_v_logit, k))) = receiver.recv() {
                        v_logit = nn_v_logit as f64;
                        k_val = k;
                        node_ref.raw_nn_policy = Some(policy);
                    }
                }
            }
        }

        if config.enable_material_value {
            if v_logit.is_finite() {
                // NN path: combine v_logit + k * delta_M
                let final_value = (v_logit + k_val as f64 * delta_m as f64).tanh();
                if let Some(log) = logger {
                    log.log_tier3_neural(final_value, k_val);
                }
                node_ref.v_logit = Some(v_logit);
                node_ref.nn_value = Some(final_value);
                if node_ref.origin == NodeOrigin::Unknown {
                    node_ref.origin = NodeOrigin::Neural;
                }
                stats.nn_evaluations += 1;
            } else {
                // Classical fallback: v_logit = 0.0, k = 0.5, rely on material only
                // k=0.5 matches NN init: softplus(0)/(2*ln2) ≈ 0.5
                let classical_v_logit = 0.0;
                let classical_k: f32 = 0.5;
                let final_value = (classical_v_logit + classical_k as f64 * delta_m as f64).tanh();
                if let Some(log) = logger {
                    log.log_classical_eval(delta_m, final_value);
                }
                node_ref.v_logit = Some(classical_v_logit);
                node_ref.nn_value = Some(final_value);
                k_val = classical_k;
            }
        } else {
            // Pure AlphaZero: no material integration
            if v_logit.is_finite() {
                let final_value = v_logit.tanh();
                if let Some(log) = logger {
                    log.log_tier3_neural(final_value, k_val);
                }
                node_ref.v_logit = Some(v_logit);
                node_ref.nn_value = Some(final_value);
                if node_ref.origin == NodeOrigin::Unknown {
                    node_ref.origin = NodeOrigin::Neural;
                }
                stats.nn_evaluations += 1;
            } else {
                // Classical fallback = 0.0 (no material, no NN = uniform/random)
                node_ref.v_logit = Some(0.0);
                node_ref.nn_value = Some(0.0);
                k_val = 0.0;
            }
        }
        node_ref.k_val = k_val;
    }

    node_ref.nn_value.unwrap()
}

fn evaluate_and_expand_node(
    node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    stats: &mut TacticalMctsStats,
    _config: &TacticalMctsConfig,
    _logger: Option<&Arc<SearchLogger>>,
) {
    let mut node_ref = node.borrow_mut();

    if !node_ref.children.is_empty() || node_ref.is_game_terminal() {
        return;
    }

    let (captures, non_captures) = move_gen.gen_pseudo_legal_moves(&node_ref.state);

    for mv in captures.iter().chain(non_captures.iter()) {
        let new_board = node_ref.state.apply_move_to_board(*mv);
        if new_board.is_legal(move_gen) {
            let child_node = MctsNode::new_child(Rc::downgrade(&node), *mv, new_board, move_gen);
            node_ref.children.push(child_node);
            stats.nodes_expanded += 1;
        }
    }
}

fn select_best_move_from_root(root: Rc<RefCell<MctsNode>>, move_gen: &MoveGen) -> Option<Move> {
    let root_ref = root.borrow();

    // Prefer win-in-1: terminal child where child's STM loses (opponent just won)
    for child in &root_ref.children {
        let cr = child.borrow();
        if cr.terminal_or_mate_value.map_or(false, |v| v < -0.5) {
            return cr.action;
        }
    }

    if let Some(mate_move) = root_ref.mate_move {
        // Validation check
        if root_ref.state.is_pseudo_legal(mate_move, move_gen)
            && root_ref.state.is_legal_after_move(mate_move, move_gen)
        {
            return Some(mate_move);
        }
    }

    let mut best_move = None;
    let mut best_visits = 0;

    for child in &root_ref.children {
        let child_ref = child.borrow();
        if child_ref.visits > best_visits {
            best_visits = child_ref.visits;
            best_move = child_ref.action;
        }
    }
    best_move
}

pub struct MctsTrainingResult {
    pub best_move: Option<Move>,
    pub root_policy: Vec<(Move, u32)>,
    pub root_value_prediction: f64,
    pub stats: TacticalMctsStats,
    /// The root node of the search tree, for tree reuse on the next move.
    pub root_node: Rc<RefCell<MctsNode>>,
}

pub fn tactical_mcts_search_for_training(
    board: Board,
    move_gen: &MoveGen,
    config: TacticalMctsConfig,
) -> MctsTrainingResult {
    let start_time = Instant::now();
    let mut stats = TacticalMctsStats::default();
    let root_node = MctsNode::new_root(board, move_gen);

    if root_node.borrow().is_game_terminal() {
        return MctsTrainingResult {
            best_move: None,
            root_policy: vec![],
            root_value_prediction: 0.0,
            stats,
            root_node,
        };
    }

    let logger = config.logger.as_ref();
    evaluate_and_expand_node(root_node.clone(), move_gen, &mut stats, &config, logger);
    let mut transposition_table = TranspositionTable::new();
    for iteration in 0..config.max_iterations {
        if start_time.elapsed() > config.time_limit {
            break;
        }

        if let Some(log) = logger {
            log.log_iteration_start(iteration + 1);
        }

        let leaf = select_leaf_node(root_node.clone(), move_gen, &config, &mut stats, logger);
        let value = evaluate_leaf_node(
            leaf.clone(),
            move_gen,
            &config,
            &mut transposition_table,
            &mut stats,
            logger,
        );
        if !leaf.borrow().is_game_terminal()
            && leaf.borrow().visits == 0
            && leaf.borrow().terminal_or_mate_value.is_none()
        {
            evaluate_and_expand_node(leaf.clone(), move_gen, &mut stats, &config, logger);
        }
        MctsNode::backpropagate(leaf, value);
        stats.iterations = iteration + 1;

        if let Some(log) = logger {
            let best = select_best_move_from_root(root_node.clone(), move_gen);
            log.log_iteration_summary(iteration + 1, best, root_node.borrow().visits);
        }

        // Early termination on forced KOTH win
        if config.enable_koth {
            let root_ref = root_node.borrow();
            let has_win_in_1 = root_ref.children.iter().any(|c| {
                c.borrow()
                    .terminal_or_mate_value
                    .map_or(false, |v| v < -0.5)
            });
            if has_win_in_1 {
                drop(root_ref);
                break;
            }
            let has_forced_win = root_ref.children.iter().any(|c| {
                let cr = c.borrow();
                cr.visits > 0 && (cr.total_value / cr.visits as f64) > 0.99
            });
            if has_forced_win {
                let all_visited = root_ref.children.iter().all(|c| c.borrow().visits > 0);
                if all_visited {
                    drop(root_ref);
                    break;
                }
            }
        }
    }

    let (policy, root_val) = {
        let root = root_node.borrow();
        let mut policy = Vec::new();
        for child in &root.children {
            if let Some(mv) = child.borrow().action {
                policy.push((mv, child.borrow().visits));
            }
        }
        let root_val = if root.visits > 0 {
            -(root.total_value / root.visits as f64)
        } else {
            0.0
        };
        (policy, root_val)
    };

    let best_move = select_best_move_from_root(root_node.clone(), move_gen);

    MctsTrainingResult {
        best_move,
        root_policy: policy,
        root_value_prediction: root_val,
        stats,
        root_node,
    }
}

/// Search for training with optional tree reuse and shared transposition table.
pub fn tactical_mcts_search_for_training_with_reuse(
    board: Board,
    move_gen: &MoveGen,
    config: TacticalMctsConfig,
    reused_root: Option<Rc<RefCell<MctsNode>>>,
    transposition_table: &mut TranspositionTable,
) -> MctsTrainingResult {
    let start_time = Instant::now();
    let mut stats = TacticalMctsStats::default();

    // Use reused root if available, otherwise create new
    let root_node = if let Some(reused) = reused_root {
        // Clear stale heuristic values from previous search.
        // terminal_or_mate_value may have been set by mate_search or koth_center_in_3
        // on a leaf deep in a previous tree. If this node is not truly terminal
        // (checkmate/stalemate), the cached value would cause select_leaf_node to
        // always return the root as a leaf, preventing any child from being visited.
        {
            let mut node = reused.borrow_mut();
            if !node.is_terminal {
                node.terminal_or_mate_value = None;
                node.mate_move = None;
            }
        }
        reused
    } else {
        MctsNode::new_root(board, move_gen)
    };

    if root_node.borrow().is_game_terminal() {
        return MctsTrainingResult {
            best_move: None,
            root_policy: vec![],
            root_value_prediction: 0.0,
            stats,
            root_node,
        };
    }

    let logger = config.logger.as_ref();
    if root_node.borrow().children.is_empty() {
        evaluate_and_expand_node(root_node.clone(), move_gen, &mut stats, &config, logger);
    }

    for iteration in 0..config.max_iterations {
        if start_time.elapsed() > config.time_limit {
            break;
        }

        if let Some(log) = logger {
            log.log_iteration_start(iteration + 1);
        }

        let leaf = select_leaf_node(root_node.clone(), move_gen, &config, &mut stats, logger);
        let value = evaluate_leaf_node(
            leaf.clone(),
            move_gen,
            &config,
            transposition_table,
            &mut stats,
            logger,
        );
        if !leaf.borrow().is_game_terminal()
            && leaf.borrow().visits == 0
            && leaf.borrow().terminal_or_mate_value.is_none()
        {
            evaluate_and_expand_node(leaf.clone(), move_gen, &mut stats, &config, logger);
        }
        MctsNode::backpropagate(leaf, value);
        stats.iterations = iteration + 1;

        if let Some(log) = logger {
            let best = select_best_move_from_root(root_node.clone(), move_gen);
            log.log_iteration_summary(iteration + 1, best, root_node.borrow().visits);
        }

        // Early termination on forced KOTH win
        if config.enable_koth {
            let root_ref = root_node.borrow();
            let has_win_in_1 = root_ref.children.iter().any(|c| {
                c.borrow()
                    .terminal_or_mate_value
                    .map_or(false, |v| v < -0.5)
            });
            if has_win_in_1 {
                drop(root_ref);
                break;
            }
            let has_forced_win = root_ref.children.iter().any(|c| {
                let cr = c.borrow();
                cr.visits > 0 && (cr.total_value / cr.visits as f64) > 0.99
            });
            if has_forced_win {
                let all_visited = root_ref.children.iter().all(|c| c.borrow().visits > 0);
                if all_visited {
                    drop(root_ref);
                    break;
                }
            }
        }
    }

    let (policy, root_val) = {
        let root = root_node.borrow();
        let mut policy = Vec::new();
        for child in &root.children {
            if let Some(mv) = child.borrow().action {
                policy.push((mv, child.borrow().visits));
            }
        }
        let root_val = if root.visits > 0 {
            -(root.total_value / root.visits as f64)
        } else {
            0.0
        };
        (policy, root_val)
    };

    let best_move = select_best_move_from_root(root_node.clone(), move_gen);

    MctsTrainingResult {
        best_move,
        root_policy: policy,
        root_value_prediction: root_val,
        stats,
        root_node,
    }
}

/// Reuse a subtree from a previous MCTS search by finding the child
/// corresponding to the played move and detaching it as a new root.
///
/// Returns None if the move is not found among the root's children.
pub fn reuse_subtree(
    root: Rc<RefCell<MctsNode>>,
    played_move: Move,
) -> Option<Rc<RefCell<MctsNode>>> {
    let root_ref = root.borrow();
    let child = root_ref
        .children
        .iter()
        .find(|c| c.borrow().action == Some(played_move))?
        .clone();
    drop(root_ref);

    // Detach child from parent to make it the new root
    child.borrow_mut().parent = None;

    Some(child)
}

/// Apply Dirichlet noise to root node's move priors (AlphaZero-style exploration).
/// Mixes: prior = (1 - epsilon) * prior + epsilon * noise
/// where noise ~ Dir(alpha, alpha, ..., alpha).
pub fn apply_dirichlet_noise(root: &Rc<RefCell<MctsNode>>, alpha: f64, epsilon: f64) {
    let mut node = root.borrow_mut();
    let moves: Vec<Move> = node.move_priorities.keys().cloned().collect();
    if moves.is_empty() {
        return;
    }

    // Generate Dirichlet noise via Gamma(alpha, 1) draws + normalization
    let mut rng = rand::thread_rng();
    let gamma = Gamma::new(alpha, 1.0).unwrap();
    let raw: Vec<f64> = (0..moves.len()).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = raw.iter().sum();
    if sum == 0.0 {
        return;
    }
    let noise: Vec<f64> = raw.iter().map(|x| x / sum).collect();

    // Mix noise into priors
    for (i, mv) in moves.iter().enumerate() {
        let prior = node.move_priorities.get(mv).copied().unwrap_or(0.0);
        let noisy = (1.0 - epsilon) * prior + epsilon * noise[i];
        node.move_priorities.insert(*mv, noisy);
    }
}

pub fn print_search_stats(stats: &TacticalMctsStats, best_move: Option<Move>) {
    println!("🎯 Tactical-First MCTS Search Complete");
    println!("   Iterations: {}", stats.iterations);
    println!("   Time: {}ms", stats.search_time.as_millis());
    println!("   Nodes expanded: {}", stats.nodes_expanded);
    println!("   Mates found: {}", stats.mates_found);
    if let Some(mv) = best_move {
        println!("   Best move: {}", mv.to_uci());
    }
}
