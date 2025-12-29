//! Tactical-First MCTS Implementation
//!
//! This module implements the main tactical-first MCTS search algorithm that combines:
//! 1. Mate search for exact forced sequences
//! 2. Tactical move prioritization (captures, checks, forks)
//! 3. Lazy neural network policy evaluation
//! 4. Strategic move exploration using UCB

use crate::board::Board;
use crate::boardstack::BoardStack;
use crate::eval::{PestoEval, extrapolate_value};
use crate::mcts::node::{MctsNode, NodeOrigin};
use crate::mcts::selection::select_child_with_tactical_priority;
use crate::mcts::inference_server::InferenceServer;
use crate::mcts::search_logger::{SearchLogger, GateReason};
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::neural_net::NeuralNetPolicy;
use crate::search::mate_search;
use crate::search::koth::koth_center_in_3;
use crate::search::quiescence::quiescence_search_tactical;
use crate::transposition::TranspositionTable;
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
    pub exploration_constant: f64,
    pub use_neural_policy: bool,
    pub inference_server: Option<Arc<InferenceServer>>,
    /// Search logger for stream of consciousness output
    pub logger: Option<Arc<SearchLogger>>,
}

impl Default for TacticalMctsConfig {
    fn default() -> Self {
        TacticalMctsConfig {
            max_iterations: 1000,
            time_limit: Duration::from_secs(5),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: true,
            inference_server: None,
            logger: None,
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
}

pub fn tactical_mcts_search(
    board: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
    config: TacticalMctsConfig,
) -> (Option<Move>, TacticalMctsStats, Rc<RefCell<MctsNode>>) {
    let mut transposition_table = TranspositionTable::new();
    tactical_mcts_search_with_tt(board, move_gen, pesto_eval, nn_policy, config, &mut transposition_table)
}

pub fn tactical_mcts_search_with_tt(
    board: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
    config: TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
) -> (Option<Move>, TacticalMctsStats, Rc<RefCell<MctsNode>>) {
    let start_time = Instant::now();
    let mut stats = TacticalMctsStats::default();
    
    // Get logger reference (or silent default)
    let logger = config.logger.as_ref();
    
    if koth_center_in_3(&board, move_gen) {
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
        for m in captures.iter().chain(moves.iter()) {
            let next = board.apply_move_to_board(*m);
            if next.is_legal(move_gen) && next.is_koth_win().0 == board.w_to_move {
                if let Some(log) = logger {
                    log.log_tier1_gate(&GateReason::KothWin, Some(*m));
                }
                stats.search_time = start_time.elapsed();
                let root_node = MctsNode::new_root(board, move_gen);
                root_node.borrow_mut().origin = NodeOrigin::Gate;
                return (Some(*m), stats, root_node);
            }
        }
    }

    let mate_move_result = if config.mate_search_depth > 0 {
        if let Some(log) = logger {
            log.log_mate_search_start(config.mate_search_depth);
        }
        
        let mut mate_search_stack = BoardStack::with_board(board.clone());
        let (mate_score, mate_move, nodes) = mate_search(&mut mate_search_stack, move_gen, config.mate_search_depth, false);
        
        if let Some(log) = logger {
            log.log_mate_search_result(mate_score, mate_move, nodes as u64);
        }
        
        if mate_score >= 1_000_000 {
            stats.mates_found += 1;
             if let Some(log) = logger {
                log.log_tier1_gate(
                    &GateReason::MateFound { 
                        depth: config.mate_search_depth, 
                        score: mate_score 
                    },
                    Some(mate_move)
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
    
    evaluate_and_expand_node(root_node.clone(), move_gen, pesto_eval, &mut stats, logger);
    
    for iteration in 0..config.max_iterations {
        if start_time.elapsed() > config.time_limit { break; }
        
        if let Some(log) = logger {
            log.log_iteration_start(iteration + 1);
        }
        
        let leaf_node = select_leaf_node(root_node.clone(), move_gen, nn_policy, &config, &mut stats, logger);
        let value = evaluate_leaf_node(leaf_node.clone(), move_gen, pesto_eval, nn_policy, &config, transposition_table, &mut stats, logger);
        
        if !leaf_node.borrow().is_game_terminal() && leaf_node.borrow().visits == 0 {
            evaluate_and_expand_node(leaf_node.clone(), move_gen, pesto_eval, &mut stats, logger);
        }
        
        backpropagate_value(leaf_node, value);
        stats.iterations = iteration + 1;
        
        if let Some(log) = logger {
            let best = select_best_move_from_root(root_node.clone(), &config);
            log.log_iteration_summary(iteration + 1, best, root_node.borrow().visits);
        }
        
        if iteration % 100 == 0 && start_time.elapsed() > config.time_limit { break; }
    }
    
    stats.search_time = start_time.elapsed();
    let best_move = select_best_move_from_root(root_node.clone(), &config);
    
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
    nn_policy: &mut Option<NeuralNetPolicy>,
    config: &TacticalMctsConfig,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) -> Rc<RefCell<MctsNode>> {
    let mut depth = 0;
    loop {
        if let Some(log) = logger {
            log.log_enter_node(&current.borrow(), depth);
        }
        
        let is_terminal = current.borrow().is_game_terminal();
        let has_children = !current.borrow().children.is_empty();
        
        if is_terminal || !has_children {
            return current;
        }
        
        if let Some(child) = select_child_with_tactical_priority(
            current.clone(),
            config.exploration_constant,
            move_gen,
            nn_policy,
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
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
    config: &TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) -> f64 {
    let mut node_ref = node.borrow_mut();
    let board = &node_ref.state;
    
    if let Some(cached_value) = node_ref.terminal_or_mate_value {
        if cached_value >= -1.0 { return cached_value; }
    }

    if koth_center_in_3(board, move_gen) {
        let win_val = 1.0; // StM wins
        if let Some(log) = logger {
            log.log_koth_check(true, Some(0)); // Distance 0 as place holder or calculate it
        }
        node_ref.terminal_or_mate_value = Some(win_val);
        return win_val;
    }

    if config.mate_search_depth > 0 {
        if let Some((mate_depth, _mate_move)) = transposition_table.probe_mate(board, config.mate_search_depth) {
            stats.tt_mate_hits += 1;
            if mate_depth != 0 {
                if let Some(log) = logger {
                    log.log_tier1_gate(
                        &GateReason::TtMateHit { depth: mate_depth },
                        None
                    );
                }
                let mate_value = if mate_depth > 0 { 1.0 } else { -1.0 };
                node_ref.terminal_or_mate_value = Some(mate_value);
                node_ref.origin = NodeOrigin::Gate;
                return mate_value;
            }
        } else {
            stats.tt_mate_misses += 1;
            let mut board_stack = BoardStack::with_board(board.clone());
            let mate_result = mate_search(&mut board_stack, move_gen, config.mate_search_depth, false);
            transposition_table.store_mate_result(board, mate_result.0.abs(), mate_result.1, config.mate_search_depth);
            
            if mate_result.0 != 0 {
                if let Some(log) = logger {
                    log.log_mate_search_result(mate_result.0, mate_result.1, 0);
                 }
                let mate_value = if mate_result.0 > 0 { 1.0 } else { -1.0 };
                node_ref.terminal_or_mate_value = Some(mate_value);
                return mate_value;
            }
        }
    }
    
    if node_ref.nn_value.is_none() {
        let mut k_val = 0.5;
        let mut raw_val = -999.0;

        // 1. Batched Inference
        if let Some(server) = &config.inference_server {
            if config.use_neural_policy {
                let receiver = server.predict_async(node_ref.state.clone());
                if let Ok(Some((_policy, value, k))) = receiver.recv() {
                    raw_val = value as f64;
                    k_val = k;
                }
            }
        } 
        // Direct Inference or Batched Response Handling
        if raw_val > -2.0 {
            // Assert NN values are in range [-1, 1] to catch issues with model heads immediately.
            if raw_val < -1.01 || raw_val > 1.01 {
                panic!("Neural Network produced value out of range [-1, 1]: {}. issues must be caught immediately.", raw_val);
            }
            if let Some(log) = logger {
                log.log_tier3_neural(raw_val, k_val);
            }
            node_ref.nn_value = Some(raw_val);
            if node_ref.origin == NodeOrigin::Unknown {
                node_ref.origin = NodeOrigin::Neural;
            }
        } else {
            // If the user explicitly requested neural policy, do not fail silently to classical eval.
            if config.use_neural_policy {
                panic!("Neural network policy enabled but no valid prediction obtained!");
            }

            // Classical path: Use Pesto evaluation mapped to tanh domain for search consistency.
            let eval_score = pesto_eval.eval(&node_ref.state, move_gen);
            let tanh_val = (eval_score as f64 / 400.0).tanh();
            if let Some(log) = logger {
                log.log_classical_eval(eval_score, tanh_val);
            }
            node_ref.nn_value = Some(tanh_val);
        }
        node_ref.k_val = k_val;
    }
    
    node_ref.nn_value.unwrap()
}

fn evaluate_and_expand_node(
    node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) {
    let mut node_ref = node.borrow_mut();
    
    if !node_ref.children.is_empty() || node_ref.is_game_terminal() {
        return;
    }

    if let Some(log) = logger {
        log.log_qs_start(&node_ref.state);
    }
    let mut board_stack = BoardStack::with_board(node_ref.state.clone());
    let tactical_tree = quiescence_search_tactical(&mut board_stack, move_gen, pesto_eval);

    if !tactical_tree.siblings.is_empty() {
        if let Some(log) = logger {
             log.log_qs_pv(&tactical_tree.principal_variation, tactical_tree.leaf_score);
        }
        let parent_eval = pesto_eval.eval(&node_ref.state, move_gen);
        let parent_v = node_ref.nn_value.unwrap_or(0.0); // Neutral is 0.0 in [-1, 1]
        let k = node_ref.k_val;

        for (mv, qs_score) in tactical_tree.siblings {
            let material_delta = qs_score - parent_eval;
            let extrapolated_v = extrapolate_value(parent_v, material_delta, k);
            
            if let Some(log) = logger {
                log.log_tier2_graft(mv, extrapolated_v, k);
            }
            
            node_ref.tactical_values.insert(mv, extrapolated_v);
            
            if tactical_tree.principal_variation.contains(&mv) {
                let next_board = node_ref.state.apply_move_to_board(mv);
                let child = MctsNode::new_child(Rc::downgrade(&node), mv, next_board, move_gen);
                // extrapolated_v is relative to Parent (Side to Move at Node).
                // child.nn_value must be relative to Child (Side to Move at Child).
                // These are opposite sides, so we negate.
                child.borrow_mut().nn_value = Some(-extrapolated_v);
                child.borrow_mut().k_val = k; 
                child.borrow_mut().is_tactical_node = true;
                child.borrow_mut().origin = NodeOrigin::Grafted;
                node_ref.children.push(child);
                stats.nodes_expanded += 1;
            }
        }
        node_ref.tactical_resolution_done = true;
    }

    let (captures, non_captures) = move_gen.gen_pseudo_legal_moves(&node_ref.state);
    for mv in captures.iter().chain(non_captures.iter()) {
        if node_ref.children.iter().any(|c| c.borrow().action == Some(*mv)) {
            continue;
        }

        let new_board = node_ref.state.apply_move_to_board(*mv);
        if new_board.is_legal(move_gen) {
            let child_node = MctsNode::new_child(Rc::downgrade(&node), *mv, new_board, move_gen);
            node_ref.children.push(child_node);
            stats.nodes_expanded += 1;
        }
    }
}

fn backpropagate_value(mut node: Rc<RefCell<MctsNode>>, mut value: f64) {
    loop {
        {
            let mut node_ref = node.borrow_mut();
            node_ref.visits += 1;
            
            // value is relative to node.state.side_to_move (StM).
            // node.total_value is relative to the player who reached this node (The player who just moved).
            // The player who just moved is the OPPONENT of StM.
            // So if StM is winning (+value), it's bad for the player who just moved (-value).
            let reward = -value;
            
            node_ref.total_value += reward;
            node_ref.total_value_squared += reward * reward;
        }

        // Move to parent. 
        // Value is currently relative to Node StM.
        // Parent StM is the Opponent of Node StM.
        // So we flip the value to be relative to Parent StM.
        let parent = {
            let node_ref = node.borrow();
            if let Some(parent_weak) = &node_ref.parent { parent_weak.upgrade() } else { None }
        };
        
        if let Some(parent_node) = parent { 
            node = parent_node; 
            value = -value;
        } else { 
            break; 
        }
    }
}

fn select_best_move_from_root(
    root: Rc<RefCell<MctsNode>>,
    _config: &TacticalMctsConfig,
) -> Option<Move> {
    let root_ref = root.borrow();
    if let Some(mate_move) = root_ref.mate_move { return Some(mate_move); }
    
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
}

pub fn tactical_mcts_search_for_training(
    board: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
    config: TacticalMctsConfig,
) -> MctsTrainingResult {
    let start_time = Instant::now();
    let mut stats = TacticalMctsStats::default();
    let root_node = MctsNode::new_root(board, move_gen);
    
    if root_node.borrow().is_game_terminal() {
        return MctsTrainingResult { best_move: None, root_policy: vec![], root_value_prediction: 0.0, stats };
    }
    
    evaluate_and_expand_node(root_node.clone(), move_gen, pesto_eval, &mut stats, None);
    let mut transposition_table = TranspositionTable::new();
    for iteration in 0..config.max_iterations {
        if start_time.elapsed() > config.time_limit { break; }
        let leaf = select_leaf_node(root_node.clone(), move_gen, nn_policy, &config, &mut stats, None);
        let value = evaluate_leaf_node(leaf.clone(), move_gen, pesto_eval, nn_policy, &config, &mut transposition_table, &mut stats, None);
        if !leaf.borrow().is_game_terminal() && leaf.borrow().visits == 0 {
            evaluate_and_expand_node(leaf.clone(), move_gen, pesto_eval, &mut stats, None);
        }
        backpropagate_value(leaf, value);
        stats.iterations = iteration + 1;
    }
    
    let root = root_node.borrow();
    let mut policy = Vec::new();
    for child in &root.children {
        if let Some(mv) = child.borrow().action {
            policy.push((mv, child.borrow().visits));
        }
    }
    
    let best_move = select_best_move_from_root(root_node.clone(), &config);
    // root.total_value stores rewards relative to side that just moved (parent's parent).
    // For root, this is the opponent of the side to move.
    // So we negate it to get the side to move's perspective.
    let root_val = if root.visits > 0 { -(root.total_value / root.visits as f64) } else { 0.0 };
    
    MctsTrainingResult { best_move, root_policy: policy, root_value_prediction: root_val, stats }
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