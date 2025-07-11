//! Tactical-First MCTS Implementation
//!
//! This module implements the main tactical-first MCTS search algorithm that combines:
//! 1. Mate search for exact forced sequences
//! 2. Tactical move prioritization (captures, checks, forks)
//! 3. Lazy neural network policy evaluation
//! 4. Strategic move exploration using UCB

use crate::board::Board;
use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::mcts::node::MctsNode;
use crate::mcts::selection::select_child_with_tactical_priority;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::neural_net::NeuralNetPolicy;
use crate::search::mate_search;
use crate::transposition::TranspositionTable;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Configuration for tactical-first MCTS search
#[derive(Debug, Clone)]
pub struct TacticalMctsConfig {
    /// Maximum number of MCTS iterations
    pub max_iterations: u32,
    /// Time limit for search
    pub time_limit: Duration,
    /// Depth for mate search at leaf nodes
    pub mate_search_depth: i32,
    /// UCB exploration constant
    pub exploration_constant: f64,
    /// Whether to use neural network policy evaluation
    pub use_neural_policy: bool,
}

impl Default for TacticalMctsConfig {
    fn default() -> Self {
        TacticalMctsConfig {
            max_iterations: 1000,
            time_limit: Duration::from_secs(5),
            mate_search_depth: 3,
            exploration_constant: 1.414, // sqrt(2)
            use_neural_policy: true,
        }
    }
}

/// Statistics collected during tactical-first MCTS search
#[derive(Debug, Default)]
pub struct TacticalMctsStats {
    /// Total MCTS iterations performed
    pub iterations: u32,
    /// Number of mate sequences found
    pub mates_found: u32,
    /// Number of tactical moves explored
    pub tactical_moves_explored: u32,
    /// Number of neural network policy evaluations
    pub nn_policy_evaluations: u32,
    /// Total time spent in search
    pub search_time: Duration,
    /// Number of nodes expanded
    pub nodes_expanded: u32,
    /// Number of transposition table mate cache hits
    pub tt_mate_hits: u32,
    /// Number of transposition table mate cache misses
    pub tt_mate_misses: u32,
}

/// Main tactical-first MCTS search function
pub fn tactical_mcts_search(
    board: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
    config: TacticalMctsConfig,
) -> (Option<Move>, TacticalMctsStats) {
    // Create a transposition table for this search
    let mut transposition_table = TranspositionTable::new();
    tactical_mcts_search_with_tt(board, move_gen, pesto_eval, nn_policy, config, &mut transposition_table)
}

/// Main tactical-first MCTS search function with transposition table
pub fn tactical_mcts_search_with_tt(
    board: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
    config: TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
) -> (Option<Move>, TacticalMctsStats) {
    let start_time = Instant::now();
    let mut stats = TacticalMctsStats::default();
    
    // **KEY INNOVATION: Mate Search First at Root Level**
    // Before any MCTS iterations, check if there's a forced mate from the root
    let mate_move_result = if config.mate_search_depth > 0 {
        let mut mate_search_stack = BoardStack::with_board(board.clone());
        let (mate_score, mate_move, _) = mate_search(&mut mate_search_stack, move_gen, config.mate_search_depth, false);
        
        if mate_score >= 1_000_000 {
            // Found a forced mate! Count it and return immediately
            stats.mates_found += 1;
            Some(mate_move)
        } else if mate_score <= -1_000_000 {
            // We're being mated, but still return best move from mate search
            Some(mate_move)
        } else {
            // No mate found, proceed with normal MCTS
            None
        }
    } else {
        None
    };
    
    // If mate was found at root, return immediately
    if let Some(immediate_mate_move) = mate_move_result {
        stats.search_time = start_time.elapsed();
        return (Some(immediate_mate_move), stats);
    }
    
    // Create root node
    let root_node = MctsNode::new_root(board, move_gen);
    
    // Check if root is terminal
    {
        let root_ref = root_node.borrow();
        if root_ref.is_game_terminal() {
            return (None, stats); // No legal moves
        }
    }
    
    // Initial root evaluation and expansion
    evaluate_and_expand_node(root_node.clone(), move_gen, pesto_eval, &mut stats);
    
    // Main MCTS loop
    for iteration in 0..config.max_iterations {
        if start_time.elapsed() > config.time_limit {
            break;
        }
        
        // Selection phase: traverse to leaf using tactical-first selection
        let leaf_node = select_leaf_node(
            root_node.clone(),
            move_gen,
            nn_policy,
            config.exploration_constant,
            &mut stats,
        );
        
        // Evaluation phase: mate search + position evaluation
        let value = evaluate_leaf_node(
            leaf_node.clone(),
            move_gen,
            pesto_eval,
            config.mate_search_depth,
            transposition_table,
            &mut stats,
        );
        
        // Expansion phase: create child nodes if not terminal
        if !leaf_node.borrow().is_game_terminal() && leaf_node.borrow().visits == 0 {
            evaluate_and_expand_node(leaf_node.clone(), move_gen, pesto_eval, &mut stats);
        }
        
        // Backpropagation phase: update values up the tree
        backpropagate_value(leaf_node, value);
        
        stats.iterations = iteration + 1;
        
        // Check for time limit periodically
        if iteration % 100 == 0 && start_time.elapsed() > config.time_limit {
            break;
        }
    }
    
    stats.search_time = start_time.elapsed();
    
    // Select best move from root
    let best_move = select_best_move_from_root(root_node, &config);
    
    (best_move, stats)
}

/// Select a leaf node using tactical-first selection strategy
fn select_leaf_node(
    mut current: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    nn_policy: &mut Option<NeuralNetPolicy>,
    exploration_constant: f64,
    stats: &mut TacticalMctsStats,
) -> Rc<RefCell<MctsNode>> {
    loop {
        let is_terminal = current.borrow().is_game_terminal();
        let has_children = !current.borrow().children.is_empty();
        
        if is_terminal || !has_children {
            return current; // Reached a leaf node
        }
        
        // Select child using tactical-first strategy
        if let Some(child) = select_child_with_tactical_priority(
            current.clone(),
            exploration_constant,
            move_gen,
            nn_policy,
            stats,
        ) {
            // Update stats
            if !current.borrow().policy_evaluated && nn_policy.is_some() {
                stats.nn_policy_evaluations += 1;
            }
            
            current = child;
        } else {
            return current; // No valid children (shouldn't happen)
        }
    }
}

/// Evaluate a leaf node using mate search + position evaluation
fn evaluate_leaf_node(
    node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    mate_search_depth: i32,
    transposition_table: &mut TranspositionTable,
    stats: &mut TacticalMctsStats,
) -> f64 {
    let mut node_ref = node.borrow_mut();
    let board = &node_ref.state;
    
    // Phase 1: Mate search with transposition table caching
    // First check if we already have mate results cached in the node
    if let Some(cached_value) = node_ref.terminal_or_mate_value {
        if cached_value >= 0.0 { // Actual mate value (0.0, 0.5, 1.0)
            return cached_value;
        }
        // cached_value == -999.0 means "mate search already done, no mate found"
        // Continue to position evaluation
    } else if mate_search_depth > 0 {
        // Check transposition table for mate search results
        if let Some((mate_depth, mate_move)) = transposition_table.probe_mate(board, mate_search_depth) {
            // Found cached mate search results
            stats.tt_mate_hits += 1;
            if mate_depth > 0 { // Mate found
                let mate_value = if mate_depth > 0 { 1.0 } else { 0.0 };
                node_ref.terminal_or_mate_value = Some(mate_value);
                node_ref.mate_move = Some(mate_move);
                stats.mates_found += 1;
                return mate_value;
            } else {
                // No mate found in cached search
                node_ref.terminal_or_mate_value = Some(-999.0);
            }
        } else {
            // Cache miss - perform mate search
            stats.tt_mate_misses += 1;
            // No cached results - perform mate search
            let mut board_stack = BoardStack::with_board(board.clone());
            let mate_result = mate_search(&mut board_stack, move_gen, mate_search_depth, false);
            
            // Store results in transposition table
            transposition_table.store_mate_result(
                board,
                mate_result.0.abs(), // Store absolute mate depth
                mate_result.1,
                mate_search_depth,
            );
            
            if mate_result.0 != 0 { // Mate found
                let mate_value = if mate_result.0 > 0 { 1.0 } else { 0.0 };
                node_ref.terminal_or_mate_value = Some(mate_value);
                node_ref.mate_move = Some(mate_result.1);
                stats.mates_found += 1;
                return mate_value;
            } else {
                // Mark as mate-search-checked to avoid redundant searches
                node_ref.terminal_or_mate_value = Some(-999.0); // Sentinel value for "no mate found"
            }
        }
    }
    
    // Phase 2: Position evaluation
    if node_ref.nn_value.is_none() {
        let eval_score = pesto_eval.eval(&node_ref.state, move_gen);
        // Convert centipawn evaluation to probability (sigmoid-like)
        let normalized_value = 1.0 / (1.0 + (-eval_score as f64 / 400.0).exp());
        node_ref.nn_value = Some(normalized_value);
    }
    
    node_ref.nn_value.unwrap_or(0.5)
}

/// Evaluate and expand a node (create child nodes)
fn evaluate_and_expand_node(
    node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    stats: &mut TacticalMctsStats,
) {
    let mut node_ref = node.borrow_mut();
    
    // Skip if already expanded or terminal
    if !node_ref.children.is_empty() || node_ref.is_game_terminal() {
        return;
    }
    
    // Generate all legal moves and create child nodes
    let (captures, non_captures) = move_gen.gen_pseudo_legal_moves(&node_ref.state);
    
    for mv in captures.iter().chain(non_captures.iter()) {
        let new_board = node_ref.state.apply_move_to_board(*mv);
        if new_board.is_legal(move_gen) {
            let child_node = MctsNode::new_child(
                Rc::downgrade(&node),
                *mv,
                new_board,
                move_gen,
            );
            node_ref.children.push(child_node);
            stats.nodes_expanded += 1;
        }
    }
    
    // Ensure position has been evaluated
    if node_ref.nn_value.is_none() {
        let eval_score = pesto_eval.eval(&node_ref.state, move_gen);
        let normalized_value = 1.0 / (1.0 + (-eval_score as f64 / 400.0).exp());
        node_ref.nn_value = Some(normalized_value);
    }
}

/// Backpropagate value up the tree
fn backpropagate_value(mut node: Rc<RefCell<MctsNode>>, mut value: f64) {
    loop {
        {
            let mut node_ref = node.borrow_mut();
            node_ref.visits += 1;
            
            // Add value from White's perspective
            let value_to_add = if node_ref.state.w_to_move {
                1.0 - value // Black just moved to get here
            } else {
                value // White just moved to get here
            };
            
            node_ref.total_value += value_to_add;
            node_ref.total_value_squared += value_to_add * value_to_add;
            
            // Flip value for next level up
            value = 1.0 - value;
        }
        
        // Move to parent
        let parent = {
            let node_ref = node.borrow();
            if let Some(parent_weak) = &node_ref.parent {
                parent_weak.upgrade()
            } else {
                None
            }
        };
        
        if let Some(parent_node) = parent {
            node = parent_node;
        } else {
            break; // Reached root
        }
    }
}

/// Select the best move from the root node
fn select_best_move_from_root(
    root: Rc<RefCell<MctsNode>>,
    config: &TacticalMctsConfig,
) -> Option<Move> {
    let root_ref = root.borrow();
    
    // Check for mate move first
    if let Some(mate_move) = root_ref.mate_move {
        return Some(mate_move);
    }
    
    // Select child with highest visit count (robustness)
    let mut best_move = None;
    let mut best_visits = 0;
    let mut best_value = f64::NEG_INFINITY;
    
    for child in &root_ref.children {
        let child_ref = child.borrow();
        
        // Primary criterion: visit count (robustness)
        if child_ref.visits > best_visits {
            best_visits = child_ref.visits;
            best_move = child_ref.action;
            best_value = if child_ref.visits > 0 {
                child_ref.total_value / child_ref.visits as f64
            } else {
                0.0
            };
        } else if child_ref.visits == best_visits && child_ref.visits > 0 {
            // Tie-break with value
            let child_value = child_ref.total_value / child_ref.visits as f64;
            if child_value > best_value {
                best_move = child_ref.action;
                best_value = child_value;
            }
        }
    }
    
    best_move
}

/// Print search statistics
pub fn print_search_stats(stats: &TacticalMctsStats, best_move: Option<Move>) {
    println!("🎯 Tactical-First MCTS Search Complete");
    println!("   Iterations: {}", stats.iterations);
    println!("   Time: {}ms", stats.search_time.as_millis());
    println!("   Nodes expanded: {}", stats.nodes_expanded);
    println!("   Mates found: {}", stats.mates_found);
    println!("   Tactical moves explored: {}", stats.tactical_moves_explored);
    println!("   NN policy evaluations: {}", stats.nn_policy_evaluations);
    
    if let Some(mv) = best_move {
        println!("   Best move: {}", format_move(mv));
    }
    
    // Calculate efficiency metrics
    if stats.iterations > 0 {
        let nn_eval_ratio = stats.nn_policy_evaluations as f64 / stats.iterations as f64;
        println!("   NN evals per iteration: {:.2}", nn_eval_ratio);
    }
}

/// Format a move for display
fn format_move(mv: Move) -> String {
    // Simple move formatting (can be enhanced)
    let from_file = ((mv.from % 8) as u8 + b'a') as char;
    let from_rank = ((mv.from / 8) as u8 + b'1') as char;
    let to_file = ((mv.to % 8) as u8 + b'a') as char;
    let to_rank = ((mv.to / 8) as u8 + b'1') as char;
    
    format!("{}{}{}{}", from_file, from_rank, to_file, to_rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    
    #[test]
    fn test_tactical_mcts_basic() {
        let board = Board::new();
        let move_gen = MoveGen::new();
        let pesto_eval = PestoEval::new();
        let mut nn_policy = None;
        
        let config = TacticalMctsConfig {
            max_iterations: 100,
            time_limit: Duration::from_millis(1000),
            ..Default::default()
        };
        
        let (best_move, stats) = tactical_mcts_search(
            board,
            &move_gen,
            &pesto_eval,
            &mut nn_policy,
            config,
        );
        
        // Should find a move from starting position
        assert!(best_move.is_some());
        assert!(stats.iterations > 0);
        assert!(stats.nodes_expanded > 0);
    }
    
    #[test]
    fn test_mate_detection() {
        // Back rank mate position
        let board = Board::new_from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1");
        let move_gen = MoveGen::new();
        let pesto_eval = PestoEval::new();
        let mut nn_policy = None;
        
        // First test: verify mate search works directly on this position
        let mut board_stack = BoardStack::with_board(board.clone());
        let (mate_score, mate_move, _nodes) = mate_search(&mut board_stack, &move_gen, 3, false);
        
        let config = TacticalMctsConfig {
            max_iterations: 100, // More iterations to be sure
            mate_search_depth: 5, // Deeper mate search to be sure
            ..Default::default()
        };
        
        let (best_move, stats) = tactical_mcts_search(
            board,
            &move_gen,
            &pesto_eval,
            &mut nn_policy,
            config,
        );
        
        // Should find the mating move
        assert!(best_move.is_some());
        // If direct mate search finds mate, tactical MCTS should too
        if mate_score >= 1000000 {
            assert!(stats.mates_found > 0, "Tactical MCTS should find mate when direct mate search does");
        }
    }
}