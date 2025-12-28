//! Tactical-First MCTS Selection
//!
//! This module implements the tactical-first selection strategy where tactical moves
//! (captures, checks, forks) are prioritized before strategic moves using neural network
//! policy guidance. This reduces neural network evaluations while maintaining tactical accuracy.

use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::mcts::node::MctsNode;
use crate::mcts::tactical::identify_tactical_moves;
use crate::mcts::tactical_mcts::TacticalMctsStats;
use crate::neural_net::NeuralNetPolicy;
use std::cell::RefCell;
use std::rc::Rc;

/// Select a child node using tactical-first prioritization
pub fn select_child_with_tactical_priority(
    node: Rc<RefCell<MctsNode>>,
    exploration_constant: f64,
    move_gen: &MoveGen,
    nn_policy: &mut Option<NeuralNetPolicy>,
    stats: &mut TacticalMctsStats,
) -> Option<Rc<RefCell<MctsNode>>> {
    // First, ensure the node has been expanded (children created)
    ensure_node_expanded(node.clone(), move_gen);
    
    {
        let node_ref = node.borrow();
        if node_ref.children.is_empty() {
            return None; // No legal moves (terminal position)
        }
    }
    
    // Phase 1: Check for unexplored tactical moves
    if let Some(tactical_child) = select_unexplored_tactical_move(node.clone(), move_gen, stats) {
        return Some(tactical_child);
    }
    
    // Phase 2: All tactical moves explored, use UCB with policy values
    select_ucb_with_policy(node, exploration_constant, nn_policy)
}

/// Ensure the node has been expanded with all child nodes
fn ensure_node_expanded(node: Rc<RefCell<MctsNode>>, move_gen: &MoveGen) {
    let mut node_ref = node.borrow_mut();
    
    // If children already created, nothing to do
    if !node_ref.children.is_empty() {
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
        }
    }
}

/// Select an unexplored tactical move, if any exist
fn select_unexplored_tactical_move(
    node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    stats: &mut TacticalMctsStats,
) -> Option<Rc<RefCell<MctsNode>>> {
    let mut node_ref = node.borrow_mut();
    
    // Ensure tactical moves have been identified
    if node_ref.tactical_moves.is_none() {
        let tactical_moves = identify_tactical_moves(&node_ref.state, move_gen);
        node_ref.tactical_moves = Some(tactical_moves);
    }
    
    // Find first unexplored tactical move
    let mut move_to_explore = None;
    if let Some(ref tactical_moves) = node_ref.tactical_moves {
        for tactical_move in tactical_moves {
            let mv = tactical_move.get_move();
            if !node_ref.tactical_moves_explored.contains(&mv) {
                // Find the corresponding child node
                let child = find_child_for_move(&node_ref.children, mv);
                if child.is_some() {
                    move_to_explore = Some(mv);
                    break;
                }
            }
        }
    }
    
    // Mark move as explored and return child
    if let Some(mv) = move_to_explore {
        node_ref.tactical_moves_explored.insert(mv);
        stats.tactical_moves_explored += 1; // Track in global statistics
        let child = find_child_for_move(&node_ref.children, mv);
        return child.cloned();
    }
    
    None // No unexplored tactical moves
}

/// Select using UCB formula with neural network policy values
fn select_ucb_with_policy(
    node: Rc<RefCell<MctsNode>>,
    exploration_constant: f64,
    nn_policy: &mut Option<NeuralNetPolicy>,
) -> Option<Rc<RefCell<MctsNode>>> {
    // Ensure policy has been evaluated
    ensure_policy_evaluated(node.clone(), nn_policy);
    
    let node_ref = node.borrow();
    if node_ref.children.is_empty() {
        return None;
    }
    
    let parent_visits = node_ref.visits;
    let num_legal_moves = node_ref.children.len();
    
    // Calculate UCB values for all children
    let mut best_child = None;
    let mut best_ucb = f64::NEG_INFINITY;
    
    for child in &node_ref.children {
        let child_ref = child.borrow();
        
        // Get policy value for this move
        let policy_value = if let Some(mv) = child_ref.action {
            // Tier 2: Use shadow priors from QS if available
            if let Some(shadow_v) = node_ref.shadow_priors.get(&mv) {
                *shadow_v // Use the extrapolated tactical value
            } else {
                node_ref.move_priorities.get(&mv).copied().unwrap_or(1.0 / num_legal_moves as f64)
            }
        } else {
            1.0 / num_legal_moves as f64
        };
        
        let ucb_value = calculate_ucb_value(
            &child_ref,
            parent_visits,
            policy_value,
            exploration_constant,
        );
        
        if ucb_value > best_ucb {
            best_ucb = ucb_value;
            best_child = Some(child.clone());
        }
    }
    
    best_child
}

/// Calculate UCB value for a child node
fn calculate_ucb_value(
    child: &MctsNode,
    parent_visits: u32,
    policy_value: f64,
    exploration_constant: f64,
) -> f64 {
    if child.visits == 0 {
        // Unvisited node gets high exploration value
        return f64::INFINITY;
    }
    
    // Q value: Average reward from this child's perspective (already relative to side that just moved)
    let q_value = child.total_value / child.visits as f64;
    
    // U value: Exploration term
    let exploration_term = exploration_constant
        * policy_value
        * (parent_visits as f64).sqrt()
        / (1.0 + child.visits as f64);
    
    q_value + exploration_term
}

/// Ensure neural network policy has been evaluated for the node
fn ensure_policy_evaluated(
    node: Rc<RefCell<MctsNode>>,
    nn_policy: &mut Option<NeuralNetPolicy>,
) {
    let mut node_ref = node.borrow_mut();
    
    if node_ref.policy_evaluated {
        return; // Already evaluated
    }
    
    // Strategic Tag: If this node was part of a tactical PV graft,
    // we still need policy priors for the *other* (quiet) moves eventually.
    // However, if we're just starting, we use the NN.

    // Collect moves first to avoid borrowing conflicts
    let mut moves_to_prioritize = Vec::new();
    for child in &node_ref.children {
        if let Some(mv) = child.borrow().action {
            moves_to_prioritize.push(mv);
        }
    }
    
    if moves_to_prioritize.is_empty() {
        node_ref.policy_evaluated = true;
        return;
    }

    // If neural network available, use it
    if let Some(nn) = nn_policy {
        if nn.is_available() {
            if let Some((policy_probs, _value, k)) = nn.predict(&node_ref.state) {
                let priors = nn.policy_to_move_priors(&policy_probs, &moves_to_prioritize);
                for (mv, prob) in priors {
                    node_ref.move_priorities.insert(mv, prob as f64);
                }
                node_ref.k_val = k; // Store k for tactical expansion
                node_ref.policy_evaluated = true;
                return;
            }
        }
    }
    
    // Fallback: uniform priors
    let uniform_prior = 1.0 / moves_to_prioritize.len() as f64;
    for mv in moves_to_prioritize {
        node_ref.move_priorities.insert(mv, uniform_prior);
    }
    
    node_ref.policy_evaluated = true;
}

/// Find a child node corresponding to a specific move
fn find_child_for_move(
    children: &[Rc<RefCell<MctsNode>>],
    mv: Move,
) -> Option<&Rc<RefCell<MctsNode>>> {
    children.iter().find(|child| {
        child.borrow().action == Some(mv)
    })
}

/// Get statistics about tactical vs strategic exploration
pub fn get_tactical_statistics(node: &MctsNode) -> TacticalStatistics {
    let total_tactical_moves = node.tactical_moves.as_ref().map_or(0, |tm| tm.len());
    let explored_tactical_moves = node.tactical_moves_explored.len();
    let tactical_phase_complete = explored_tactical_moves == total_tactical_moves;
    
    TacticalStatistics {
        total_tactical_moves,
        explored_tactical_moves,
        tactical_phase_complete,
        policy_evaluated: node.policy_evaluated,
    }
}

/// Statistics about tactical exploration for a node
#[derive(Debug, Clone)]
pub struct TacticalStatistics {
    pub total_tactical_moves: usize,
    pub explored_tactical_moves: usize,
    pub tactical_phase_complete: bool,
    pub policy_evaluated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    
    #[test]
    fn test_tactical_move_selection() {
        let board = Board::new();
        let move_gen = MoveGen::new();
        let root = MctsNode::new_root(board, &move_gen);
        
        // In starting position, should have no tactical moves
        let mut stats = TacticalMctsStats::default();
        let tactical_child = select_unexplored_tactical_move(root.clone(), &move_gen, &mut stats);
        assert!(tactical_child.is_none());
    }
    
    #[test]
    fn test_node_expansion() {
        let board = Board::new();
        let move_gen = MoveGen::new();
        let root = MctsNode::new_root(board, &move_gen);
        
        ensure_node_expanded(root.clone(), &move_gen);
        
        // Starting position should have 20 legal moves
        let node_ref = root.borrow();
        assert_eq!(node_ref.children.len(), 20);
    }
    
    #[test]
    fn test_ucb_calculation() {
        // Create a mock child node for testing
        let board = Board::new();
        let move_gen = MoveGen::new();
        let parent = MctsNode::new_root(board.clone(), &move_gen);
        
        let child_board = board.apply_move_to_board(Move::new(12, 28, None)); // e2-e4
        let child = MctsNode::new_child(
            Rc::downgrade(&parent),
            Move::new(12, 28, None),
            child_board,
            &move_gen,
        );
        
        // Test UCB calculation for unvisited node
        let child_ref = child.borrow();
        let ucb = calculate_ucb_value(&child_ref, 10, 0.1, 1.414);
        assert!(ucb.is_infinite()); // Unvisited nodes get infinite UCB
    }
}