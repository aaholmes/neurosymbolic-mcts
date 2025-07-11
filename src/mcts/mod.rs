//! Tactical-First Monte Carlo Tree Search (MCTS) Module
//!
//! This module implements a sophisticated **Tactical-First MCTS** architecture that combines
//! classical chess principles with modern AI techniques. The core innovation is a three-tier
//! prioritization system that ensures tactical completeness while maintaining computational efficiency.
//!
//! ## Architecture Overview
//!
//! ### Three-Tier Search Prioritization
//!
//! 1. **Mate Search First** (`tactical_mcts.rs`)
//!    - Exhaustive forced-mate analysis before any other evaluation
//!    - Immediate return if mate found (no further MCTS needed)
//!
//! 2. **Tactical Move Priority** (`tactical.rs`, `selection.rs`)
//!    - Classical heuristics explore forcing moves before strategic moves
//!    - MVV-LVA ordering for captures
//!    - Knight/pawn fork detection with value calculation
//!    - Check move prioritization with centrality bonuses
//!
//! 3. **Lazy Neural Policy** (`selection.rs`)
//!    - Neural network policy evaluation deferred until after tactical exploration
//!    - Substantially reduces expensive neural network computational overhead
//!    - UCB selection with policy priors for strategic moves
//!
//! ## Key Components
//!
//! - **`tactical_mcts`**: Main tactical-first search algorithm
//! - **`tactical`**: Tactical move detection and prioritization
//! - **`selection`**: Tactical-first node selection strategy
//! - **`node`**: Enhanced MCTS node structure with tactical fields
//! - **`neural_mcts`**: Classical neural-guided MCTS implementation
//! - **`policy`**: Neural network policy interface
//!
//! ## Usage
//!
//! ```rust
//! use kingfisher::mcts::{tactical_mcts_search, TacticalMctsConfig};
//! use std::time::Duration;
//!
//! let config = TacticalMctsConfig {
//!     max_iterations: 1000,
//!     time_limit: Duration::from_millis(5000),
//!     mate_search_depth: 3,
//!     exploration_constant: 1.414,
//!     use_neural_policy: true,
//! };
//!
//! let (best_move, stats) = tactical_mcts_search(
//!     board, &move_gen, &pesto_eval, &mut nn_policy, config
//! );
//! ```
//!
//! This architecture successfully implements the chess principle of "examine all checks,
//! captures, and threats" while dramatically reducing computational overhead through
//! intelligent move ordering and lazy evaluation.

pub mod neural_mcts;
pub mod nn_counter;
pub mod node;
pub mod policy;
pub mod selection;
pub mod simulation; // Keep for testing/alternative use
pub mod tactical;
pub mod tactical_mcts;

// Additional test modules are included within existing tests module at the bottom

use crate::board::Board;
use crate::boardstack::BoardStack; // Needed for mate_search
use crate::eval::PestoEval; // Import PestoEval
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::neural_net::NeuralNetPolicy; // Import neural network
use crate::search::mate_search; // Import mate_search function
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant}; // Use trait from submodule

// Import necessary components from submodules
pub use self::neural_mcts::neural_mcts_search;
pub use self::node::{select_leaf_for_expansion, MctsNode, MoveCategory};
pub use self::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig, TacticalMctsStats, print_search_stats};

// pub use self::policy::PolicyNetwork; // No longer needed for pesto search
                                     // pub use self::simulation::simulate_random_playout; // Don't export by default

// Constants
const EXPLORATION_CONSTANT: f64 = 1.414; // sqrt(2), a common value for UCT/PUCT
const PESSIMISTIC_FACTOR: f64 = 1.0; // Factor 'k' for pessimistic value calculation (Q - k * std_err)

/// Backpropagates an evaluation result through the tree.
/// Updates visits, total_value, and total_value_squared for each node.
/// Assumes 'value' is from White's perspective [0.0, 1.0].
fn backpropagate(node: Rc<RefCell<MctsNode>>, value: f64) {
    let mut current_node_opt = Some(node);
    while let Some(current_node_rc) = current_node_opt {
        {
            // Borrow scope
            let mut current_node = current_node_rc.borrow_mut();
            current_node.visits += 1;
            // Add value relative to the player whose turn it *was* in the parent
            let reward_to_add = if current_node.state.w_to_move {
                1.0 - value // Black just moved to get here, add Black's score
            } else {
                value // White just moved to get here, add White's score
            };
            current_node.total_value += reward_to_add;
            current_node.total_value_squared += reward_to_add.powi(2); // Accumulate squared reward
        } // End borrow scope

        // Move to parent
        current_node_opt = {
            let current_node = current_node_rc.borrow();
            if let Some(parent_weak) = &current_node.parent {
                parent_weak.upgrade()
            } else {
                None // Reached root
            }
        };
    }
}

/// Performs MCTS search using Policy/Value evaluation and Mate Search First strategy.
///
/// # Arguments
/// * `root_state` - The initial board state.
/// * `move_gen` - The move generator.
/// * `policy_network` - Implementation of the PolicyNetwork trait.
/// * `mate_search_depth` - Depth for the classical mate search check. Set <= 0 to disable.
/// * `iterations` - Optional number of MCTS iterations to run.
/// * `time_limit` - Optional time duration limit for the search.
///
/// # Returns
/// The best move found for the root state based on visit counts. Returns None if no legal moves from root.
pub fn mcts_pesto_search(
    root_state: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval, // Use PestoEval
    mate_search_depth: i32,
    iterations: Option<u32>,
    time_limit: Option<Duration>,
) -> Option<Move> {
    if iterations.is_none() && time_limit.is_none() {
        panic!("MCTS search requires either iterations or time_limit to be set.");
    }
    if mate_search_depth < 0 {
        panic!("Mate search depth cannot be negative.");
    }

    let start_time = Instant::now();
    
    // **KEY INNOVATION: Mate Search First at Root Level**
    // Before any MCTS iterations, check if there's a forced mate from the root
    let mate_move_result = if mate_search_depth > 0 {
        let mut mate_search_stack = BoardStack::with_board(root_state.clone());
        let (mate_score, mate_move, _) = mate_search(&mut mate_search_stack, move_gen, mate_search_depth, false);
        
        if mate_score >= 1_000_000 {
            // Found a forced mate! Return immediately without MCTS
            println!("Mate found at root level in {} plies, returning immediately: {:?}", mate_search_depth, mate_move);
            Some(mate_move)
        } else if mate_score <= -1_000_000 {
            // We're being mated, but still return best move from mate search
            println!("Being mated, but returning best defensive move: {:?}", mate_move);
            Some(mate_move)
        } else {
            // No mate found, proceed with normal MCTS
            println!("No mate found at depth {}, proceeding with MCTS", mate_search_depth);
            None
        }
    } else {
        None
    };
    
    // If mate was found at root, return immediately
    if let Some(immediate_mate_move) = mate_move_result {
        return Some(immediate_mate_move);
    }

    let root_node_rc = MctsNode::new_root(root_state, move_gen);

    // Handle case where root node is already terminal
    if root_node_rc.borrow().is_terminal {
        return None; // No moves possible
    }

    // Evaluate and categorize the root node immediately to enable expansion
    {
        let mut root_node = root_node_rc.borrow_mut();
        if root_node.nn_value.is_none() && root_node.terminal_or_mate_value.is_none() {
            let score_cp = pesto_eval.eval(&root_node.state, move_gen);
            let value_current_player = 1.0 / (1.0 + (-score_cp as f64 / 400.0).exp());
            let value_white_pov = if root_node.state.w_to_move {
                value_current_player
            } else {
                1.0 - value_current_player
            };
            root_node.nn_value = Some(value_white_pov);
            root_node.categorize_and_store_moves(move_gen);
        }
    }

    let mut iteration_count = 0;

    loop {
        // --- Check Termination Conditions ---
        if let Some(max_iter) = iterations {
            if iteration_count >= max_iter {
                break;
            }
        }
        if let Some(limit) = time_limit {
            if (iteration_count & 63) == 0 && start_time.elapsed() >= limit {
                // Check time periodically
                break;
            }
        }

        // --- MCTS Cycle ---
        // 1. Selection: Find a leaf node suitable for expansion/evaluation.
        let leaf_node_rc = select_leaf_for_expansion(root_node_rc.clone(), EXPLORATION_CONSTANT);

        // --- 2. Mate Check / Evaluation / Expansion ---
        let mut node_to_propagate_from: Rc<RefCell<MctsNode>>;
        let mut value_to_propagate: f64; // Value relative to White [0.0, 1.0]

        {
            // Scope for borrowing leaf_node
            let mut leaf_node = leaf_node_rc.borrow_mut();

            // --- 2a. Check Terminal/Mate Status (only once per node) ---
            if leaf_node.terminal_or_mate_value.is_none() {
                let (is_mate, is_stalemate) = leaf_node.state.is_checkmate_or_stalemate(move_gen);
                if is_stalemate {
                    leaf_node.terminal_or_mate_value = Some(0.5);
                } else if is_mate {
                    leaf_node.terminal_or_mate_value =
                        Some(if leaf_node.state.w_to_move { 0.0 } else { 1.0 });
                // White's perspective
                } else if mate_search_depth > 0 {
                    // Not terminal, run mate search first (key innovation!)
                    let mut mate_search_stack = BoardStack::with_board(leaf_node.state.clone());
                    let (mate_score, mate_move, _) = mate_search(&mut mate_search_stack, move_gen, mate_search_depth, false);
                    if mate_score >= 1_000_000 {
                        // Mate found for current player
                        leaf_node.terminal_or_mate_value =
                            Some(if leaf_node.state.w_to_move { 1.0 } else { 0.0 });
                        leaf_node.mate_move = Some(mate_move); // Store the mating move!
                        println!("Mate found during MCTS leaf expansion: {:?}", mate_move);
                    } else if mate_score <= -1_000_000 {
                        // Mated
                        leaf_node.terminal_or_mate_value =
                            Some(if leaf_node.state.w_to_move { 0.0 } else { 1.0 });
                        leaf_node.mate_move = Some(mate_move); // Store the best defensive move
                        println!("Being mated during MCTS leaf expansion, best defense: {:?}", mate_move);
                    } else {
                        leaf_node.terminal_or_mate_value = Some(-999.0); // Sentinel: No mate found
                    }
                } else {
                    leaf_node.terminal_or_mate_value = Some(-999.0); // Mate search disabled
                }
            }

            // --- 2b. Decide Value and Node for Backpropagation ---
            if let Some(exact_value) = leaf_node.terminal_or_mate_value {
                if exact_value >= 0.0 {
                    // Mate found or terminal state (0.0, 0.5, 1.0)
                    value_to_propagate = exact_value;
                    node_to_propagate_from = leaf_node_rc.clone(); // Backpropagate exact value from leaf
                } else {
                    // No mate found (sentinel -999.0), proceed to evaluate/expand
                    // --- 2c. Evaluate Leaf (if not already done) ---
                    if leaf_node.nn_value.is_none() { // Reuse nn_value field for Pesto value
                        // Evaluate using Pesto
                        let score_cp = pesto_eval.eval(&leaf_node.state, move_gen);
                        // Convert centipawns to win probability [0.0, 1.0] for the current player
                        // Using a sigmoid function: 1 / (1 + exp(-score / k))
                        // k=400 is a common choice (maps +/- 400cp to ~75%/25% win prob)
                        let value_current_player = 1.0 / (1.0 + (-score_cp as f64 / 400.0).exp());

                        // Convert to White's perspective for storage and backpropagation
                        let value_white_pov = if leaf_node.state.w_to_move {
                            value_current_player
                        } else {
                            1.0 - value_current_player
                        };
                        leaf_node.nn_value = Some(value_white_pov); // Store Pesto-derived value

                        // Now that the node is evaluated, categorize its moves for expansion
                        leaf_node.categorize_and_store_moves(move_gen);

                        value_to_propagate = value_white_pov;
                        node_to_propagate_from = leaf_node_rc.clone(); // Backpropagate evaluated value from leaf
                    } else {
                        // --- 2d. Expand Leaf ---
                        // Node has already been evaluated (nn_value is Some). Try to expand.
                        // Use the prioritized move selection.
                        if let Some(action_to_expand) = leaf_node.get_next_move_to_explore() { // Use the correct function
                            let next_state = leaf_node.state.apply_move_to_board(action_to_expand);
                            let parent_weak = Rc::downgrade(&leaf_node_rc);
                            let new_child_rc = MctsNode::new_child(
                                parent_weak,
                                action_to_expand,
                                next_state,
                                move_gen,
                            );
                            leaf_node.children.push(new_child_rc.clone()); // Add child

                            // Backpropagate from the newly created child so it gets a visit
                            value_to_propagate = leaf_node.nn_value.unwrap();
                            node_to_propagate_from = new_child_rc; // Start backprop from the new child
                        } else {
                            // Node evaluated but fully explored. This can happen.
                            // Backpropagate the known value again to reinforce parent nodes.
                            value_to_propagate = leaf_node.nn_value.unwrap();
                            node_to_propagate_from = leaf_node_rc.clone();
                        }
                    }
                }
            } else {
                panic!("Node terminal/mate status should have been determined");
            }
        } // End borrow scope for leaf_node

        // --- 3. Backpropagation ---
        backpropagate(node_to_propagate_from, value_to_propagate);

        iteration_count += 1;
    }

    // --- Select Best Move --- (Using Pessimistic Value: Q - k * StdErr)
    let best_move = {
        let root_node = root_node_rc.borrow();
        root_node.children.iter()
            .filter(|child_rc| child_rc.borrow().visits > 0) // Only consider visited children
            .max_by(|a_rc, b_rc| {
                let a = a_rc.borrow();
                let b = b_rc.borrow();
                
                // **MATE PRIORITIZATION**: If either child has a mate move, prioritize it
                let a_has_mate = a.mate_move.is_some() && a.terminal_or_mate_value.is_some() && a.terminal_or_mate_value.unwrap() >= 0.0;
                let b_has_mate = b.mate_move.is_some() && b.terminal_or_mate_value.is_some() && b.terminal_or_mate_value.unwrap() >= 0.0;
                
                if a_has_mate && !b_has_mate {
                    return std::cmp::Ordering::Greater; // A has mate, B doesn't - choose A
                }
                if b_has_mate && !a_has_mate {
                    return std::cmp::Ordering::Less; // B has mate, A doesn't - choose B
                }
                // If both have mates or neither have mates, use normal evaluation
                
                let visits_a = a.visits as f64;
                let visits_b = b.visits as f64;

                // Calculate Q (average value from White's perspective)
                let q_a = a.total_value / visits_a;
                let q_b = b.total_value / visits_b;

                // Calculate Standard Error = sqrt(Var(X) / N) = sqrt( (E[X^2] - E[X]^2) / N )
                let var_a = (a.total_value_squared / visits_a) - q_a.powi(2);
                let var_b = (b.total_value_squared / visits_b) - q_b.powi(2);
                // Ensure variance is non-negative due to potential floating point inaccuracies
                let std_err_a = (var_a.max(0.0) / visits_a).sqrt();
                let std_err_b = (var_b.max(0.0) / visits_b).sqrt();

                // Calculate Pessimistic Score (Lower Confidence Bound from White's perspective)
                let pessimistic_a = q_a - PESSIMISTIC_FACTOR * std_err_a;
                let pessimistic_b = q_b - PESSIMISTIC_FACTOR * std_err_b;

                // Compare based on whose turn it is at the root
                if root_node.state.w_to_move {
                    // White wants to maximize the pessimistic score
                    pessimistic_a.partial_cmp(&pessimistic_b).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    // Black wants to minimize White's pessimistic score
                    // This is equivalent to maximizing Black's pessimistic score: (1-Q) - k*StdErr
                    // Or minimizing White's Upper Confidence Bound: Q + k*StdErr
                    let ucb_a = q_a + PESSIMISTIC_FACTOR * std_err_a;
                    let ucb_b = q_b + PESSIMISTIC_FACTOR * std_err_b;
                    // Black chooses the move with the *lowest* UCB for White
                    ucb_b.partial_cmp(&ucb_a).unwrap_or(std::cmp::Ordering::Equal)
                }
            })
            .map(|best_child| best_child.borrow().action.expect("Child node must have an action"))
    };

    best_move
}

#[cfg(test)]
mod unit_tests {
    mod tactical_tests;
    mod selection_tests; 
    mod mcts_integration_tests;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::eval::PestoEval;
    use crate::move_generation::MoveGen;
    // Using Move::from_uci for creating Move objects
    use std::time::Duration;

    // Helper function to initialize common test components
    fn setup_test_env() -> (Board, MoveGen, PestoEval) {
        let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let move_gen = MoveGen::new();
        let pesto_eval = PestoEval::new();
        (board, move_gen, pesto_eval)
    }

    #[test]
    fn test_mcts_pesto_basic_run() {
        let (board, move_gen, pesto_eval) = setup_test_env();

        // Run for a small number of iterations
        let result = mcts_pesto_search(
            board,
            &move_gen,
            &pesto_eval,
            0, // Disable mate search for this basic test
            Some(20), // Low iteration count
            None,
        );

        // Should return *some* move from the starting position
        assert!(result.is_some(), "MCTS should find a move from the start position");
    }

    #[test]
    fn test_mcts_pesto_time_limit() {
         let (board, move_gen, pesto_eval) = setup_test_env();

         // Run with a time limit
         let result = mcts_pesto_search(
             board,
             &move_gen,
             &pesto_eval,
             0, // Disable mate search
             None, // No iteration limit
             Some(Duration::from_millis(50)), // Short time limit
         );

         // Should still return a move
         assert!(result.is_some(), "MCTS with time limit should find a move");
    }

#[test]
    fn test_mcts_pesto_favors_better_eval() {
        // Position: White to move. White queen on d4 can capture black queen on b6 or black pawn on f6.
        // Capturing the queen is much better according to Pesto.
        let board = Board::new_from_fen("rnbk2nr/pppp1ppp/1q3p2/8/3Q4/8/PPPPPPPP/RNB1KBNR w KQ - 0 1");
        let move_gen = MoveGen::new();
        let pesto_eval = PestoEval::new();

        // Moves available: Qxb6 (capture queen), Qxf6 (capture pawn), and others.
        let queen_capture_move = Move::from_uci("d4b6").unwrap(); // Capture queen
        let pawn_capture_move = Move::from_uci("d4f6").unwrap(); // Capture pawn (less good)

        // Run for enough iterations to distinguish
        let best_move = mcts_pesto_search(
            board.clone(),
            &move_gen,
            &pesto_eval,
            0, // Disable mate search
            Some(2000), // More iterations to allow evaluation difference to propagate
            None,
        );

        assert!(best_move.is_some(), "Search should return a move");

        // We expect the search to prefer capturing the queen
        assert_eq!(
            best_move.unwrap(),
            queen_capture_move,
            "MCTS with Pesto should prefer capturing the queen (Qxb6) over the pawn (Qxf6)"
        );

        // Optional: Verify both moves are actually legal
        let mut legal_moves = Vec::new();
        let (captures, non_captures) = move_gen.gen_pseudo_legal_moves(&board);
        legal_moves.extend(captures);
        legal_moves.extend(non_captures);
        legal_moves.retain(|&mv| board.apply_move_to_board(mv).is_legal(&move_gen));
        assert!(legal_moves.contains(&queen_capture_move), "Queen capture (Qxb6) should be legal");
        assert!(legal_moves.contains(&pawn_capture_move), "Pawn capture (Qxf6) should be legal");
    }
     #[test]
     #[should_panic]
     fn test_mcts_pesto_no_limits_panic() {
         let (board, move_gen, pesto_eval) = setup_test_env();
         // This should panic because neither iterations nor time limit is set
         mcts_pesto_search(board, &move_gen, &pesto_eval, 0, None, None);
     }

     #[test]
     #[should_panic]
     fn test_mcts_pesto_negative_mate_depth_panic() {
         let (board, move_gen, pesto_eval) = setup_test_env();
         // This should panic due to negative mate search depth
         mcts_pesto_search(board, &move_gen, &pesto_eval, -1, Some(10), None);
     }

    // --- More tests to be added below ---

}
