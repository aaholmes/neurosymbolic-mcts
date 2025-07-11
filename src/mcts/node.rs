//! Defines the Node structure for the MCTS tree.

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::mcts::tactical::TacticalMove;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::f64;
use std::rc::{Rc, Weak};
                         // use super::policy::PolicyNetwork; // Not needed directly in this file if PolicyNetwork passed to mcts_search
                         // use crate::search::see::see; // Import SEE if used for categorization

// Define Move Categories (Lower discriminant = higher priority)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MoveCategory {
    Check = 0,   // Highest priority
    Capture = 1, // Includes promotions
    Quiet = 2,   // Lowest priority
}

/// A node in the Monte Carlo Search Tree
#[derive(Debug)]
pub struct MctsNode {
    /// The chess position at this node
    pub state: Board,

    /// The move that led to this state (None for root)
    pub action: Option<Move>,

    /// Number of times this node has been visited
    pub visits: u32,

    /// Total value accumulated through this node (always from White's perspective, 0.0 to 1.0)
    pub total_value: f64,
    /// Sum of squared values accumulated through this node (for variance calculation).
    pub total_value_squared: f64,

    // --- Evaluation / Mate Status ---
    /// Stores the exact value if determined by terminal state check or mate search (0.0, 0.5, 1.0 for White).
    /// Also used as a flag to indicate if mate search has been performed. None = not checked, Some(-999.0) = checked, no mate.
    pub terminal_or_mate_value: Option<f64>,
    /// If mate search found a mate, stores the mating move
    pub mate_move: Option<Move>,
    /// Stores the value from the evaluation (Pesto or NN) (0.0 to 1.0 for White) when node is first evaluated.
    pub nn_value: Option<f64>, // Reusing this field for Pesto value
    /// Neural network prior probability for this move (from parent's perspective)
    pub prior_probability: f32,
    // /// Stores the policy priors for all legal moves from this state, evaluated once.
    // pub policy_priors: Option<HashMap<Move, f64>>, // REMOVED - Not used in Pesto MCTS

    /// Reference to parent node (None for root)
    pub parent: Option<Weak<RefCell<MctsNode>>>,
    /// Child nodes (explored actions)
    pub children: Vec<Rc<RefCell<MctsNode>>>,

    // --- Expansion / Selection Control ---
    /// Stores unexplored legal moves, categorized by priority. Populated once after evaluation.
    pub unexplored_moves_by_cat: HashMap<MoveCategory, Vec<Move>>,
    /// Tracks the current highest-priority category being explored. Initialized after evaluation.
    pub current_priority_category: Option<MoveCategory>,
    /// Number of legal moves from this state, used for uniform prior in PUCT. Set during categorization.
    pub num_legal_moves: Option<usize>,
    // `untried_actions` Vec removed, replaced by the map above.
    /// Whether this is a terminal state (checkmate, stalemate) - based on initial check
    pub is_terminal: bool,

    // --- Tactical-First MCTS Fields ---
    /// Cached tactical moves from this position (computed once)
    pub tactical_moves: Option<Vec<TacticalMove>>,
    /// Set of tactical moves that have been explored
    pub tactical_moves_explored: HashSet<Move>,
    /// Whether the neural network policy has been evaluated for this node
    pub policy_evaluated: bool,
    /// Move priorities for UCB selection (after tactical phase)
    pub move_priorities: HashMap<Move, f64>,
}

impl MctsNode {
    /// Creates a new root node for MCTS
    pub fn new_root(state: Board, move_gen: &MoveGen) -> Rc<RefCell<Self>> {
        // Check if the state is already terminal
        let (is_checkmate, is_stalemate) = state.is_checkmate_or_stalemate(move_gen);
        let is_terminal = is_checkmate || is_stalemate;

        // Determine initial terminal/mate value
        let initial_terminal_value = if is_stalemate {
            Some(0.5)
        } else if is_checkmate {
            Some(if state.w_to_move { 0.0 } else { 1.0 }) // Mate value from White's perspective
        } else {
            None // Not terminal initially
        };

        Rc::new(RefCell::new(Self {
            state,
            action: None,
            visits: 0,
            total_value: 0.0,
            total_value_squared: 0.0, // Initialize new field
            parent: None,
            children: Vec::new(),
            terminal_or_mate_value: initial_terminal_value,
            mate_move: None,                         // Set if mate search finds a mate
            nn_value: None,                          // Evaluated later
            prior_probability: 0.0,                  // Default prior
            // policy_priors: None,                  // REMOVED
            unexplored_moves_by_cat: HashMap::new(), // Populated later
            current_priority_category: None,         // Set later
            num_legal_moves: None,                   // Set later
            is_terminal,                             // Store initial terminal status
            
            // Initialize tactical-first fields
            tactical_moves: None,
            tactical_moves_explored: HashSet::new(),
            policy_evaluated: false,
            move_priorities: HashMap::new(),
        }))
    }

    /// Creates a new child node. Note: Priors/Categorization happen later.
    pub fn new_child(
        // Made public
        parent: Weak<RefCell<MctsNode>>,
        action: Move,
        new_state: Board, // Pass the already calculated new state
        move_gen: &MoveGen,
    ) -> Rc<RefCell<Self>> {
        // Check if the new state is terminal
        let (is_checkmate, is_stalemate) = new_state.is_checkmate_or_stalemate(move_gen);
        let is_terminal = is_checkmate || is_stalemate;

        // Determine initial terminal/mate value
        let initial_terminal_value = if is_stalemate {
            Some(0.5)
        } else if is_checkmate {
            Some(if new_state.w_to_move { 0.0 } else { 1.0 }) // Mate value from White's perspective
        } else {
            None // Not terminal initially
        };

        Rc::new(RefCell::new(Self {
            state: new_state,
            action: Some(action),
            visits: 0,
            total_value: 0.0,
            total_value_squared: 0.0, // Initialize new field
            parent: Some(parent),
            children: Vec::new(),
            terminal_or_mate_value: initial_terminal_value,
            mate_move: None,                         // Set if mate search finds a mate
            nn_value: None,
            prior_probability: 0.0,                  // Default prior
            // policy_priors: None,                  // REMOVED
            unexplored_moves_by_cat: HashMap::new(),
            current_priority_category: None,
            num_legal_moves: None,                   // Set later
            is_terminal,
            
            // Initialize tactical-first fields
            tactical_moves: None,
            tactical_moves_explored: HashSet::new(),
            policy_evaluated: false,
            move_priorities: HashMap::new(),
        }))
    }

    /// Returns true if this node's state is terminal (checkmate/stalemate).
    pub fn is_game_terminal(&self) -> bool {
        self.is_terminal
    }

    /// Returns true if the node has been evaluated (by NN or mate search) OR is terminal.
    pub fn is_evaluated_or_terminal(&self) -> bool {
        self.nn_value.is_some() || self.terminal_or_mate_value.is_some()
    }

    /// Checks if all move categories have been fully explored.
    pub fn is_fully_explored(&self) -> bool {
        // Considered fully explored if categorization has happened (num_legal_moves is Some)
        // and the map of unexplored moves is empty.
        self.num_legal_moves.is_some() && self.unexplored_moves_by_cat.is_empty()
    }

    /// PUCT value calculation - used for selecting among *explored* children.
    /// PUCT = Q + U
    /// Q = Average value for the selector (parent)
    /// U = exploration_constant * P * sqrt(parent_visits) / (1 + child_visits)
    /// P = Uniform prior = 1.0 / num_legal_moves (for the parent node)
    pub fn puct_value(&self, parent_visits: u32, parent_num_legal_moves: usize, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            // For unvisited children, PUCT relies heavily on the prior 'P'.
            // Calculate U term with visits = 0. Q is effectively 0 for the selector.
            let prior_p = if parent_num_legal_moves > 0 {
                1.0 / parent_num_legal_moves as f64
            } else {
                0.0 // No legal moves from parent? Should not happen if child exists.
            };
            let u_value = exploration_constant * prior_p * (parent_visits as f64).sqrt(); // Denominator is 1 + 0 = 1
            return u_value; // Return exploration term only for unvisited nodes
        }

        // Q value: Average value from White's perspective
        let avg_value_white = self.total_value / self.visits as f64;

        // Adjust Q value based on whose turn it is at the *parent* node (who is selecting)
        let q_value_for_selector = if let Some(parent_weak) = &self.parent {
            if let Some(parent_rc) = parent_weak.upgrade() {
                if parent_rc.borrow().state.w_to_move { avg_value_white } else { 1.0 - avg_value_white }
            } else { 0.5 } // Parent dropped? Default to draw value
        } else { 0.5 }; // Root node? Should not be selected via PUCT. Default draw.

        // U value: Exploration term (PUCT)
        let prior_p = if parent_num_legal_moves > 0 {
            1.0 / parent_num_legal_moves as f64
        } else {
            0.0
        };
        let exploration_term = exploration_constant
            * prior_p
            * (parent_visits as f64).sqrt()
            / (1.0 + self.visits as f64);

        q_value_for_selector + exploration_term
    }


    /// UCT value calculation - used for selecting among *explored* children.
    /// Note: This uses standard UCT. PUCT logic is implicitly handled by the
    /// prioritized expansion strategy, which determines *which* nodes get expanded,
    /// rather than modifying the selection value formula itself.
    pub fn uct_value(&self, parent_visits: u32, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY; // Should not happen if selecting among explored children
        }

        // Q value: Average value from White's perspective
        let avg_value_white = self.total_value / self.visits as f64;

        // Adjust Q value based on whose turn it is at the *parent* node (who is selecting)
        let q_value_for_selector = if let Some(parent_weak) = &self.parent {
            if let Some(parent_rc) = parent_weak.upgrade() {
                // If parent was White, use Q directly. If parent was Black, use 1-Q.
                if parent_rc.borrow().state.w_to_move {
                    avg_value_white
                } else {
                    1.0 - avg_value_white
                }
            } else {
                0.5
            } // Parent dropped? Default to draw value
        } else {
            0.5
        }; // Root node? Should not be selected via UCT. Default draw.

        // U value: Exploration term (standard UCT)
        let exploration_term =
            exploration_constant * ((parent_visits as f64).ln() / self.visits as f64).sqrt();

        q_value_for_selector + exploration_term
    }

    /// Selects the best *explored* child node according to the PUCT formula.
    /// Panics if called on a terminal node or a node with no children.
    pub fn select_best_explored_child(&self, exploration_constant: f64) -> Rc<RefCell<MctsNode>> {
        let parent_visits = self.visits;
        // Parent needs its own num_legal_moves for the PUCT calculation
        let parent_num_legal = self.num_legal_moves.unwrap_or(1); // Default to 1 if not set (should be set)

        self.children
            .iter()
            .max_by(|a, b| {
                let puct_a = a.borrow().puct_value(parent_visits, parent_num_legal, exploration_constant);
                let puct_b = b.borrow().puct_value(parent_visits, parent_num_legal, exploration_constant);
                puct_a
                    .partial_cmp(&puct_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .expect("select_best_explored_child called on node with no children")
    }


    /// Generates legal moves, categorizes them, sorts them within categories,
    /// and stores them for prioritized expansion. Called once after evaluation.
    pub fn categorize_and_store_moves(
        &mut self,
        move_gen: &MoveGen,
        // Pass killers if needed for categorization
        // killers: &[[Move; 2]; MAX_PLY],
        // ply: usize,
    ) {
        if self.num_legal_moves.is_some() {
             return; // Already done
        }

        let legal_moves = MctsNode::get_legal_moves(&self.state, move_gen);
        self.num_legal_moves = Some(legal_moves.len()); // Store count for PUCT

        let mut categorized_moves: HashMap<MoveCategory, Vec<Move>> = HashMap::new();

        // Get MVV-LVA scores for captures beforehand
        let mut capture_scores: HashMap<Move, i32> = HashMap::new();
        for mv in &legal_moves {
             let opponent_color = !self.state.w_to_move as usize;
             let is_capture = (self.state.pieces_occ[opponent_color] & (1u64 << mv.to)) != 0 || mv.is_en_passant();
             if is_capture || mv.is_promotion() {
                 // Use the implemented mvv_lva method from MoveGen
                 capture_scores.insert(*mv, move_gen.mvv_lva(&self.state, mv.from, mv.to));
             }
        }


        // Categorize moves
        for mv in legal_moves {
            let category = self.categorize_move(&mv, move_gen);
            categorized_moves.entry(category).or_default().push(mv);
        }

        // Sort moves within each category
        for (category, moves) in categorized_moves.iter_mut() {
            match category {
                MoveCategory::Check | MoveCategory::Quiet => {
                    // No policy priors. Could shuffle or use simple heuristics later.
                    // For now, keep the order from move generation (often somewhat reasonable).
                    // Or potentially shuffle them:
                    // use rand::seq::SliceRandom;
                    // use rand::thread_rng;
                    // moves.shuffle(&mut thread_rng());
                }
                MoveCategory::Capture => {
                    // Sort by MVV-LVA score (descending)
                    moves.sort_unstable_by_key(|mv| {
                        std::cmp::Reverse(capture_scores.get(mv).cloned().unwrap_or(0)) // Use Reverse for descending
                    });
                }
            }
        }

        // Sort categories by priority to determine the starting category
        let mut sorted_categories: Vec<_> = categorized_moves.keys().cloned().collect();
        sorted_categories.sort(); // Sorts by enum discriminant (priority)

        self.unexplored_moves_by_cat = categorized_moves; // Store the map with sorted Vec<Move>
        self.current_priority_category = sorted_categories.first().cloned();
    }


    /// Categorizes a move for prioritizing expansion.
    /// Helper to categorize a single move into Tactical/Killer/Quiet.
    /// TODO: Needs access to killers, history, SEE results for proper categorization.
    /// TODO: Needs efficient fork detection logic.
    fn categorize_move(&self, mv: &Move, move_gen: &MoveGen) -> MoveCategory {
        // Check detection (potentially expensive)
        // Apply the move temporarily to check if the opponent is in check
        let next_state = self.state.apply_move_to_board(*mv);
        // is_check checks if the *new* side to move (opponent) is in check
        if next_state.is_check(move_gen) {
             return MoveCategory::Check;
        }

        // Captures (includes promotions as they often capture or are high value)
        let opponent_color = !self.state.w_to_move as usize;
        let is_capture = (self.state.pieces_occ[opponent_color] & (1u64 << mv.to)) != 0 || mv.is_en_passant();
        if is_capture || mv.is_promotion() {
             // TODO: Could refine with SEE later (WinningCapture, LosingCapture, EqualCapture)
             return MoveCategory::Capture;
        }

        // 3. Checks (Tactical) - Check if the move puts the opponent's king in check
        // This requires applying the move and checking legality/check status of the resulting board.
        // This might be too slow to do for every move during categorization.
        // Alternative: Use MoveGen::gives_check(&self.state, mv) if such a function exists.
        // Placeholder: Assume checks are not categorized as Tactical for now due to performance concerns.
        // let next_state = self.state.apply_move_to_board(*mv);
        // if next_state.is_check(move_gen) { // is_check checks if the *current* player to move is in check
        //     return MoveCategory::Tactical;
        // }

        // 4. Forks (Tactical) - Placeholder
        // TODO: Implement fork detection (e.g., check if pawn/knight move attacks multiple valuable pieces)

        // 5. Killers (Killer) - Placeholder
        // TODO: Check if mv matches killer moves for the current ply (requires passing killer table context)
        // if is_killer(mv, ply, killers) { return MoveCategory::Killer; }

        // Otherwise, it's a quiet move
        MoveCategory::Quiet
    }

    /// Gets the next highest-priority unexplored move according to tactical categories
    /// (Check > Capture > Quiet) and removes it from the map.
    /// Advances the current priority category if necessary.
    pub fn get_next_move_to_explore(&mut self) -> Option<Move> {
        while let Some(current_cat) = self.current_priority_category {
            if let Some(moves_in_cat) = self.unexplored_moves_by_cat.get_mut(&current_cat) {
                // Pop moves from the end (highest MVV-LVA for captures, last for others)
                if let Some(mv) = moves_in_cat.pop() {
                    // If category becomes empty after pop, remove it and advance category
                    if moves_in_cat.is_empty() {
                        self.unexplored_moves_by_cat.remove(&current_cat);
                        self.advance_priority_category();
                    }
                    return Some(mv);
                } else {
                    // Category was already empty, remove it and advance
                    self.unexplored_moves_by_cat.remove(&current_cat);
                    self.advance_priority_category();
                    // Continue loop to check next category
                }
            } else {
                // Category key exists but no vector? Should not happen. Advance.
                self.advance_priority_category();
                // Continue loop
            }
        }
        None // No unexplored moves left in any category
    }

    /// Advances `current_priority_category` to the next non-empty one.
    fn advance_priority_category(&mut self) {
        if let Some(current_cat) = self.current_priority_category {
            // Find the next category in the sorted order
            let mut next_cat_found = false;
            // Iterate through simplified enum variants by discriminant value
            // Iterate through refined enum variants by discriminant value
            // Iterate through enum variants by discriminant value, starting from the next one
            for cat_num in (current_cat as usize + 1)..=(MoveCategory::Quiet as usize) { // Ensure range includes all categories
                // This unsafe transmute assumes a standard C-like enum layout and discriminant order
                let next_possible_cat = match cat_num {
                    0 => MoveCategory::Check,
                    1 => MoveCategory::Capture,
                    2 => MoveCategory::Quiet,
                    _ => continue, // Should not happen
                };
                 if self
                    .unexplored_moves_by_cat
                    .contains_key(&next_possible_cat)
                {
                    self.current_priority_category = Some(next_possible_cat);
                    next_cat_found = true;
                    break;
                }
            }
            if !next_cat_found {
                self.current_priority_category = None; // All categories exhausted
            }
        }
    }

    /// Backpropagates a simulation result through the tree.
    /// Updates visits and values for each node from the given leaf to the root.
    pub fn backpropagate(node: Rc<RefCell<MctsNode>>, value: f64) {
        let mut current_node_opt = Some(node);

        while let Some(current_node_rc) = current_node_opt {
            // Update statistics for the current node
            {
                let mut current_node = current_node_rc.borrow_mut();
                current_node.visits += 1;
                current_node.total_value += value;
            }

            // Move to parent if there is one
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

    /// Helper function to generate legal moves for a given state.
    /// Filters pseudo-legal moves by checking legality using apply_move_to_board.
    pub fn get_legal_moves(state: &Board, move_gen: &MoveGen) -> Vec<Move> {
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(state);
        let mut legal_moves = Vec::with_capacity(captures.len() + moves.len());
        for m in captures.into_iter().chain(moves.into_iter()) {
            // Use the existing apply_move_to_board method which returns a new Board state.
            // Ensure apply_move_to_board is public in board.rs or its defining module.
            // This check is inefficient if done repeatedly; ideally MoveGen provides legal moves.
            let next_state = state.apply_move_to_board(m); // Use actual method
            if next_state.is_legal(move_gen) {
                // Check legality of the resulting state
                legal_moves.push(m);
            }
        }
        legal_moves
    }
}

/// Determines if a node should be expanded (has unevaluated/unexplored moves)
/// or if selection should continue down the tree.
fn should_expand_not_select(node: &MctsNode) -> bool {
    // If the node hasn't been evaluated yet (Pesto value not calculated),
    // it's the target leaf for evaluation, not expansion yet.
    if node.nn_value.is_none() {
        return false; // Don't expand, evaluate this node first.
    }

    // If the node has been evaluated AND still has unexplored moves, it's ready for expansion.
    // Check num_legal_moves.is_some() to ensure categorization has happened.
    if node.num_legal_moves.is_some() && !node.unexplored_moves_by_cat.is_empty() {
         return true; // Expand this node.
    }

    // If evaluated and no unexplored moves left, continue selection.
    false
}

/// Selects a leaf node for expansion in MCTS.
pub fn select_leaf_for_expansion(
    root: Rc<RefCell<MctsNode>>,
    exploration_constant: f64,
) -> Rc<RefCell<MctsNode>> {
    let mut current = root;

    loop {
        // Check if this node is terminal or ready for expansion
        let should_expand;
        {
            let node_borrow = current.borrow();
            if node_borrow.terminal_or_mate_value.is_some() || node_borrow.is_terminal {
                break;
            }
            should_expand = should_expand_not_select(&node_borrow);
        }

        if should_expand {
            break; // This node has unexplored moves, stop here for expansion
        }

        // If no children yet but we decided not to expand, there's likely an issue
        let has_children = !current.borrow().children.is_empty();
        if !has_children {
            break; // Leaf node reached
        }

        // Select child with highest PUCT score
        let next = current
            .borrow()
            .select_best_explored_child(exploration_constant); // Uses PUCT now
        current = next;
    }

    current
}

// Note: Expansion logic is now split between MctsNode helpers and the main mcts_search loop.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    // use crate::eval::PestoEval; // Not strictly needed for these tests
    use crate::move_generation::MoveGen;
    // Using Move::from_uci for test moves
    use std::cell::RefCell;
    use std::rc::Rc;

    // Helper to create a node, categorize moves, and return the node + movegen
    fn setup_node_for_priority_test(fen: &str) -> (Rc<RefCell<MctsNode>>, MoveGen) {
        let board = Board::new_from_fen(fen);
        let move_gen = MoveGen::new();
        let node_rc = MctsNode::new_root(board, &move_gen);
        // We need to simulate the evaluation step to trigger categorization
        {
            let mut node = node_rc.borrow_mut();
            // Assign a dummy evaluation value and categorize
            // In the real search, this happens if mate isn't found and evaluation is done.
            if node.terminal_or_mate_value.is_none() || node.terminal_or_mate_value == Some(-999.0) {
                 node.nn_value = Some(0.5); // Assign a dummy evaluation value
                 node.categorize_and_store_moves(&move_gen);
            } else {
                // If node is already terminal, categorization won't happen,
                // which might be okay for some tests, but not for priority testing.
                // Panic here to indicate a bad test setup if the FEN is terminal.
                panic!("FEN provided for priority test is already terminal: {}", fen);
            }
        }
        (node_rc, move_gen)
    }

    #[test]
    fn test_move_priority_check_capture_quiet() {
        // Simpler test: just verify that if there are checks, they have priority over captures/quiet
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let (node_rc, _move_gen) = setup_node_for_priority_test(fen);

        let mut node = node_rc.borrow_mut();

        // Verify initial state after categorization - just check that priority is set correctly
        // If the system works, it should prioritize in this order: Check > Capture > Quiet
        if node.unexplored_moves_by_cat.contains_key(&MoveCategory::Check) && !node.unexplored_moves_by_cat[&MoveCategory::Check].is_empty() {
            assert_eq!(node.current_priority_category, Some(MoveCategory::Check), "Initial priority should be Check if checks are available");
        } else if node.unexplored_moves_by_cat.contains_key(&MoveCategory::Capture) && !node.unexplored_moves_by_cat[&MoveCategory::Capture].is_empty() {
            assert_eq!(node.current_priority_category, Some(MoveCategory::Capture), "Initial priority should be Capture if no checks but captures available");
        } else {
            assert_eq!(node.current_priority_category, Some(MoveCategory::Quiet), "Initial priority should be Quiet if only quiet moves available");
        }


        // Test that prioritization works - highest priority category should be selected
        assert!(node.current_priority_category.is_some(), "Should have a priority category set");
    }

    // --- More tests can be added below ---
}
