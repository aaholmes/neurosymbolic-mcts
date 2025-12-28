//! Defines the Node structure for the MCTS tree.

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::mcts::tactical::TacticalMove;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::f64;
use std::rc::{Rc, Weak};

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

    /// Total value accumulated through this node (Relative to side that just moved, [-visits, visits])
    pub total_value: f64,
    /// Sum of squared values accumulated through this node (for variance calculation).
    pub total_value_squared: f64,

    // --- Evaluation / Mate Status ---
    /// Stores the exact value if determined by terminal state check or mate search (1.0 for White Win, -1.0 for White Loss).
    /// Also used as a flag to indicate if mate search has been performed. None = not checked, Some(-999.0) = checked, no mate.
    pub terminal_or_mate_value: Option<f64>,
    /// If mate search found a mate, stores the mating move
    pub mate_move: Option<Move>,
    /// Stores the value from the evaluation (Pesto or NN) ([-1, 1] for White) when node is first evaluated.
    pub nn_value: Option<f64>, 
    /// The material confidence scalar (k) predicted by the neural network for this node.
    /// Used for extrapolating values of child tactical nodes.
    pub k_val: f32,
    /// Neural network prior probability for this move (from parent's perspective)
    pub prior_probability: f32,

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
    /// Shadow priors for tactical moves that were not the best PV (logit space or raw values)
    pub shadow_priors: HashMap<Move, f64>,
    /// Whether Tier 2 tactical resolution (QS) has been performed for this node
    pub tactical_resolution_done: bool,
    /// Whether this node was created as part of a tactical PV graft
    pub is_tactical_node: bool,
}

impl MctsNode {
    /// Creates a new root node for MCTS
    pub fn new_root(state: Board, move_gen: &MoveGen) -> Rc<RefCell<Self>> {
        let (is_checkmate, is_stalemate) = state.is_checkmate_or_stalemate(move_gen);
        let is_terminal = is_checkmate || is_stalemate;

        let initial_terminal_value = if is_stalemate {
            Some(0.0)
        } else if is_checkmate {
            Some(if state.w_to_move { -1.0 } else { 1.0 })
        } else {
            None
        };

        Rc::new(RefCell::new(Self {
            state,
            action: None,
            visits: 0,
            total_value: 0.0,
            total_value_squared: 0.0,
            parent: None,
            children: Vec::new(),
            terminal_or_mate_value: initial_terminal_value,
            mate_move: None,
            nn_value: None,
            k_val: 0.5, // Default confidence
            prior_probability: 0.0,
            unexplored_moves_by_cat: HashMap::new(),
            current_priority_category: None,
            num_legal_moves: None,
            is_terminal,
            tactical_moves: None,
            tactical_moves_explored: HashSet::new(),
            policy_evaluated: false,
            move_priorities: HashMap::new(),
            shadow_priors: HashMap::new(),
            tactical_resolution_done: false,
            is_tactical_node: false,
        }))
    }

    /// Creates a new child node.
    pub fn new_child(
        parent: Weak<RefCell<MctsNode>>,
        action: Move,
        new_state: Board,
        move_gen: &MoveGen,
    ) -> Rc<RefCell<Self>> {
        let (is_checkmate, is_stalemate) = new_state.is_checkmate_or_stalemate(move_gen);
        let is_terminal = is_checkmate || is_stalemate;

        let initial_terminal_value = if is_stalemate {
            Some(0.0)
        } else if is_checkmate {
            Some(if new_state.w_to_move { -1.0 } else { 1.0 })
        } else {
            None
        };

        Rc::new(RefCell::new(Self {
            state: new_state,
            action: Some(action),
            visits: 0,
            total_value: 0.0,
            total_value_squared: 0.0,
            parent: Some(parent),
            children: Vec::new(),
            terminal_or_mate_value: initial_terminal_value,
            mate_move: None,
            nn_value: None,
            k_val: 0.5, // Default confidence
            prior_probability: 0.0,
            unexplored_moves_by_cat: HashMap::new(),
            current_priority_category: None,
            num_legal_moves: None,
            is_terminal,
            tactical_moves: None,
            tactical_moves_explored: HashSet::new(),
            policy_evaluated: false,
            move_priorities: HashMap::new(),
            shadow_priors: HashMap::new(),
            tactical_resolution_done: false,
            is_tactical_node: false,
        }))
    }

    pub fn is_game_terminal(&self) -> bool {
        self.is_terminal
    }

    pub fn is_evaluated_or_terminal(&self) -> bool {
        self.nn_value.is_some() || self.terminal_or_mate_value.is_some()
    }

    pub fn is_fully_explored(&self) -> bool {
        self.num_legal_moves.is_some() && self.unexplored_moves_by_cat.is_empty()
    }

    pub fn puct_value(&self, parent_visits: u32, parent_num_legal_moves: usize, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            let prior_p = if parent_num_legal_moves > 0 {
                1.0 / parent_num_legal_moves as f64
            } else {
                0.0
            };
            let u_value = exploration_constant * prior_p * (parent_visits as f64).sqrt();
            return u_value;
        }

        // total_value is relative to side that just moved (parent's perspective during selection)
        let q_value = self.total_value / self.visits as f64;

        let prior_p = if parent_num_legal_moves > 0 {
            1.0 / parent_num_legal_moves as f64
        } else {
            0.0
        };
        let exploration_term = exploration_constant
            * prior_p
            * (parent_visits as f64).sqrt()
            / (1.0 + self.visits as f64);

        q_value + exploration_term
    }

    pub fn uct_value(&self, parent_visits: u32, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }

        // total_value is relative to side that just moved (parent's perspective during selection)
        let q_value = self.total_value / self.visits as f64;

        let exploration_term =
            exploration_constant * ((parent_visits as f64).ln() / self.visits as f64).sqrt();

        q_value + exploration_term
    }

    pub fn select_best_explored_child(&self, exploration_constant: f64) -> Rc<RefCell<MctsNode>> {
        let parent_visits = self.visits;
        let parent_num_legal = self.num_legal_moves.unwrap_or(1);

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

    pub fn categorize_and_store_moves(
        &mut self,
        move_gen: &MoveGen,
    ) {
        if self.num_legal_moves.is_some() {
             return;
        }

        let legal_moves = MctsNode::get_legal_moves(&self.state, move_gen);
        self.num_legal_moves = Some(legal_moves.len());

        let mut categorized_moves: HashMap<MoveCategory, Vec<Move>> = HashMap::new();

        let mut capture_scores: HashMap<Move, i32> = HashMap::new();
        for mv in &legal_moves {
             let opponent_color = !self.state.w_to_move as usize;
             let is_capture = (self.state.pieces_occ[opponent_color] & (1u64 << mv.to)) != 0 || mv.is_en_passant();
             if is_capture || mv.is_promotion() {
                 capture_scores.insert(*mv, move_gen.mvv_lva(&self.state, mv.from, mv.to));
             }
        }

        for mv in legal_moves {
            let category = self.categorize_move(&mv, move_gen);
            categorized_moves.entry(category).or_default().push(mv);
        }

        for (category, moves) in categorized_moves.iter_mut() {
            match category {
                MoveCategory::Check | MoveCategory::Quiet => {}
                MoveCategory::Capture => {
                    moves.sort_unstable_by_key(|mv| {
                        std::cmp::Reverse(capture_scores.get(mv).cloned().unwrap_or(0))
                    });
                }
            }
        }

        let mut sorted_categories: Vec<_> = categorized_moves.keys().cloned().collect();
        sorted_categories.sort();

        self.unexplored_moves_by_cat = categorized_moves;
        self.current_priority_category = sorted_categories.first().cloned();
    }

    fn categorize_move(&self, mv: &Move, move_gen: &MoveGen) -> MoveCategory {
        let next_state = self.state.apply_move_to_board(*mv);
        if next_state.is_check(move_gen) {
             return MoveCategory::Check;
        }

        let opponent_color = !self.state.w_to_move as usize;
        let is_capture = (self.state.pieces_occ[opponent_color] & (1u64 << mv.to)) != 0 || mv.is_en_passant();
        if is_capture || mv.is_promotion() {
             return MoveCategory::Capture;
        }

        MoveCategory::Quiet
    }

    pub fn get_next_move_to_explore(&mut self) -> Option<Move> {
        while let Some(current_cat) = self.current_priority_category {
            if let Some(moves_in_cat) = self.unexplored_moves_by_cat.get_mut(&current_cat) {
                if let Some(mv) = moves_in_cat.pop() {
                    if moves_in_cat.is_empty() {
                        self.unexplored_moves_by_cat.remove(&current_cat);
                        self.advance_priority_category();
                    }
                    return Some(mv);
                } else {
                    self.unexplored_moves_by_cat.remove(&current_cat);
                    self.advance_priority_category();
                }
            } else {
                self.advance_priority_category();
            }
        }
        None
    }

    fn advance_priority_category(&mut self) {
        if let Some(current_cat) = self.current_priority_category {
            for cat_num in (current_cat as usize + 1)..=(MoveCategory::Quiet as usize) {
                let next_possible_cat = match cat_num {
                    0 => MoveCategory::Check,
                    1 => MoveCategory::Capture,
                    2 => MoveCategory::Quiet,
                    _ => continue,
                };
                 if self
                    .unexplored_moves_by_cat
                    .contains_key(&next_possible_cat)
                {
                    self.current_priority_category = Some(next_possible_cat);
                    return;
                }
            }
            self.current_priority_category = None;
        }
    }

    pub fn backpropagate(node: Rc<RefCell<MctsNode>>, value: f64) {
        let mut current_node_opt = Some(node);

        while let Some(current_node_rc) = current_node_opt {
            {
                let mut current_node = current_node_rc.borrow_mut();
                current_node.visits += 1;
                current_node.total_value += value;
            }

            current_node_opt = {
                let current_node = current_node_rc.borrow();
                if let Some(parent_weak) = &current_node.parent {
                    parent_weak.upgrade()
                } else {
                    None
                }
            };
        }
    }

    pub fn get_legal_moves(state: &Board, move_gen: &MoveGen) -> Vec<Move> {
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(state);
        let mut legal_moves = Vec::with_capacity(captures.len() + moves.len());
        for m in captures.into_iter().chain(moves.into_iter()) {
            let next_state = state.apply_move_to_board(m);
            if next_state.is_legal(move_gen) {
                legal_moves.push(m);
            }
        }
        legal_moves
    }

    /// Exports the MCTS tree to a Graphviz DOT string for visualization.
    pub fn export_dot(&self, depth_limit: usize) -> String {
        let mut output = String::from("digraph MCTS {\n");
        output.push_str("  node [shape=record, style=filled, fontname=\"Arial\"];\n");
        
        let mut id_counter = 0;
        self.recursive_dot(&mut output, 0, &mut id_counter, 0, depth_limit);
        
        output.push_str("}");
        output
    }

    fn recursive_dot(&self, out: &mut String, my_id: usize, id_counter: &mut usize, depth: usize, limit: usize) {
        if depth > limit { return; }

        // 1. Determine Color
        // Red (firebrick1): Tier 1 Solution (Mate/KOTH found). Check terminal/mate value.
        // Important: Gate solutions have visits == 0 (solved before simulation).
        // Standard solved nodes (visits > 0) are also red but maybe a different shade? 
        // Instructions say: "Tier 1 ONLY if terminal value and visits == 0".
        // Actually, if visits > 0 and solved, it's just a solved node. 
        // Let's stick to the instruction: Red if Tier 1 Gate (visits == 0 && terminal value).
        
        // However, if we found mate deep in the tree, terminal_or_mate_value is set.
        // If it's a leaf node that we just evaluated as mate, visits might be 0.
        // Let's implement logic:
        
        let color = if self.terminal_or_mate_value.is_some() && self.visits == 0 {
            "firebrick1" // Tier 1 Gate (Solved immediately)
        } else if self.is_tactical_node {
            "gold"       // Tier 2 Graft
        } else {
            "lightblue"  // Tier 3 / Standard
        };

        // 2. Format Label
        let move_str = self.action.map_or("Root".to_string(), |m| m.to_uci());
        let val = if self.visits > 0 { self.total_value / self.visits as f64 } else { 0.0 };
        // Show Visits (N), Value (Q), and maybe Evaluation (NN/Pesto)
        let eval_str = if let Some(ev) = self.nn_value { format!("{:.2}", ev) } else { "?".to_string() };
        
        let label = format!("{{ {} | N:{} | Q:{:.2} | Eval:{} }}", 
            move_str, self.visits, val, eval_str);

        // 3. Write Node Definition
        out.push_str(&format!("  {} [label=\"{}\", fillcolor={}];\n", my_id, label, color));

        // 4. Handle Real Children
        for child in &self.children {
            *id_counter += 1;
            let child_id = *id_counter;
            
            out.push_str(&format!("  {} -> {};\n", my_id, child_id));
            child.borrow().recursive_dot(out, child_id, id_counter, depth + 1, limit);
        }

        // 5. Handle Shadow Priors (Ghost Nodes)
        for (mv, score) in &self.shadow_priors {
            // Need unique IDs for ghosts too
            *id_counter += 1;
            let ghost_id = *id_counter;
            
            let ghost_label = format!("{{ {} | QS:{:.2} | (Refuted) }}", mv.to_uci(), score);
            
            // Write Ghost Node
            out.push_str(&format!("  {} [label=\"{}\", style=dashed, fillcolor=lightgrey];\n", ghost_id, ghost_label));
            // Write Dashed Edge
            out.push_str(&format!("  {} -> {} [style=dashed];\n", my_id, ghost_id));
        }
    }
}

fn should_expand_not_select(node: &MctsNode) -> bool {
    if node.nn_value.is_none() {
        return false;
    }

    if node.num_legal_moves.is_some() && !node.unexplored_moves_by_cat.is_empty() {
         return true;
    }

    false
}

pub fn select_leaf_for_expansion(
    root: Rc<RefCell<MctsNode>>,
    exploration_constant: f64,
) -> Rc<RefCell<MctsNode>> {
    let mut current = root;

    loop {
        let should_expand;
        {
            let node_borrow = current.borrow();
            if node_borrow.terminal_or_mate_value.is_some() || node_borrow.is_terminal {
                break;
            }
            should_expand = should_expand_not_select(&node_borrow);
        }

        if should_expand {
            break;
        }

        let has_children = !current.borrow().children.is_empty();
        if !has_children {
            break;
        }

        let next = current
            .borrow()
            .select_best_explored_child(exploration_constant);
        current = next;
    }

    current
}