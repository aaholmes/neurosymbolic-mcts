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

/// Tracks how this node was created/evaluated for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeOrigin {
    #[default]
    Unknown,
    /// Tier 1: Solved by mate search or KOTH gate
    Gate,
    /// Tier 3: Standard neural network evaluation
    Neural,
}

impl NodeOrigin {
    pub fn to_color(&self) -> &'static str {
        match self {
            NodeOrigin::Gate => "firebrick1",
            NodeOrigin::Neural => "lightblue",
            NodeOrigin::Unknown => "white",
        }
    }

    pub fn to_label(&self) -> &'static str {
        match self {
            NodeOrigin::Gate => "T1:Gate",
            NodeOrigin::Neural => "T3:NN",
            NodeOrigin::Unknown => "?",
        }
    }
}

/// A node in the Monte Carlo Search Tree
#[derive(Debug)]
pub struct MctsNode {
    /// The chess position at this node
    pub state: Board,

    /// The move that led to this state (None for root)
    pub action: Option<Move>,

    /// Origin/tier of this node for visualization
    pub origin: NodeOrigin,

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
    /// Stores the final combined value ([-1, 1]) = tanh(v_logit + k * delta_M).
    pub nn_value: Option<f64>,
    /// Raw positional logit from NN (unbounded, before material adjustment + tanh).
    pub v_logit: Option<f64>,
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
    /// Q-init values for capture/promotion moves (MVV-LVA normalized to [-1, 1])
    pub tactical_values: HashMap<Move, f64>,
    /// Cached raw NN policy vector from InferenceServer (used by ensure_policy_evaluated)
    pub raw_nn_policy: Option<Vec<f32>>,
}

impl MctsNode {
    pub fn new_root(state: Board, move_gen: &MoveGen) -> Rc<RefCell<Self>> {
        let (is_checkmate, is_stalemate) = state.is_checkmate_or_stalemate(move_gen);
        let is_terminal = is_checkmate || is_stalemate;

        let initial_terminal_value = if is_stalemate {
            Some(0.0)
        } else if is_checkmate {
            Some(-1.0)
        } else {
            None
        };

        Rc::new(RefCell::new(Self {
            state,
            action: None,
            origin: NodeOrigin::Unknown,
            visits: 0,
            total_value: 0.0,
            total_value_squared: 0.0,
            parent: None,
            children: Vec::new(),
            terminal_or_mate_value: initial_terminal_value,
            mate_move: None,
            nn_value: None,
            v_logit: None,
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
            tactical_values: HashMap::new(),
            raw_nn_policy: None,
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
            Some(-1.0)
        } else {
            None
        };

        Rc::new(RefCell::new(Self {
            state: new_state,
            action: Some(action),
            origin: NodeOrigin::Unknown,
            visits: 0,
            total_value: 0.0,
            total_value_squared: 0.0,
            parent: Some(parent),
            children: Vec::new(),
            terminal_or_mate_value: initial_terminal_value,
            mate_move: None,
            nn_value: None,
            v_logit: None,
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
            tactical_values: HashMap::new(),
            raw_nn_policy: None,
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
        if self.action.is_some() && self.state.get_piece(self.action.unwrap().from).is_none() {
             return f64::NEG_INFINITY;
        }
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
        if self.action.is_some() && self.state.get_piece(self.action.unwrap().from).is_none() {
             return f64::NEG_INFINITY;
        }
        if self.visits == 0 {
            return f64::INFINITY;
        }

        // total_value is relative to side that just moved (parent's perspective during selection)
        let q_value = self.total_value / self.visits as f64;

        let exploration_term =
            exploration_constant * ((parent_visits as f64).ln() / self.visits as f64).sqrt();

        q_value + exploration_term
    }

    pub fn select_best_explored_child(&self, _move_gen: &MoveGen, exploration_constant: f64) -> Rc<RefCell<MctsNode>> {
        let parent_visits = self.visits;
        let parent_num_legal = self.num_legal_moves.unwrap_or(1);

        // Children are already legality-checked during expansion, so we just
        // pick the highest-PUCT child directly without re-validating.
        self.children.iter()
            .max_by(|a, b| {
                let puct_a = a.borrow().puct_value(parent_visits, parent_num_legal, exploration_constant);
                let puct_b = b.borrow().puct_value(parent_visits, parent_num_legal, exploration_constant);
                puct_a.partial_cmp(&puct_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .expect("select_best_explored_child: no children found!")
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
             if self.state.get_piece(mv.from).is_none() {
                 continue;
             }
             let opponent_color = if self.state.w_to_move { 1 } else { 0 };
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
        if self.state.get_piece(mv.from).is_none() {
            return MoveCategory::Quiet;
        }
        // Use gives_check to avoid cloning the board
        if self.state.gives_check(*mv, move_gen) {
             return MoveCategory::Check;
        }

        let opponent_color = if self.state.w_to_move { 1 } else { 0 };
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

    pub fn backpropagate(node: Rc<RefCell<MctsNode>>, mut value: f64) {
        let mut current_node_opt = Some(node);

        while let Some(current_node_rc) = current_node_opt {
            let next_parent = {
                let mut current_node = current_node_rc.borrow_mut();
                current_node.visits += 1;
                
                // value is relative to side to move at current_node
                // node.total_value is relative to side that just moved (parent's side)
                let reward = -value;
                current_node.total_value += reward;
                current_node.total_value_squared += reward * reward;
                
                // Flip value for parent (opponent's perspective)
                value = -value;
                
                current_node.parent.as_ref().and_then(|w| w.upgrade())
            };
            current_node_opt = next_parent;
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
    /// 
    /// # Arguments
    /// * `depth_limit` - Maximum tree depth to export
    /// * `min_visits` - Minimum visits for a node to be included (filters noise)
    /// 
    /// # Returns
    /// A DOT format string suitable for `dot -Tpng -o tree.png`
    pub fn export_dot(&self, depth_limit: usize, min_visits: u32) -> String {
        let mut output = String::from("digraph MCTS {\n");
        output.push_str("  rankdir=TB;\n");
        output.push_str("  node [shape=record, style=filled, fontname=\"Helvetica\", fontsize=10];\n");
        output.push_str("  edge [fontsize=8];\n");
        output.push_str("\n  // Legend\n");
        output.push_str("  subgraph cluster_legend {\n");
        output.push_str("    label=\"Legend\";\n");
        output.push_str("    style=dashed;\n");
        output.push_str("    leg_gate [label=\"Tier 1: Gate\\n(Mate/KOTH)\", fillcolor=firebrick1];\n");
        output.push_str("    leg_neural [label=\"Tier 3: Neural\", fillcolor=lightblue];\n");
        output.push_str("  }\n\n");
        
        let mut id_counter = 0;
        self.recursive_dot(&mut output, 0, &mut id_counter, 0, depth_limit, min_visits);
        
        output.push_str("}\n");
        output
    }

    fn recursive_dot(
        &self, 
        out: &mut String, 
        my_id: usize, 
        id_counter: &mut usize, 
        depth: usize, 
        limit: usize,
        min_visits: u32,
    ) {
        if depth > limit {
            return;
        }
        
        // Filter low-visit nodes (except root)
        if depth > 0 && self.visits < min_visits {
            return;
        }

        // Determine color based on origin
        let color = self.determine_node_color();
        
        // Format node label
        let move_str = self.action.map_or("Root".to_string(), |m| m.to_uci());
        let q_value = if self.visits > 0 { 
            self.total_value / self.visits as f64 
        } else { 
            0.0 
        };
        let eval_str = self.nn_value
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "—".to_string());
        
        let tier_label = self.origin.to_label();
        
        // Multi-line label with escape sequences for Graphviz
        let label = format!(
            "{{ {} | {} | N: {} | Q: {:.2} | V: {} }}",
            move_str,
            tier_label,
            self.visits,
            q_value,
            eval_str
        );

        // Write node definition
        out.push_str(&format!(
            "  {} [label=\"{}\", fillcolor={}];\n",
            my_id, label, color
        ));

        // Process real children
        for child in &self.children {
            let child_ref = child.borrow();
            
            // Skip low-visit children
            if child_ref.visits < min_visits && min_visits > 0 {
                continue;
            }

            // Safety check for move legality
            if let Some(mv) = child_ref.action {
                let mg = MoveGen::new();
                if self.state.get_piece(mv.from).is_none() || !self.state.apply_move_to_board(mv).is_legal(&mg) {
                    continue;
                }
            }
            
            *id_counter += 1;
            let child_id = *id_counter;
            
            // Edge label shows prior probability if available
            let edge_label = if let Some(mv) = child_ref.action {
                if let Some(prob) = self.move_priorities.get(&mv) {
                    format!(" [label=\"P:{:.2}\"]", prob)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            
            out.push_str(&format!("  {} -> {}{};\n", my_id, child_id, edge_label));
            child_ref.recursive_dot(out, child_id, id_counter, depth + 1, limit, min_visits);
        }

    }
    
    fn determine_node_color(&self) -> &'static str {
        // Priority: explicit origin > inferred from state
        match self.origin {
            NodeOrigin::Gate => "firebrick1",
            NodeOrigin::Neural => "lightblue",
            NodeOrigin::Unknown => {
                // Infer from node state
                if self.terminal_or_mate_value.is_some() && self.visits == 0 {
                    "firebrick1" // Solved by gate before expansion
                } else if self.nn_value.is_some() {
                    "lightblue" // Has neural evaluation
                } else {
                    "white" // Unexpanded
                }
            }
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
    move_gen: &MoveGen,
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
            .select_best_explored_child(move_gen, exploration_constant);
        current = next;
    }

    current
}