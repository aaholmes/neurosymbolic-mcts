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
use crate::mcts::inference_server::InferenceServer;
use crate::mcts::node::{MctsNode, NodeOrigin};
use crate::mcts::search_logger::{GateReason, SearchLogger};
use crate::mcts::selection::select_child_with_tactical_priority;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::search::koth_center_in_n_counted;
use crate::search::mate_search;
use crate::search::{
    forced_cap1_pesto_balance, forced_ext_pesto_balance_counted, forced_principal_exchange,
};
use crate::search::{koth_best_move, koth_center_in_n};
use crate::transposition::TranspositionTable;
use crossbeam_channel::Receiver;
use rand_distr::{Distribution, Gamma};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Material-trust scalar `k` used when no NN is available (classical fallback)
/// and as the initial `k` before an NN result arrives. Matches the network's
/// init: `0.47 * softplus(0) = 0.47 * ln(2) ≈ 0.326`.
pub const CLASSICAL_K: f32 = 0.326;

/// Which quiescence search variant to use for computing ΔM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QSearchVariant {
    /// Principal exchange: follow the single best MVV-LVA capture at each node.
    /// A straight line (not a tree). ~1–5 nodes. GPU-friendly.
    PrincipalExchange,
    /// Cap=1: best MVV-LVA capture + one checking move per side + full check
    /// evasions. ~5–90 nodes.
    Cap1,
    /// Full extended q-search: captures + checks + forks + null-move threat
    /// detection + mystery-square recapture. ~50–500+ nodes.
    Extended,
}

impl QSearchVariant {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pe" | "principal-exchange" => Some(QSearchVariant::PrincipalExchange),
            "cap1" => Some(QSearchVariant::Cap1),
            "extended" | "ext" => Some(QSearchVariant::Extended),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            QSearchVariant::PrincipalExchange => "principal-exchange",
            QSearchVariant::Cap1 => "cap1",
            QSearchVariant::Extended => "extended",
        }
    }
}

/// The network's two quiescence inputs for a position: the quiescence score in
/// pawns and whether that search finished, with the search's size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QsearchInputs {
    pub score: f32,
    pub completed: bool,
    pub nodes: u32,
    pub depth: u8,
}

/// The quiescence inputs the search gives the network for `board`, and that
/// training data records for it. With the material score off no search runs:
/// the network gets a score of 0 and a finished flag.
pub fn qsearch_inputs(
    board: &Board,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    variant: QSearchVariant,
    enable_material: bool,
) -> QsearchInputs {
    if !enable_material {
        return QsearchInputs {
            score: 0.0,
            completed: true,
            nodes: 0,
            depth: 0,
        };
    }
    let mut stack = BoardStack::with_board(board.clone());
    let (score, completed, nodes, depth) = match variant {
        QSearchVariant::PrincipalExchange => {
            let (s, n) = forced_principal_exchange(&mut stack, move_gen, pesto);
            (s, true, n, 0) // a single line of captures always finishes
        }
        QSearchVariant::Cap1 => forced_cap1_pesto_balance(&mut stack, move_gen, pesto),
        QSearchVariant::Extended => forced_ext_pesto_balance_counted(&mut stack, move_gen, pesto),
    };
    QsearchInputs {
        score,
        completed,
        nodes,
        depth,
    }
}

/// Configuration for tactical-first MCTS search
#[derive(Debug, Clone)]
pub struct TacticalMctsConfig {
    pub max_iterations: u32,
    pub time_limit: Duration,
    pub mate_search_depth: i32,
    /// Max mate-in-N for exhaustive (all-moves) mate search. Depths above this use checks-only.
    /// Default 0 = all checks-only. Set to 2 for exhaustive mate-in-1/2, checks-only mate-in-3+.
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
    /// Maximum depth for KOTH center-in-N search (default 3)
    pub koth_depth: u8,
    /// Enable material integration in value: V = tanh(V_logit + k·ΔM)
    /// When false (pure AlphaZero): V = tanh(V_logit), no Q-search material
    pub enable_material_value: bool,
    /// Which q-search variant to use for ΔM computation
    pub qsearch_variant: QSearchVariant,
    /// Dirichlet noise alpha (0.0 = disabled, 0.3 for chess training)
    pub dirichlet_alpha: f64,
    /// Dirichlet noise epsilon (0.0 = disabled, 0.25 for chess training)
    pub dirichlet_epsilon: f64,
    /// Shuffle children after expansion to break move-generation-order bias
    /// Enable for training (self-play), disable for deterministic analysis/evaluation
    pub randomize_move_order: bool,
    /// PeSTO evaluator for q-search (constructed once, reused across all searches)
    pub pesto: PestoEval,
    /// Leaves selected per iteration before waiting for the network. Above 1,
    /// virtual loss steers the selections apart and their network requests are
    /// sent together, so one game keeps several positions waiting. 1 = one
    /// leaf at a time (standard MCTS).
    pub leaves_per_iteration: u32,
    /// With tier 1 off, give a leaf whose side to move has a mate this search
    /// finds the exact value +1 instead of a network evaluation. (Tier 1 has
    /// its own mate search.)
    pub leaf_mate: crate::egtb::MateSearch,
    /// Syzygy tables: a leaf they cover gets its exact value instead of a
    /// network evaluation. Not used in King of the Hill, where the tables'
    /// results do not hold.
    pub tablebase: Option<crate::egtb::Tablebase>,
}

impl Default for TacticalMctsConfig {
    fn default() -> Self {
        TacticalMctsConfig {
            max_iterations: 1000,
            time_limit: Duration::from_secs(5),
            mate_search_depth: 5,
            exhaustive_mate_depth: 0,
            exploration_constant: 1.414,
            use_neural_policy: true,
            inference_server: None,
            logger: None,
            // All tiers enabled by default
            enable_tier1_gate: true,
            enable_tier3_neural: true,
            enable_koth: false,
            koth_depth: 3,
            enable_material_value: true,
            qsearch_variant: QSearchVariant::Extended,
            dirichlet_alpha: 0.0,
            dirichlet_epsilon: 0.0,
            randomize_move_order: false,
            pesto: PestoEval::new(),
            leaves_per_iteration: 1,
            leaf_mate: crate::egtb::MateSearch::OFF,
            tablebase: None,
        }
    }
}

/// Welford's online algorithm for streaming mean/variance of call durations.
#[derive(Debug, Clone, Default)]
pub struct TimingAccumulator {
    pub count: u64,
    pub total: Duration,
    mean: f64, // running mean in microseconds
    m2: f64,   // running sum of squared deviations
}

impl TimingAccumulator {
    pub fn record(&mut self, elapsed: Duration) {
        self.total += elapsed;
        self.count += 1;
        let x = elapsed.as_secs_f64() * 1_000_000.0; // microseconds
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn mean_us(&self) -> f64 {
        self.mean
    }

    pub fn std_us(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count - 1) as f64).sqrt()
        }
    }

    pub fn merge(&mut self, other: &TimingAccumulator) {
        if other.count == 0 {
            return;
        }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean = (self.mean * self.count as f64 + other.mean * other.count as f64)
            / combined_count as f64;
        self.m2 += other.m2
            + delta * delta * (self.count as f64 * other.count as f64) / combined_count as f64;
        self.count = combined_count;
        self.total += other.total;
    }
}

/// Welford's online algorithm for streaming mean/variance of arbitrary f64 values.
#[derive(Debug, Clone, Default)]
pub struct StatsAccumulator {
    pub count: u64,
    pub total: f64,
    mean: f64,
    m2: f64,
}

impl StatsAccumulator {
    pub fn record(&mut self, value: f64) {
        self.total += value;
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn mean_val(&self) -> f64 {
        self.mean
    }

    pub fn std_val(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count - 1) as f64).sqrt()
        }
    }

    pub fn merge(&mut self, other: &StatsAccumulator) {
        if other.count == 0 {
            return;
        }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean = (self.mean * self.count as f64 + other.mean * other.count as f64)
            / combined_count as f64;
        self.m2 += other.m2
            + delta * delta * (self.count as f64 * other.count as f64) / combined_count as f64;
        self.count = combined_count;
        self.total += other.total;
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
    /// Leaves given an exact value by `leaf_mate` or the tablebase
    pub exact_leaf_values: u32,

    // Per-operation timing
    pub koth_timing: TimingAccumulator,
    pub mate_search_timing: TimingAccumulator,
    pub qsearch_timing: TimingAccumulator,
    pub nn_timing: TimingAccumulator,

    // Per-operation node counts
    pub mate_search_nodes: StatsAccumulator,
    pub qsearch_nodes: StatsAccumulator,
    pub koth_nodes: StatsAccumulator,

    // Q-search depth distribution
    pub qsearch_depth: StatsAccumulator,
    pub qsearch_completed_count: u64,
    pub qsearch_hit_limit_count: u64,
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
        s.push('\n');
        s.push_str(r"\toprule");
        s.push('\n');
        s.push_str(r"Metric & Value \\");
        s.push('\n');
        s.push_str(r"\midrule");
        s.push('\n');
        s.push_str(&format!(r"Iterations & {} \\", self.iterations));
        s.push('\n');
        s.push_str(&format!(r"Nodes Expanded & {} \\", self.nodes_expanded));
        s.push('\n');
        s.push_str(&format!(r"NN Evaluations & {} \\", self.nn_evaluations));
        s.push('\n');
        s.push_str(&format!(
            r"NN Saved (Tier 1) & {} \\",
            self.nn_saved_by_tier1
        ));
        s.push('\n');
        s.push_str(&format!(
            r"Reduction & {:.1}\% \\",
            self.nn_reduction_percentage()
        ));
        s.push('\n');
        s.push_str(r"\bottomrule");
        s.push('\n');
        s.push_str(r"\end{tabular}");
        s.push('\n');
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
        // Check if opponent already occupies a center square (instant loss)
        let (w_won, b_won) = board.is_koth_win();
        let opponent_won = (board.w_to_move && b_won) || (!board.w_to_move && w_won);
        if opponent_won {
            stats.tier1_solutions += 1;
            stats.search_time = start_time.elapsed();
            let root_node = MctsNode::new_root(board, move_gen);
            root_node.borrow_mut().terminal_or_mate_value = Some(-1.0);
            root_node.borrow_mut().terminal_distance = Some(0);
            root_node.borrow_mut().is_terminal = true;
            return (None, stats, root_node);
        }

        if let Some(dist) = koth_center_in_n(&board, move_gen, config.koth_depth) {
            if let Some(best) = koth_best_move(&board, move_gen) {
                if let Some(log) = logger {
                    log.log_tier1_gate(&GateReason::KothWin { distance: dist }, Some(best));
                }
                stats.tier1_solutions += 1;
                stats.search_time = start_time.elapsed();
                let root_node = MctsNode::new_root(board, move_gen);
                root_node.borrow_mut().origin = NodeOrigin::Gate;
                root_node.borrow_mut().mate_move = Some(best);
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

        let (mate_score, mate_move, nodes) = mate_search(
            &board,
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

    let mut iteration = 0;
    while iteration < config.max_iterations {
        if start_time.elapsed() > config.time_limit {
            break;
        }

        iteration += run_mcts_iteration(
            &root_node,
            move_gen,
            &config,
            transposition_table,
            &mut stats,
            logger,
            iteration,
        );

        if iteration % 100 == 0 && start_time.elapsed() > config.time_limit {
            break;
        }

        if should_early_terminate(&root_node, &config) {
            break;
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

/// Run one MCTS iteration: select a leaf, evaluate it, expand it if it is a
/// fresh non-terminal node, then backpropagate. With `leaves_per_iteration`
/// above 1, several leaves are selected under virtual loss and their network
/// requests are sent before waiting for any answer. Returns the number of
/// simulations run. Shared by all three search entrypoints to keep the tier
/// ladder byte-for-byte identical.
#[allow(clippy::too_many_arguments)]
fn run_mcts_iteration(
    root_node: &Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    config: &TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
    iteration: u32,
) -> u32 {
    let leaves = config
        .leaves_per_iteration
        .clamp(1, config.max_iterations.saturating_sub(iteration).max(1));
    if leaves == 1 {
        run_single_leaf(
            root_node,
            move_gen,
            config,
            transposition_table,
            stats,
            logger,
            iteration,
        );
        return 1;
    }

    // 1. Select the leaves, each under virtual loss from the ones before it
    let mut selected: Vec<Rc<RefCell<MctsNode>>> = Vec::with_capacity(leaves as usize);
    for _ in 0..leaves {
        let leaf = select_leaf_node(root_node.clone(), move_gen, config, stats, logger);
        MctsNode::apply_virtual_loss(&leaf);
        selected.push(leaf);
    }
    for leaf in &selected {
        MctsNode::remove_virtual_loss(leaf);
    }

    // 2. Start each distinct leaf's evaluation; network requests go out without waiting
    let mut unique: Vec<Rc<RefCell<MctsNode>>> = Vec::new();
    for leaf in &selected {
        if !unique.iter().any(|u| Rc::ptr_eq(u, leaf)) {
            unique.push(leaf.clone());
        }
    }
    let started: Vec<LeafEval> = unique
        .iter()
        .map(|leaf| {
            begin_leaf_evaluation(
                leaf.clone(),
                move_gen,
                config,
                transposition_table,
                stats,
                logger,
            )
        })
        .collect();

    // 3. Wait for the answers
    let values: Vec<f64> = unique
        .iter()
        .zip(started)
        .map(|(leaf, ev)| match ev {
            LeafEval::Done(v) => v,
            LeafEval::Pending(p) => finish_leaf_evaluation(leaf.clone(), p, config, stats, logger),
        })
        .collect();

    // 4. Expand and back up every selection (a leaf selected twice is backed up twice)
    for leaf in &selected {
        let i = unique.iter().position(|u| Rc::ptr_eq(u, leaf)).unwrap();
        if !leaf.borrow().is_game_terminal()
            && leaf.borrow().visits == 0
            && leaf.borrow().terminal_or_mate_value.is_none()
        {
            evaluate_and_expand_node(leaf.clone(), move_gen, stats, config, logger);
        }
        MctsNode::backpropagate(leaf.clone(), values[i]);
    }
    stats.iterations = iteration + leaves;
    leaves
}

/// One leaf per iteration (standard MCTS).
#[allow(clippy::too_many_arguments)]
fn run_single_leaf(
    root_node: &Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    config: &TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
    iteration: u32,
) {
    if let Some(log) = logger {
        log.log_iteration_start(iteration + 1);
    }

    let leaf = select_leaf_node(root_node.clone(), move_gen, config, stats, logger);
    let value = evaluate_leaf_node(
        leaf.clone(),
        move_gen,
        config,
        transposition_table,
        stats,
        logger,
    );

    if !leaf.borrow().is_game_terminal()
        && leaf.borrow().visits == 0
        && leaf.borrow().terminal_or_mate_value.is_none()
    {
        evaluate_and_expand_node(leaf.clone(), move_gen, stats, config, logger);
    }

    MctsNode::backpropagate(leaf, value);
    stats.iterations = iteration + 1;

    if let Some(log) = logger {
        let best = select_best_move_from_root(root_node.clone(), move_gen);
        log.log_iteration_summary(iteration + 1, best, root_node.borrow().visits);
    }
}

/// Early-termination check on forced win/loss. Driven by the Tier-1 gate
/// (mate/KOTH), which sets terminal_or_mate_value — not KOTH alone, so proven
/// mates short-circuit in standard chess too. Returns true if the search loop
/// should break. No-op (returns false) when the Tier-1 gate is disabled.
fn should_early_terminate(root_node: &Rc<RefCell<MctsNode>>, config: &TacticalMctsConfig) -> bool {
    if !config.enable_tier1_gate {
        return false;
    }
    let root_ref = root_node.borrow();
    let has_win_in_1 = root_ref
        .children
        .iter()
        .any(|c| c.borrow().terminal_or_mate_value.is_some_and(|v| v < -0.5));
    if has_win_in_1 {
        return true;
    }
    let has_forced_win = root_ref.children.iter().any(|c| {
        let cr = c.borrow();
        cr.visits > 0 && (cr.total_value / cr.visits as f64) > 0.99
    });
    if has_forced_win {
        let all_visited = root_ref.children.iter().all(|c| c.borrow().visits > 0);
        if all_visited {
            return true;
        }
    }
    // Early termination: all moves are forced losses
    let all_visited = root_ref.children.iter().all(|c| c.borrow().visits > 0);
    let all_losing = !root_ref.children.is_empty()
        && root_ref
            .children
            .iter()
            .all(|c| c.borrow().terminal_or_mate_value.is_some_and(|v| v > 0.0));
    all_visited && all_losing
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

/// Result of starting a leaf evaluation.
enum LeafEval {
    /// The value is known (terminal, Tier 1, cached, or no network in use).
    Done(f64),
    /// A network request has been sent; finish with `finish_leaf_evaluation`.
    Pending(PendingNn),
}

struct PendingNn {
    receiver: Receiver<Option<(Vec<f32>, f32, f32)>>,
    /// The legal moves sent with the request; the answer's priors follow this order.
    moves: Vec<Move>,
    q_result: f32,
    started: Instant,
}

fn evaluate_leaf_node(
    node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    config: &TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) -> f64 {
    match begin_leaf_evaluation(
        node.clone(),
        move_gen,
        config,
        transposition_table,
        stats,
        logger,
    ) {
        LeafEval::Done(v) => v,
        LeafEval::Pending(p) => finish_leaf_evaluation(node, p, config, stats, logger),
    }
}

/// Tiers 1 and 2 for a leaf, then send its network request without waiting.
fn begin_leaf_evaluation(
    node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    config: &TacticalMctsConfig,
    transposition_table: &mut TranspositionTable,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) -> LeafEval {
    let mut node_ref = node.borrow_mut();
    let board = &node_ref.state;

    if let Some(cached_value) = node_ref.terminal_or_mate_value {
        // A cached terminal/mate value is always exact (in [-1, 1]); reuse it
        // and skip the NN entirely.
        stats.nn_saved_by_tier1 += 1;
        return LeafEval::Done(cached_value);
    }

    if config.enable_koth && config.enable_tier1_gate {
        let koth_start = Instant::now();
        // Opponent already on center = STM loses (KOTH terminal)
        let (w_won, b_won) = board.is_koth_win();
        let opponent_won = (board.w_to_move && b_won) || (!board.w_to_move && w_won);
        if opponent_won {
            stats.koth_timing.record(koth_start.elapsed());
            node_ref.terminal_or_mate_value = Some(-1.0);
            node_ref.terminal_distance = Some(0);
            node_ref.is_terminal = true;
            stats.nn_saved_by_tier1 += 1;
            stats.tier1_solutions += 1;
            return LeafEval::Done(-1.0);
        }

        let (koth_result, koth_nodes) =
            koth_center_in_n_counted(board, move_gen, config.koth_depth);
        stats.koth_timing.record(koth_start.elapsed());
        stats.koth_nodes.record(koth_nodes as f64);
        if let Some(dist) = koth_result {
            if let Some(log) = logger {
                log.log_koth_check(true, Some(dist as u32));
            }
            node_ref.terminal_or_mate_value = Some(1.0);
            node_ref.terminal_distance = Some(dist);
            stats.nn_saved_by_tier1 += 1;
            stats.tier1_solutions += 1;
            return LeafEval::Done(1.0);
        }
    }

    // The root is always searched: marking it solved would leave no move.
    let is_root = node_ref.parent.is_none();

    if let Some(tb) = config
        .tablebase
        .as_ref()
        .filter(|_| !config.enable_koth && !is_root)
    {
        if let Some(v) = tb.probe_value(board) {
            let v = v as f64;
            node_ref.terminal_or_mate_value = Some(v);
            node_ref.origin = NodeOrigin::Gate;
            stats.exact_leaf_values += 1;
            return LeafEval::Done(v);
        }
    }

    // Tier 1 has its own mate search; without it, `leaf_mate` is used if on.
    let leaf_mate = !config.enable_tier1_gate && config.leaf_mate.is_on() && !is_root;
    let (mate_search_depth, exhaustive_mate_depth) = if config.enable_tier1_gate {
        (config.mate_search_depth, config.exhaustive_mate_depth)
    } else if leaf_mate {
        (config.leaf_mate.depth, config.leaf_mate.exhaustive)
    } else {
        (0, 0)
    };

    if mate_search_depth > 0 {
        if let Some((mate_depth, _mate_move)) =
            transposition_table.probe_mate(board, mate_search_depth)
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
                    // A leaf search records only that a mate exists, not its length.
                    if !leaf_mate {
                        node_ref.terminal_distance = Some(mate_depth.unsigned_abs() as u8);
                    }
                    node_ref.origin = NodeOrigin::Gate;
                    stats.nn_saved_by_tier1 += 1;
                    if leaf_mate {
                        stats.exact_leaf_values += 1;
                    }
                    return LeafEval::Done(mate_value);
                }
            }
        } else if leaf_mate {
            // Only whether a mate exists matters here, not its move or length.
            stats.tt_mate_misses += 1;
            let mate_start = Instant::now();
            let (found, nodes) = config.leaf_mate.search(board, move_gen);
            stats.mate_search_timing.record(mate_start.elapsed());
            stats.mate_search_nodes.record(nodes as f64);
            // Stored as a mate in the full depth, with no move.
            let score = if found {
                1_000_000 + 2 * mate_search_depth - 1
            } else {
                0
            };
            transposition_table.store_mate_result(board, score, Move::null(), mate_search_depth);
            if found {
                node_ref.terminal_or_mate_value = Some(1.0);
                node_ref.origin = NodeOrigin::Gate;
                stats.nn_saved_by_tier1 += 1;
                stats.exact_leaf_values += 1;
                return LeafEval::Done(1.0);
            }
        } else {
            stats.tt_mate_misses += 1;
            let mate_start = Instant::now();
            let mate_result = mate_search(
                board,
                move_gen,
                mate_search_depth,
                false,
                exhaustive_mate_depth,
            );
            stats.mate_search_timing.record(mate_start.elapsed());
            stats.mate_search_nodes.record(mate_result.2 as f64);
            transposition_table.store_mate_result(
                board,
                mate_result.0.abs(),
                mate_result.1,
                mate_search_depth,
            );

            if mate_result.0 != 0 {
                if let Some(log) = logger {
                    log.log_mate_search_result(mate_result.0, mate_result.1, 0);
                }
                let mate_value = if mate_result.0 > 0 { 1.0 } else { -1.0 };
                node_ref.terminal_or_mate_value = Some(mate_value);
                node_ref.terminal_distance = Some(mate_result.0.unsigned_abs() as u8);
                stats.nn_saved_by_tier1 += 1;
                if leaf_mate {
                    stats.exact_leaf_values += 1;
                } else {
                    stats.tier1_solutions += 1;
                }
                return LeafEval::Done(mate_value);
            }
        }
    }

    if node_ref.nn_value.is_none() {
        let k_val: f32 = CLASSICAL_K;
        let v_logit: f64 = f64::NEG_INFINITY; // sentinel: no NN result

        // 1. Run Q-search FIRST so completion flag is available for NN inference
        let qsearch_start = Instant::now();
        let q = qsearch_inputs(
            &node_ref.state,
            move_gen,
            &config.pesto,
            config.qsearch_variant,
            config.enable_material_value,
        );
        if config.enable_material_value {
            stats.qsearch_timing.record(qsearch_start.elapsed());
            stats.qsearch_nodes.record(q.nodes as f64);
            stats.qsearch_depth.record(q.depth as f64);
            if q.completed {
                stats.qsearch_completed_count += 1;
            } else {
                stats.qsearch_hit_limit_count += 1;
            }
        }
        let (q_result, qsearch_completed) = (q.score, q.completed);

        // 2. Batched Inference — NN now returns raw v_logit (unbounded)
        if config.enable_tier3_neural {
            if let Some(server) = &config.inference_server {
                if config.use_neural_policy {
                    let started = Instant::now();
                    let moves = legal_moves(&node_ref.state, move_gen);
                    let receiver = server.predict_legal_async(
                        node_ref.state.clone(),
                        moves.clone(),
                        qsearch_completed,
                        q_result,
                    );
                    return LeafEval::Pending(PendingNn {
                        receiver,
                        moves,
                        q_result,
                        started,
                    });
                }
            }
        }
        set_leaf_value(
            &mut node_ref,
            v_logit,
            k_val,
            q_result,
            config,
            stats,
            logger,
        );
    }

    LeafEval::Done(node_ref.nn_value.unwrap())
}

/// Wait for a leaf's network answer and set its value.
fn finish_leaf_evaluation(
    node: Rc<RefCell<MctsNode>>,
    pending: PendingNn,
    config: &TacticalMctsConfig,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) -> f64 {
    let mut node_ref = node.borrow_mut();
    let mut k_val: f32 = CLASSICAL_K;
    let mut v_logit: f64 = f64::NEG_INFINITY; // sentinel: no NN result
    let reply = {
        let _wait = crate::threads::span(c"kf.game.wait");
        pending.receiver.recv()
    };
    if let Ok(Some((policy, nn_v_logit, k))) = reply {
        v_logit = nn_v_logit as f64;
        k_val = k;
        if policy.len() == pending.moves.len() {
            node_ref.nn_priors = Some(pending.moves.into_iter().zip(policy).collect());
        }
    }
    stats.nn_timing.record(pending.started.elapsed());
    set_leaf_value(
        &mut node_ref,
        v_logit,
        k_val,
        pending.q_result,
        config,
        stats,
        logger,
    );
    node_ref.nn_value.unwrap()
}

/// Combine the network output (or the classical fallback) with the material
/// score and store the leaf's value.
fn set_leaf_value(
    node_ref: &mut MctsNode,
    v_logit: f64,
    mut k_val: f32,
    q_result: f32,
    config: &TacticalMctsConfig,
    stats: &mut TacticalMctsStats,
    logger: Option<&Arc<SearchLogger>>,
) {
    {
        if config.enable_material_value {
            if v_logit.is_finite() {
                // NN path: combine v_logit + k * q_result
                let final_value = (v_logit + k_val as f64 * q_result as f64).tanh();
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
                // Classical fallback: v_logit = 0.0, k = CLASSICAL_K, rely on material only
                let classical_v_logit = 0.0;
                let classical_k: f32 = CLASSICAL_K;
                let final_value = (classical_v_logit + classical_k as f64 * q_result as f64).tanh();
                if let Some(log) = logger {
                    log.log_classical_eval((q_result * 100.0) as i32, final_value);
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

    for mv in legal_moves(&node_ref.state, move_gen) {
        let new_board = node_ref.state.apply_move_to_board(mv);
        let child_node = MctsNode::new_child(Rc::downgrade(&node), mv, new_board, move_gen);
        node_ref.children.push(child_node);
        stats.nodes_expanded += 1;
    }
}

/// Legal moves of `board`: captures first, then quiet moves, each in move
/// generator order (the order children are created in).
fn legal_moves(board: &Board, move_gen: &MoveGen) -> Vec<Move> {
    let (captures, non_captures) = move_gen.gen_pseudo_legal_moves(board);
    let mut legal = Vec::with_capacity(captures.len() + non_captures.len());
    legal.extend(
        captures
            .into_iter()
            .chain(non_captures)
            .filter(|mv| board.apply_move_to_board(*mv).is_legal(move_gen)),
    );
    legal
}

pub fn select_best_move_from_root(root: Rc<RefCell<MctsNode>>, move_gen: &MoveGen) -> Option<Move> {
    let root_ref = root.borrow();

    // Prefer winning children (child STM loses = parent wins).
    // Among multiple wins, pick the one with the shortest terminal_distance (fastest win).
    {
        let mut best_win: Option<(Move, u8)> = None;
        for child in &root_ref.children {
            let cr = child.borrow();
            if cr.terminal_or_mate_value.is_some_and(|v| v < -0.5) {
                let dist = cr.terminal_distance.unwrap_or(0);
                if best_win.is_none() || dist < best_win.unwrap().1 {
                    if let Some(action) = cr.action {
                        best_win = Some((action, dist));
                    }
                }
            }
        }
        if let Some((mv, _)) = best_win {
            return Some(mv);
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

    // If ALL children are proven losses, prefer the one with the largest terminal_distance
    // (delay the loss as long as possible).
    {
        let all_losing = !root_ref.children.is_empty()
            && root_ref
                .children
                .iter()
                .all(|c| c.borrow().terminal_or_mate_value.is_some_and(|v| v > 0.0));
        if all_losing {
            let mut best_move: Option<(Move, u8)> = None;
            for child in &root_ref.children {
                let cr = child.borrow();
                let dist = cr.terminal_distance.unwrap_or(0);
                if best_move.is_none() || dist > best_move.unwrap().1 {
                    if let Some(action) = cr.action {
                        best_move = Some((action, dist));
                    }
                }
            }
            if let Some((mv, _)) = best_move {
                return Some(mv);
            }
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
    let mut iteration = 0;
    while iteration < config.max_iterations {
        if start_time.elapsed() > config.time_limit {
            break;
        }

        iteration += run_mcts_iteration(
            &root_node,
            move_gen,
            &config,
            &mut transposition_table,
            &mut stats,
            logger,
            iteration,
        );

        if should_early_terminate(&root_node, &config) {
            break;
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
///
/// The root ends with about `config.max_iterations` visits: a reused root keeps
/// its visits and gets only enough new simulations to reach the budget.
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
                node.terminal_distance = None;
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

    // `max_iterations` is the visit count the root should reach: a reused
    // subtree's visits count toward it, so only the rest are new simulations.
    let mut config = config;
    config.max_iterations = config
        .max_iterations
        .saturating_sub(root_node.borrow().visits);

    let logger = config.logger.as_ref();
    if root_node.borrow().children.is_empty() {
        evaluate_and_expand_node(root_node.clone(), move_gen, &mut stats, &config, logger);
    }

    let mut iteration = 0;
    while iteration < config.max_iterations {
        if start_time.elapsed() > config.time_limit {
            break;
        }

        iteration += run_mcts_iteration(
            &root_node,
            move_gen,
            &config,
            transposition_table,
            &mut stats,
            logger,
            iteration,
        );

        if should_early_terminate(&root_node, &config) {
            break;
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
