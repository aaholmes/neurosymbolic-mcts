//! Stream of Consciousness Logger for MCTS Search
//! 
//! Provides real-time narration of search decisions, tier overrides,
//! and selection logic for debugging and educational purposes.

use crate::board::Board;
use crate::move_types::Move;
use crate::mcts::node::{MctsNode, NodeOrigin};
use crate::mcts::tactical::TacticalMove;
use std::cell::RefCell;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{self, Write};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Verbosity level for the search logger
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    /// No output
    Silent = 0,
    /// Only tier overrides and final results
    Minimal = 1,
    /// Selection decisions and evaluations
    Normal = 2,
    /// Full trace including backpropagation
    Verbose = 3,
    /// Debug-level with internal state dumps
    Debug = 4,
}

impl Default for Verbosity {
    fn default() -> Self {
        Verbosity::Normal
    }
}

/// Output destination for log messages
#[derive(Debug)]
pub enum LogSink {
    /// Write to stdout
    Console,
    /// Write to a file
    File(Mutex<File>),
    /// Accumulate in a string buffer (for testing)
    Buffer(Mutex<String>),
    /// Multiple sinks (fan-out)
    Multi(Vec<Box<LogSink>>),
}

impl LogSink {
    pub fn write(&self, msg: &str) {
        match self {
            LogSink::Console => {
                print!("{}", msg);
                io::stdout().flush().ok();
            }
            LogSink::File(f) => {
                if let Ok(mut file) = f.lock() {
                    write!(file, "{}", msg).ok();
                }
            }
            LogSink::Buffer(buf) => {
                if let Ok(mut b) = buf.lock() {
                    b.push_str(msg);
                }
            }
            LogSink::Multi(sinks) => {
                for sink in sinks {
                    sink.write(msg);
                }
            }
        }
    }
    
    pub fn writeln(&self, msg: &str) {
        self.write(msg);
        self.write("\n");
    }
}

/// Reason for a Tier 1 gate activation
#[derive(Debug, Clone)]
pub enum GateReason {
    /// Mate found via mate search
    MateFound { depth: i32, score: i32 },
    /// King of the Hill instant win
    KothWin,
    /// Terminal position (checkmate/stalemate)
    Terminal { is_checkmate: bool },
    /// Mate found in transposition table
    TtMateHit { depth: i32 },
}

impl GateReason {
    pub fn description(&self) -> String {
        match self {
            GateReason::MateFound { depth, score } => {
                if *score > 0 {
                    format!("Mate in {} found (winning)", depth)
                } else {
                    format!("Getting mated in {} (losing)", depth.abs())
                }
            }
            GateReason::KothWin => "King of the Hill center reachable in ≤3 moves".to_string(),
            GateReason::Terminal { is_checkmate } => {
                if *is_checkmate {
                    "Position is checkmate".to_string()
                } else {
                    "Position is stalemate".to_string()
                }
            }
            GateReason::TtMateHit { depth } => {
                format!("Mate at depth {} found in transposition table", depth)
            }
        }
    }
}

/// Reason for a selection decision
#[derive(Debug, Clone)]
pub enum SelectionReason {
    /// Chose unexplored tactical move (Tier 2 priority)
    TacticalPriority { move_type: String, score: f64 },
    /// Chose via UCB/PUCT formula
    UcbSelection { q_value: f64, u_value: f64, total: f64 },
    /// Random exploration
    Exploration,
    /// Forced (only one legal move)
    ForcedMove,
}

/// The main search logger
#[derive(Debug)]
pub struct SearchLogger {
    /// Current verbosity level
    verbosity: Verbosity,
    /// Output destination
    sink: LogSink,
    /// Start time for elapsed timestamps
    start_time: Mutex<Instant>,
    /// Current search depth (for indentation)
    current_depth: AtomicUsize,
    /// Whether to use emoji in output
    use_emoji: bool,
    /// Iteration counter
    iteration: AtomicU32,
    /// Enable/disable flag (for conditional logging)
    enabled: AtomicBool,
}

impl SearchLogger {
    /// Create a new logger with console output
    pub fn new(verbosity: Verbosity) -> Self {
        SearchLogger {
            verbosity,
            sink: LogSink::Console,
            start_time: Mutex::new(Instant::now()),
            current_depth: AtomicUsize::new(0),
            use_emoji: true,
            iteration: AtomicU32::new(0),
            enabled: AtomicBool::new(true),
        }
    }
    
    /// Create a silent logger (no output)
    pub fn silent() -> Self {
        SearchLogger {
            verbosity: Verbosity::Silent,
            sink: LogSink::Console,
            start_time: Mutex::new(Instant::now()),
            current_depth: AtomicUsize::new(0),
            use_emoji: false,
            iteration: AtomicU32::new(0),
            enabled: AtomicBool::new(false),
        }
    }
    
    /// Create a logger that writes to a buffer (for testing)
    pub fn buffered(verbosity: Verbosity) -> Self {
        SearchLogger {
            verbosity,
            sink: LogSink::Buffer(Mutex::new(String::new())),
            start_time: Mutex::new(Instant::now()),
            current_depth: AtomicUsize::new(0),
            use_emoji: false,
            iteration: AtomicU32::new(0),
            enabled: AtomicBool::new(true),
        }
    }
    
    /// Get the buffered output (panics if not a buffer sink)
    pub fn get_buffer(&self) -> String {
        match &self.sink {
            LogSink::Buffer(buf) => buf.lock().unwrap().clone(),
            _ => panic!("get_buffer called on non-buffer sink"),
        }
    }
    
    /// Set the output sink
    pub fn with_sink(mut self, sink: LogSink) -> Self {
        self.sink = sink;
        self
    }
    
    /// Set whether to use emoji
    pub fn with_emoji(mut self, use_emoji: bool) -> Self {
        self.use_emoji = use_emoji;
        self
    }
    
    /// Reset the start time
    pub fn reset_timer(&self) {
        if let Ok(mut start) = self.start_time.lock() {
            *start = Instant::now();
        }
        self.iteration.store(0, Ordering::SeqCst);
    }
    
    /// Enable or disable logging
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }
    
    /// Check if logging is enabled at the given verbosity
    fn should_log(&self, required: Verbosity) -> bool {
        self.enabled.load(Ordering::SeqCst) && self.verbosity >= required
    }
    
    /// Get elapsed time as a formatted string
    fn elapsed(&self) -> String {
        let elapsed = self.start_time.lock().unwrap().elapsed();
        format!("{:>6.1}ms", elapsed.as_secs_f64() * 1000.0)
    }
    
    /// Get indentation for current depth
    fn indent(&self) -> String {
        "  ".repeat(self.current_depth.load(Ordering::SeqCst))
    }
    
    /// Format a move for display
    fn fmt_move(&self, mv: Move) -> String {
        mv.to_uci()
    }
    
    /// Format a board position summary
    fn fmt_board_summary(&self, board: &Board) -> String {
        let stm = if board.w_to_move { "White" } else { "Black" };
        format!("{} to move", stm)
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Tier 1: Gate Events
    // ═══════════════════════════════════════════════════════════════
    
    /// Log a Tier 1 gate activation
    pub fn log_tier1_gate(&self, reason: &GateReason, chosen_move: Option<Move>) {
        if !self.should_log(Verbosity::Minimal) {
            return;
        }
        
        let emoji = if self.use_emoji { "🚨 " } else { "[GATE] " };
        let move_str = chosen_move
            .map(|m| format!(" → {}", self.fmt_move(m)))
            .unwrap_or_default();
        
        self.sink.writeln(&format!(
            "{}{}TIER 1 GATE: {}{}",
            self.elapsed(),
            emoji,
            reason.description(),
            move_str
        ));
    }
    
    /// Log KOTH detection
    pub fn log_koth_check(&self, reachable: bool, distance: Option<u32>) {
        if !self.should_log(Verbosity::Verbose) {
            return;
        }
        
        let emoji = if self.use_emoji { "👑 " } else { "[KOTH] " };
        if reachable {
            self.sink.writeln(&format!(
                "{}{}KOTH: Center reachable in {} moves",
                self.indent(),
                emoji,
                distance.unwrap_or(0)
            ));
        }
    }
    
    /// Log mate search initiation
    pub fn log_mate_search_start(&self, depth: i32) {
        if !self.should_log(Verbosity::Verbose) {
            return;
        }
        
        let emoji = if self.use_emoji { "🔍 " } else { "[MATE] " };
        self.sink.writeln(&format!(
            "{}{}Starting mate search at depth {}",
            self.indent(),
            emoji,
            depth
        ));
    }
    
    /// Log mate search result
    pub fn log_mate_search_result(&self, score: i32, best_move: Move, nodes: u64) {
        if !self.should_log(Verbosity::Normal) {
            return;
        }
        
        let emoji = if self.use_emoji { "🔍 " } else { "[MATE] " };
        let result = if score >= 1_000_000 {
            format!("MATE FOUND: {} (score: {})", self.fmt_move(best_move), score)
        } else if score <= -1_000_000 {
            format!("GETTING MATED: {} (score: {})", self.fmt_move(best_move), score)
        } else {
            format!("No mate (best: {}, nodes: {})", self.fmt_move(best_move), nodes)
        };
        
        self.sink.writeln(&format!(
            "{}{}Mate search: {}",
            self.indent(),
            emoji,
            result
        ));
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Tier 2: Tactical Grafting Events
    // ═══════════════════════════════════════════════════════════════
    
    /// Log start of quiescence search
    pub fn log_qs_start(&self, board: &Board) {
        if !self.should_log(Verbosity::Verbose) {
            return;
        }
        
        let emoji = if self.use_emoji { "⚔️ " } else { "[QS] " };
        self.sink.writeln(&format!(
            "{}{}Starting quiescence search ({})",
            self.indent(),
            emoji,
            self.fmt_board_summary(board)
        ));
    }
    
    /// Log tactical moves identified
    pub fn log_tactical_moves_found(&self, moves: &[TacticalMove]) {
        if !self.should_log(Verbosity::Normal) {
            return;
        }
        
        if moves.is_empty() {
            return;
        }
        
        let emoji = if self.use_emoji { "⚔️ " } else { "[TACT] " };
        
        let mut summary = String::new();
        let mut captures = 0;
        let mut checks = 0;
        let mut forks = 0;
        
        for tm in moves {
            match tm {
                TacticalMove::Capture(_, _) => captures += 1,
                TacticalMove::Check(_, _) => checks += 1,
                TacticalMove::Fork(_, _) => forks += 1,
                TacticalMove::Pin(_, _) => {} // Not counted in summary
            }
        }
        
        if captures > 0 { write!(summary, "{} captures ", captures).ok(); }
        if checks > 0 { write!(summary, "{} checks ", checks).ok(); }
        if forks > 0 { write!(summary, "{} forks", forks).ok(); }
        
        self.sink.writeln(&format!(
            "{}{}Tactical moves: {}",
            self.indent(),
            emoji,
            summary.trim()
        ));
    }
    
    /// Log a tactical graft
    pub fn log_tier2_graft(&self, move_: Move, extrapolated_value: f64, k_val: f32) {
        if !self.should_log(Verbosity::Normal) {
            return;
        }
        
        let emoji = if self.use_emoji { "🌿 " } else { "[GRAFT] " };
        self.sink.writeln(&format!(
            "{}{}TIER 2 GRAFT: {} extrapolated to {:.3} (k={:.2})",
            self.indent(),
            emoji,
            self.fmt_move(move_),
            extrapolated_value,
            k_val
        ));
    }
    
    /// Log principal variation from QS
    pub fn log_qs_pv(&self, pv: &[Move], terminal_eval: i32) {
        if !self.should_log(Verbosity::Verbose) {
            return;
        }
        
        let emoji = if self.use_emoji { "📍 " } else { "[PV] " };
        let pv_str: String = pv.iter() 
            .map(|m| self.fmt_move(*m))
            .collect::<Vec<_>>()
            .join(" ");
        
        self.sink.writeln(&format!(
            "{}{}QS PV: {} (eval: {}cp)",
            self.indent(),
            emoji,
            pv_str,
            terminal_eval
        ));
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Tier 3: Neural Network Events
    // ═══════════════════════════════════════════════════════════════
    
    /// Log neural network evaluation
    pub fn log_tier3_neural(&self, value: f64, k_val: f32) {
        if !self.should_log(Verbosity::Normal) {
            return;
        }
        
        let emoji = if self.use_emoji { "🧠 " } else { "[NN] " };
        self.sink.writeln(&format!(
            "{}{}TIER 3 NEURAL: value={:.3}, k={:.2}",
            self.indent(),
            emoji,
            value,
            k_val
        ));
    }
    
    /// Log top policy moves from neural network
    pub fn log_nn_policy(&self, top_moves: &[(Move, f32)]) {
        if !self.should_log(Verbosity::Verbose) {
            return;
        }
        
        let emoji = if self.use_emoji { "🧠 " } else { "[POLICY] " };
        let moves_str: String = top_moves.iter() 
            .take(5)
            .map(|(m, p)| format!("{}:{:.1}%", self.fmt_move(*m), p * 100.0))
            .collect::<Vec<_>>()
            .join(", ");
        
        self.sink.writeln(&format!(
            "{}{}Policy priors: {}",
            self.indent(),
            emoji,
            moves_str
        ));
    }
    
    /// Log classical (Pesto) evaluation fallback
    pub fn log_classical_eval(&self, eval_cp: i32, tanh_value: f64) {
        if !self.should_log(Verbosity::Verbose) {
            return;
        }
        
        let emoji = if self.use_emoji { "📊 " } else { "[EVAL] " };
        self.sink.writeln(&format!(
            "{}{}Classical eval: {}cp → tanh={:.3}",
            self.indent(),
            emoji,
            eval_cp,
            tanh_value
        ));
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Selection & Tree Traversal Events
    // ═══════════════════════════════════════════════════════════════
    
    /// Log iteration start
    pub fn log_iteration_start(&self, iteration: u32) {
        self.iteration.store(iteration, Ordering::SeqCst);
        
        if !self.should_log(Verbosity::Debug) {
            return;
        }
        
        let emoji = if self.use_emoji { "🔄 " } else { "[ITER] " };
        self.sink.writeln(&format!(
            "{}{}Iteration {}",
            self.elapsed(),
            emoji,
            iteration
        ));
    }
    
    /// Log node selection
    pub fn log_selection(&self, chosen_move: Move, reason: &SelectionReason, depth: usize) {
        if !self.should_log(Verbosity::Verbose) {
            return;
        }
        
        let emoji = if self.use_emoji { "➡️ " } else { "[SEL] " };
        let reason_str = match reason {
            SelectionReason::TacticalPriority { move_type, score } => {
                format!("tactical priority ({}: {:.2})", move_type, score)
            }
            SelectionReason::UcbSelection { q_value, u_value, total } => {
                format!("UCB (Q={:.3} + U={:.3} = {:.3})", q_value, u_value, total)
            }
            SelectionReason::Exploration => "exploration".to_string(),
            SelectionReason::ForcedMove => "only legal move".to_string(),
        };
        
        let indent = "  ".repeat(depth);
        self.sink.writeln(&format!(
            "{}{}{}Select {} [{}]",
            indent,
            emoji,
            "", // Placeholder for potential future additions
            self.fmt_move(chosen_move),
            reason_str
        ));
    }
    
    /// Log entering a node during selection
    pub fn log_enter_node(&self, node: &MctsNode, depth: usize) {
        self.current_depth.store(depth, Ordering::SeqCst);
        
        if !self.should_log(Verbosity::Debug) {
            return;
        }
        
        let move_str = node.action
            .map(|m| self.fmt_move(m))
            .unwrap_or_else(|| "Root".to_string());
        
        self.sink.writeln(&format!(
            "{}Enter: {} (N={}, Q={:.3})",
            self.indent(),
            move_str,
            node.visits,
            if node.visits > 0 { node.total_value / node.visits as f64 } else { 0.0 }
        ));
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Backpropagation Events
    // ═══════════════════════════════════════════════════════════════
    
    /// Log backpropagation
    pub fn log_backprop(&self, path_length: usize, leaf_value: f64) {
        if !self.should_log(Verbosity::Debug) {
            return;
        }
        
        let emoji = if self.use_emoji { "⬆️ " } else { "[BACK] " };
        self.sink.writeln(&format!(
            "{}{}Backprop: value={:.3} through {} nodes",
            self.elapsed(),
            emoji,
            leaf_value,
            path_length
        ));
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Summary Events
    // ═══════════════════════════════════════════════════════════════
    
    /// Log iteration summary (periodic)
    pub fn log_iteration_summary(&self, iteration: u32, best_move: Option<Move>, root_visits: u32) {
        if !self.should_log(Verbosity::Minimal) {
            return;
        }
        
        // Only log every 100 iterations at Minimal, every 10 at Normal
        let interval = match self.verbosity {
            Verbosity::Minimal => 100,
            Verbosity::Normal => 50,
            _ => 10,
        };
        
        if iteration % interval != 0 {
            return;
        }
        
        let emoji = if self.use_emoji { "📈 " } else { "[STAT] " };
        let move_str = best_move
            .map(|m| self.fmt_move(m))
            .unwrap_or_else(|| "?".to_string());
        
        self.sink.writeln(&format!(
            "{}{}Iter {}: best={}, root_N={}",
            self.elapsed(),
            emoji,
            iteration,
            move_str,
            root_visits
        ));
    }
    
    /// Log final search result
    pub fn log_search_complete(
        &self,
        best_move: Option<Move>,
        iterations: u32,
        nodes_expanded: u32,
        mates_found: u32,
    ) {
        if !self.should_log(Verbosity::Minimal) {
            return;
        }
        
        let emoji = if self.use_emoji { "✅ " } else { "[DONE] " };
        let move_str = best_move
            .map(|m| self.fmt_move(m))
            .unwrap_or_else(|| "none".to_string());
        
        self.sink.writeln(&format!(
            "\n{}{}Search complete!",
            self.elapsed(),
            emoji
        ));
        self.sink.writeln(&format!("   Best move: {}", move_str));
        self.sink.writeln(&format!("   Iterations: {}", iterations));
        self.sink.writeln(&format!("   Nodes expanded: {}", nodes_expanded));
        if mates_found > 0 {
            self.sink.writeln(&format!("   Mates found: {}", mates_found));
        }
    }
    
    /// Log tier override explanation
    pub fn log_tier_override(&self, from_tier: &str, to_tier: &str, reason: &str) {
        if !self.should_log(Verbosity::Normal) {
            return;
        }
        
        let emoji = if self.use_emoji { "⚡ " } else { "[OVERRIDE] " };
        self.sink.writeln(&format!(
            "{}{}OVERRIDE: {} → {} ({})",
            self.elapsed(),
            emoji,
            from_tier,
            to_tier,
            reason
        ));
    }
}

impl Default for SearchLogger {
    fn default() -> Self {
        SearchLogger::silent()
    }
}

// Global logger instance for convenience
thread_local! {
    static GLOBAL_LOGGER: RefCell<SearchLogger> = RefCell::new(SearchLogger::silent());
}

/// Set the global logger for the current thread
pub fn set_global_logger(logger: SearchLogger) {
    GLOBAL_LOGGER.with(|l| {
        *l.borrow_mut() = logger;
    });
}

/// Get a reference to the global logger
pub fn with_global_logger<F, R>(f: F) -> R
where
    F: FnOnce(&SearchLogger) -> R,
{
    GLOBAL_LOGGER.with(|l| f(&l.borrow()))
}

/// Get a mutable reference to the global logger
pub fn with_global_logger_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut SearchLogger) -> R,
{
    GLOBAL_LOGGER.with(|l| f(&mut l.borrow_mut()))
}
