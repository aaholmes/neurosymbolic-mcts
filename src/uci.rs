//! UCI interface module for the chess engine
//!
//! This module implements the UCI interface for the chess engine. It allows the user to interact with the engine using UCI commands.

use crate::agent::{Agent, HumanlikeAgent, SimpleAgent}; // Import agents and trait
use crate::boardstack::BoardStack;
use crate::egtb::EgtbProber; // Import EGTB Prober
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use lazy_static::lazy_static;
use crate::move_types::Move;
// Removed unused search imports, agent will handle search
// use crate::search::iterative_deepening_ab_search;
// use crate::transposition::TranspositionTable;
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

// Static instances to avoid lifetime issues
lazy_static! {
    static ref GLOBAL_MOVE_GEN: MoveGen = MoveGen::new();
    static ref GLOBAL_PESTO_EVAL: PestoEval = PestoEval::new();
}

pub struct UCIEngine {
    board: BoardStack,
    egtb_prober: Option<EgtbProber>,
    time_left: Duration,
    increment: Duration,
    moves_to_go: Option<u32>,
    depth: Option<i32>, // UCI depth limit (may or may not be used by agent)
    nodes: Option<u64>, // UCI node limit (may or may not be used by agent)
    mate: Option<i32>,  // UCI mate search depth (may or may not be used by agent)
    movetime: Option<Duration>, // UCI move time limit
    agent_type: String,
    agent: Box<dyn Agent + 'static>, // Store the selected agent

    // Agent configuration (using defaults for now, UCI options needed later)
    mate_search_depth: i32,
    ab_search_depth: i32,
    q_search_max_depth: i32,
    mcts_iterations: u32,
    mcts_time_limit_ms: u64,
    // placeholder_ab_depth: i32, // Included in HumanlikeAgent::new
    // placeholder_q_depth: i32, // Included in HumanlikeAgent::new
}

impl UCIEngine {
    pub fn new() -> Self {
        // EGTB prober disabled by default - requires path configuration
        // TODO: Add UCI option for EGTB path
        let egtb_prober: Option<EgtbProber> = None;

        let agent_type = "Humanlike".to_string(); // Default agent type

        // Default configurations
        let mate_search_depth = 4;
        let ab_search_depth = 6;
        let q_search_max_depth = 4;
        let mcts_iterations = 10000;
        let mcts_time_limit_ms = 5000;

        let mut engine = UCIEngine {
            board: BoardStack::new(),
            egtb_prober,
            time_left: Duration::from_secs(0),
            increment: Duration::from_secs(0),
            moves_to_go: None,
            depth: None,
            nodes: None,
            mate: None,
            movetime: None,
            agent_type, // Store default agent type string
            // Placeholder agent, will be replaced by update_agent
            agent: Box::new(SimpleAgent::new(0, 0, 0, false, &*GLOBAL_MOVE_GEN, &*GLOBAL_PESTO_EVAL)), // Temporary dummy
            mate_search_depth,
            ab_search_depth,
            q_search_max_depth,
            mcts_iterations,
            mcts_time_limit_ms,
        };
        engine.update_agent(); // Create the actual agent based on agent_type
        engine
    }

    /// Creates and updates the `agent` field based on the current `agent_type`.
    fn update_agent(&mut self) {
        // Use static instances to avoid lifetime issues
        let temp_egtb_prober = self.egtb_prober.clone();

        let agent: Box<dyn Agent + 'static> = match self.agent_type.as_str() {
            "AlphaBeta" => Box::new(SimpleAgent::new(
                self.mate_search_depth,
                self.ab_search_depth,
                self.q_search_max_depth,
                false, // verbose - could be another UCI option
                &*GLOBAL_MOVE_GEN, // Use static instance
                &*GLOBAL_PESTO_EVAL,    // Use static instance
            )),
            "Humanlike" | _ => {
                // Default to Humanlike if type is unknown
                if self.agent_type != "Humanlike" {
                    println!(
                        "info string Unknown AgentType '{}', defaulting to Humanlike",
                        self.agent_type
                    );
                    self.agent_type = "Humanlike".to_string(); // Correct the stored type
                }
                Box::new(HumanlikeAgent::new(
                    &*GLOBAL_MOVE_GEN, // Use static instance
                    &*GLOBAL_PESTO_EVAL,    // Use static instance
                    temp_egtb_prober, // Use cloned Option<EgtbProber>
                    self.mate_search_depth,
                    self.mcts_iterations,
                    self.mcts_time_limit_ms,
                    self.ab_search_depth, // Using ab_search_depth for placeholder
                    self.q_search_max_depth, // Using q_search_max_depth for placeholder
                ))
            }
        };
        self.agent = agent; // Assign the newly created agent
        println!("info string Agent set to {}", self.agent_type); // Debug info
    }


    pub fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let input = line.unwrap();
            let tokens: Vec<&str> = input.split_whitespace().collect();

            if tokens.is_empty() {
                continue;
            }

            match tokens[0] {
                "uci" => {
                    println!("id name Caissawary");
                    println!("id author Adam Holmes");
                    // Define the AgentType option
                    println!("option name AgentType type combo default Humanlike var AlphaBeta var Humanlike");
                    println!("uciok");
                }
                "isready" => println!("readyok"),
                "ucinewgame" => self.handle_ucinewgame(), // Changed to handle potential agent re-init
                "position" => self.handle_position(&tokens[1..]),
                "setoption" => self.handle_setoption(&tokens[1..]), // Added handler for setoption
                "go" => self.handle_go(&tokens[1..]),
                "debug" => {
                    if tokens.len() >= 2 && tokens[1] == "viz" {
                        let filename = if tokens.len() >= 3 { tokens[2] } else { "debug.dot" };
                        if let Some(root) = self.agent.get_last_search_tree() {
                            let dot = root.borrow().export_dot(10, 0);
                            match std::fs::write(filename, dot) {
                                Ok(_) => println!("info string Tree dumped to {}. Use xdot or edotor.net to view.", filename),
                                Err(e) => println!("info string Failed to write viz file: {}", e),
                            }
                        } else {
                            println!("info string No search tree available. Run a search first!");
                        }
                    }
                }
                "quit" => break,
                _ => println!("Unknown command: {}", tokens[0]),
            }

            io::stdout().flush().unwrap();
        }
    }

    fn handle_position(&mut self, args: &[&str]) {
        if args.is_empty() {
            return;
        }

        if args[0] == "startpos" {
            self.board = BoardStack::new();
            let moves_idx = 2;

            if args.len() > 1 && args[1] == "moves" {
                for move_str in &args[moves_idx..] {
                    if let Some(chess_move) = Move::from_uci(move_str) {
                        self.board.make_move(chess_move);
                    }
                }
            }
        } else if args[0] == "fen" {
            // Find the index where "moves" starts, if present
            let moves_idx = args
                .iter()
                .position(|&x| x == "moves")
                .unwrap_or(args.len());

            // Join the FEN parts
            let fen = args[1..moves_idx].join(" ");

            // Create a new board from the FEN
            self.board = BoardStack::new_from_fen(&fen);

            // Apply moves if present
            if moves_idx < args.len() {
                for move_str in &args[moves_idx + 1..] {
                    if let Some(chess_move) = Move::from_uci(move_str) {
                        self.board.make_move(chess_move);
                    }
                }
            }
        } else {
            println!("info string Invalid position command");
        }
    }

    fn handle_go(&mut self, args: &[&str]) {
        self.parse_go_command(args); // Parses time controls, depth, nodes etc. into self

        // Note: The current Agent trait doesn't pass time controls.
        // The agents use their internal logic or fixed depths/iterations for now.
        // TODO: Enhance Agent trait or agent implementations to use UCI time controls.
        let _allocated_time = self.calculate_allocated_time(); // Calculate but not used directly by agent.get_move yet
        let start_time = Instant::now();

        // Get the move from the currently selected agent
        // The agent's get_move implementation handles the search (AB, MCTS, etc.)
        // We pass a mutable reference to the board stack.
        let best_move = self.agent.get_move(&mut self.board);

        let elapsed = start_time.elapsed();

        // Print info - We don't get depth, score, nodes back from the generic agent easily.
        // We could modify the Agent trait to return this info, or have agents print UCI info internally.
        // For now, just print the best move and time.
        // Agents might print their own info during search.
        println!(
            "info time {}", // Basic info from UCI loop
            elapsed.as_millis()
            // Example if agent printed info: info depth 10 score cp 150 nodes 12345 time 1500 pv e2e4 e7e5 ...
        );

        println!("bestmove {}", &best_move.print_algebraic());
    }

    fn parse_go_command(&mut self, args: &[&str]) {
        self.time_left = Duration::from_secs(0);
        self.increment = Duration::from_secs(0);
        self.moves_to_go = None;
        self.depth = None;
        self.nodes = None;
        self.mate = None;
        self.movetime = None;

        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "wtime" | "btime" => {
                    if (args[i] == "wtime" && self.board.current_state().w_to_move)
                        || (args[i] == "btime" && !self.board.current_state().w_to_move)
                    {
                        self.time_left = Duration::from_millis(args[i + 1].parse().unwrap_or(0));
                    }
                    i += 2;
                }
                "winc" | "binc" => {
                    if (args[i] == "winc" && self.board.current_state().w_to_move)
                        || (args[i] == "binc" && !self.board.current_state().w_to_move)
                    {
                        self.increment = Duration::from_millis(args[i + 1].parse().unwrap_or(0));
                    }
                    i += 2;
                }
                "movestogo" => {
                    self.moves_to_go = Some(args[i + 1].parse().unwrap_or(30));
                    i += 2;
                }
                "depth" => {
                    self.depth = Some(args[i + 1].parse().unwrap_or(100));
                    i += 2;
                }
                "nodes" => {
                    self.nodes = Some(args[i + 1].parse().unwrap_or(0));
                    i += 2;
                }
                "mate" => {
                    self.mate = Some(args[i + 1].parse().unwrap_or(0));
                    i += 2;
                }
                "movetime" => {
                    self.movetime = Some(Duration::from_millis(args[i + 1].parse().unwrap_or(0)));
                    i += 2;
                }
                _ => i += 1,
            }
        }
    }
    
    // Renamed from direct assignment in the match arm to potentially handle agent re-initialization
    fn handle_ucinewgame(&mut self) {
            self.board = BoardStack::new();
            // Potentially re-initialize agent state here if needed based on self.agent_type
        }
    
    fn handle_setoption(&mut self, args: &[&str]) {
            if args.len() >= 4 && args[0] == "name" && args[2] == "value" {
                let name = args[1];
                let value = args[3];
                match name {
                    "AgentType" => {
                        if value == "AlphaBeta" || value == "Humanlike" {
                            if self.agent_type != value { // Only update if changed
                                self.agent_type = value.to_string();
                                self.update_agent(); // Re-create the agent instance
                            }
                        } else {
                            println!("info string Invalid value for AgentType: {}", value);
                        }
                    }
                    "ModelPath" => {
                        // This would need to be passed to the agent.
                        // For now, let's just log it, as we need to thread this path 
                        // through update_agent -> HumanlikeAgent -> NeuralNetPolicy
                        println!("info string Setting ModelPath to {} (Not fully wired yet)", value);
                        // TODO: Store this path in UCIEngine struct and pass to HumanlikeAgent::new
                    }
                    _ => {
                        // println!("info string Unknown option: {}", name);
                    }
                }
            } else {
                println!("info string Invalid setoption command format");
            }
        }

    /// This function calculates the allocated time for a chess move based on the time control settings.
    ///
    /// Formula: time left per move until time control is reached (or 5% of time left if no time
    /// control is specified) + 50% of increment
    /// Defaults to 5 seconds if no time control is specified.
    fn calculate_allocated_time(&self) -> Duration {
        if let Some(movetime) = self.movetime {
            return movetime;
        }

        if self.time_left.as_millis() == 0 {
            return Duration::from_secs(5); // Default to 5 seconds if no time control is specified
        }

        let moves_left = self.moves_to_go.unwrap_or(20) as f32;
        let base_time = self.time_left.as_secs_f32() / moves_left;
        let bonus_time = self.increment.as_secs_f32();

        Duration::from_secs_f32(base_time + bonus_time * 0.5)
    }

    // Make internal handlers public for testing

    #[cfg(test)]
    pub fn get_board(&self) -> &BoardStack {
        &self.board
    }

    #[cfg(test)]
    pub fn get_agent_type(&self) -> &str {
        &self.agent_type
    }

    #[cfg(test)]
    pub fn get_time_limits(&self) -> (Duration, Duration, Option<Duration>) {
        (self.time_left, self.increment, self.movetime)
    }

    #[cfg(test)]
    pub fn get_search_limits(&self) -> (Option<i32>, Option<u64>, Option<i32>) {
        (self.depth, self.nodes, self.mate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board; // Assuming Board is needed for state checks

    #[test]
    fn test_handle_position_startpos() {
        let mut engine = UCIEngine::new();
        engine.handle_position(&["startpos"]);
        // Check if the board is in the initial state
        assert_eq!(engine.get_board().current_state().to_fen(), Some("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()));
    }

    #[test]
    fn test_handle_position_startpos_moves() {
        let mut engine = UCIEngine::new();
        engine.handle_position(&["startpos", "moves", "e2e4", "e7e5"]);
        // Check if the board is in the correct state after moves
        assert_eq!(engine.get_board().current_state().to_fen(), Some("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2".to_string()));
    }

    #[test]
    fn test_handle_position_fen() {
        let mut engine = UCIEngine::new();
        let fen = "r3k2r/p1ppqpbp/bnNppnp1/8/4P3/2N2Q2/PPP2PPP/R3K2R w KQkq - 0 1";
        engine.handle_position(&["fen", "r3k2r/p1ppqpbp/bnNppnp1/8/4P3/2N2Q2/PPP2PPP/R3K2R", "w", "KkQq", "-", "0", "1"]);
        // Check if the board is in the state specified by the FEN
        assert_eq!(engine.get_board().current_state().to_fen(), Some(fen.to_string()));
    }

    #[test]
    fn test_handle_position_fen_moves() {
        let mut engine = UCIEngine::new();
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        engine.handle_position(&["fen", fen, "moves", "e2e4", "e7e5", "g1f3"]);
        // Check if the board is in the correct state after FEN and moves
        assert_eq!(engine.get_board().current_state().to_fen(), Some("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2".to_string()));
    }

    #[test]
    fn test_parse_go_movetime() {
        let mut engine = UCIEngine::new();
        engine.parse_go_command(&["go", "movetime", "1000"]);
        let (time_left, increment, movetime) = engine.get_time_limits();
        assert_eq!(movetime, Some(Duration::from_millis(1000)));
        assert_eq!(time_left, Duration::from_secs(0));
        assert_eq!(increment, Duration::from_secs(0));
    }

    #[test]
    fn test_parse_go_wtime_btime_winc_binc() {
        let mut engine = UCIEngine::new();
        // Test for white to move
        engine.handle_position(&["startpos"]); // White to move
        engine.parse_go_command(&["go", "wtime", "60000", "btime", "50000", "winc", "1000", "binc", "500"]);
        let (time_left, increment, movetime) = engine.get_time_limits();
        assert_eq!(time_left, Duration::from_millis(60000));
        assert_eq!(increment, Duration::from_millis(1000));
        assert_eq!(movetime, None);

        // Test for black to move
        engine.handle_position(&["startpos", "moves", "e2e4"]); // Black to move
        engine.parse_go_command(&["go", "wtime", "59000", "btime", "49000", "winc", "1000", "binc", "500"]);
        let (time_left, increment, movetime) = engine.get_time_limits();
        assert_eq!(time_left, Duration::from_millis(49000));
        assert_eq!(increment, Duration::from_millis(500));
        assert_eq!(movetime, None);
    }

    #[test]
    fn test_parse_go_depth() {
        let mut engine = UCIEngine::new();
        engine.parse_go_command(&["go", "depth", "10"]);
        let (depth, nodes, mate) = engine.get_search_limits();
        assert_eq!(depth, Some(10));
        assert_eq!(nodes, None);
        assert_eq!(mate, None);
    }

    #[test]
    fn test_parse_go_nodes() {
        let mut engine = UCIEngine::new();
        engine.parse_go_command(&["go", "nodes", "100000"]);
        let (depth, nodes, mate) = engine.get_search_limits();
        assert_eq!(depth, None);
        assert_eq!(nodes, Some(100000));
        assert_eq!(mate, None);
    }

    #[test]
    fn test_parse_go_mate() {
        let mut engine = UCIEngine::new();
        engine.parse_go_command(&["go", "mate", "5"]);
        let (depth, nodes, mate) = engine.get_search_limits();
        assert_eq!(depth, None);
        assert_eq!(nodes, None);
        assert_eq!(mate, Some(5));
    }

    #[test]
    fn test_parse_go_infinite() {
        let mut engine = UCIEngine::new();
        engine.parse_go_command(&["go", "infinite"]);
        let (time_left, increment, movetime) = engine.get_time_limits();
        assert_eq!(time_left, Duration::from_secs(0));
        assert_eq!(increment, Duration::from_secs(0));
        assert_eq!(movetime, None); // infinite doesn't set movetime
        let (depth, nodes, mate) = engine.get_search_limits();
        assert_eq!(depth, None);
        assert_eq!(nodes, None);
        assert_eq!(mate, None);
    }

    #[test]
    fn test_handle_setoption_agenttype() {
        let mut engine = UCIEngine::new();
        // Default is Humanlike
        assert_eq!(engine.get_agent_type(), "Humanlike");

        // Set to AlphaBeta
        engine.handle_setoption(&["name", "AgentType", "value", "AlphaBeta"]);
        assert_eq!(engine.get_agent_type(), "AlphaBeta");

        // Set back to Humanlike
        engine.handle_setoption(&["name", "AgentType", "value", "Humanlike"]);
        assert_eq!(engine.get_agent_type(), "Humanlike");

        // Try setting an invalid value (should remain Humanlike)
        engine.handle_setoption(&["name", "AgentType", "value", "InvalidAgent"]);
        assert_eq!(engine.get_agent_type(), "Humanlike");
    }

    #[test]
    fn test_handle_setoption_unknown() {
        let mut engine = UCIEngine::new();
        // Setting an unknown option should not change the agent type
        engine.handle_setoption(&["name", "UnknownOption", "value", "SomeValue"]);
        assert_eq!(engine.get_agent_type(), "Humanlike"); // Default agent type should remain
    }

    #[test]
    fn test_handle_setoption_invalid_format() {
        let mut engine = UCIEngine::new();
        // Invalid format should not change the agent type
        engine.handle_setoption(&["name", "AgentType", "InvalidValue"]);
        assert_eq!(engine.get_agent_type(), "Humanlike"); // Default agent type should remain

        engine.handle_setoption(&["AgentType", "value", "Humanlike"]);
        assert_eq!(engine.get_agent_type(), "Humanlike"); // Default agent type should remain
    }
}
