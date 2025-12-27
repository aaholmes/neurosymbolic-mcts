//! Main entry point for the Kingfisher chess engine.
//!
//! This module sets up the chess engine components and runs a sample game
//! between two simple agents.

extern crate kingfisher;
use kingfisher::agent::SimpleAgent;
use kingfisher::arena::Match;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::uci::UCIEngine;

/// The main function that sets up and runs a sample chess game.
fn run_simple_game() {
    // Note: We'd need to expose the global instances or just not run this demo in main
}

fn main() {
    let mut engine = UCIEngine::new();
    engine.run();
}