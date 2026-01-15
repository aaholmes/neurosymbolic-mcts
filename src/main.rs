//! Main entry point for the Kingfisher chess engine.
//!
//! Runs the UCI (Universal Chess Interface) protocol handler for
//! communication with chess GUIs.

use kingfisher::uci::UCIEngine;

fn main() {
    let mut engine = UCIEngine::new();
    engine.run();
}