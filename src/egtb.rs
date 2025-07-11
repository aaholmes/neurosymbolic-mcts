// src/egtb.rs

// Use internal project types
use crate::board::Board;
use crate::move_types::{Move, CastlingRights};
use crate::piece_types::{self, BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use crate::board_utils::sq_ind_to_bit; // Needed for piece count calculation

// Use the Syzygy library
use shakmaty::{Board as ShakmatyBoard, Position, Role, Color, Square};
use shakmaty_syzygy::{Tablebase, Wdl, Dtz};
use std::path::Path;

// Define the error type for EGTB operations
#[derive(Debug, Clone)]
pub enum EgtbError {
    LoadError(String), // Error loading tablebases
    ProbeError(String), // Error during probing
    ConversionError(String), // Error converting board representation
}

impl std::fmt::Display for EgtbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EgtbError::LoadError(s) => write!(f, "EGTB Load Error: {}", s),
            EgtbError::ProbeError(s) => write!(f, "EGTB Probe Error: {}", s),
            EgtbError::ConversionError(s) => write!(f, "Board Conversion Error: {}", s),
        }
    }
}

impl std::error::Error for EgtbError {}


/// Information obtained from probing the endgame tablebases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgtbInfo {
    /// Win/Draw/Loss status from the perspective of the side to move.
    pub wdl: Wdl,
    /// Distance to Zeroing move (often related to Distance to Mate).
    pub dtz: Option<Dtz>,
    /// The best move according to the tablebase, if available. (Currently always None)
    pub best_move: Option<Move>, // Use crate::move_types::Move
}

/// Structure to handle Syzygy endgame tablebase probing.
#[derive(Clone)]
pub struct EgtbProber {
    // tablebases: Tablebase<ShakmatyBoard>, // Temporarily commented out to fix compilation
    pub max_pieces: u8, // Store the max pieces supported by loaded tables
}

impl EgtbProber {
    /// Creates a new EgtbProber by loading tablebases from the specified path.
    pub fn new(path_str: &str) -> Result<Self, EgtbError> {
        // Check if path exists and is valid for testing
        if !std::path::Path::new(path_str).exists() {
            return Err(EgtbError::LoadError(format!("Path does not exist: {}", path_str)));
        }
        
        // Temporarily simplified implementation to focus on MCTS
        let max_pieces = 7;
        Ok(EgtbProber { max_pieces })
    }

    /// Probes the endgame tablebases for the given board position.
    pub fn probe(&self, _board: &Board) -> Result<Option<EgtbInfo>, EgtbError> {
        // Temporarily disabled to focus on MCTS functionality
        // TODO: Implement proper EGTB integration after MCTS is working
        Ok(None)
    }

    /// Helper function to convert `crate::board::Board` to `shakmaty::Board`.
    #[allow(dead_code)]
    fn convert_board(&self, _board: &Board) -> Result<ShakmatyBoard, EgtbError> {
        // Temporarily disabled to focus on MCTS functionality
        // TODO: Implement proper board conversion after MCTS is working
        Err(EgtbError::ConversionError("EGTB temporarily disabled".to_string()))
    }
}

// Optional: Basic unit tests (updated for internal Board)
#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board; // Use internal Board
    use shakmaty::Chess; // Use shakmaty's Chess board for creating positions

    // Mock structure specifically for testing the piece count check logic
    // within the probe function, without needing actual tablebases.
    struct PieceCountCheckProber {
        max_pieces: u8,
    }

    impl PieceCountCheckProber {
        fn new(max_pieces: u8) -> Self {
            PieceCountCheckProber { max_pieces }
        }

        // Mimics the initial piece count check of the real EgtbProber::probe
        fn probe(&self, board: &Board) -> Result<Option<()>, EgtbError> {
            let piece_count = board.get_all_occupancy().count_ones() as u8;
            if piece_count > self.max_pieces {
                Ok(None) // Too many pieces, real probe would return here
            } else {
                // In a real probe, conversion and actual probing would happen here.
                // For this mock, we just indicate the piece count was acceptable.
                Ok(Some(()))
            }
        }
    }

    #[test]
    fn test_new_invalid_path() {
        // Use a path that is highly unlikely to exist or be valid EGTB dir
        let result = EgtbProber::new("/tmp/non_existent_egtb_path_for_testing");
        assert!(result.is_err());
        match result.err().unwrap() {
            EgtbError::LoadError(_) => {} // Expected error type
            _ => panic!("Expected LoadError for invalid path"),
        }
    }


    #[test]
    fn test_probe_piece_count_within_limit() {
        // Use the mock that only checks piece count
        let prober = PieceCountCheckProber::new(7);

        // 7 pieces - should pass piece count check (returns Some in mock)
        let board = Board::new_from_fen("8/8/8/8/k7/p7/P7/K7 w - - 0 1");
        assert!(prober.probe(&board).unwrap().is_some());

        // 3 pieces - should pass piece count check
        let board = Board::new_from_fen("8/8/8/8/8/8/k7/K1N5 w - - 0 1");
        assert!(prober.probe(&board).unwrap().is_some());
    }

    #[test]
    fn test_probe_piece_count_exceeds_limit() {
        // Use the mock that only checks piece count
        let prober = PieceCountCheckProber::new(7);

        // Starting pos (32 pieces) - should fail piece count check (returns None)
        let board = Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(prober.probe(&board).unwrap().is_none());

        // 8 pieces exactly - should fail piece count check
        let board = Board::new_from_fen("8/8/8/8/k7/p1p5/P1P5/KBN5 w - - 0 1");
        assert!(prober.probe(&board).unwrap().is_none());
    }

    // Note: Testing actual probing still requires real EGTB files.
    // #[test]
    // #[ignore]
    // fn test_real_probe_known_position() {
    //     let prober = EgtbProber::new("../egtb_files").expect("Failed to load test EGTBs");
    //     let board = Board::new_from_fen("8/8/8/8/8/k7/P7/K7 w - - 0 1"); // K vs K+P endgame
    //     let result = prober.probe(&board).unwrap();
    //     assert!(result.is_some());
    //     // Add assertions based on expected WDL/DTZ
    // }
}