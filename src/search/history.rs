//! History heuristic for move ordering in alpha-beta search.
//!
//! The history heuristic tracks which moves have historically caused beta cutoffs
//! (fail-highs) in the search tree. Moves that frequently cause cutoffs are likely
//! to be good moves, so they are searched earlier in subsequent positions.
//!
//! This is a "quiet move" ordering technique - it helps order non-tactical moves
//! that don't have obvious features like captures or checks.
//!
//! # Implementation
//!
//! Uses a 64x64 table indexed by (from_square, to_square). When a move causes
//! a beta cutoff, its history score is incremented by depth². Higher depths
//! give larger bonuses since deeper cutoffs save more work.

use crate::move_types::Move;

/// Maximum ply depth for history table
pub const MAX_PLY: usize = 64;

/// History table for move ordering
pub struct HistoryTable {
    /// History scores for each from-to square combination
    /// Indexed by [from_square][to_square]
    table: [[i32; 64]; 64],
}

impl Default for HistoryTable {
    fn default() -> Self {
        Self {
            table: [[0; 64]; 64],
        }
    }
}

impl HistoryTable {
    /// Create a new history table with all scores initialized to 0
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the history score for a move that caused a beta cutoff
    pub fn update(&mut self, mv: &Move, depth: i32) {
        let bonus = depth * depth;
        let from = mv.from;
        let to = mv.to;
        // Saturating add to prevent overflow
        self.table[from][to] = self.table[from][to].saturating_add(bonus);
    }

    /// Get the history score for a move
    pub fn get_score(&self, mv: &Move) -> i32 {
        self.table[mv.from][mv.to]
    }

    /// Get the history score for a move specified by from and to squares
    pub fn get_score_from_squares(&self, from: usize, to: usize) -> i32 {
        self.table[from][to]
    }

    /// Clear all history scores
    pub fn clear(&mut self) {
        for from in 0..64 {
            for to in 0..64 {
                self.table[from][to] = 0;
            }
        }
    }
}
