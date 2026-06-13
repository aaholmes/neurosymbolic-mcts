//! Transposition table for storing previously seen positions.
//!
//! This module implements a transposition table, which is used to cache and retrieve
//! information about previously analyzed chess positions, improving search efficiency.

use crate::board::Board;
use crate::move_types::Move;
use std::collections::HashMap;

/// Represents an entry in the transposition table.
#[derive(PartialEq, Clone)]
pub struct TranspositionEntry {
    /// The depth at which this position was searched.
    pub(crate) depth: i32,
    /// The evaluation score for this position.
    pub(crate) score: i32,
    /// The best move found for this position.
    pub(crate) best_move: Move,
    /// Mate search results: (mate_depth, mate_move, searched_depth)
    /// None means mate search not performed yet
    /// Some((0, move, depth)) means no mate found at depth
    /// Some((mate_depth, move, depth)) means mate found in mate_depth moves
    pub(crate) mate_result: Option<(i32, Move, i32)>,
}

/// A transposition table for caching chess positions and their evaluations.
pub struct TranspositionTable {
    /// The underlying hash map storing positions and their corresponding entries.
    table: HashMap<u64, TranspositionEntry>,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TranspositionTable {
    /// Creates a new transposition table.
    pub fn new() -> Self {
        TranspositionTable {
            table: HashMap::new(),
        }
    }

    /// Checks the table for a given board position and search depth.
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the `Bitboard` position to look up.
    /// * `depth` - The current search depth.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the `TranspositionEntry` if found and the stored depth
    /// is greater than or equal to the current depth, otherwise `None`.
    pub fn probe(&self, board: &Board, depth: i32) -> Option<&TranspositionEntry> {
        // Check the table for a given board position and search depth
        // If it exists, return a reference to the entry
        // Else, return None
        let out = self.table.get(&board.zobrist_hash);
        out?;
        let entry = out.unwrap();
        if entry.depth >= depth {
            Some(entry)
        } else {
            None
        }
    }

    /// Adds a position to the transposition table or updates an existing entry.
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the `Bitboard` position to store.
    /// * `depth` - The depth at which this position was searched.
    /// * `score` - The evaluation score for this position.
    /// * `best_move` - The best move found for this position, if any.
    pub fn store(&mut self, board: &Board, depth: i32, score: i32, best_move: Move) {
        self.store_with_mate(board, depth, score, best_move, None);
    }

    /// Adds a position with mate search results to the transposition table.
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the `Bitboard` position to store.
    /// * `depth` - The depth at which this position was searched.
    /// * `score` - The evaluation score for this position.
    /// * `best_move` - The best move found for this position.
    /// * `mate_result` - Mate search results: (mate_depth, mate_move, searched_depth)
    pub fn store_with_mate(
        &mut self,
        board: &Board,
        depth: i32,
        score: i32,
        best_move: Move,
        mate_result: Option<(i32, Move, i32)>,
    ) {
        // Add a position to the table
        // If the position already exists, update it if the depth is greater
        let entry = self.table.get(&board.zobrist_hash);
        if entry.is_none() {
            self.table.insert(
                board.zobrist_hash,
                TranspositionEntry {
                    depth,
                    score,
                    best_move,
                    mate_result,
                },
            );
        } else {
            let entry = entry.unwrap();
            if depth > entry.depth {
                // Update with new search results
                self.table.insert(
                    board.zobrist_hash,
                    TranspositionEntry {
                        depth,
                        score,
                        best_move,
                        mate_result,
                    },
                );
            } else if mate_result.is_some() && entry.mate_result.is_none() {
                // Keep existing entry but add mate result if we didn't have one
                let mut updated_entry = entry.clone();
                updated_entry.mate_result = mate_result;
                self.table.insert(board.zobrist_hash, updated_entry);
            }
        }
    }

    /// Probe for mate search results in the transposition table.
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the board position to look up.
    /// * `mate_depth` - The depth for mate search.
    ///
    /// # Returns
    ///
    /// An `Option` containing mate search results if found and the stored depth
    /// is greater than or equal to the requested depth, otherwise `None`.
    pub fn probe_mate(&self, board: &Board, mate_depth: i32) -> Option<(i32, Move)> {
        if let Some(entry) = self.table.get(&board.zobrist_hash) {
            if let Some((stored_mate_depth, mate_move, searched_depth)) = entry.mate_result {
                if searched_depth >= mate_depth {
                    return Some((stored_mate_depth, mate_move));
                }
            }
        }
        None
    }

    /// Store mate search results in the transposition table.
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the board position.
    /// * `mate_depth` - The depth of mate found (0 if no mate).
    /// * `mate_move` - The best move (or null move if no mate).
    /// * `searched_depth` - The depth at which mate search was performed.
    pub fn store_mate_result(
        &mut self,
        board: &Board,
        mate_depth: i32,
        mate_move: Move,
        searched_depth: i32,
    ) {
        let mate_result = Some((mate_depth, mate_move, searched_depth));

        // Get existing entry or create a new one
        if let Some(existing_entry) = self.table.get(&board.zobrist_hash) {
            // Update existing entry with mate result
            let mut updated_entry = existing_entry.clone();
            updated_entry.mate_result = mate_result;
            self.table.insert(board.zobrist_hash, updated_entry);
        } else {
            // Create new entry with mate result only
            self.table.insert(
                board.zobrist_hash,
                TranspositionEntry {
                    depth: 0, // No regular search depth
                    score: 0, // No evaluation score
                    best_move: mate_move,
                    mate_result,
                },
            );
        }
    }

    /// Get statistics about the transposition table.
    ///
    /// # Returns
    ///
    /// A tuple containing (size, entries_with_mate_results, total_capacity)
    pub fn stats(&self) -> (usize, usize) {
        let size = self.table.len();
        let entries_with_mate = self
            .table
            .values()
            .filter(|entry| entry.mate_result.is_some())
            .count();
        (size, entries_with_mate)
    }

    /// Clears the transposition table.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}
