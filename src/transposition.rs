//! Transposition table for storing previously seen positions.
//!
//! This module implements a transposition table, which is used to cache and retrieve
//! information about previously analyzed chess positions, improving search efficiency.

use crate::board::Board;
use crate::move_types::Move;
use std::collections::HashMap;

/// The kind of bound a stored score represents relative to the alpha-beta
/// window in which it was produced.
///
/// Without this flag a fail-hard alpha-beta search stores `beta` on a cutoff
/// and the last move's score on a fail-low, both of which are *bounds*, not
/// exact values. Reusing them as exact scores returns wrong results.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Bound {
    /// The score is exact (`alpha < score < beta`, a move improved alpha).
    Exact,
    /// The score is a lower bound (fail-high / beta cutoff: `score >= beta`).
    /// The true value is at least `score`.
    Lower,
    /// The score is an upper bound (fail-low: no move improved alpha).
    /// The true value is at most `score`.
    Upper,
}

/// Represents an entry in the transposition table.
#[derive(PartialEq, Clone)]
pub struct TranspositionEntry {
    /// The depth at which this position was searched.
    pub(crate) depth: i32,
    /// The evaluation score for this position.
    pub(crate) score: i32,
    /// The bound type of the stored score relative to its search window.
    pub(crate) bound: Bound,
    /// The best move found for this position.
    pub(crate) best_move: Move,
    /// Mate search results: (mate_depth, mate_move, searched_depth)
    /// None means mate search not performed yet
    /// Some((0, move, depth)) means no mate found at depth
    /// Some((mate_depth, move, depth)) means mate found in mate_depth moves
    pub(crate) mate_result: Option<(i32, Move, i32)>,
}

impl TranspositionEntry {
    /// The bound type of the stored score.
    pub fn bound(&self) -> Bound {
        self.bound
    }

    /// The stored evaluation score.
    pub fn score(&self) -> i32 {
        self.score
    }

    /// The best move stored for this position.
    pub fn best_move(&self) -> Move {
        self.best_move
    }
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

    /// Look up the raw entry regardless of depth (used for move ordering).
    pub fn probe_entry(&self, board: &Board) -> Option<&TranspositionEntry> {
        self.table.get(&board.zobrist_hash)
    }

    /// Probe for a directly usable score given the current `[alpha, beta]`
    /// search window.
    ///
    /// A stored score may only be returned as the value of the node when its
    /// bound is *compatible* with the window:
    /// - `Exact` is always usable.
    /// - `Lower` (`score >= beta` when stored) is usable only if it still
    ///   produces a beta cutoff in the current window (`score >= beta`).
    /// - `Upper` (`score <= alpha` when stored) is usable only if it still
    ///   produces a fail-low in the current window (`score <= alpha`).
    ///
    /// The stored depth must be at least `depth`. Mate handling is left to the
    /// caller; stored scores are returned verbatim. Returns `None` when the
    /// entry exists but can only be used for move ordering.
    pub fn probe_value(&self, board: &Board, depth: i32, alpha: i32, beta: i32) -> Option<i32> {
        let entry = self.table.get(&board.zobrist_hash)?;
        if entry.depth < depth {
            return None;
        }
        match entry.bound {
            Bound::Exact => Some(entry.score),
            Bound::Lower if entry.score >= beta => Some(entry.score),
            Bound::Upper if entry.score <= alpha => Some(entry.score),
            _ => None,
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
        self.store_with_bound(board, depth, score, Bound::Exact, best_move, None);
    }

    /// Store a position with an explicit score bound (Exact/Lower/Upper).
    pub fn store_with_bound(
        &mut self,
        board: &Board,
        depth: i32,
        score: i32,
        bound: Bound,
        best_move: Move,
        mate_result: Option<(i32, Move, i32)>,
    ) {
        let entry = self.table.get(&board.zobrist_hash);
        if entry.is_none() {
            self.table.insert(
                board.zobrist_hash,
                TranspositionEntry {
                    depth,
                    score,
                    bound,
                    best_move,
                    mate_result,
                },
            );
        } else {
            let entry = entry.unwrap();
            if depth > entry.depth {
                self.table.insert(
                    board.zobrist_hash,
                    TranspositionEntry {
                        depth,
                        score,
                        bound,
                        best_move,
                        mate_result,
                    },
                );
            } else if mate_result.is_some() && entry.mate_result.is_none() {
                let mut updated_entry = entry.clone();
                updated_entry.mate_result = mate_result;
                self.table.insert(board.zobrist_hash, updated_entry);
            }
        }
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
        self.store_with_bound(board, depth, score, Bound::Exact, best_move, mate_result);
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
                    bound: Bound::Exact,
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
