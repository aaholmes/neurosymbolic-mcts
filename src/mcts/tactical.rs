//! Tactical Move Detection and Prioritization
//!
//! This module implements tactical move identification and prioritization for the
//! tactical-first MCTS approach. It identifies captures and promotions, then
//! prioritizes them using MVV-LVA scoring.
//!
//! Features position-based caching to avoid redundant tactical move computation.

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::piece_types::{BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK};
use std::cell::RefCell;
use std::collections::HashMap;

/// Represents a tactical move with its associated priority score
#[derive(Debug, Clone)]
pub enum TacticalMove {
    /// Capture or promotion move with MVV-LVA score
    Capture(Move, f64),
}

impl TacticalMove {
    /// Get the underlying move
    pub fn get_move(&self) -> Move {
        let TacticalMove::Capture(mv, _) = self;
        *mv
    }

    /// Get the priority score for this tactical move
    pub fn score(&self) -> f64 {
        let TacticalMove::Capture(_, score) = self;
        *score
    }

    /// Get the tactical move type as a string
    pub fn move_type(&self) -> &'static str {
        "Capture"
    }
}

/// Position-based cache for tactical moves to avoid redundant computation
#[derive(Debug)]
pub struct TacticalMoveCache {
    /// Cache mapping zobrist hash to computed tactical moves
    cache: HashMap<u64, Vec<TacticalMove>>,
    /// Maximum number of entries to keep in cache
    max_size: usize,
    /// Cache hit statistics for monitoring
    pub hits: u64,
    pub misses: u64,
}

impl TacticalMoveCache {
    /// Create a new tactical move cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Create a new cache with default size (1000 positions)
    pub fn new_default() -> Self {
        Self::new(1000)
    }

    /// Get cached tactical moves or compute and cache them
    pub fn get_or_compute(&mut self, board: &Board, move_gen: &MoveGen) -> Vec<TacticalMove> {
        let zobrist = board.zobrist_hash;

        if let Some(cached_moves) = self.cache.get(&zobrist) {
            self.hits += 1;
            cached_moves.clone()
        } else {
            self.misses += 1;

            // Evict entries if cache is full
            if self.cache.len() >= self.max_size {
                self.evict_oldest();
            }

            // Compute tactical moves
            let tactical_moves = identify_tactical_moves_internal(board, move_gen);

            // Cache the result
            self.cache.insert(zobrist, tactical_moves.clone());

            tactical_moves
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize, u64, u64, f64) {
        let total_requests = self.hits + self.misses;
        let hit_rate = if total_requests > 0 {
            self.hits as f64 / total_requests as f64
        } else {
            0.0
        };
        (
            self.cache.len(),
            self.max_size,
            self.hits,
            self.misses,
            hit_rate,
        )
    }

    /// Evict the oldest entries (simple FIFO eviction)
    /// For better performance, consider implementing LRU in the future
    fn evict_oldest(&mut self) {
        let entries_to_remove = self.cache.len() / 4; // Remove 25% of entries
        let keys_to_remove: Vec<u64> = self.cache.keys().take(entries_to_remove).copied().collect();
        for key in keys_to_remove {
            self.cache.remove(&key);
        }
    }
}

// Thread-local cache for tactical moves
thread_local! {
    static TACTICAL_CACHE: RefCell<TacticalMoveCache> = RefCell::new(TacticalMoveCache::new_default());
}

/// Identify all tactical moves from a given position (with caching)
/// This is the main public interface that uses position-based caching
pub fn identify_tactical_moves(board: &Board, move_gen: &MoveGen) -> Vec<TacticalMove> {
    TACTICAL_CACHE.with(|cache| cache.borrow_mut().get_or_compute(board, move_gen))
}

/// Get tactical move cache statistics for monitoring performance
pub fn get_tactical_cache_stats() -> (usize, usize, u64, u64, f64) {
    TACTICAL_CACHE.with(|cache| cache.borrow().stats())
}

/// Clear the tactical move cache (useful for benchmarking or testing)
pub fn clear_tactical_cache() {
    TACTICAL_CACHE.with(|cache| cache.borrow_mut().clear())
}

/// Identify all tactical moves from a given position (without caching)
/// This is the internal implementation used by the cache
fn identify_tactical_moves_internal(board: &Board, move_gen: &MoveGen) -> Vec<TacticalMove> {
    let mut tactical_moves = Vec::new();
    let (captures, _non_captures) = move_gen.gen_pseudo_legal_moves(board);

    // Captures list from gen_pseudo_legal_moves already includes promotions
    for mv in &captures {
        if board.is_legal_after_move(*mv, move_gen) {
            let score = calculate_mvv_lva(*mv, board);
            tactical_moves.push(TacticalMove::Capture(*mv, score));
        }
    }

    // Sort by priority score (highest first)
    tactical_moves.sort_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    tactical_moves
}

/// Calculate MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score
/// For promotions, adds the promoted piece value as a bonus.
pub fn calculate_mvv_lva(mv: Move, board: &Board) -> f64 {
    let victim_value = get_piece_value_at_square(board, mv.to());
    let attacker_value = get_piece_value_at_square(board, mv.from());

    // MVV-LVA: prioritize valuable victims, deprioritize valuable attackers
    // Use 10x multiplier for victim to ensure victim value dominates
    let mut score = (victim_value * 10.0) - attacker_value;

    // Promotion bonus: add the value of the promoted piece
    if let Some(promo_piece) = mv.promotion() {
        score += get_piece_type_value(promo_piece);
    }

    score
}

/// Get the value of a piece at a given square
fn get_piece_value_at_square(board: &Board, square: usize) -> f64 {
    for color in 0..2 {
        for piece_type in 0..6 {
            if board.pieces[color][piece_type] & (1u64 << square) != 0 {
                return get_piece_type_value(piece_type);
            }
        }
    }
    0.0 // Empty square
}

/// Get the standard value of a piece type
fn get_piece_type_value(piece_type: usize) -> f64 {
    match piece_type {
        PAWN => 1.0,
        KNIGHT => 3.0,
        BISHOP => 3.0,
        ROOK => 5.0,
        QUEEN => 9.0,
        KING => 0.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_tactical_move_identification() {
        let board = Board::new(); // Starting position
        let move_gen = MoveGen::new();

        let tactical_moves = identify_tactical_moves(&board, &move_gen);

        // Starting position should have no tactical moves
        assert!(tactical_moves.is_empty());
    }

    #[test]
    fn test_tactical_cache_functionality() {
        let board = Board::new();
        let move_gen = MoveGen::new();

        // Clear cache to start fresh
        clear_tactical_cache();

        // First call should be a cache miss
        let moves1 = identify_tactical_moves(&board, &move_gen);
        let (cache_size, _, hits, misses, hit_rate) = get_tactical_cache_stats();

        assert_eq!(misses, 1);
        assert_eq!(hits, 0);
        assert!(hit_rate < 0.1);
        assert_eq!(cache_size, 1);

        // Second call should be a cache hit
        let moves2 = identify_tactical_moves(&board, &move_gen);
        let (_, _, hits, misses, hit_rate) = get_tactical_cache_stats();

        assert_eq!(misses, 1);
        assert_eq!(hits, 1);
        assert!((hit_rate - 0.5).abs() < 0.1);

        // Results should be identical
        assert_eq!(moves1.len(), moves2.len());
    }

    #[test]
    fn test_mvv_lva_calculation() {
        let board =
            Board::new_from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");

        // Pawn takes pawn: (1.0 * 10) - 1.0 = 9.0
        let mv = Move::new(28, 36, None);
        let score = calculate_mvv_lva(mv, &board);
        assert!((score - 9.0).abs() < 0.1);
    }

    #[test]
    fn test_mvv_lva_promotion_bonus() {
        // White pawn on a7, no piece on a8 — quiet queen promotion
        let board = Board::new_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");

        let mv = Move::new(48, 56, Some(QUEEN)); // a7-a8=Q
        let score = calculate_mvv_lva(mv, &board);
        // victim=0, attacker=1.0(pawn), promo=9.0 → 0 - 1.0 + 9.0 = 8.0
        assert!(
            (score - 8.0).abs() < 0.1,
            "Quiet queen promotion score should be 8.0, got {}",
            score
        );

        // Underpromotion to knight
        let mv_knight = Move::new(48, 56, Some(KNIGHT));
        let score_knight = calculate_mvv_lva(mv_knight, &board);
        // 0 - 1.0 + 3.0 = 2.0
        assert!(
            (score_knight - 2.0).abs() < 0.1,
            "Knight underpromotion score should be 2.0, got {}",
            score_knight
        );
    }

    #[test]
    fn test_captures_include_all_no_see_filter() {
        // Position where bishop captures a defended pawn — previously filtered by SEE
        let board = Board::new_from_fen("4k3/8/8/3p4/8/5B2/8/4K3 w - - 0 1");
        let move_gen = MoveGen::new();

        clear_tactical_cache();
        let tactical_moves = identify_tactical_moves(&board, &move_gen);

        // Bxd5 should be present even though it may be a "losing" capture
        let has_bxd5 = tactical_moves.iter().any(|t| {
            let mv = t.get_move();
            mv.from() == 21 && mv.to() == 35 // f3=21, d5=35
        });
        assert!(has_bxd5, "Bxd5 should be included (no SEE filtering)");
    }
}
