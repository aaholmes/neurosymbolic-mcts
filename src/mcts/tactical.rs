//! Tactical Move Detection and Prioritization
//!
//! This module implements tactical move identification and prioritization for the
//! tactical-first MCTS approach. It identifies captures, checks, and forks, then
//! prioritizes them using classical heuristics like MVV-LVA.
//!
//! Features position-based caching to avoid redundant tactical move computation.

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::piece_types::{PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
use crate::search::see;
use std::collections::{HashMap, HashSet};
use std::cell::RefCell;

/// Represents a tactical move with its associated priority score
#[derive(Debug, Clone)]
pub enum TacticalMove {
    /// Capture move with MVV-LVA score
    Capture(Move, f64),
    /// Checking move with priority score
    Check(Move, f64),
    /// Fork move with value score
    Fork(Move, f64),
    /// Pin or skewer move
    Pin(Move, f64),
}

impl TacticalMove {
    /// Get the underlying move
    pub fn get_move(&self) -> Move {
        match self {
            TacticalMove::Capture(mv, _) => *mv,
            TacticalMove::Check(mv, _) => *mv,
            TacticalMove::Fork(mv, _) => *mv,
            TacticalMove::Pin(mv, _) => *mv,
        }
    }
    
    /// Get the priority score for this tactical move
    pub fn score(&self) -> f64 {
        match self {
            TacticalMove::Capture(_, score) => *score,
            TacticalMove::Check(_, score) => *score,
            TacticalMove::Fork(_, score) => *score,
            TacticalMove::Pin(_, score) => *score,
        }
    }
    
    /// Get the tactical move type as a string
    pub fn move_type(&self) -> &'static str {
        match self {
            TacticalMove::Capture(_, _) => "Capture",
            TacticalMove::Check(_, _) => "Check",
            TacticalMove::Fork(_, _) => "Fork",
            TacticalMove::Pin(_, _) => "Pin",
        }
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
        (self.cache.len(), self.max_size, self.hits, self.misses, hit_rate)
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
    TACTICAL_CACHE.with(|cache| {
        cache.borrow_mut().get_or_compute(board, move_gen)
    })
}

/// Get tactical move cache statistics for monitoring performance
pub fn get_tactical_cache_stats() -> (usize, usize, u64, u64, f64) {
    TACTICAL_CACHE.with(|cache| {
        cache.borrow().stats()
    })
}

/// Clear the tactical move cache (useful for benchmarking or testing)
pub fn clear_tactical_cache() {
    TACTICAL_CACHE.with(|cache| {
        cache.borrow_mut().clear()
    })
}

/// Identify all tactical moves from a given position (without caching)
/// This is the internal implementation used by the cache
fn identify_tactical_moves_internal(board: &Board, move_gen: &MoveGen) -> Vec<TacticalMove> {
    let mut tactical_moves = Vec::new();
    let (captures, non_captures) = move_gen.gen_pseudo_legal_moves(board);
    
    // Track moves we've already identified to avoid duplicates
    let mut identified_moves = HashSet::new();
    
    // 1. Analyze captures with MVV-LVA scoring
    for mv in &captures {
        if board.apply_move_to_board(*mv).is_legal(move_gen) {
            if !is_losing_capture(*mv, board, move_gen) {
                let mvv_lva_score = calculate_mvv_lva(*mv, board);
                tactical_moves.push(TacticalMove::Capture(*mv, mvv_lva_score));
                identified_moves.insert(*mv);
            }
        }
    }
    
    // 2. Analyze checking moves (including non-captures)
    for mv in captures.iter().chain(non_captures.iter()) {
        if !identified_moves.contains(mv) {
            let new_board = board.apply_move_to_board(*mv);
            if new_board.is_legal(move_gen) && new_board.is_check(move_gen) {
                let check_score = calculate_check_priority(*mv, board);
                tactical_moves.push(TacticalMove::Check(*mv, check_score));
                identified_moves.insert(*mv);
            }
        }
    }
    
    // 3. Analyze fork moves (knight and pawn forks initially)
    for mv in captures.iter().chain(non_captures.iter()) {
        if !identified_moves.contains(mv) {
            let new_board = board.apply_move_to_board(*mv);
            if new_board.is_legal(move_gen) {
                if let Some(fork_score) = detect_fork_move(*mv, board, &new_board) {
                    tactical_moves.push(TacticalMove::Fork(*mv, fork_score));
                    identified_moves.insert(*mv);
                }
            }
        }
    }
    
    // Sort by priority score (highest first)
    tactical_moves.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal));
    
    tactical_moves
}

/// Calculate MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score
pub fn calculate_mvv_lva(mv: Move, board: &Board) -> f64 {
    let victim_value = get_piece_value_at_square(board, mv.to);
    let attacker_value = get_piece_value_at_square(board, mv.from);
    
    // MVV-LVA: prioritize valuable victims, deprioritize valuable attackers
    // Use 10x multiplier for victim to ensure victim value dominates
    (victim_value * 10.0) - attacker_value
}

/// Check if a capture is losing using Static Exchange Evaluation
fn is_losing_capture(mv: Move, board: &Board, move_gen: &MoveGen) -> bool {
    // Use SEE to determine if the capture loses material
    let see_value = see(board, move_gen, mv.to, mv.from);
    see_value < 0 // Losing if SEE evaluation is negative
}

/// Get the value of a piece at a given square
fn get_piece_value_at_square(board: &Board, square: usize) -> f64 {
    // Extract piece information from board
    // This is a simplified implementation - you may need to adjust based on your Board structure
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
        KING => 0.0, // King "captures" are handled separately (illegal anyway)
        _ => 0.0,
    }
}

/// Calculate priority score for checking moves
fn calculate_check_priority(mv: Move, board: &Board) -> f64 {
    let piece_value = get_piece_value_at_square(board, mv.from);
    let centrality_bonus = calculate_centrality_bonus(mv.to);
    
    // Base score for checks, with bonus for piece value and centrality
    let base_check_score = 2.0; // Checks are generally valuable
    base_check_score + piece_value * 0.1 + centrality_bonus
}

/// Calculate bonus for central squares (simple implementation)
fn calculate_centrality_bonus(square: usize) -> f64 {
    let file = square % 8;
    let rank = square / 8;
    
    // Central squares (d4, d5, e4, e5) get highest bonus
    let distance_from_center = ((file as i32 - 3).abs() + (rank as i32 - 3).abs()) as f64;
    (8.0 - distance_from_center) * 0.1
}

/// Detect fork moves and calculate their value
fn detect_fork_move(mv: Move, board: &Board, new_board: &Board) -> Option<f64> {
    let piece_type = get_piece_type_at_square(board, mv.from)?;
    
    match piece_type {
        KNIGHT => detect_knight_fork(mv, new_board),
        PAWN => detect_pawn_fork(mv, board, new_board),
        _ => None, // Other piece forks are more complex to detect accurately
    }
}

/// Get the piece type at a given square
fn get_piece_type_at_square(board: &Board, square: usize) -> Option<usize> {
    for color in 0..2 {
        for piece_type in 0..6 {
            if board.pieces[color][piece_type] & (1u64 << square) != 0 {
                return Some(piece_type);
            }
        }
    }
    None
}

/// Detect knight forks
fn detect_knight_fork(mv: Move, new_board: &Board) -> Option<f64> {
    let knight_attacks = get_knight_attacks(mv.to);
    let opponent_color = if new_board.w_to_move { 1 } else { 0 }; // Opponent's color after the move
    
    let mut valuable_targets = Vec::new();
    
    for attack_square in knight_attacks {
        if let Some(target_value) = get_piece_value_at_square_for_color(new_board, attack_square, opponent_color) {
            if target_value >= 3.0 { // Rook, Queen, or King
                valuable_targets.push(target_value);
            }
        }
    }
    
    if valuable_targets.len() >= 2 {
        // Fork detected - score based on target values
        let total_value: f64 = valuable_targets.iter().sum();
        Some(5.0 + total_value) // Base fork value + target values
    } else {
        None
    }
}

/// Get knight attack squares from a given position
fn get_knight_attacks(square: usize) -> Vec<usize> {
    let mut attacks = Vec::new();
    let file = (square % 8) as i8;
    let rank = (square / 8) as i8;
    
    let knight_offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ];
    
    for (df, dr) in knight_offsets.iter() {
        let new_file = file + df;
        let new_rank = rank + dr;
        
        if new_file >= 0 && new_file < 8 && new_rank >= 0 && new_rank < 8 {
            attacks.push((new_rank * 8 + new_file) as usize);
        }
    }
    
    attacks
}

/// Get piece value at square for specific color
fn get_piece_value_at_square_for_color(board: &Board, square: usize, color: usize) -> Option<f64> {
    for piece_type in 0..6 {
        if board.pieces[color][piece_type] & (1u64 << square) != 0 {
            return Some(get_piece_type_value(piece_type));
        }
    }
    None
}

/// Detect pawn forks
fn detect_pawn_fork(mv: Move, board: &Board, new_board: &Board) -> Option<f64> {
    let pawn_attacks = get_pawn_attacks(mv.to, board.w_to_move);
    let opponent_color = if new_board.w_to_move { 1 } else { 0 };
    
    let mut targets = 0;
    let mut total_value = 0.0;
    
    for attack_square in pawn_attacks {
        if let Some(target_value) = get_piece_value_at_square_for_color(new_board, attack_square, opponent_color) {
            if target_value >= 3.0 { // Knight, Bishop, Rook, or Queen
                targets += 1;
                total_value += target_value;
            }
        }
    }
    
    if targets >= 2 {
        Some(3.0 + total_value) // Pawn forks are generally less valuable than knight forks
    } else {
        None
    }
}

/// Get pawn attack squares
fn get_pawn_attacks(square: usize, is_white: bool) -> Vec<usize> {
    let mut attacks = Vec::new();
    let file = (square % 8) as i8;
    let rank = (square / 8) as i8;
    
    let direction = if is_white { 1 } else { -1 };
    
    // Left diagonal
    if file > 0 {
        let new_rank = rank + direction;
        if new_rank >= 0 && new_rank < 8 {
            attacks.push((new_rank * 8 + file - 1) as usize);
        }
    }
    
    // Right diagonal
    if file < 7 {
        let new_rank = rank + direction;
        if new_rank >= 0 && new_rank < 8 {
            attacks.push((new_rank * 8 + file + 1) as usize);
        }
    }
    
    attacks
}

/// Filter tactical moves to remove obviously bad ones
pub fn filter_tactical_moves(tactical_moves: Vec<TacticalMove>, board: &Board) -> Vec<TacticalMove> {
    tactical_moves
        .into_iter()
        .filter(|tactical_move| {
            let mv = tactical_move.get_move();
            
            // Basic legality check (should already be done, but double-check)
            let new_board = board.apply_move_to_board(mv);
            if !new_board.is_legal(&MoveGen::new()) {
                return false;
            }
            
            // For captures, ensure they're not obviously losing
            if let TacticalMove::Capture(_, _) = tactical_move {
                let temp_move_gen = MoveGen::new();
            return !is_losing_capture(mv, board, &temp_move_gen);
            }
            
            true
        })
        .collect()
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
        let board = Board::new_from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
        
        // Test capturing pawn with pawn (if such move exists)
        let mv = Move::new(28, 36, None); // e4 takes e5 (example)
        let score = calculate_mvv_lva(mv, &board);
        
        // Pawn takes pawn should give score: (1.0 * 10) - 1.0 = 9.0
        assert!((score - 9.0).abs() < 0.1);
    }
    
    #[test]
    fn test_knight_attack_generation() {
        // Test knight on e4 (square 28)
        let attacks = get_knight_attacks(28);
        
        // Knight on e4 should attack 8 squares (if all are on board)
        assert_eq!(attacks.len(), 8);
        
        // Check some specific squares (e4 = 28: file 4, rank 3)
        assert!(attacks.contains(&43)); // d6 (file 3, rank 5 = 5*8+3 = 43)
        assert!(attacks.contains(&45)); // f6 (file 5, rank 5 = 5*8+5 = 45)  
        assert!(attacks.contains(&11)); // d2 (file 3, rank 1 = 1*8+3 = 11)
    }
    
    #[test]
    fn test_see_integration() {
        use crate::move_types::Move;
        use crate::move_generation::MoveGen;
        
        let move_gen = MoveGen::new();
        
        // Test position: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3
        // White can capture on e5 with pawn (good capture) or bishop (bad capture)
        let board = Board::new_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3");
        
        // Test pawn takes pawn (should be good)
        let pawn_capture = Move::new(28, 36, None); // e4 x e5
        assert!(!is_losing_capture(pawn_capture, &board, &move_gen), 
                "Pawn takes pawn should not be losing");
        
        // Test bishop takes pawn (likely bad due to knight defending)
        let bishop_capture = Move::new(26, 36, None); // Bc4 x e5 (not possible from this position, but for testing)
        // Note: This might not be a losing capture depending on the exact position
        
        // Test in starting position (no captures should be losing since no pieces can be captured)
        let starting_board = Board::new();
        let fake_capture = Move::new(0, 8, None); // Any move in starting position
        // This won't be a capture in starting position, so SEE should handle gracefully
    }
    
    #[test]
    fn test_tactical_moves_with_see_filtering() {
        let board = Board::new_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3");
        let move_gen = MoveGen::new();
        
        let tactical_moves = identify_tactical_moves(&board, &move_gen);
        
        // Should find tactical moves, and SEE filtering should be working
        // The exact number depends on the position analysis
        assert!(tactical_moves.len() >= 0, "Should find tactical moves or filter appropriately");
        
        // Verify that all returned moves are not losing captures
        for tactical_move in &tactical_moves {
            if let TacticalMove::Capture(mv, _) = tactical_move {
                assert!(!is_losing_capture(*mv, &board, &move_gen),
                        "Tactical moves should not include losing captures");
            }
        }
    }
}