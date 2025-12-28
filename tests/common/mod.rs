/// Shared test utilities for Kingfisher test suite

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use std::collections::HashSet;

/// Initialize a board from FEN
pub fn board_from_fen(fen: &str) -> Board {
    Board::new_from_fen(fen)
}

/// Get all legal moves as a HashSet for easy comparison
pub fn legal_moves_set(board: &Board, move_gen: &MoveGen) -> HashSet<Move> {
    generate_legal_moves(board, move_gen).into_iter().collect()
}

/// Generate legal moves (helper)
pub fn generate_legal_moves(board: &Board, move_gen: &MoveGen) -> Vec<Move> {
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
    let mut legal_moves = Vec::with_capacity(captures.len() + moves.len());
    for m in captures.into_iter().chain(moves.into_iter()) {
        let next_state = board.apply_move_to_board(m);
        if next_state.is_legal(move_gen) {
            legal_moves.push(m);
        }
    }
    legal_moves
}

/// Assert two f64 values are approximately equal
pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64, context: &str) {
    assert!(
        (a - b).abs() < epsilon,
        "{}: expected {} ≈ {}, difference {} exceeds epsilon {}",
        context, a, b, (a - b).abs(), epsilon
    );
}

/// Assert a value is in tanh domain [-1, 1]
pub fn assert_in_tanh_domain(v: f64, context: &str) {
    assert!(
        v >= -1.0 && v <= 1.0,
        "{}: value {} outside tanh domain [-1, 1]",
        context, v
    );
}

/// Standard test positions with known properties
pub mod positions {
    pub const STARTING: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    pub const MATE_IN_1_WHITE: &str = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"; // Re8#
    pub const MATE_IN_1_BLACK: &str = "4r1k1/8/8/8/8/8/5PPP/6K1 b - - 0 1"; // Re1#
    pub const STALEMATE: &str = "k7/1R6/K7/8/8/8/8/8 b - - 0 1"; // Black stalemated
    pub const KOTH_WIN_AVAILABLE: &str = "8/8/3K4/8/8/8/8/k7 w - - 0 1"; // Kd5/Ke5 wins
    pub const EN_PASSANT: &str = "8/8/8/pP6/8/8/8/K6k w - a6 0 1";
    pub const CASTLING_BOTH: &str = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    pub const PROMOTION: &str = "8/P7/8/8/8/8/8/K6k w - - 0 1";
    
    // Tactical positions
    pub const WINNING_CAPTURE: &str = "8/8/8/3q4/4N3/8/8/K6k w - - 0 1"; // Nxd5 wins queen
    pub const FORK_AVAILABLE: &str = "r3k2r/8/8/8/3N4/8/8/K7 w - - 0 1"; // Nc6 forks rooks
    
    // Material evaluation positions
    pub const EQUAL_MATERIAL: &str = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
    pub const WHITE_UP_QUEEN: &str = "4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1";
    pub const BLACK_UP_QUEEN: &str = "3qk3/8/8/8/8/8/8/4K3 w - - 0 1";
}
