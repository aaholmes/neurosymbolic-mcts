//! Chess piece type and color constants.
//!
//! Defines the fundamental constants used throughout the engine for indexing
//! piece bitboards and identifying piece types. These constants are used as
//! array indices into the `Board.pieces[color][piece_type]` bitboard array.
//!
//! # Piece Type Indices
//! - `PAWN = 0`, `KNIGHT = 1`, `BISHOP = 2`, `ROOK = 3`, `QUEEN = 4`, `KING = 5`
//!
//! # Color Indices
//! - `WHITE = 0`, `BLACK = 1`

/// Represents the type of a chess piece.
pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

/// Represents the color of a chess piece.
pub const WHITE: usize = 0;
pub const BLACK: usize = 1;
