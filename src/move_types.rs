//! Defines the Move struct and related methods for representing chess moves.
//!
//! This module provides the core `Move` type used throughout the chess engine
//! to represent and manipulate chess moves.

use crate::board_utils::sq_ind_to_algebraic;
use crate::piece_types::{BISHOP, KNIGHT, QUEEN, ROOK};
use std::fmt;
use std::hash::{Hash, Hasher};

// Constants for castling squares
const C1: usize = 2;  // White queenside castle destination
const E1: usize = 4;  // King starting square (white)
const G1: usize = 6;  // White kingside castle destination
const C8: usize = 58; // Black queenside castle destination
const E8: usize = 60; // King starting square (black)
const G8: usize = 62; // Black kingside castle destination

/// Represents a chess move.
///
/// This struct contains information about the source square, destination square,
/// and any promotion that occurs as a result of the move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Move {
    /// The index of the square the piece is moving from (0-63).
    pub from: usize,
    /// The index of the square the piece is moving to (0-63).
    pub to: usize,
    /// The type of piece to promote to, if this move results in a promotion.
    /// `None` if the move does not result in a promotion.
    pub promotion: Option<usize>,
}

// Implement Hash to enable using Move as a HashMap key
impl Hash for Move {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.from.hash(state);
        self.to.hash(state);
        if let Some(promotion) = self.promotion {
            promotion.hash(state);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CastlingRights {
    pub white_kingside: bool,
    pub white_queenside: bool,
    pub black_kingside: bool,
    pub black_queenside: bool,
}

impl CastlingRights {
    pub(crate) fn default() -> CastlingRights {
        CastlingRights {
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
        }
    }
}

/// A constant representing a null move, used in search algorithms
pub const NULL_MOVE: Move = Move {
    from: 0,
    to: 0,
    promotion: None,
};

impl Move {
    /// Creates a new `Move` instance.
    ///
    /// # Arguments
    ///
    /// * `from` - The index of the source square (0-63).
    /// * `to` - The index of the destination square (0-63).
    /// * `promotion` - The type of piece to promote to, if applicable. Use `None` if not a promotion.
    ///
    /// # Returns
    ///
    /// A new `Move` instance with the specified parameters.
    pub fn new(from: usize, to: usize, promotion: Option<usize>) -> Move {
        // New move
        Move {
            from,
            to,
            promotion,
        }
    }

    /// Creates a new `Move` instance from a UCI string.
    ///
    /// # Arguments
    ///
    /// * `uci` - A string representing the move in UCI format (e.g., "e2e4", "e7e8q").
    ///
    /// # Returns
    ///
    /// An `Option<Move>` which is `Some(Move)` if the UCI string is valid, or `None` if it's invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use kingfisher::move_types::Move;
    /// let move1 = Move::from_uci("e2e4").unwrap();
    /// let move2 = Move::from_uci("e7e8q").unwrap(); // Promotion to queen
    /// assert!(Move::from_uci("invalid").is_none());
    /// ```
    pub fn from_uci(uci: &str) -> Option<Move> {
        if uci.len() < 4 || uci.len() > 5 {
            return None;
        }

        let from_file = (uci.chars().nth(0)? as u8).wrapping_sub(b'a');
        let from_rank = (uci.chars().nth(1)? as u8).wrapping_sub(b'1');
        let to_file = (uci.chars().nth(2)? as u8).wrapping_sub(b'a');
        let to_rank = (uci.chars().nth(3)? as u8).wrapping_sub(b'1');

        if from_file > 7 || from_rank > 7 || to_file > 7 || to_rank > 7 {
            return None;
        }

        let from: usize = (from_rank * 8 + from_file) as usize;
        let to: usize = (to_rank * 8 + to_file) as usize;

        let promotion = if uci.len() == 5 {
            // Make sure the promotion is on the last rank
            if to_rank != 7 && to_rank != 0 {
                return None;
            }
            match uci.chars().nth(4)? {
                'n' => Some(KNIGHT),
                'b' => Some(BISHOP),
                'r' => Some(ROOK),
                'q' => Some(QUEEN),
                _ => return None,
            }
        } else {
            None
        };

        Some(Move {
            from,
            to,
            promotion,
        })
    }

    /// Creates a null move.
    ///
    /// A null move is a special move used in chess engines to pass the turn
    /// without making an actual move on the board. It's typically used in
    /// null move pruning and other search techniques.
    ///
    /// # Returns
    ///
    /// A `Move` instance representing a null move, with `from` and `to` set to 0
    /// and `promotion` set to `None`.
    pub fn null() -> Move {
        // Null move
        Move {
            from: 0,
            to: 0,
            promotion: None,
        }
    }

    /// Checks if the move is a promotion.
    ///
    /// # Returns
    ///
    /// True if the move has a promotion piece specified, false otherwise.
    pub fn is_promotion(&self) -> bool {
        self.promotion.is_some()
    }

    /// Checks if the move is an en passant capture.
    ///
    /// This is determined by looking at the characteristic pattern of an en passant move:
    /// - A pawn move (from 6th/3rd rank to 5th/4th rank)
    /// - Moving diagonally
    ///
    /// Note: This is a heuristic and should be used alongside board state to confirm.
    ///
    /// # Returns
    ///
    /// True if the move appears to be an en passant capture, false otherwise.
    pub fn is_en_passant(&self) -> bool {
        // Check if moving diagonally (file difference of 1)
        let from_file = self.from % 8;
        let to_file = self.to % 8;
        let file_diff = if from_file > to_file { from_file - to_file } else { to_file - from_file };
        
        if file_diff != 1 {
            return false;
        }
        
        // Check if it's a pawn move from rank 4 to 5 (white) or rank 3 to 2 (black)
        let from_rank = self.from / 8;
        let to_rank = self.to / 8;
        
        (from_rank == 4 && to_rank == 5) || (from_rank == 3 && to_rank == 2)
    }

    /// Checks if the move is kingside castling.
    ///
    /// # Returns
    ///
    /// True if the move is kingside castling, false otherwise.
    pub fn is_kingside_castle(&self) -> bool {
        (self.from == E1 && self.to == G1) || // White kingside (e1g1)
        (self.from == E8 && self.to == G8)    // Black kingside (e8g8)
    }

    /// Checks if the move is queenside castling.
    ///
    /// # Returns
    ///
    /// True if the move is queenside castling, false otherwise.
    pub fn is_queenside_castle(&self) -> bool {
        (self.from == E1 && self.to == C1) || // White queenside (e1c1)
        (self.from == E8 && self.to == C8)    // Black queenside (e8c8)
    }

    /// Checks if the move is any type of castling.
    ///
    /// # Returns
    ///
    /// True if the move is either kingside or queenside castling, false otherwise.
    pub fn is_castle(&self) -> bool {
        self.is_kingside_castle() || self.is_queenside_castle()
    }

    /// Convert move to UCI notation
    pub fn to_uci(&self) -> String {
        let from = sq_ind_to_algebraic(self.from);
        let to = sq_ind_to_algebraic(self.to);
        let mut result = format!("{}{}", from, to);
        
        if let Some(promotion) = self.promotion {
            let promotion_char = match promotion {
                n if n == KNIGHT => 'n',
                n if n == BISHOP => 'b', 
                n if n == ROOK => 'r',
                n if n == QUEEN => 'q',
                _ => 'q', // Default to queen
            };
            result.push(promotion_char);
        }
        
        result
    }

    /// Returns a new move with squares flipped vertically.
    pub fn flip_vertical(&self) -> Move {
        use crate::board_utils::flip_sq_ind_vertically;
        Move {
            from: flip_sq_ind_vertically(self.from),
            to: flip_sq_ind_vertically(self.to),
            promotion: self.promotion,
        }
    }

    /// Change the way a move is printed so that it uses algebraic notation
    pub fn print_algebraic(&self) -> String {
        let from = sq_ind_to_algebraic(self.from);
        let to = sq_ind_to_algebraic(self.to);
        let mut promotion = String::from("");
        if self.promotion.is_some() {
            promotion = String::from("=");
            match self.promotion.unwrap() {
                n if n == KNIGHT => promotion.push('N'),
                n if n == BISHOP => promotion.push('B'),
                n if n == ROOK => promotion.push('R'),
                n if n == QUEEN => promotion.push('Q'),
                _ => panic!("Invalid promotion piece"),
            }
        }
        format!("{}{}{}", from, to, promotion)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.print_algebraic())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_from_uci() {
        assert_eq!(
            Move::from_uci("e2e4"),
            Some(Move {
                from: 12,
                to: 28,
                promotion: None
            })
        );
        assert_eq!(
            Move::from_uci("a7a8q"),
            Some(Move {
                from: 48,
                to: 56,
                promotion: Some(QUEEN)
            })
        );
        assert_eq!(
            Move::from_uci("h2h1n"),
            Some(Move {
                from: 15,
                to: 7,
                promotion: Some(KNIGHT)
            })
        );
        assert_eq!(
            Move::from_uci("e1g1"),
            Some(Move {
                from: 4,
                to: 6,
                promotion: None
            })
        ); // Castling
        assert_eq!(Move::from_uci("invalid"), None);
        assert_eq!(Move::from_uci("e2e9"), None); // Invalid square
        assert_eq!(Move::from_uci("e2e4q"), None); // Invalid promotion (not on last rank)
    }

    #[test]
    fn test_move_special_methods() {
        // Test promotion detection
        let promotion_move = Move::new(48, 56, Some(QUEEN)); // a7a8q
        assert!(promotion_move.is_promotion());
        let normal_move = Move::new(12, 28, None); // e2e4
        assert!(!normal_move.is_promotion());

        // Test en passant detection
        let white_ep = Move::new(36, 45, None); // e5xd6 (e.p.)
        assert!(white_ep.is_en_passant());
        let black_ep = Move::new(27, 18, None); // d4xe3 (e.p.)
        assert!(black_ep.is_en_passant());
        assert!(!normal_move.is_en_passant());

        // Test castling detection
        let white_kingside = Move::new(E1, G1, None); // e1g1
        assert!(white_kingside.is_kingside_castle());
        assert!(!white_kingside.is_queenside_castle());
        assert!(white_kingside.is_castle());

        let black_queenside = Move::new(E8, C8, None); // e8c8
        assert!(black_queenside.is_queenside_castle());
        assert!(!black_queenside.is_kingside_castle());
        assert!(black_queenside.is_castle());

        assert!(!normal_move.is_castle());
    }
}
