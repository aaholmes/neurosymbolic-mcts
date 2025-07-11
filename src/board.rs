//! This module defines the Bitboard structure and associated functions for chess board representation.

use crate::board_utils::{algebraic_to_sq_ind, bit_to_sq_ind, coords_to_sq_ind, sq_ind_to_bit};
use crate::move_generation::MoveGen;
use crate::move_types::CastlingRights;
use crate::piece_types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};

/// Represents the chess board using bitboards.
///
/// Each piece type and color has its own 64-bit unsigned integer,
/// where each bit represents a square on the chess board.
#[derive(Clone, Debug)]
pub struct Board {
    pub(crate) pieces: [[u64; 6]; 2], // [Color as usize][PieceType as usize]
    pub(crate) pieces_occ: [u64; 2],  // Total occupancy for each color
    pub w_to_move: bool,
    pub(crate) en_passant: Option<u8>,
    pub castling_rights: CastlingRights,
    pub(crate) halfmove_clock: u8,
    pub(crate) fullmove_number: u8,
    pub(crate) zobrist_hash: u64,
    pub(crate) eval: i32,
    pub game_phase: i32,
}

impl Board {
    pub fn new() -> Board {
        let mut board = Board {
            pieces: [[0; 6]; 2],
            pieces_occ: [0; 2],
            w_to_move: true,
            en_passant: None,
            castling_rights: CastlingRights::default(),
            halfmove_clock: 0,
            fullmove_number: 1,
            zobrist_hash: 0,
            eval: 0,
            game_phase: 24,
        };
        board.init_position();
        board.zobrist_hash = board.compute_zobrist_hash();
        board
    }

    fn init_position(&mut self) {
        // Set up pieces in the starting position
        self.pieces[WHITE][PAWN] = 0x000000000000FF00;
        self.pieces[BLACK][PAWN] = 0x00FF000000000000;
        self.pieces[WHITE][KNIGHT] = 0x0000000000000042;
        self.pieces[BLACK][KNIGHT] = 0x4200000000000000;
        self.pieces[WHITE][BISHOP] = 0x0000000000000024;
        self.pieces[BLACK][BISHOP] = 0x2400000000000000;
        self.pieces[WHITE][ROOK] = 0x0000000000000081;
        self.pieces[BLACK][ROOK] = 0x8100000000000000;
        self.pieces[WHITE][QUEEN] = 0x0000000000000008;
        self.pieces[BLACK][QUEEN] = 0x0800000000000000;
        self.pieces[WHITE][KING] = 0x0000000000000010;
        self.pieces[BLACK][KING] = 0x1000000000000000;
        self.pieces_occ[WHITE] = 0x000000000000FFFF;
        self.pieces_occ[BLACK] = 0xFFFF000000000000;

        // Update occupancy bitboards
        self.update_occupancy();
    }

    pub(crate) fn update_occupancy(&mut self) {
        for color in [WHITE, BLACK] {
            self.pieces_occ[color] = self.pieces[color].iter().fold(0, |acc, &x| acc | x);
        }
    }

    /// Creates a new Bitboard from a FEN (Forsyth–Edwards Notation) string.
    ///
    /// # Arguments
    ///
    /// * `fen` - A string slice that holds the FEN representation of a chess position.
    ///
    /// # Returns
    ///
    /// A new Bitboard struct representing the chess position described by the FEN string.
    pub fn new_from_fen(fen: &str) -> Board {
        let parts = fen.split(' ').collect::<Vec<&str>>();
        let mut board = Board::new();
        board.pieces = [[0; 6]; 2];
        board.pieces_occ = [0; 2];
        board.castling_rights.white_kingside = false;
        board.castling_rights.white_queenside = false;
        board.castling_rights.black_kingside = false;
        board.castling_rights.black_queenside = false;
        let mut rank = 7;
        let mut file = 0;
        for c in parts[0].chars() {
            if c == '/' {
                rank -= 1;
                file = 0;
            } else if c.is_ascii_digit() {
                file += c.to_digit(10).unwrap() as usize;
            } else {
                let sq_ind = coords_to_sq_ind(file, rank);
                let bit = sq_ind_to_bit(sq_ind);
                match c {
                    'P' => board.pieces[WHITE][PAWN] ^= bit,
                    'p' => board.pieces[BLACK][PAWN] ^= bit,
                    'N' => board.pieces[WHITE][KNIGHT] ^= bit,
                    'n' => board.pieces[BLACK][KNIGHT] ^= bit,
                    'B' => board.pieces[WHITE][BISHOP] ^= bit,
                    'b' => board.pieces[BLACK][BISHOP] ^= bit,
                    'R' => board.pieces[WHITE][ROOK] ^= bit,
                    'r' => board.pieces[BLACK][ROOK] ^= bit,
                    'Q' => board.pieces[WHITE][QUEEN] ^= bit,
                    'q' => board.pieces[BLACK][QUEEN] ^= bit,
                    'K' => board.pieces[WHITE][KING] ^= bit,
                    'k' => board.pieces[BLACK][KING] ^= bit,
                    _ => panic!("Invalid FEN"),
                }
                file += 1;
            }
        }
        match parts[1] {
            "w" => board.w_to_move = true,
            "b" => board.w_to_move = false,
            _ => panic!("Invalid FEN"),
        }
        match parts[2] {
            "-" => (),
            _ => {
                for c in parts[2].chars() {
                    match c {
                        'K' => board.castling_rights.white_kingside = true,
                        'Q' => board.castling_rights.white_queenside = true,
                        'k' => board.castling_rights.black_kingside = true,
                        'q' => board.castling_rights.black_queenside = true,
                        _ => panic!("Invalid FEN"),
                    }
                }
            }
        }
        match parts[3] {
            "-" => (),
            _ => {
                let sq_ind = algebraic_to_sq_ind(parts[3]);
                board.en_passant = Some(sq_ind as u8);
            }
        }
        match parts[4] {
            "0" => (),
            _ => {
                board.halfmove_clock = parts[4].parse::<u8>().unwrap();
            }
        }
        match parts[5] {
            "1" => (),
            _ => {
                board.fullmove_number = parts[5].parse::<u8>().unwrap();
            }
        }
        for color in 0..2 {
            board.pieces_occ[color] = board.pieces[color][PAWN];
            for piece in 1..6 {
                board.pieces_occ[color] |= board.pieces[color][piece];
            }
        }
        board
    }

    /// Prints a visual representation of the chess board to the console.
    pub fn print(&self) {
        println!("  +-----------------+");
        for rank in (0..8).rev() {
            print!("{} | ", rank + 1);
            for file in 0..8 {
                let sq_ind = coords_to_sq_ind(file, rank);
                let bit = sq_ind_to_bit(sq_ind);
                if bit & self.pieces[WHITE][PAWN] != 0 {
                    print!("P ");
                } else if bit & self.pieces[BLACK][PAWN] != 0 {
                    print!("p ");
                } else if bit & self.pieces[WHITE][KNIGHT] != 0 {
                    print!("N ");
                } else if bit & self.pieces[BLACK][KNIGHT] != 0 {
                    print!("n ");
                } else if bit & self.pieces[WHITE][BISHOP] != 0 {
                    print!("B ");
                } else if bit & self.pieces[BLACK][BISHOP] != 0 {
                    print!("b ");
                } else if bit & self.pieces[WHITE][ROOK] != 0 {
                    print!("R ");
                } else if bit & self.pieces[BLACK][ROOK] != 0 {
                    print!("r ");
                } else if bit & self.pieces[WHITE][QUEEN] != 0 {
                    print!("Q ");
                } else if bit & self.pieces[BLACK][QUEEN] != 0 {
                    print!("q ");
                } else if bit & self.pieces[WHITE][KING] != 0 {
                    print!("K ");
                } else if bit & self.pieces[BLACK][KING] != 0 {
                    print!("k ");
                } else {
                    print!(". ");
                }
            }
            println!("|");
        }
        println!("  +-----------------+");
        println!("    a b c d e f g h");
    }

    /// Gets the piece type at a given square index.
    ///
    /// # Arguments
    ///
    /// * `sq_ind` - The square index to check (0-63)
    ///
    /// # Returns
    ///
    /// An Option containing the piece type if a piece is present, or None if the square is empty.
    pub fn get_piece(&self, sq_ind: usize) -> Option<(usize, usize)> {
        let bit = sq_ind_to_bit(sq_ind);
        if bit & self.pieces[WHITE][PAWN] != 0 {
            Some((WHITE, PAWN))
        } else if bit & self.pieces[BLACK][PAWN] != 0 {
            Some((BLACK, PAWN))
        } else if bit & self.pieces[WHITE][KNIGHT] != 0 {
            Some((WHITE, KNIGHT))
        } else if bit & self.pieces[BLACK][KNIGHT] != 0 {
            Some((BLACK, KNIGHT))
        } else if bit & self.pieces[WHITE][BISHOP] != 0 {
            Some((WHITE, BISHOP))
        } else if bit & self.pieces[BLACK][BISHOP] != 0 {
            Some((BLACK, BISHOP))
        } else if bit & self.pieces[WHITE][ROOK] != 0 {
            Some((WHITE, ROOK))
        } else if bit & self.pieces[BLACK][ROOK] != 0 {
            Some((BLACK, ROOK))
        } else if bit & self.pieces[WHITE][QUEEN] != 0 {
            Some((WHITE, QUEEN))
        } else if bit & self.pieces[BLACK][QUEEN] != 0 {
            Some((BLACK, QUEEN))
        } else if bit & self.pieces[WHITE][KING] != 0 {
            Some((WHITE, KING))
        } else if bit & self.pieces[BLACK][KING] != 0 {
            Some((BLACK, KING))
        } else {
            None
        }
    }

    /// Returns the bitboard for a specific piece type and color.
    ///
    /// # Arguments
    ///
    /// * `color` - The color of the piece (White or Black)
    /// * `piece_type` - The type of the piece (Pawn, Knight, Bishop, Rook, Queen, or King)
    ///
    /// # Returns
    ///
    /// A 64-bit unsigned integer representing the bitboard for the specified piece type and color.
    ///
    /// # Examples
    ///
    /// ```
    /// use kingfisher::board::Board;
    /// use kingfisher::piece_types::{PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK};
    /// let board = Board::new(); // Assume this creates a standard starting position
    /// let white_pawns = board.get_piece_bitboard(WHITE, PAWN);
    /// let black_pawns = board.get_piece_bitboard(BLACK, PAWN);
    /// assert_eq!(white_pawns, 0x000000000000FF00); // All white pawns on their starting squares
    /// assert_eq!(black_pawns, 0x00FF000000000000); // All black pawns on their starting squares
    /// ```
    pub fn get_piece_bitboard(&self, color: usize, piece_type: usize) -> u64 {
        self.pieces[color][piece_type]
    }

    /// Gets the occupancy bitboard for a color.
    ///
    /// # Arguments
    ///
    /// * `color` - The color (White or Black)
    ///
    /// # Returns
    ///
    /// A 64-bit unsigned integer representing all pieces of the specified color.
    pub fn get_color_occupancy(&self, color: usize) -> u64 {
        self.pieces_occ[color]
    }

    /// Gets the combined occupancy of all pieces on the board.
    ///
    /// # Returns
    ///
    /// A 64-bit unsigned integer representing all pieces on the board.
    pub fn get_all_occupancy(&self) -> u64 {
        self.pieces_occ[WHITE] | self.pieces_occ[BLACK]
    }

    /// Determines whether the current position is legal.
    ///
    /// A position is considered legal if the side to move cannot capture the opponent's king.
    ///
    /// # Arguments
    ///
    /// * `move_gen` - A reference to a MoveGen struct for generating potential moves.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the current position is legal.
    pub fn is_legal(&self, move_gen: &MoveGen) -> bool {
        let king_sq_ind: usize;
        if self.w_to_move {
            king_sq_ind = bit_to_sq_ind(self.pieces[BLACK][KING]);
            if king_sq_ind == 64 {
                println!("No black king");
                self.print();
            }
        } else {
            king_sq_ind = bit_to_sq_ind(self.pieces[WHITE][KING]);
            if king_sq_ind == 64 {
                println!("No white king");
                self.print();
            }
        }
        !self.is_square_attacked(king_sq_ind, self.w_to_move, move_gen)
    }

    /// Determines whether the current position is checkmate or stalemate.
    ///
    /// A position is considered checkmate if the side to move is in check and has no legal moves.
    /// A position is considered stalemate if the side to move is not in check but has no legal moves.
    ///
    /// # Arguments
    ///
    /// * `move_gen` - A reference to a MoveGen struct for generating potential moves.
    ///
    /// # Returns
    ///
    /// A tuple (bool, bool) where:
    /// - The first boolean is true if the position is checkmate, false otherwise.
    /// - The second boolean is true if the position is stalemate, false otherwise.
    pub fn is_checkmate_or_stalemate(&self, move_gen: &MoveGen) -> (bool, bool) {
        // Generate all pseudo-legal moves
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(self);

        // Check if any of the captures are legal
        if !captures.is_empty() {
            for c in captures {
                let new_board = self.apply_move_to_board(c);
                if new_board.is_legal(move_gen) {
                    return (false, false);
                }
            }
        }

        // Check if any of the non-capture moves are legal
        if !moves.is_empty() {
            for m in moves {
                let new_board = self.apply_move_to_board(m);
                if new_board.is_legal(move_gen) {
                    return (false, false);
                }
            }
        }

        // If we get here, there are no legal moves.
        // Check if the current position is in check
        let is_check = self.is_check(move_gen);
        if is_check {
            (true, false) // Checkmate
        } else {
            (false, true) // Stalemate
        }
    }

    /// Checks if the king of the side to move is in check.
    ///
    /// # Arguments
    ///
    /// * `move_gen` - A reference to a MoveGen struct for generating potential moves.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the king of the side to move is in check.
    pub fn is_check(&self, move_gen: &MoveGen) -> bool {
        let king_sq_ind: usize;
        if self.w_to_move {
            if self.pieces[WHITE][KING] == 0 {
                // No king on the board for the side to move - can't be in check
                return false;
            }
            king_sq_ind = bit_to_sq_ind(self.pieces[WHITE][KING]);
        } else {
            if self.pieces[BLACK][KING] == 0 {
                // No king on the board for the side to move - can't be in check
                return false;
            }
            king_sq_ind = bit_to_sq_ind(self.pieces[BLACK][KING]);
        }
        self.is_square_attacked(king_sq_ind, !self.w_to_move, move_gen)
    }

    /// Convert board to FEN notation
    pub fn to_fen(&self) -> Option<String> {
        let mut fen = String::new();
        
        // 1. Piece placement (from rank 8 to rank 1)
        for rank in (0..8).rev() {
            let mut empty_count = 0;
            for file in 0..8 {
                let square = rank * 8 + file;
                let square_bit = 1u64 << square;
                
                let mut piece_found = false;
                for color in 0..2 {
                    for piece_type in 0..6 {
                        if self.pieces[color][piece_type] & square_bit != 0 {
                            if empty_count > 0 {
                                fen.push_str(&empty_count.to_string());
                                empty_count = 0;
                            }
                            let piece_char = match (color, piece_type) {
                                (WHITE, PAWN) => 'P', (BLACK, PAWN) => 'p',
                                (WHITE, ROOK) => 'R', (BLACK, ROOK) => 'r',
                                (WHITE, KNIGHT) => 'N', (BLACK, KNIGHT) => 'n',
                                (WHITE, BISHOP) => 'B', (BLACK, BISHOP) => 'b',
                                (WHITE, QUEEN) => 'Q', (BLACK, QUEEN) => 'q',
                                (WHITE, KING) => 'K', (BLACK, KING) => 'k',
                                _ => '?'
                            };
                            fen.push(piece_char);
                            piece_found = true;
                            break;
                        }
                    }
                    if piece_found { break; }
                }
                if !piece_found {
                    empty_count += 1;
                }
            }
            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        
        // 2. Active color
        fen.push(' ');
        fen.push(if self.w_to_move { 'w' } else { 'b' });
        
        // 3. Castling availability
        fen.push(' ');
        let mut castling = String::new();
        if self.castling_rights.white_kingside { castling.push('K'); }
        if self.castling_rights.white_queenside { castling.push('Q'); }
        if self.castling_rights.black_kingside { castling.push('k'); }
        if self.castling_rights.black_queenside { castling.push('q'); }
        if castling.is_empty() { castling.push('-'); }
        fen.push_str(&castling);
        
        // 4. En passant target square
        fen.push(' ');
        match self.en_passant {
            Some(sq) => {
                let file = (sq % 8) as u8;
                let rank = (sq / 8) as u8;
                fen.push((b'a' + file) as char);
                fen.push((b'1' + rank) as char);
            }
            None => fen.push('-')
        }
        
        // 5. Halfmove clock and fullmove number
        fen.push_str(&format!(" {} {}", self.halfmove_clock, self.fullmove_number));
        
        Some(fen)
    }

    /// Checks if a square is attacked by a given side.
    ///
    /// # Arguments
    ///
    /// * `sq_ind`- The square index (0-63) to check.
    /// * `by_white` - If true, check if the square is attacked by white; if false, check if it's attacked by black.
    /// * `move_gen` - A reference to a MoveGen struct for generating potential moves.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the square is attacked by the specified side.
    pub fn is_square_attacked(&self, sq_ind: usize, by_white: bool, move_gen: &MoveGen) -> bool {
        // Find out if the square is attacked by a given side (white if by_white is true, black if by_white is false).
        if by_white {
            // Can the king reach an enemy bishop or queen by a bishop move?
            if (move_gen.gen_bishop_potential_captures(self, sq_ind)
                & (self.pieces[WHITE][BISHOP] | self.pieces[WHITE][QUEEN]))
                != 0
            {
                return true;
            }
            // Can the king reach an enemy rook or queen by a rook move?
            if (move_gen.gen_rook_potential_captures(self, sq_ind)
                & (self.pieces[WHITE][ROOK] | self.pieces[WHITE][QUEEN]))
                != 0
            {
                return true;
            }
            // Can the king reach an enemy knight by a knight move?
            if (move_gen.n_move_bitboard[sq_ind] & self.pieces[WHITE][KNIGHT]) != 0 {
                return true;
            }
            // Can the king reach an enemy pawn by a pawn move?
            if (move_gen.bp_capture_bitboard[sq_ind] & self.pieces[WHITE][PAWN]) != 0 {
                return true;
            }
            // Can the king reach an enemy king by a king move?
            if (move_gen.k_move_bitboard[sq_ind] & self.pieces[WHITE][KING]) != 0 {
                return true;
            }
            false
        } else {
            // Can the king reach an enemy bishop or queen by a bishop move?
            if (move_gen.gen_bishop_potential_captures(self, sq_ind)
                & (self.pieces[BLACK][BISHOP] | self.pieces[BLACK][QUEEN]))
                != 0
            {
                return true;
            }
            // Can the king reach an enemy rook or queen by a rook move?
            if (move_gen.gen_rook_potential_captures(self, sq_ind)
                & (self.pieces[BLACK][ROOK] | self.pieces[BLACK][QUEEN]))
                != 0
            {
                return true;
            }
            // Can the king reach an enemy knight by a knight move?
            if (move_gen.n_move_bitboard[sq_ind] & self.pieces[BLACK][KNIGHT]) != 0 {
                return true;
            }
            // Can the king reach an enemy pawn by a pawn move?
            if (move_gen.wp_capture_bitboard[sq_ind] & self.pieces[BLACK][PAWN]) != 0 {
                return true;
            }
            // Can the king reach an enemy king by a king move?
            if (move_gen.k_move_bitboard[sq_ind] & self.pieces[BLACK][KING]) != 0 {
                return true;
            }
            false
        }
    }
}
