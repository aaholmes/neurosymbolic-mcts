//! This module defines the Bitboard structure and associated functions for chess board representation.

use crate::bits::popcnt;
use crate::board_utils::{algebraic_to_sq_ind, bit_to_sq_ind, coords_to_sq_ind, sq_ind_to_bit};
use crate::move_generation::MoveGen;
use crate::move_types::{CastlingRights, Move};
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

pub const KOTH_CENTER: u64 = 0x0000001818000000; // d4, e4, d5, e5

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
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

    /// Checks if either king has reached the center (King of the Hill win condition)
    pub fn is_koth_win(&self) -> (bool, bool) {
        let white_win = (self.pieces[WHITE][KING] & KOTH_CENTER) != 0;
        let black_win = (self.pieces[BLACK][KING] & KOTH_CENTER) != 0;
        (white_win, black_win)
    }

    /// Calculates material imbalance from side to move's perspective
    pub fn material_imbalance(&self) -> i32 {
        let piece_values = [1, 3, 3, 5, 9, 0]; // P, N, B, R, Q, K
        let mut white_mat = 0;
        let mut black_mat = 0;

        for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN] {
            white_mat += popcnt(self.pieces[WHITE][piece_type]) * piece_values[piece_type];
            black_mat += popcnt(self.pieces[BLACK][piece_type]) * piece_values[piece_type];
        }

        if self.w_to_move {
            white_mat - black_mat
        } else {
            black_mat - white_mat
        }
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
        board.zobrist_hash = board.compute_zobrist_hash();
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

    /// Gets the en passant target square.
    pub fn en_passant(&self) -> Option<u8> {
        self.en_passant
    }

    /// Gets the halfmove clock (number of half-moves since last pawn push or capture).
    pub fn halfmove_clock(&self) -> u8 {
        self.halfmove_clock
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
        // is_legal is called AFTER w_to_move has been flipped.
        // So we check the king of the side that JUST MOVED.
        if self.w_to_move {
            // Check if Black king is attacked by White
            king_sq_ind = bit_to_sq_ind(self.pieces[BLACK][KING]);
            if king_sq_ind == 64 {
                println!("DEBUG: Black king missing!");
                return false;
            }
            !self.is_square_attacked(king_sq_ind, true, move_gen)
        } else {
            // Check if White king is attacked by Black
            king_sq_ind = bit_to_sq_ind(self.pieces[WHITE][KING]);
            if king_sq_ind == 64 {
                println!("DEBUG: White king missing!");
                return false;
            }
            !self.is_square_attacked(king_sq_ind, false, move_gen)
        }
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

        // Check if any move is legal using lightweight legality check
        for mv in captures.iter().chain(moves.iter()) {
            if self.is_legal_after_move(*mv, move_gen) {
                return (false, false);
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

    /// Checks if a move is pseudo-legal in the current position without generating
    /// all pseudo-legal moves. Verifies:
    /// 1. There is a piece of the correct color at `mv.from`
    /// 2. The destination is reachable by that piece type
    /// 3. No own-piece at destination (for non-castling moves)
    ///
    /// This is used to validate TT-probed moves without full move generation.
    pub fn is_pseudo_legal(&self, mv: Move, move_gen: &MoveGen) -> bool {
        let stm = if self.w_to_move { WHITE } else { BLACK };
        let opp = 1 - stm;
        let from_bit = 1u64 << mv.from;
        let to_bit = 1u64 << mv.to;

        // Check there's a piece of the correct color at from
        if self.pieces_occ[stm] & from_bit == 0 {
            return false;
        }

        // Find piece type at from
        let piece_type = match self.get_piece(mv.from) {
            Some((color, pt)) if color == stm => pt,
            _ => return false,
        };

        // Can't capture own piece (except castling is handled specially)
        if !mv.is_castle() && self.pieces_occ[stm] & to_bit != 0 {
            return false;
        }

        let all_occ = self.pieces_occ[WHITE] | self.pieces_occ[BLACK];

        match piece_type {
            PAWN => {
                let from_rank = mv.from / 8;
                let to_rank = mv.to / 8;
                let from_file = mv.from % 8;
                let to_file = mv.to % 8;

                if self.w_to_move {
                    // White pawn
                    if from_file == to_file {
                        // Push
                        if mv.to == mv.from + 8 && all_occ & to_bit == 0 {
                            return true;
                        }
                        // Double push
                        if from_rank == 1
                            && mv.to == mv.from + 16
                            && all_occ & (1u64 << (mv.from + 8)) == 0
                            && all_occ & to_bit == 0
                        {
                            return true;
                        }
                    } else if (to_file as i32 - from_file as i32).abs() == 1
                        && to_rank == from_rank + 1
                    {
                        // Capture (including en passant)
                        if self.pieces_occ[opp] & to_bit != 0 {
                            return true;
                        }
                        if self.en_passant == Some(mv.to as u8) {
                            return true;
                        }
                    }
                } else {
                    // Black pawn
                    if from_file == to_file {
                        if mv.from >= 8 && mv.to == mv.from - 8 && all_occ & to_bit == 0 {
                            return true;
                        }
                        if from_rank == 6
                            && mv.from >= 16
                            && mv.to == mv.from - 16
                            && all_occ & (1u64 << (mv.from - 8)) == 0
                            && all_occ & to_bit == 0
                        {
                            return true;
                        }
                    } else if mv.from >= 8
                        && (to_file as i32 - from_file as i32).abs() == 1
                        && to_rank + 1 == from_rank
                    {
                        if self.pieces_occ[opp] & to_bit != 0 {
                            return true;
                        }
                        if self.en_passant == Some(mv.to as u8) {
                            return true;
                        }
                    }
                }
                false
            }
            KNIGHT => move_gen.n_move_bitboard[mv.from] & to_bit != 0,
            BISHOP => {
                let attacks = move_gen.gen_bishop_attacks_occ(mv.from, all_occ);
                attacks & to_bit != 0
            }
            ROOK => {
                let attacks = move_gen.gen_rook_attacks_occ(mv.from, all_occ);
                attacks & to_bit != 0
            }
            QUEEN => {
                let attacks = move_gen.gen_bishop_attacks_occ(mv.from, all_occ)
                    | move_gen.gen_rook_attacks_occ(mv.from, all_occ);
                attacks & to_bit != 0
            }
            KING => {
                // Normal king move
                if move_gen.k_move_bitboard[mv.from] & to_bit != 0 {
                    return true;
                }
                // Castling: already validated by is_castle()
                if mv.is_castle() {
                    // We trust that move generation already validated castling prerequisites.
                    // Just check the basic structure: correct king position and castling rights.
                    if self.w_to_move {
                        if mv.from == 4 && mv.to == 6 && self.castling_rights.white_kingside {
                            return true;
                        }
                        if mv.from == 4 && mv.to == 2 && self.castling_rights.white_queenside {
                            return true;
                        }
                    } else {
                        if mv.from == 60 && mv.to == 62 && self.castling_rights.black_kingside {
                            return true;
                        }
                        if mv.from == 60 && mv.to == 58 && self.castling_rights.black_queenside {
                            return true;
                        }
                    }
                }
                false
            }
            _ => false,
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
                                (WHITE, PAWN) => 'P',
                                (BLACK, PAWN) => 'p',
                                (WHITE, ROOK) => 'R',
                                (BLACK, ROOK) => 'r',
                                (WHITE, KNIGHT) => 'N',
                                (BLACK, KNIGHT) => 'n',
                                (WHITE, BISHOP) => 'B',
                                (BLACK, BISHOP) => 'b',
                                (WHITE, QUEEN) => 'Q',
                                (BLACK, QUEEN) => 'q',
                                (WHITE, KING) => 'K',
                                (BLACK, KING) => 'k',
                                _ => '?',
                            };
                            fen.push(piece_char);
                            piece_found = true;
                            break;
                        }
                    }
                    if piece_found {
                        break;
                    }
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
        if self.castling_rights.white_kingside {
            castling.push('K');
        }
        if self.castling_rights.white_queenside {
            castling.push('Q');
        }
        if self.castling_rights.black_kingside {
            castling.push('k');
        }
        if self.castling_rights.black_queenside {
            castling.push('q');
        }
        if castling.is_empty() {
            castling.push('-');
        }
        fen.push_str(&castling);

        // 4. En passant target square
        fen.push(' ');
        match self.en_passant {
            Some(sq) => {
                let file = sq % 8;
                let rank = sq / 8;
                fen.push((b'a' + file) as char);
                fen.push((b'1' + rank) as char);
            }
            None => fen.push('-'),
        }

        // 5. Halfmove clock and fullmove number
        fen.push_str(&format!(
            " {} {}",
            self.halfmove_clock, self.fullmove_number
        ));

        Some(fen)
    }

    /// Determines if a move gives check to the opponent's king, without cloning the board.
    ///
    /// Checks for:
    /// 1. Direct check: the piece on its destination square attacks the king
    /// 2. Discovered check: removing the piece from its origin reveals a sliding attack
    /// 3. Special moves: promotions, en passant, castling
    ///
    /// For special moves (en passant, castling, promotion), falls back to apply_move + is_check.
    pub fn gives_check(&self, mv: Move, move_gen: &MoveGen) -> bool {
        use crate::board_utils::bit_to_sq_ind;

        // Handle special moves with fallback (these are rare)
        if mv.promotion.is_some() || mv.is_castle() {
            let new_board = self.apply_move_to_board(mv);
            return new_board.is_check(move_gen);
        }

        // En passant: also fallback (pawn disappears from a different square)
        if self.en_passant.is_some() && mv.to == self.en_passant.unwrap() as usize {
            if let Some((_, PAWN)) = self.get_piece(mv.from) {
                let new_board = self.apply_move_to_board(mv);
                return new_board.is_check(move_gen);
            }
        }

        let stm_color = if self.w_to_move { WHITE } else { BLACK };
        let opp_color = 1 - stm_color;

        // Find opponent's king square
        let opp_king_bb = self.pieces[opp_color][KING];
        if opp_king_bb == 0 {
            return false;
        }
        let king_sq = bit_to_sq_ind(opp_king_bb);

        let from_bit = 1u64 << mv.from;
        let to_bit = 1u64 << mv.to;

        // Build occupancy with the piece moved (remove from `from`, add to `to`,
        // and remove any captured piece at `to`)
        let all_occ = (self.pieces_occ[0] | self.pieces_occ[1]) & !from_bit | to_bit;

        // 1. Direct check: does the moved piece attack the king from its destination?
        let piece_type = match self.get_piece(mv.from) {
            Some((_, pt)) => pt,
            None => return false,
        };

        let direct_check = match piece_type {
            PAWN => {
                if stm_color == WHITE {
                    // White pawn attacks diagonally upward
                    move_gen.wp_capture_bitboard[mv.to] & opp_king_bb != 0
                } else {
                    // Black pawn attacks diagonally downward
                    move_gen.bp_capture_bitboard[mv.to] & opp_king_bb != 0
                }
            }
            KNIGHT => move_gen.n_move_bitboard[mv.to] & opp_king_bb != 0,
            BISHOP => {
                let bishop_attacks = move_gen.gen_bishop_attacks_occ(mv.to, all_occ);
                bishop_attacks & opp_king_bb != 0
            }
            ROOK => {
                let rook_attacks = move_gen.gen_rook_attacks_occ(mv.to, all_occ);
                rook_attacks & opp_king_bb != 0
            }
            QUEEN => {
                let bishop_attacks = move_gen.gen_bishop_attacks_occ(mv.to, all_occ);
                let rook_attacks = move_gen.gen_rook_attacks_occ(mv.to, all_occ);
                (bishop_attacks | rook_attacks) & opp_king_bb != 0
            }
            KING => {
                // King moves can't directly give check (king can't attack another king in check sense)
                false
            }
            _ => false,
        };

        if direct_check {
            return true;
        }

        // 2. Discovered check: does removing the piece from `from` reveal a sliding attack?
        // Check if there's a rook/queen on the same rank/file as from and king
        let rook_queen = self.pieces[stm_color][ROOK] | self.pieces[stm_color][QUEEN];
        if rook_queen != 0 {
            let rook_attacks_to_king = move_gen.gen_rook_attacks_occ(king_sq, all_occ);
            if rook_attacks_to_king & rook_queen & !from_bit != 0 {
                return true;
            }
        }

        // Check if there's a bishop/queen on the same diagonal as from and king
        let bishop_queen = self.pieces[stm_color][BISHOP] | self.pieces[stm_color][QUEEN];
        if bishop_queen != 0 {
            let bishop_attacks_to_king = move_gen.gen_bishop_attacks_occ(king_sq, all_occ);
            if bishop_attacks_to_king & bishop_queen & !from_bit != 0 {
                return true;
            }
        }

        false
    }

    /// Checks if a pseudo-legal move is legal (doesn't leave own king in check)
    /// without cloning the board. Works by building modified occupancy on the stack
    /// and checking for attacks against the king.
    ///
    /// This is functionally equivalent to `apply_move_to_board(mv).is_legal(move_gen)`
    /// but avoids the board clone and hash recomputation.
    pub fn is_legal_after_move(&self, mv: Move, move_gen: &MoveGen) -> bool {
        let stm = if self.w_to_move { WHITE } else { BLACK };
        let opp = 1 - stm;

        let from_bit = 1u64 << mv.from;
        let to_bit = 1u64 << mv.to;

        // Determine the moving piece type
        let piece_type = match self.get_piece(mv.from) {
            Some((_, pt)) => pt,
            None => return false,
        };

        // Find the king square (may have moved)
        let king_sq: usize;
        if piece_type == KING {
            king_sq = mv.to;
        } else {
            let king_bb = self.pieces[stm][KING];
            if king_bb == 0 {
                return false;
            }
            king_sq = king_bb.trailing_zeros() as usize;
        }

        // Build modified occupancy: remove piece from `from`, add at `to`, remove captured piece
        let mut all_occ = (self.pieces_occ[WHITE] | self.pieces_occ[BLACK]) & !from_bit | to_bit;

        // Handle en passant: also remove the captured pawn
        let mut ep_captured_sq: Option<usize> = None;
        if piece_type == PAWN
            && self.en_passant.is_some()
            && mv.to == self.en_passant.unwrap() as usize
        {
            let cap_sq = if self.w_to_move { mv.to - 8 } else { mv.to + 8 };
            ep_captured_sq = Some(cap_sq);
            all_occ &= !(1u64 << cap_sq);
        }

        // For castling, we need to also check that the king doesn't pass through check.
        // Castling legality is already checked by move generation (which verifies
        // the king doesn't start in check, doesn't pass through attacked squares,
        // and doesn't end in check). So for castling we just need to verify the
        // final king position isn't attacked (which is what we do below).
        // The rook also moves, so update occupancy for castling.
        if piece_type == KING {
            if mv.from == 4 && mv.to == 6 {
                // White O-O
                all_occ = (all_occ & !(1u64 << 7)) | (1u64 << 5);
            } else if mv.from == 4 && mv.to == 2 {
                // White O-O-O
                all_occ = (all_occ & !(1u64 << 0)) | (1u64 << 3);
            } else if mv.from == 60 && mv.to == 62 {
                // Black O-O
                all_occ = (all_occ & !(1u64 << 63)) | (1u64 << 61);
            } else if mv.from == 60 && mv.to == 58 {
                // Black O-O-O
                all_occ = (all_occ & !(1u64 << 56)) | (1u64 << 59);
            }
        }

        // Check if king_sq is attacked by opponent using modified occupancy.
        // We need to account for:
        // - The moving piece is no longer at `from` (already handled via all_occ)
        // - There may be a captured piece at `to` (already handled via all_occ including to_bit)
        // - For EP, the captured pawn is removed from its actual square

        // Build opponent piece bitboards accounting for captures
        let opp_pawns = self.pieces[opp][PAWN]
            & !to_bit
            & if let Some(ep_sq) = ep_captured_sq {
                !(1u64 << ep_sq)
            } else {
                !0u64
            };
        let opp_knights = self.pieces[opp][KNIGHT] & !to_bit;
        let opp_bishops = self.pieces[opp][BISHOP] & !to_bit;
        let opp_rooks = self.pieces[opp][ROOK] & !to_bit;
        let opp_queens = self.pieces[opp][QUEEN] & !to_bit;
        let opp_king = self.pieces[opp][KING] & !to_bit;

        // Check pawn attacks
        if stm == WHITE {
            // White king attacked by black pawns: black pawns attack downward
            if move_gen.wp_capture_bitboard[king_sq] & opp_pawns != 0 {
                return false;
            }
        } else {
            // Black king attacked by white pawns: white pawns attack upward
            if move_gen.bp_capture_bitboard[king_sq] & opp_pawns != 0 {
                return false;
            }
        }

        // Check knight attacks
        if move_gen.n_move_bitboard[king_sq] & opp_knights != 0 {
            return false;
        }

        // Check king attacks (opponent king)
        if move_gen.k_move_bitboard[king_sq] & opp_king != 0 {
            return false;
        }

        // Check sliding piece attacks using modified occupancy
        let bishop_attacks = move_gen.gen_bishop_attacks_occ(king_sq, all_occ);
        if bishop_attacks & (opp_bishops | opp_queens) != 0 {
            return false;
        }

        let rook_attacks = move_gen.gen_rook_attacks_occ(king_sq, all_occ);
        if rook_attacks & (opp_rooks | opp_queens) != 0 {
            return false;
        }

        true
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
