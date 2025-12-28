use crate::board::Board;
use crate::move_types::Move;
use std::collections::{HashMap, VecDeque};

/// Represents a stack of boards for undoing moves.
#[derive(Clone)]
pub struct BoardStack {
    pub position_history: HashMap<u64, u8>,
    pub(crate) state_stack: VecDeque<Board>,
    move_stack: VecDeque<Move>,
}

impl BoardStack {
    /// Create a new boardstack
    pub fn new() -> Self {
        let initial_state = Board::new();
        let mut board = BoardStack {
            position_history: HashMap::new(),
            state_stack: VecDeque::new(),
            move_stack: VecDeque::new(),
        };

        board.position_history.insert(initial_state.zobrist_hash, 1);
        board.state_stack.push_front(initial_state);
        board
    }

    /// Generate a new boardstack whose starting board is given by the fen string
    pub fn new_from_fen(fen: &str) -> Self {
        let mut board = BoardStack::new();
        let fen_position = Board::new_from_fen(fen);

        // Remove the starting position from the state stack and position history
        board.position_history = HashMap::new();
        board.state_stack.pop_front();

        // Add the starting position to the state stack and position history
        board.position_history.insert(fen_position.zobrist_hash, 1);
        board.state_stack.push_front(fen_position);

        board
    }

    /// Generate a new boardstack whose starting board is the given board
    pub fn with_board(initial_board: Board) -> Self {
        let mut board = BoardStack {
            position_history: HashMap::new(),
            state_stack: VecDeque::new(),
            move_stack: VecDeque::new(),
        };
        board.position_history.insert(initial_board.zobrist_hash, 1);
        board.state_stack.push_front(initial_board);
        board
    }

    /// Return the current state by peeking at the board stack
    pub fn current_state(&self) -> &Board {
        &self.state_stack.front().unwrap()
    }

    /// Applies a move to the boardstack
    pub fn make_move(&mut self, mv: Move) {
        // Push the move onto the move stack
        self.move_stack.push_front(mv);

        // Apply the move to the current state
        let new_board = self.current_state().apply_move_to_board(mv);

        // Update position history
        *self
            .position_history
            .entry(new_board.zobrist_hash)
            .or_insert(0) += 1;

        // Push the new board onto the stack
        self.state_stack.push_front(new_board);
    }

    /// Undoes the last move in the move stack.
    pub fn undo_move(&mut self) -> Option<Move> {
        if let (Some(popped_board), Some(mv)) = (self.state_stack.pop_front(), self.move_stack.pop_front()) {
            // Update position history for the position we just removed
            let hash = popped_board.zobrist_hash;
            if let Some(count) = self.position_history.get_mut(&hash) {
                if *count == 1 {
                    self.position_history.remove(&hash);
                } else {
                    *count -= 1;
                }
            }
            Some(mv)
        } else {
            None
        }
    }

    /// Undoes the last `n` moves in the move stack.
    ///
    /// Returns a vector of undone moves.
    pub fn undo_moves(&mut self, n: usize) -> Vec<Move> {
        let mut undone_moves = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(mv) = self.undo_move() {
                undone_moves.push(mv);
            } else {
                break;
            }
        }
        undone_moves
    }

    /// Checks if the current position has occurred three times, resulting in a draw by repetition.
    ///
    /// This method considers a position to be repeated if:
    /// - The piece positions are identical
    /// - The side to move is the same
    /// - The castling rights are the same
    /// - The en passant possibilities are the same
    ///
    /// # Returns
    ///
    /// `true` if the current position has occurred three times, `false` otherwise.
    ///
    /// # Note
    ///
    /// This method relies on the Zobrist hash of the position, which includes all
    /// relevant aspects of the chess position.
    pub fn is_draw_by_repetition(&self) -> bool {
        // Get Zobrist hash of current position
        let hash = self.current_state().zobrist_hash;

        // Check if there are 3 or more repetitions of the same hash
        *self.position_history.get(&hash).unwrap() >= 3
    }

    /// Checks if the side to move is in check
    ///
    /// # Arguments
    ///
    /// * `move_gen` - A reference to the move generator
    ///
    /// # Returns
    ///
    /// `true` if the side to move is in check, `false` otherwise.
    pub fn is_check(&self, move_gen: &crate::move_generation::MoveGen) -> bool {
        self.current_state().is_check(move_gen)
    }

    /// Makes a null move (passes the turn to the opponent)
    ///
    /// This is used for null move pruning in the search algorithm.
    /// It doesn't change the board position except for the side to move.
    pub fn make_null_move(&mut self) {
        // Create a new board that's the same as the current one but with the side to move flipped
        let mut new_board = self.current_state().clone();
        new_board.w_to_move = !new_board.w_to_move;

        // Reset en passant target if any
        new_board.en_passant = None;

        // Update zobrist hash for the side to move change and en passant reset
        new_board.zobrist_hash = new_board.compute_zobrist_hash();

        // Push the new board onto the stack and record a null move
        *self
            .position_history
            .entry(new_board.zobrist_hash)
            .or_insert(0) += 1;
        self.state_stack.push_front(new_board);
        self.move_stack.push_front(Move::null());
    }

    /// Undoes a null move
    ///
    /// This is the counterpart to make_null_move.
    pub fn undo_null_move(&mut self) {
        self.undo_move();
    }
}
