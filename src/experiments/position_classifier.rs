// src/experiments/position_classifier.rs
//! Automatic classification of chess positions for experiment analysis.

use crate::board::Board;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionType {
    /// Forced mate exists
    ForcedMate,
    /// Significant tactical opportunities (captures, forks, etc.)
    Tactical,
    /// Quiet positional play
    Positional,
    /// Simplified endgame
    Endgame,
    /// Under attack / defensive position
    Defensive,
}

pub struct PositionClassifier<'a> {
    move_gen: &'a MoveGen,
    _pesto: &'a PestoEval,
}

impl<'a> PositionClassifier<'a> {
    pub fn new(move_gen: &'a MoveGen, pesto: &'a PestoEval) -> Self {
        PositionClassifier {
            move_gen,
            _pesto: pesto,
        }
    }

    pub fn classify(&self, board: &Board) -> PositionType {
        let piece_count = board.get_all_occupancy().count_ones();

        // Endgame: few pieces
        if piece_count <= 10 {
            return PositionType::Endgame;
        }

        // Check for tactical features
        let (captures, _) = self.move_gen.gen_pseudo_legal_moves(board);

        // Many captures available suggests tactical position
        let legal_captures: Vec<_> = captures
            .iter()
            .filter(|&&m| board.apply_move_to_board(m).is_legal(self.move_gen))
            .collect();

        if legal_captures.len() >= 3 {
            return PositionType::Tactical;
        }

        // Check if in check
        if board.is_check(self.move_gen) {
            return PositionType::Defensive;
        }

        // Default to positional
        PositionType::Positional
    }

    /// Estimate if position has a forced mate (shallow check)
    pub fn has_likely_forced_mate(&self, board: &Board, depth: i32) -> bool {
        use crate::boardstack::BoardStack;
        use crate::search::mate_search;

        let mut stack = BoardStack::with_board(board.clone());
        let (score, _, _) = mate_search(&mut stack, self.move_gen, depth, false, 3);

        score.abs() >= 1_000_000
    }
}
