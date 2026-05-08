//! NNUE evaluator — drop-in replacement for `PestoEval::pst_eval_cp` inside the Q-search.
//!
//! Architecture (planned, multi-phase):
//!
//! 1. **Scaffolding** (this commit): `NnueEvaluator::new_stub()` returns a
//!    placeholder evaluator that delegates to PeSTO. Lets the rest of the
//!    integration land without blocking on the actual NNUE work.
//! 2. **HalfKP feature encoder**: 64 king squares × 641 (piece, square) tuples
//!    = 41,024 features per perspective. Two perspectives concatenated → 82,048
//!    sparse one-hot features into the noru accumulator.
//! 3. **Stockfish .nnue weight loader**: parse the binary format, remap into
//!    `noru::network::NnueWeights` layout.
//! 4. **Incremental updates**: per-thread accumulator stack, push/pop on
//!    `BoardStack::make_move` / `undo_move`. Avoids the 41K-feature scan on
//!    every Q-search node.
//!
//! The output is **centipawns from STM perspective**, matching the existing
//! `PestoEval::pst_eval_cp` contract — the rest of the engine doesn't need to
//! know which evaluator produced the value, only that the scale and sign
//! convention match.

use crate::board::Board;
use crate::eval::PestoEval;

/// Static evaluator producing centipawn scores from STM perspective.
///
/// In the scaffolding phase this delegates to PeSTO so the rest of the engine
/// keeps working while the NNUE infrastructure is built out underneath.
pub struct NnueEvaluator {
    /// PeSTO fallback used during scaffolding and as a sanity reference once
    /// the real NNUE path is in place. Held by value rather than reference so
    /// the evaluator is self-contained.
    fallback: PestoEval,
}

impl NnueEvaluator {
    /// Create a stub evaluator that delegates to PeSTO. Intended for the
    /// scaffolding phase only — replace with `from_nnue_file` once the .nnue
    /// loader lands.
    pub fn new_stub() -> Self {
        Self {
            fallback: PestoEval::new(),
        }
    }

    /// Static evaluation in centipawns, STM perspective. Matches the contract
    /// of `PestoEval::pst_eval_cp`.
    pub fn eval_cp(&self, board: &Board) -> i32 {
        self.fallback.pst_eval_cp(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_starting_position_is_zero() {
        let nnue = NnueEvaluator::new_stub();
        let board = Board::new();
        // Starting position is symmetric; PeSTO returns 0 from STM (white) view.
        assert_eq!(nnue.eval_cp(&board), 0);
    }

    #[test]
    fn stub_matches_pesto() {
        let nnue = NnueEvaluator::new_stub();
        let pesto = PestoEval::new();
        // After 1.e4 e5 2.Nf3, evaluator should match exactly (it IS PeSTO under the hood).
        let board = Board::new_from_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        );
        assert_eq!(nnue.eval_cp(&board), pesto.pst_eval_cp(&board));
    }
}
