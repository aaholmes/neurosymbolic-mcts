use crate::board::Board;

pub mod data_loader;
pub mod texel;

#[derive(Debug, Clone)]
pub struct TexelPosition {
    pub board: Board,
    pub game_result: f64, // 0.0 = loss, 0.5 = draw, 1.0 = win (from White's perspective)
    pub game_phase: f64,  // 0.0 = endgame, 1.0 = opening
    pub description: String,
}

impl TexelPosition {
    pub fn new(fen: &str, result: f64, description: String) -> Option<Self> {
        let board = Board::new_from_fen(fen);

        // Calculate game phase based on material
        let game_phase = calculate_game_phase(&board);

        Some(TexelPosition {
            board,
            game_result: result,
            game_phase,
            description,
        })
    }

    pub fn from_pgn_result(fen: &str, pgn_result: &str, description: String) -> Option<Self> {
        let result = match pgn_result {
            "1-0" => 1.0,     // White wins
            "0-1" => 0.0,     // Black wins
            "1/2-1/2" => 0.5, // Draw
            _ => return None, // Unknown result
        };

        Self::new(fen, result, description)
    }
}

/// Calculate game phase based on piece values (0.0 = endgame, 1.0 = opening)
fn calculate_game_phase(board: &Board) -> f64 {
    use crate::piece_types::*;

    let mut total_material = 0;

    // Count material for both sides
    for color in [WHITE, BLACK] {
        total_material += board.get_piece_bitboard(color, QUEEN).count_ones() * 9;
        total_material += board.get_piece_bitboard(color, ROOK).count_ones() * 5;
        total_material += board.get_piece_bitboard(color, BISHOP).count_ones() * 3;
        total_material += board.get_piece_bitboard(color, KNIGHT).count_ones() * 3;
        total_material += board.get_piece_bitboard(color, PAWN).count_ones();
    }

    // Starting position has ~78 points of material (2*Q + 4*R + 4*B + 4*N + 16*P = 2*9 + 4*5 + 4*3 + 4*3 + 16*1 = 78)
    let max_material = 78.0;

    (total_material as f64 / max_material).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texel_position_creation() {
        let position = TexelPosition::new(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0.5,
            "Starting position".to_string(),
        );

        assert!(position.is_some());
        let pos = position.unwrap();
        assert_eq!(pos.game_result, 0.5);
        assert!(pos.game_phase > 0.9); // Should be near opening
    }

    #[test]
    fn test_game_phase_calculation() {
        // Starting position should be ~1.0 (opening)
        let start_board =
            Board::new_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let start_phase = calculate_game_phase(&start_board);
        assert!(start_phase > 0.9);

        // King and pawn endgame should be ~0.0 (endgame)
        let endgame_board = Board::new_from_fen("8/8/8/4k3/4P3/4K3/8/8 w - - 0 1");
        let endgame_phase = calculate_game_phase(&endgame_board);
        assert!(endgame_phase < 0.1);
    }

    #[test]
    fn test_pgn_result_parsing() {
        let pos = TexelPosition::from_pgn_result(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "1-0",
            "Test".to_string(),
        );
        assert!(pos.is_some());
        assert_eq!(pos.unwrap().game_result, 1.0);

        let pos = TexelPosition::from_pgn_result(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "0-1",
            "Test".to_string(),
        );
        assert!(pos.is_some());
        assert_eq!(pos.unwrap().game_result, 0.0);

        let pos = TexelPosition::from_pgn_result(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "1/2-1/2",
            "Test".to_string(),
        );
        assert!(pos.is_some());
        assert_eq!(pos.unwrap().game_result, 0.5);
    }
}
