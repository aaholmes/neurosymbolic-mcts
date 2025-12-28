//! Property-based tests for system invariants

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::node::MctsNode;
use kingfisher::piece_types::{WHITE, BLACK, KING};
use proptest::prelude::*;
use crate::common::positions;

// Strategy to generate random legal positions
fn random_position() -> impl Strategy<Value = Board> {
    // Start from known positions and make random legal moves
    prop::sample::select(vec![
        positions::STARTING,
        positions::CASTLING_BOTH,
        "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    ]).prop_map(|fen| Board::from_fen(fen).unwrap())
}

proptest! {
    #[test]
    fn test_move_gen_legal_moves_are_legal(board in random_position()) {
        let move_gen = MoveGen::new();
        let moves = move_gen.generate_legal_moves(&board);
        
        for mv in moves {
            let mut new_board = board.clone();
            new_board.make_move(&mv);
            
            // After a legal move, the opponent should not be able to capture our king
            let opp_moves = move_gen.generate_legal_moves(&new_board);
            let our_king_sq = if board.w_to_move {
                new_board.get_piece_bitboard(WHITE, KING).trailing_zeros() as usize
            } else {
                new_board.get_piece_bitboard(BLACK, KING).trailing_zeros() as usize
            };
            
            for opp_mv in &opp_moves {
                prop_assert!(opp_mv.to != our_king_sq, 
                    "Legal move resulted in king capture being possible");
            }
        }
    }
    
    #[test]
    fn test_value_domain_invariant(board in random_position()) {
        let move_gen = MoveGen::new();
        let node = MctsNode::new_root(board, &move_gen);
        let node_ref = node.borrow();
        
        if let Some(v) = node_ref.terminal_or_mate_value {
            prop_assert!(v >= -1.0 && v <= 1.0, 
                "Terminal value {} outside [-1, 1]", v);
        }
    }
}

#[cfg(feature = "neural")]
mod tensor_props {
    use super::*;
    use kingfisher::neural_net::NeuralNetPolicy;
    use tch::Tensor;

    fn extract_plane(tensor: &Tensor, plane: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; 64];
        tensor.narrow(0, plane as i64, 1).view([64]).copy_data(&mut result, 64);
        result
    }

    proptest! {
        #[test]
        fn test_tensor_all_values_binary(board in random_position()) {
            let nn = NeuralNetPolicy::new();
            let tensor = nn.board_to_tensor(&board);
            
            // Piece planes (0-11) should only contain 0.0 or 1.0
            for plane in 0..12 {
                let values = extract_plane(&tensor, plane);
                for v in values {
                    prop_assert!(v == 0.0 || v == 1.0,
                        "Piece plane {} contains non-binary value {}", plane, v);
                }
            }
        }
    }
}
