//! Tests for tensor construction and StM flipping logic

#![cfg(feature = "neural")]

use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::board::Board;
use kingfisher::piece_types::{WHITE, BLACK, KING};
use crate::common::{board_from_fen, positions};
use tch::Tensor;

/// Test that tensor construction respects side-to-move flipping
#[test]
fn test_tensor_stm_symmetry() {
    let nn = NeuralNetPolicy::new(); 
    
    // Position where White has king on e1, Black has king on e8
    let white_to_move = board_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let tensor_w = nn.board_to_tensor(&white_to_move);
    
    // Same position but Black to move
    let black_to_move = board_from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1");
    let tensor_b = nn.board_to_tensor(&black_to_move);
    
    // In the tensor for white_to_move:
    //   - Plane 5 (our king) should have a 1 at e1 (row 7, col 4 in tensor coords? No, let's check mapping)
    // In src/neural_net.rs: 
    // rank = i / 8; tensor_rank = 7 - rank (for White).
    // e1 is index 4. Rank 0. tensor_rank = 7 - 0 = 7.
    // So yes, row 7.
    
    // In the tensor for black_to_move:
    //   - Plane 5 (our king = black king) should have a 1 at e8 (index 60).
    //     e8 is rank 7. 
    //     For Black stm, tensor_rank = rank = 7.
    //     So it appears at row 7.
    
    // The key invariant: "our king" is always at the same tensor position
    // regardless of which color we are (relative to the board flip)
    let our_king_plane = 5;
    
    // Extract values
    let w_our_king = extract_plane(&tensor_w, our_king_plane);
    let b_our_king = extract_plane(&tensor_b, our_king_plane);
    
    // The "our king" pattern should be identical in both tensors
    // (both should show king at tensor row 7, col 4)
    assert_eq!(w_our_king, b_our_king, 
        "StM flipping broken: 'our king' plane differs between White and Black perspectives");
}

#[test]
fn test_tensor_dimensions() {
    let nn = NeuralNetPolicy::new();
    let board = board_from_fen(positions::STARTING);
    let tensor = nn.board_to_tensor(&board);
    
    let shape = tensor.size();
    assert_eq!(shape, &[17, 8, 8], "Tensor shape should be [17, 8, 8]"); // Guide said [1, 17, 8, 8] but src uses view([17, 8, 8])
}

#[test]
fn test_castling_planes() {
    let nn = NeuralNetPolicy::new();
    
    // Position with all castling rights
    let all_castling = board_from_fen(positions::CASTLING_BOTH);
    let tensor = nn.board_to_tensor(&all_castling);
    
    // Planes 13-16 are castling rights (StM-KS, StM-QS, Opp-KS, Opp-QS)
    // All should be 1.0 (filled planes) when rights exist
    for plane_idx in 13..17 {
        let plane = extract_plane(&tensor, plane_idx);
        let sum: f32 = plane.iter().sum();
        assert_eq!(sum, 64.0, "Castling plane {} should be all 1s when right exists", plane_idx);
    }
    
    // Position with no castling rights
    let no_castling = board_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1");
    let tensor_no = nn.board_to_tensor(&no_castling);
    
    for plane_idx in 13..17 {
        let plane = extract_plane(&tensor_no, plane_idx);
        let sum: f32 = plane.iter().sum();
        assert_eq!(sum, 0.0, "Castling plane {} should be all 0s when right missing", plane_idx);
    }
}

#[test]
fn test_en_passant_plane() {
    let nn = NeuralNetPolicy::new();
    let board = board_from_fen(positions::EN_PASSANT);
    let tensor = nn.board_to_tensor(&board);
    
    // Plane 12 is en passant
    let ep_plane = extract_plane(&tensor, 12);
    let nonzero_count: usize = ep_plane.iter().filter(|&&x| x > 0.0).count();
    
    assert_eq!(nonzero_count, 1, "EP plane should have exactly 1 square marked");
}

fn extract_plane(tensor: &Tensor, plane: usize) -> Vec<f32> {
    // Extract a single 8x8 plane from the tensor
    let mut result = vec![0.0f32; 64];
    // tensor is [17, 8, 8]
    // narrow(dim, start, length)
    tensor.narrow(0, plane as i64, 1).view([64]).copy_data(&mut result, 64);
    result
}
