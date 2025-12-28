//! Verify Neural Network Input Feature Construction
//!
//! This utility helps debug the board-to-tensor conversion by visualizing
//! the active bits in each plane of the input tensor.

use kingfisher::board::Board;
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::piece_types::{PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
// use tch::{Tensor, Device}; // NeuralNetPolicy handles tensor creation internals

fn main() {
    println!("🧪 Neural Network Feature Visualizer");
    println!("===================================");

    // Setup a position with diverse features:
    // - White castling rights (K, Q)
    // - Black castling rights (k)
    // - En passant target
    // - Pieces of all types
    // FEN: r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 (KiwiPete)
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    println!("Position: {}", fen);
    let board = Board::new_from_fen(fen);
    
    // We need to access the tensor conversion logic.
    // Since board_to_tensor is an internal helper of NeuralNetPolicy (or part of its impl),
    // and might use `tch`, we rely on NeuralNetPolicy to expose it or we test via a public method if available.
    // `NeuralNetPolicy` has `board_to_tensor` but it returns a `Tensor`.
    // We need to convert that `Tensor` back to CPU data to visualize.
    
    #[cfg(feature = "neural")]
    {
        use tch::{Device, Kind};
        let nn = NeuralNetPolicy::new(); // Don't need to load weights, just need structure
        let tensor = nn.board_to_tensor(&board);
        
        println!("Tensor Shape: {:?}", tensor.size());
        
        // Extract data
        // Shape: [17, 8, 8]
        let mut flat_data = vec![0.0f32; 17 * 8 * 8];
        tensor.view([-1]).to_device(Device::Cpu).copy_data(&mut flat_data, 17 * 8 * 8);
        
        let plane_names = [
            "White Pawn", "White Knight", "White Bishop", "White Rook", "White Queen", "White King",
            "Black Pawn", "Black Knight", "Black Bishop", "Black Rook", "Black Queen", "Black King",
            "En Passant", 
            "W King-side Castle", "W Queen-side Castle", 
            "B King-side Castle", "B Queen-side Castle"
        ];
        
        for (i, name) in plane_names.iter().enumerate() {
            println!("\nPlane {}: {}", i, name);
            println!("  +-----------------+");
            for rank in (0..8).rev() { // Print rank 8 at top
                print!("{} | ", rank + 1);
                for file in 0..8 {
                    // Index calculation matches neural_net.rs:
                    let tensor_row = 7 - rank;
                    let idx = (i * 64) + (tensor_row * 8) + file;
                    
                    let val = flat_data[idx];
                    if val > 0.5 {
                        print!("1 ");
                    } else {
                        print!(". ");
                    }
                }
                println!("|");
            }
            println!("  +-----------------+");
            println!("    a b c d e f g h");
        }
        
        println!("\n✅ Verification Complete: Check above planes against FEN.");
    }

    #[cfg(not(feature = "neural"))]
    println!("⚠️  Please run with --features neural to verify tensor construction.");
}
