//! Verify Neural Network Input Feature Construction
//!
//! This utility helps debug the board-to-tensor conversion by visualizing
//! the active bits in each plane of the input tensor.

use kingfisher::board::Board;
use kingfisher::neural_net::NeuralNetPolicy;
use kingfisher::piece_types::{BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK};

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
    // we rely on NeuralNetPolicy to expose it.
    // `NeuralNetPolicy` has `board_to_tensor` but it returns a `Tensor`.
    // We need to convert that `Tensor` back to CPU data to visualize.

    #[cfg(feature = "neural")]
    {
        let nn = NeuralNetPolicy::new(); // Don't need to load weights, just need structure
        let tensor = nn.board_to_tensor(&board);

        println!("Tensor Shape: {:?}", tensor.dims());

        // Shape: [17, 8, 8]
        let flat_data: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();

        let plane_names = [
            "Us Pawn",
            "Us Knight",
            "Us Bishop",
            "Us Rook",
            "Us Queen",
            "Us King",
            "Them Pawn",
            "Them Knight",
            "Them Bishop",
            "Them Rook",
            "Them Queen",
            "Them King",
            "En Passant",
            "Us King-side Castle",
            "Us Queen-side Castle",
            "Them King-side Castle",
            "Them Queen-side Castle",
        ];

        for (i, name) in plane_names.iter().enumerate() {
            println!("\nPlane {}: {}", i, name);
            println!("  +-----------------+");
            for rank in (0..8).rev() {
                // Print rank 8 at top
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
