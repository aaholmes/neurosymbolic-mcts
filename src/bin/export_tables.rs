//! Export move generation tables as binary files for CUDA consumption.
//!
//! Writes the following files to `cuda/tables/`:
//! - rook_attacks.bin:   64 * 4096 u64 values (2 MB), padded per-square
//! - bishop_attacks.bin: 64 * 4096 u64 values (2 MB), padded per-square
//! - knight_moves.bin:   64 u64 values
//! - king_moves.bin:     64 u64 values
//! - wp_captures.bin:    64 u64 values (white pawn capture bitboards)
//! - bp_captures.bin:    64 u64 values (black pawn capture bitboards)

use kingfisher::move_generation::MoveGen;
use std::fs;
use std::io::Write;
use std::path::Path;

fn write_u64_array(path: &Path, data: &[u64]) {
    let mut file = fs::File::create(path).expect("Failed to create file");
    for &val in data {
        file.write_all(&val.to_le_bytes())
            .expect("Failed to write");
    }
    println!(
        "  Wrote {} entries ({} bytes) to {}",
        data.len(),
        data.len() * 8,
        path.display()
    );
}

fn main() {
    println!("Initializing MoveGen (computing magic tables)...");
    let move_gen = MoveGen::new();
    println!("Done.\n");

    let out_dir = Path::new("cuda/tables");
    fs::create_dir_all(out_dir).expect("Failed to create cuda/tables/");

    // --- Rook attack table: pad each square to 4096 entries ---
    println!("Exporting rook attack table...");
    let rook_table = move_gen.rook_attack_table();
    let mut rook_flat = vec![0u64; 64 * 4096];
    for sq in 0..64 {
        for (key, &val) in rook_table[sq].iter().enumerate() {
            rook_flat[sq * 4096 + key] = val;
        }
    }
    write_u64_array(&out_dir.join("rook_attacks.bin"), &rook_flat);

    // --- Bishop attack table: pad each square to 4096 entries ---
    println!("Exporting bishop attack table...");
    let bishop_table = move_gen.bishop_attack_table();
    let mut bishop_flat = vec![0u64; 64 * 4096];
    for sq in 0..64 {
        for (key, &val) in bishop_table[sq].iter().enumerate() {
            bishop_flat[sq * 4096 + key] = val;
        }
    }
    write_u64_array(&out_dir.join("bishop_attacks.bin"), &bishop_flat);

    // --- Jump piece tables ---
    println!("Exporting knight/king move tables...");
    write_u64_array(
        &out_dir.join("knight_moves.bin"),
        &move_gen.n_move_bitboard,
    );
    write_u64_array(&out_dir.join("king_moves.bin"), &move_gen.k_move_bitboard);

    // --- Pawn capture tables ---
    println!("Exporting pawn capture tables...");
    write_u64_array(
        &out_dir.join("wp_captures.bin"),
        &move_gen.wp_capture_bitboard,
    );
    write_u64_array(
        &out_dir.join("bp_captures.bin"),
        &move_gen.bp_capture_bitboard,
    );

    println!("\nAll tables exported to {}/", out_dir.display());
    println!("Total: {} bytes", 2 * 64 * 4096 * 8 + 4 * 64 * 8);
}
