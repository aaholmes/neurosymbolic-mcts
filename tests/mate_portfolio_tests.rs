use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::mate_search;
use std::time::Instant;

/// Integration tests for the checks-only mate search.
#[test]
fn test_mate_search_checks_only() {
    let move_gen = MoveGen::new();

    // Test 1: Mate via forcing checks
    // Scholar's mate pattern — every move by white is check.
    run_mate_test_case(
        "Forcing Checks Mate",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        &move_gen,
        true,
        "h5f7",
    );

    // Test 2: Quiet setup move required — exhaustive mate-in-2 finds this
    // King + Rook vs King. Requires 1. Kg6 (quiet) before Ra8#.
    run_mate_test_case(
        "Quiet Move Required (found by exhaustive)",
        "7k/R7/5K2/8/8/8/8/8 w - - 0 1",
        &move_gen,
        true,
        "f6g6",
    );

    // Test 3: Starting position — no mate exists
    run_mate_test_case(
        "Starting Position (no mate)",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &move_gen,
        false,
        "",
    );

    // Test 4: Back rank mate in 1 (single check = mate)
    run_mate_test_case(
        "Back Rank Mate in 1",
        "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1",
        &move_gen,
        true,
        "e1e8",
    );
}

fn run_mate_test_case(
    name: &str,
    fen: &str,
    move_gen: &MoveGen,
    expect_mate: bool,
    expected_move_uci: &str,
) {
    println!("\n  Running test case: {}", name);
    let board = Board::new_from_fen(fen);

    let start = Instant::now();
    let (score, best_move, _nodes) = mate_search(&board, move_gen, 5, false, 3);
    let duration = start.elapsed();

    println!(
        "   Result: Score={}, Move={:?}, Time={:.2?}",
        score, best_move, duration
    );

    if expect_mate {
        assert!(score >= 1_000_000, "Expected mate not found in {}", name);
        println!("   Mate Found! Best move: {:?}", best_move);
        if !expected_move_uci.is_empty() {
            let parsed_expected = kingfisher::move_types::Move::from_uci(expected_move_uci);
            if let Some(expected) = parsed_expected {
                if best_move != expected {
                    println!(
                        "   Note: Found move {:?} differs from expected {:?}, but mate was found.",
                        best_move, expected
                    );
                }
            }
        }
    } else {
        assert!(score.abs() < 1_000_000, "Unexpected mate found in {}", name);
        println!("   No Mate Found (as expected).");
    }
}
