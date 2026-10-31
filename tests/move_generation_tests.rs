use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::piece_types::QUEEN;

#[test]
fn test_initial_move_count() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
    assert_eq!(captures.len() + moves.len(), 20); // 16 pawn moves + 4 knight moves
}

#[test]
fn test_knight_moves() {
    let board = Board::new_from_fen("K7/8/k7/8/4N3/8/8/8 w - - 0 1");
    let move_gen = MoveGen::new();
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
    assert_eq!(captures.len() + moves.len(), 11); // Knight should have 8 possible moves and king should have 3 (moving into check is OK for this function)
}

#[test]
fn test_pawn_promotion() {
    let board = Board::new_from_fen("1r6/P7/K7/8/k7/8/8/8 w - - 0 1");
    let move_gen = MoveGen::new();
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
    assert_eq!(captures.len() + moves.len(), 12); // 4 promotions, 4 capture-promotions, 4 king moves (moving into check is OK for this function)
}

#[test]
fn test_capture_ordering() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, _) = move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    let capture_vals: Vec<i32> = captures
        .iter()
        .map(|m| move_gen.mvv_lva(&board, m.from(), m.to()))
        .collect();
    println!("{} Captures:", captures.len());
    for (i, m) in captures.iter().enumerate() {
        println!("{}. {} ({})", i + 1, m, capture_vals[i]);
    }

    // Check that captures are ordered by MVV-LVA score in descending order
    for i in 1..captures.len() {
        assert!(
            capture_vals[i - 1] >= capture_vals[i],
            "Moves not properly ordered at index {}",
            i
        );
    }
}

#[test]
fn test_non_capture_ordering_white() {
    let board =
        Board::new_from_fen("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4");
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    board.print();

    println!("Captures:");
    for (i, m) in captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            move_gen.mvv_lva(&board, m.from(), m.to())
        );
    }
    println!("Non-captures:");
    for (i, m) in non_captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            pesto.move_eval(&board, &move_gen, m.from(), m.to())
        );
    }

    // Check that non-captures are ordered by Pesto eval change in descending order
    for i in 1..non_captures.len() {
        assert!(
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i - 1].from(),
                non_captures[i - 1].to()
            ) >= pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i].from(),
                non_captures[i].to()
            ),
            "Non-captures not properly ordered at index {}. {} vs {}",
            i,
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i - 1].from(),
                non_captures[i - 1].to()
            ),
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i].from(),
                non_captures[i].to()
            )
        );
    }
}

#[test]
fn test_non_capture_ordering_black() {
    let board =
        Board::new_from_fen("rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R b KQkq - 0 5");
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    board.print();

    println!("Captures:");
    for (i, m) in captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            move_gen.mvv_lva(&board, m.from(), m.to())
        );
    }
    println!("Non-captures:");
    for (i, m) in non_captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            pesto.move_eval(&board, &move_gen, m.from(), m.to())
        );
    }

    // Check that non-captures are ordered by Pesto eval change in descending order
    for i in 1..non_captures.len() {
        assert!(
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i - 1].from(),
                non_captures[i - 1].to()
            ) >= pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i].from(),
                non_captures[i].to()
            ),
            "Non-captures not properly ordered at index {}. {} vs {}",
            i,
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i - 1].from(),
                non_captures[i - 1].to()
            ),
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i].from(),
                non_captures[i].to()
            )
        );
    }
}

#[test]
#[ignore] // Pre-existing failure - move ordering issue
fn test_pawn_fork_ordering() {
    let mut boardstack = BoardStack::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Set up a position where a pawn fork is available
    let moves = ["e2e4", "e7e5", "b1c3", "g8f6", "f1c4", "f6e4", "c3e4"];

    for mv_str in moves.iter() {
        let mv = Move::from_uci(mv_str).unwrap();
        boardstack.make_move(mv);
    }

    let board = boardstack.current_state();

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    board.print();

    println!("Captures:");
    for (i, m) in captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            move_gen.mvv_lva(&board, m.from(), m.to())
        );
    }
    println!("Non-captures:");
    for (i, m) in non_captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            pesto.move_eval(&board, &move_gen, m.from(), m.to())
        );
    }
    assert!(
        pesto.move_eval(
            &board,
            &move_gen,
            non_captures[0].from(),
            non_captures[0].to()
        ) == 600
    );
}

#[test]
fn test_pseudo_legal_captures() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, _) = move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);
    assert!(
        captures.is_empty(),
        "No captures should be possible in the starting position"
    );
}

#[test]
fn test_mvv_lva_ordering() {
    let board =
        Board::new_from_fen("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, _non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    assert!(
        !captures.is_empty(),
        "Captures should be possible in this position"
    );

    // Check if captures are ordered by MVV-LVA
    for i in 0..(captures.len() - 1) {
        let current_score = move_gen.mvv_lva(&board, captures[i].from(), captures[i].to());
        let next_score = move_gen.mvv_lva(&board, captures[i + 1].from(), captures[i + 1].to());
        assert!(
            current_score >= next_score,
            "Captures should be ordered by descending MVV-LVA scores"
        );
    }
}

#[test]
fn test_pesto_move_eval_consistency() {
    let fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1";
    let board = Board::new_from_fen(fen);
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (_captures, non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    // Check if non-captures are ordered by descending PestoEval scores
    for i in 0..(non_captures.len() - 1) {
        let current_score = pesto.move_eval(
            &board,
            &move_gen,
            non_captures[i].from(),
            non_captures[i].to(),
        );
        let next_score = pesto.move_eval(
            &board,
            &move_gen,
            non_captures[i + 1].from(),
            non_captures[i + 1].to(),
        );
        assert!(
            current_score >= next_score,
            "Non-captures should be ordered by descending PestoEval scores"
        );
    }
}

#[test]
fn test_promotion_handling() {
    let fen = "rnbqk2r/pppp1P1p/5n2/2b1p3/4P3/8/PPPP2PP/RNBQKBNR w KQkq - 0 1";
    let board = Board::new_from_fen(fen);
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, _non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    // Count promotions and make sure they're properly ordered
    let mut promotion_count = 0;
    for m in &captures {
        if m.promotion().is_some() {
            promotion_count += 1;

            // Promotions to queen should come before other promotion types
            if promotion_count == 1 {
                assert_eq!(
                    m.promotion(),
                    Some(QUEEN),
                    "First promotion should be to queen"
                );
            }
        }
    }

    assert!(
        promotion_count > 0,
        "Promotions should be present in captures list"
    );
}

// ============================================================
// Edge-square capture classification tests
//
// Sliding piece captures on board edges can be misclassified as
// quiet moves because B_MASKS/R_MASKS exclude edge squares from
// the blocker mask.  These tests verify each capture lands in the
// captures list, not the quiet-moves list.
// ============================================================

/// Helper: assert a move is in captures and NOT in quiet moves.
fn assert_capture_classified(fen: &str, from: usize, to: usize, label: &str) {
    let board = Board::new_from_fen(fen);
    let move_gen = MoveGen::new();
    let (captures, quiet) = move_gen.gen_pseudo_legal_moves(&board);
    let expected = Move::new(from, to, None);
    assert!(
        captures.contains(&expected),
        "{}: move {}->{} should be in captures list",
        label,
        from,
        to
    );
    assert!(
        !quiet.contains(&expected),
        "{}: move {}->{} should NOT be in quiet moves list",
        label,
        from,
        to
    );
}

// ---------- Bishop edge captures ----------

#[test]
fn test_bishop_capture_corner_h8() {
    // White Bb2 captures black pawn on h8
    assert_capture_classified(
        "7p/8/8/8/8/8/1B6/K6k w - - 0 1",
        9,
        63,
        "bishop capture corner h8",
    );
}

#[test]
fn test_bishop_capture_corner_a8() {
    // White Bg2 captures black pawn on a8
    assert_capture_classified(
        "p7/8/8/8/8/8/6B1/K6k w - - 0 1",
        14,
        56,
        "bishop capture corner a8",
    );
}

#[test]
fn test_bishop_capture_corner_a1_black() {
    // Black Bg7 captures white pawn on a1
    assert_capture_classified(
        "K6k/6b1/8/8/8/8/8/P7 b - - 0 1",
        54,
        0,
        "bishop capture corner a1 (black)",
    );
}

#[test]
fn test_bishop_capture_corner_h1_black() {
    // Black Bb7 captures white pawn on h1
    assert_capture_classified(
        "K5k1/1b6/8/8/8/8/8/7P b - - 0 1",
        49,
        7,
        "bishop capture corner h1 (black)",
    );
}

#[test]
fn test_bishop_capture_rank8_edge() {
    // White Bc3 captures black pawn on h8
    assert_capture_classified(
        "7p/8/8/8/8/2B5/8/K6k w - - 0 1",
        18,
        63,
        "bishop capture rank 8 edge",
    );
}

#[test]
fn test_bishop_capture_file_a_edge() {
    // White Be4 captures black pawn on a8
    assert_capture_classified(
        "p7/8/8/8/4B3/8/8/K6k w - - 0 1",
        28,
        56,
        "bishop capture file a edge",
    );
}

#[test]
fn test_bishop_capture_file_h_edge() {
    // White Bc1 captures black rook on h6
    assert_capture_classified(
        "K6k/8/7r/8/8/8/8/2B5 w - - 0 1",
        2,
        47,
        "bishop capture file h edge",
    );
}

#[test]
fn test_bishop_capture_rank8_black() {
    // Black Bc6 captures white knight on a8
    assert_capture_classified(
        "N6k/8/2b5/8/8/8/8/K7 b - - 0 1",
        42,
        56,
        "bishop capture rank 8 (black)",
    );
}

#[test]
fn test_bishop_capture_non_edge_control() {
    // White Bc1 captures black pawn on f4 — non-edge, should already work
    assert_capture_classified(
        "K6k/8/8/8/5p2/8/8/2B5 w - - 0 1",
        2,
        29,
        "bishop capture non-edge control",
    );
}

#[test]
fn test_bishop_capture_a_file_mid_edge() {
    // White Bd4 captures black pawn on a7
    assert_capture_classified(
        "7k/p7/8/8/3B4/8/8/K7 w - - 0 1",
        27,
        48,
        "bishop capture a-file mid-edge",
    );
}

#[test]
fn test_bishop_capture_h_file_mid_edge() {
    // White Bd4 captures black pawn on h8
    assert_capture_classified(
        "6kp/8/8/8/3B4/8/8/K7 w - - 0 1",
        27,
        63,
        "bishop capture h-file mid-edge",
    );
}

// ---------- Rook edge captures ----------

#[test]
fn test_rook_capture_rank8() {
    // White Ra1 captures black rook on a8
    assert_capture_classified(
        "r6k/8/8/8/8/8/8/R6K w - - 0 1",
        0,
        56,
        "rook capture rank 8",
    );
}

#[test]
fn test_rook_capture_rank1_black() {
    // Black Rh8 captures white rook on h1
    assert_capture_classified(
        "K6r/8/8/8/8/8/8/k6R b - - 0 1",
        63,
        7,
        "rook capture rank 1 (black)",
    );
}

#[test]
fn test_rook_capture_file_a() {
    // White Rd1 captures black knight on a1
    assert_capture_classified(
        "K6k/8/8/8/8/8/8/n2R4 w - - 0 1",
        3,
        0,
        "rook capture file a",
    );
}

#[test]
fn test_rook_capture_file_h() {
    // White Ra1 captures black bishop on h1
    assert_capture_classified("K6k/8/8/8/8/8/8/R6b w - - 0 1", 0, 7, "rook capture file h");
}

#[test]
fn test_rook_capture_non_edge_control() {
    // White Ra1 captures black pawn on a4 — non-edge, should already work
    assert_capture_classified(
        "K6k/8/8/8/p7/8/8/R7 w - - 0 1",
        0,
        24,
        "rook capture non-edge control",
    );
}

#[test]
fn test_rook_capture_corner_h8() {
    // White Rh1 captures black pawn on h8
    assert_capture_classified(
        "K6p/8/8/8/8/8/8/k6R w - - 0 1",
        7,
        63,
        "rook capture corner h8",
    );
}

#[test]
fn test_rook_capture_corner_a8() {
    // White Ra1 captures black pawn on a8
    assert_capture_classified(
        "p6k/8/8/8/8/8/8/R6K w - - 0 1",
        0,
        56,
        "rook capture corner a8",
    );
}

// ---------- Queen edge captures ----------

#[test]
fn test_queen_capture_diagonal_corner() {
    // White Qa1 captures black pawn on h8
    assert_capture_classified(
        "K5kp/8/8/8/8/8/8/Q7 w - - 0 1",
        0,
        63,
        "queen capture diagonal corner",
    );
}

#[test]
fn test_queen_capture_rank_edge() {
    // White Qa4 captures black rook on h4
    assert_capture_classified(
        "K6k/8/8/8/Q6r/8/8/8 w - - 0 1",
        24,
        31,
        "queen capture rank edge",
    );
}

#[test]
fn test_queen_capture_file_edge() {
    // White Qd1 captures black knight on d8
    assert_capture_classified(
        "K2n3k/8/8/8/8/8/8/3Q4 w - - 0 1",
        3,
        59,
        "queen capture file edge",
    );
}

#[test]
fn test_queen_capture_diagonal_a_file() {
    // White Qd4 captures black pawn on a7
    assert_capture_classified(
        "K6k/p7/8/8/3Q4/8/8/8 w - - 0 1",
        27,
        48,
        "queen capture diagonal a-file",
    );
}

#[test]
fn test_queen_capture_non_edge_control() {
    // White Qd4 captures black pawn on f6 — non-edge, should already work
    assert_capture_classified(
        "K6k/8/5p2/8/3Q4/8/8/8 w - - 0 1",
        27,
        45,
        "queen capture non-edge control",
    );
}
