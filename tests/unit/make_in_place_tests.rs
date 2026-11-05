//! Making a move on the board in place, and undoing it, as the quiescence
//! search does: the result must equal the board `apply_move_to_board` returns,
//! and undoing must restore the original exactly (every field, the hash too).

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;

/// Every field of a board, for exact comparison.
fn state(b: &Board) -> String {
    format!("{b:?}")
}

/// Positions with castling, en passant, promotions (plain and capturing),
/// captures of rooks on their home squares, and the two fixture files.
fn positions() -> Vec<Board> {
    let special = [
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq d3 0 3",
        "r3k2r/1P4P1/8/8/8/8/1p4p1/R3K2R w KQkq - 0 1",
        "r3k2r/1P4P1/8/8/8/8/1p4p1/R3K2R b KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/6B1/8/8/8/8/1b6/R3K2R w KQkq - 0 1",
        "r3k2r/6B1/8/8/8/8/1b6/R3K2R b KQkq - 0 1",
    ];
    let files = [
        include_str!("../fixtures/qsearch_leaves.fen"),
        include_str!("../fixtures/mate_positions.fen"),
    ];
    special
        .into_iter()
        .chain(files.iter().flat_map(|f| f.lines()))
        .filter(|l| !l.trim().is_empty())
        .map(Board::new_from_fen)
        .collect()
}

fn legal_moves(b: &Board, mg: &MoveGen) -> Vec<Move> {
    let (caps, quiets) = mg.gen_pseudo_legal_moves(b);
    caps.into_iter()
        .chain(quiets)
        .filter(|&m| b.is_legal_after_move(m, mg))
        .collect()
}

#[test]
fn a_move_made_in_place_equals_the_copied_board_and_undoes_exactly() {
    let mg = MoveGen::new();
    let mut n = 0;
    for start in positions() {
        for mv in legal_moves(&start, &mg) {
            let mut b = start.clone();
            let undo = b.make_move_in_place(mv);
            assert_eq!(state(&b), state(&start.apply_move_to_board(mv)), "{mv:?}");
            b.unmake_move(mv, undo);
            assert_eq!(state(&b), state(&start), "{mv:?}");
            n += 1;
        }
    }
    assert!(n > 5000, "{n}");
}

#[test]
fn every_special_move_kind_is_covered() {
    let mg = MoveGen::new();
    let (mut castles, mut en_passant, mut promotions, mut capture_promotions) = (0, 0, 0, 0);
    for b in positions() {
        for mv in legal_moves(&b, &mg) {
            let piece = b.get_piece(mv.from()).unwrap().1;
            let capture = b.get_piece(mv.to()).is_some();
            if piece == 5 && mv.from().abs_diff(mv.to()) == 2 {
                castles += 1;
            }
            if piece == 0 && b.en_passant() == Some(mv.to() as u8) {
                en_passant += 1;
            }
            if mv.promotion().is_some() {
                promotions += 1;
                capture_promotions += capture as u32;
            }
        }
    }
    assert!(castles >= 4 && en_passant >= 2 && promotions >= 8 && capture_promotions >= 4);
}

#[test]
fn a_null_move_made_in_place_matches_the_stack_and_undoes_exactly() {
    for start in positions() {
        let mut stack = BoardStack::with_board(start.clone());
        stack.make_null_move_without_history();
        let expected = state(stack.current_state());
        let mut b = start.clone();
        let undo = b.make_null_move_in_place();
        assert_eq!(state(&b), expected);
        b.unmake_null_move(undo);
        assert_eq!(state(&b), state(&start));
    }
}

#[test]
fn the_stack_makes_and_undoes_moves_without_history_in_place() {
    let mg = MoveGen::new();
    for start in positions().into_iter().take(60) {
        let mut stack = BoardStack::with_board(start.clone());
        // Three plies deep with a null move in the middle, then back up.
        let Some(&m1) = legal_moves(&start, &mg).first() else {
            continue;
        };
        stack.make_move_without_history(m1);
        let after1 = start.apply_move_to_board(m1);
        assert_eq!(state(stack.current_state()), state(&after1));
        stack.make_null_move_without_history();
        let after_null = state(stack.current_state());
        let mid = stack.current_state().clone();
        if let Some(&m2) = legal_moves(&mid, &mg).last() {
            stack.make_move_without_history(m2);
            assert_eq!(
                state(stack.current_state()),
                state(&mid.apply_move_to_board(m2))
            );
            // A move with history on top undoes first.
            if let Some(&m3) = legal_moves(stack.current_state(), &mg).first() {
                let before = state(stack.current_state());
                stack.make_move(m3);
                stack.undo_move();
                assert_eq!(state(stack.current_state()), before);
            }
            stack.undo_move_without_history();
        }
        assert_eq!(state(stack.current_state()), after_null);
        stack.undo_move_without_history();
        assert_eq!(state(stack.current_state()), state(&after1));
        stack.undo_move_without_history();
        assert_eq!(state(stack.current_state()), state(&start));
    }
}
