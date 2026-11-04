//! Speeding up the extended quiescence search: a frozen copy of the original
//! search is the reference, and every exact change must return the same score
//! and completed flag on real self-play leaves (both feed the network).
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::qsearch_bench::{answer_checksum, percentile};
use kingfisher::search::quiescence::forced_ext_pesto_balance_counted;
use kingfisher::search::quiescence_reference::forced_ext_pesto_balance_reference;

use crate::common::board_from_fen;

/// 100 leaf positions logged in self-play (odd-numbered games, seed 0).
const LEAVES: &str = include_str!("../fixtures/qsearch_leaves.fen");

fn leaves() -> Vec<&'static str> {
    LEAVES.lines().filter(|l| !l.trim().is_empty()).collect()
}

#[test]
fn the_fixture_holds_100_leaves() {
    assert_eq!(leaves().len(), 100);
}

#[test]
fn the_search_returns_the_reference_score_on_every_leaf() {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    for fen in leaves() {
        let board = board_from_fen(fen);
        let (s, c, _, _) = forced_ext_pesto_balance_counted(
            &mut BoardStack::with_board(board.clone()),
            &mg,
            &pesto,
        );
        let (rs, rc, _, _) =
            forced_ext_pesto_balance_reference(&mut BoardStack::with_board(board), &mg, &pesto);
        assert_eq!(s, rs, "{fen}");
        // A search can only become completed: lines that cannot change the
        // score are no longer searched, so their depth limit no longer counts.
        assert!(c || !rc, "completed in the reference but not now: {fen}");
    }
}

#[test]
fn the_threat_probe_is_skipped_where_it_cannot_change_the_score() {
    // The probe only lowers the stand-pat score, so where stand-pat cannot
    // raise alpha it is wasted work: skipping it searches far fewer positions.
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let (mut now, mut reference) = (0u64, 0u64);
    for fen in leaves() {
        let board = board_from_fen(fen);
        now += forced_ext_pesto_balance_counted(
            &mut BoardStack::with_board(board.clone()),
            &mg,
            &pesto,
        )
        .2 as u64;
        reference +=
            forced_ext_pesto_balance_reference(&mut BoardStack::with_board(board), &mg, &pesto).2
                as u64;
    }
    assert!(
        now * 10 < reference * 8,
        "{now} positions searched, reference {reference}"
    );
}

#[test]
fn the_checksum_depends_on_every_score_flag_and_their_order() {
    let a = [(1.25f32, true), (-0.5, false)];
    assert_eq!(answer_checksum(&a), answer_checksum(&a));
    assert_ne!(
        answer_checksum(&a),
        answer_checksum(&[(1.25, true), (-0.5, true)])
    );
    assert_ne!(
        answer_checksum(&a),
        answer_checksum(&[(1.26, true), (-0.5, false)])
    );
    assert_ne!(
        answer_checksum(&a),
        answer_checksum(&[(-0.5, false), (1.25, true)])
    );
}

#[test]
fn percentiles_use_the_nearest_rank() {
    let mut v: Vec<f64> = (1..=100).map(f64::from).collect();
    v.reverse();
    assert_eq!(percentile(&mut v, 50.0), 50.0);
    assert_eq!(percentile(&mut v, 99.0), 99.0);
    assert_eq!(percentile(&mut v, 100.0), 100.0);
    assert_eq!(percentile(&mut [7.0], 50.0), 7.0);
}

#[test]
fn capture_generation_matches_the_capture_list_of_full_generation() {
    // Exactly the captures and promotions, in the same order, so a search that
    // generates only captures visits the same moves in the same order.
    let mg = MoveGen::new();
    let mates = include_str!("../fixtures/mate_positions.fen");
    let extra = [
        "4k3/1P6/8/3pP3/8/8/6p1/4K2R w K d6 0 1", // en passant, promotions both ways, castling
        "r3k2r/pPpp1ppp/8/4pP2/8/8/PPPP1pPP/R3K2R b KQkq - 0 1",
    ];
    let fens = leaves()
        .into_iter()
        .chain(mates.lines().filter(|l| !l.trim().is_empty()))
        .chain(extra);
    let mut n = 0;
    for fen in fens {
        for board in [board_from_fen(fen), {
            let mut b = board_from_fen(fen);
            b.w_to_move = !b.w_to_move; // the other side too
            b
        }] {
            let (expected, _) = mg.gen_pseudo_legal_moves(&board);
            let (mut caps, mut promos) = (Vec::new(), Vec::new());
            mg.gen_captures_into(&board, &mut caps, &mut promos);
            caps.append(&mut promos);
            assert_eq!(caps, expected, "{fen} white to move: {}", board.w_to_move);
            n += 1;
        }
    }
    assert!(n > 300);
}
