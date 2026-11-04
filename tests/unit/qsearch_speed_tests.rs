//! Speeding up the extended quiescence search: a frozen copy of the original
//! search is the reference, and every exact change must return the same score
//! and completed flag on real self-play leaves (both feed the network).
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::qsearch_bench::{answer_checksum, parse_config, percentile};
use kingfisher::search::quiescence::{
    forced_ext_pesto_balance_counted, forced_ext_pesto_balance_with, ExtConfig,
};
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
fn the_exact_search_returns_the_reference_score_on_every_leaf() {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    for fen in leaves() {
        let board = board_from_fen(fen);
        let (s, c, _, _) = forced_ext_pesto_balance_with(
            &mut BoardStack::with_board(board.clone()),
            &mg,
            &pesto,
            &ExtConfig::exact(),
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
        now += forced_ext_pesto_balance_with(
            &mut BoardStack::with_board(board.clone()),
            &mg,
            &pesto,
            &ExtConfig::exact(),
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

fn total_positions(cfg: &ExtConfig) -> u64 {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    leaves()
        .into_iter()
        .map(|fen| {
            let mut stack = BoardStack::with_board(board_from_fen(fen));
            forced_ext_pesto_balance_with(&mut stack, &mg, &pesto, cfg).2 as u64
        })
        .sum()
}

#[test]
fn the_default_prunes_with_one_probe_per_side_delta_200_and_losing_captures() {
    let d = ExtConfig::default();
    assert!(d.probe_once && d.see_prune);
    assert_eq!((d.delta_margin_cp, d.max_depth), (Some(200), 20));
    let e = ExtConfig::exact();
    assert!(!e.probe_once && !e.see_prune);
    assert_eq!((e.delta_margin_cp, e.max_depth), (None, 20));
}

#[test]
fn the_default_config_is_the_current_search() {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    for fen in leaves() {
        let board = board_from_fen(fen);
        let with = forced_ext_pesto_balance_with(
            &mut BoardStack::with_board(board.clone()),
            &mg,
            &pesto,
            &ExtConfig::default(),
        );
        let counted =
            forced_ext_pesto_balance_counted(&mut BoardStack::with_board(board), &mg, &pesto);
        assert_eq!(with, counted, "{fen}");
    }
}

#[test]
fn each_pruning_rule_searches_fewer_positions() {
    let base = total_positions(&ExtConfig::exact());
    for name in ["probe1", "delta200", "see", "d12"] {
        let n = total_positions(&parse_config(name).unwrap());
        assert!(n < base, "{name}: {n} positions searched, default {base}");
    }
}

#[test]
fn a_lower_depth_limit_is_respected() {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let cfg = parse_config("d6").unwrap();
    for fen in leaves() {
        let mut stack = BoardStack::with_board(board_from_fen(fen));
        assert!(
            forced_ext_pesto_balance_with(&mut stack, &mg, &pesto, &cfg).3 <= 6,
            "{fen}"
        );
    }
}

#[test]
fn all_pruning_still_wins_a_hanging_queen_and_sees_a_defended_one() {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let cfg = parse_config("probe1+delta200+see+d12").unwrap();
    let score = |fen: &str| {
        forced_ext_pesto_balance_with(
            &mut BoardStack::with_board(board_from_fen(fen)),
            &mg,
            &pesto,
            &cfg,
        )
        .0
    };
    // White rook takes an undefended black queen.
    assert!(score("3q3k/8/8/8/8/8/8/3RK3 w - - 0 1") > 4.0);
    // The queen is defended by a pawn: RxQ pxR still wins a queen for a rook, leaving
    // White about a pawn down instead of 5.75 down.
    assert!(score("4k3/8/2p5/3q4/8/8/8/3RK3 w - - 0 1") > -2.0);
}

#[test]
fn configs_parse_from_names() {
    let all = parse_config("probe1+delta150+see+d10").unwrap();
    assert!(all.probe_once && all.see_prune);
    assert_eq!((all.delta_margin_cp, all.max_depth), (Some(150), 10));
    assert_eq!(parse_config("exact").unwrap(), ExtConfig::exact());
    assert_eq!(parse_config("default").unwrap(), ExtConfig::default());
    assert!(parse_config("delta").is_err() && parse_config("fast").is_err());
}
