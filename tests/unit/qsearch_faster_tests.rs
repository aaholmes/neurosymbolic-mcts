//! Faster quiescence search. The exact changes (quiet checks and forks
//! generated directly, a square-to-piece table, legality from pinned pieces,
//! a piece-square score kept up to date move by move, and moves made without
//! updating the hash) must leave every answer unchanged: the golden file holds
//! the answers of the search before them. The position budget and the
//! transposition table are options that may change answers.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::piece_types::{BLACK, WHITE};
use kingfisher::search::qsearch_bench::parse_config;
use kingfisher::search::quiescence::{
    forced_ext_pesto_balance_with, is_tactical_quiet, ExtConfig, TtScope,
};

const GOLDEN: &str = "tests/fixtures/qsearch_answers.txt";

fn fixture_fens() -> Vec<String> {
    let special = [
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq d3 0 3",
        "r3k2r/1P4P1/8/8/8/8/1p4p1/R3K2R w KQkq - 0 1",
        "r3k2r/6B1/8/8/8/8/1b6/R3K2R b KQkq - 0 1",
        // A pawn and a knight that can fork, and pinned pieces of both sides.
        "r3k2r/8/8/3n4/8/2N5/1P6/R3K2R w - - 0 1",
        "4k3/8/8/1q6/8/3B4/4K3/8 w - - 0 1",
        "4k3/4r3/8/8/8/8/4N3/4K3 w - - 0 1",
        "4k3/8/8/K2Pp2r/8/8/8/8 w - e6 0 2",
    ];
    let files = [
        include_str!("../fixtures/qsearch_leaves.fen"),
        include_str!("../fixtures/mate_positions.fen"),
    ];
    special
        .iter()
        .map(|s| s.to_string())
        .chain(
            files
                .iter()
                .flat_map(|f| f.lines())
                .filter(|l| !l.trim().is_empty())
                .map(String::from),
        )
        .collect()
}

/// The fixtures and each with the other side to move (when that is legal).
fn positions() -> Vec<Board> {
    let mg = MoveGen::new();
    let mut out = Vec::new();
    for fen in fixture_fens() {
        let b = Board::new_from_fen(&fen);
        // After flipping, the side not to move must not be in check.
        if !b.is_check(&mg) {
            let f: Vec<&str> = fen.split_whitespace().collect();
            let side = if f[1] == "w" { "b" } else { "w" };
            out.push(Board::new_from_fen(&format!(
                "{} {side} {} - 0 1",
                f[0], f[2]
            )));
        }
        out.push(b);
    }
    out
}

fn answer_line(cfg: &ExtConfig, fen: &str) -> String {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let mut stack = BoardStack::with_board(Board::new_from_fen(fen));
    let (score, completed, nodes, depth) =
        forced_ext_pesto_balance_with(&mut stack, &mg, &pesto, cfg);
    format!(
        "{:08x} {} {} {}",
        score.to_bits(),
        completed as u8,
        nodes,
        depth
    )
}

/// The configurations in the golden file, and the positions each is run on
/// (the unpruned search only on the 100 logged leaves, as it is slow).
fn golden_runs() -> Vec<(&'static str, Vec<String>)> {
    let all = fixture_fens();
    let leaves: Vec<String> = include_str!("../fixtures/qsearch_leaves.fen")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect();
    vec![("default", all), ("exact", leaves)]
}

fn compute_golden() -> String {
    let mut s = String::new();
    for (name, fens) in golden_runs() {
        let cfg = parse_config(name).unwrap();
        for fen in fens {
            s += &format!("{name} {} | {fen}\n", answer_line(&cfg, &fen));
        }
    }
    s
}

#[test]
fn every_answer_equals_the_golden_file() {
    // KF_WRITE_GOLDEN=1 writes the file from the current search.
    if std::env::var("KF_WRITE_GOLDEN").is_ok() {
        std::fs::write(GOLDEN, compute_golden()).unwrap();
        return;
    }
    let want = std::fs::read_to_string(GOLDEN).expect("golden file");
    let got = compute_golden();
    for (w, g) in want.lines().zip(got.lines()) {
        assert_eq!(w, g);
    }
    assert_eq!(want.lines().count(), got.lines().count());
}

// ---- Quiet checks and forks generated directly ----

#[test]
fn tactical_quiets_are_the_filtered_quiets_in_the_same_order() {
    let mg = MoveGen::new();
    let mut total = 0;
    for b in positions() {
        let (_, quiets) = mg.gen_pseudo_legal_moves(&b);
        let want: Vec<Move> = quiets
            .into_iter()
            .filter(|&m| is_tactical_quiet(&b, m, &mg))
            .collect();
        let mut got = Vec::new();
        mg.gen_tactical_quiets_into(&b, &mut got);
        assert_eq!(got, want, "{}", b.to_fen().unwrap());
        total += want.len();
    }
    assert!(total > 200, "{total}");
}

// ---- Square-to-piece table ----

fn piece_from_bitboards(b: &Board, sq: usize) -> Option<(usize, usize)> {
    for color in [WHITE, BLACK] {
        for piece in 0..6 {
            if b.get_piece_bitboard(color, piece) & (1u64 << sq) != 0 {
                return Some((color, piece));
            }
        }
    }
    None
}

fn assert_table_matches(b: &Board) {
    for sq in 0..64 {
        assert_eq!(b.get_piece(sq), piece_from_bitboards(b, sq), "square {sq}");
    }
}

#[test]
fn the_square_table_matches_the_bitboards_after_every_kind_of_move() {
    let mg = MoveGen::new();
    assert_table_matches(&Board::new());
    for start in positions() {
        assert_table_matches(&start);
        let (caps, quiets) = mg.gen_pseudo_legal_moves(&start);
        for mv in caps.into_iter().chain(quiets) {
            if !start.is_legal_after_move(mv, &mg) {
                continue;
            }
            assert_table_matches(&start.apply_move_to_board(mv));
            let mut b = start.clone();
            let undo = b.make_move_in_place(mv);
            assert_table_matches(&b);
            b.unmake_move(mv, undo);
            assert_table_matches(&b);
        }
    }
}

// ---- Legality from pinned pieces ----

#[test]
fn legality_from_pins_equals_making_the_move() {
    let mg = MoveGen::new();
    let (mut n, mut illegal) = (0, 0);
    for b in positions() {
        let pins = b.pins(&mg);
        let (caps, quiets) = mg.gen_pseudo_legal_moves(&b);
        for mv in caps.into_iter().chain(quiets) {
            let want = b.is_legal_after_move(mv, &mg);
            assert_eq!(
                b.is_legal_with_pins(mv, &pins, &mg),
                want,
                "{} {mv:?}",
                b.to_fen().unwrap()
            );
            n += 1;
            illegal += !want as u32;
        }
    }
    assert!(n > 5000 && illegal > 50, "{n} moves, {illegal} illegal");
}

// ---- Piece-square score kept up to date ----

#[test]
fn the_running_piece_square_score_equals_a_full_count() {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    for start in positions() {
        assert_eq!(pesto.pst_eval_cp(&start), pesto.pst_eval_cp_full(&start));
        let (caps, quiets) = mg.gen_pseudo_legal_moves(&start);
        for mv in caps.into_iter().chain(quiets) {
            if !start.is_legal_after_move(mv, &mg) {
                continue;
            }
            let copied = start.apply_move_to_board(mv);
            assert_eq!(pesto.pst_eval_cp(&copied), pesto.pst_eval_cp_full(&copied));
            let mut b = start.clone();
            let undo = b.make_move_in_place(mv);
            assert_eq!(pesto.pst_eval_cp(&b), pesto.pst_eval_cp_full(&b), "{mv:?}");
            b.unmake_move(mv, undo);
            assert_eq!(pesto.pst_eval_cp(&b), pesto.pst_eval_cp_full(&b));
        }
    }
}

// ---- Moves made without updating the hash ----

#[test]
fn a_move_made_without_the_hash_differs_only_in_the_hash_and_undoes_exactly() {
    let mg = MoveGen::new();
    for start in positions() {
        let (caps, quiets) = mg.gen_pseudo_legal_moves(&start);
        for mv in caps.into_iter().chain(quiets) {
            if !start.is_legal_after_move(mv, &mg) {
                continue;
            }
            let mut hashed = start.clone();
            hashed.make_move_in_place(mv);
            let mut b = start.clone();
            let undo = b.make_move_in_place_unhashed(mv);
            assert_eq!(b.to_fen(), hashed.to_fen());
            assert_eq!(b.get_all_occupancy(), hashed.get_all_occupancy());
            b.unmake_move(mv, undo);
            assert_eq!(format!("{b:?}"), format!("{start:?}"));
        }
    }
}

// ---- Position budget ----

fn leaves() -> Vec<Board> {
    include_str!("../fixtures/qsearch_leaves.fen")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(Board::new_from_fen)
        .collect()
}

fn run(cfg: &ExtConfig, b: &Board) -> (f32, bool, u32, u8) {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    forced_ext_pesto_balance_with(&mut BoardStack::with_board(b.clone()), &mg, &pesto, cfg)
}

#[test]
fn a_budget_limits_the_positions_searched() {
    let cfg = ExtConfig {
        max_nodes: Some(200),
        ..ExtConfig::default()
    };
    let mut cut = 0;
    for b in leaves() {
        let (_, completed, nodes, _) = run(&cfg, &b);
        let (_, _, free_nodes, _) = run(&ExtConfig::default(), &b);
        assert!(nodes <= 200 + 1, "{nodes}");
        if free_nodes > 200 {
            assert!(!completed);
            cut += 1;
        }
    }
    assert!(cut > 3, "{cut}");
}

#[test]
fn a_budget_never_reached_changes_nothing() {
    let cfg = ExtConfig {
        max_nodes: Some(u32::MAX),
        ..ExtConfig::default()
    };
    for b in leaves() {
        assert_eq!(run(&cfg, &b), run(&ExtConfig::default(), &b));
    }
}

// ---- Transposition table ----

#[test]
fn the_table_keeps_every_score_of_the_unpruned_search_and_searches_fewer_positions() {
    for scope in [TtScope::Call, TtScope::Thread] {
        let with = ExtConfig {
            tt: Some(scope),
            ..ExtConfig::exact()
        };
        let (mut n_with, mut n_without) = (0u64, 0u64);
        for b in leaves() {
            let a = run(&with, &b);
            let e = run(&ExtConfig::exact(), &b);
            assert_eq!(a.0, e.0, "{scope:?} {}", b.to_fen().unwrap());
            n_with += a.2 as u64;
            n_without += e.2 as u64;
        }
        assert!(n_with < n_without, "{scope:?}: {n_with} vs {n_without}");
    }
}

#[test]
fn the_new_options_parse() {
    let c = parse_config("default+tt").unwrap();
    assert_eq!(c.tt, Some(TtScope::Call));
    assert!(c.probe_once && c.see_prune);
    assert_eq!(
        parse_config("default+ttg").unwrap().tt,
        Some(TtScope::Thread)
    );
    assert_eq!(parse_config("default+n500").unwrap().max_nodes, Some(500));
    assert_eq!(parse_config("exact+n500").unwrap().delta_margin_cp, None);
    assert_eq!(ExtConfig::default().tt, None);
    assert_eq!(ExtConfig::default().max_nodes, None);
}

// ---- Choosing the engine's quiescence settings on the command line ----

fn args(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

#[test]
fn the_qsearch_flag_parses() {
    use kingfisher::search::quiescence::qsearch_config_from_args;
    assert_eq!(qsearch_config_from_args(&args("self_play 8 200")), Ok(None));
    let c = qsearch_config_from_args(&args(
        "self_play --qsearch extended --qsearch-config default+ttg+n4000 --threads 4",
    ))
    .unwrap()
    .unwrap();
    assert_eq!(c.tt, Some(TtScope::Thread));
    assert_eq!(c.max_nodes, Some(4000));
    assert!(qsearch_config_from_args(&args("self_play --qsearch-config bogus")).is_err());
    assert!(qsearch_config_from_args(&args("self_play --qsearch-config")).is_err());
}

#[test]
fn the_engine_uses_the_default_settings_unless_told_otherwise() {
    use kingfisher::search::quiescence::engine_config;
    assert_eq!(engine_config(), ExtConfig::default());
}

#[test]
fn a_move_made_without_the_score_keeps_the_pieces_and_hash_and_undoes_exactly() {
    let mg = MoveGen::new();
    for start in positions() {
        let (caps, quiets) = mg.gen_pseudo_legal_moves(&start);
        for mv in caps.into_iter().chain(quiets) {
            if !start.is_legal_after_move(mv, &mg) {
                continue;
            }
            let full = start.apply_move_to_board(mv);
            let mut b = start.clone();
            let undo = b.make_move_in_place_unscored(mv);
            assert_eq!(b.to_fen(), full.to_fen());
            assert_eq!(b.zobrist_hash(), full.zobrist_hash());
            for sq in 0..64 {
                assert_eq!(b.get_piece(sq), full.get_piece(sq));
            }
            b.unmake_move(mv, undo);
            assert_eq!(format!("{b:?}"), format!("{start:?}"));
        }
    }
}
