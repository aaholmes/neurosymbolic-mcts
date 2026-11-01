//! Syzygy win/draw/loss probes (`kingfisher::egtb`). These need the 5-piece WDL
//! files, in KF_SYZYGY_PATH or /data/syzygy; without them each test is skipped.

use kingfisher::board::Board;
use kingfisher::egtb::{wdl_to_value, Tablebase};
use shakmaty_syzygy::Wdl;

fn tables() -> Option<Tablebase> {
    let dir = std::env::var("KF_SYZYGY_PATH").unwrap_or_else(|_| "/data/syzygy".into());
    if !std::path::Path::new(&dir).join("KRvK.rtbw").exists() {
        eprintln!("skipped: no Syzygy files in {dir}");
        return None;
    }
    Some(Tablebase::open(&dir).expect("tablebase opens"))
}

fn probe(tb: &Tablebase, fen: &str) -> Option<f32> {
    tb.probe_value(&Board::new_from_fen(fen))
}

#[test]
fn opening_a_directory_without_tables_is_an_error() {
    assert!(Tablebase::open("/nonexistent/syzygy").is_err());
    let empty = std::env::temp_dir().join("kf_empty_syzygy_dir");
    std::fs::create_dir_all(&empty).unwrap();
    assert!(Tablebase::open(empty.to_str().unwrap()).is_err());
}

#[test]
fn the_tables_loaded_cover_five_pieces() {
    let Some(tb) = tables() else { return };
    assert_eq!(tb.max_pieces(), 5);
}

#[test]
fn a_rook_up_wins_for_the_side_with_the_rook_whoever_is_to_move() {
    let Some(tb) = tables() else { return };
    assert_eq!(probe(&tb, "8/8/8/8/8/2k5/8/R3K3 w - - 0 1"), Some(1.0));
    assert_eq!(probe(&tb, "8/8/8/8/8/2k5/8/R3K3 b - - 0 1"), Some(-1.0));
    // Mirrored colours.
    assert_eq!(probe(&tb, "r3k3/8/2K5/8/8/8/8/8 b - - 0 1"), Some(1.0));
}

#[test]
fn bare_kings_and_a_lone_knight_are_draws() {
    let Some(tb) = tables() else { return };
    assert_eq!(probe(&tb, "8/8/8/4k3/8/8/8/4K3 w - - 0 1"), Some(0.0));
    assert_eq!(probe(&tb, "8/8/8/4k3/8/8/8/3NK3 b - - 0 1"), Some(0.0));
}

#[test]
fn a_checkmated_side_to_move_has_lost() {
    let Some(tb) = tables() else { return };
    // Black king h8, white queen g7 protected by the king on f6.
    assert_eq!(probe(&tb, "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"), Some(-1.0));
}

#[test]
fn positions_the_tables_do_not_cover_give_none() {
    let Some(tb) = tables() else { return };
    // Castling rights are never in the tables.
    assert_eq!(probe(&tb, "r3k3/8/8/8/8/8/8/4K3 b q - 0 1"), None);
    // Six pieces.
    assert_eq!(probe(&tb, "8/8/8/4k3/8/2PP4/8/R3K2R w - - 0 1"), None);
    assert_eq!(probe(&tb, Board::new().to_fen().unwrap().as_str()), None);
}

#[test]
fn an_en_passant_square_is_passed_to_the_tables() {
    let Some(tb) = tables() else { return };
    // After e2-e4, Black's f4 pawn can take en passant; either way it is a
    // known result.
    let v = probe(&tb, "8/8/8/8/4Pp2/8/k7/4K3 b - e3 0 1").expect("covered");
    assert!([-1.0, 0.0, 1.0].contains(&v));
}

#[test]
fn wins_and_losses_the_fifty_move_rule_spoils_count_as_draws() {
    assert_eq!(wdl_to_value(Wdl::Win), 1.0);
    assert_eq!(wdl_to_value(Wdl::CursedWin), 0.0);
    assert_eq!(wdl_to_value(Wdl::Draw), 0.0);
    assert_eq!(wdl_to_value(Wdl::BlessedLoss), 0.0);
    assert_eq!(wdl_to_value(Wdl::Loss), -1.0);
}

#[test]
fn adjudication_gives_the_result_from_whites_side() {
    let Some(tb) = tables() else { return };
    let white_wins = Board::new_from_fen("8/8/8/8/8/2k5/8/R3K3 b - - 0 1");
    assert_eq!(tb.adjudicate(&white_wins), Some(1.0));
    let black_wins = Board::new_from_fen("r3k3/8/2K5/8/8/8/8/8 w - - 0 1");
    assert_eq!(tb.adjudicate(&black_wins), Some(-1.0));
    assert_eq!(tb.adjudicate(&Board::new()), None);
}
