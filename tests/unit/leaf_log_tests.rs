//! Logging the positions where self-play runs the leaf mate search, so that
//! the mate search can be timed on the positions it actually sees.

use kingfisher::board::Board;
use kingfisher::search::leaf_log::{game_from_thread_name, parse_spec, LeafLog};

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("kf_leaf_log_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn the_spec_is_a_path_and_how_often_to_log() {
    let (path, every) = parse_spec("/tmp/x/leaves.txt:200").unwrap();
    assert_eq!(path, std::path::PathBuf::from("/tmp/x/leaves.txt"));
    assert_eq!(every, 200);
    assert!(parse_spec("/tmp/leaves.txt").is_err());
    assert!(parse_spec("/tmp/leaves.txt:0").is_err());
    assert!(parse_spec("/tmp/leaves.txt:many").is_err());
    assert!(parse_spec(":5").is_err());
}

#[test]
fn the_game_comes_from_the_thread_name() {
    assert_eq!(game_from_thread_name(Some("kf-game-17")), Some(17));
    assert_eq!(game_from_thread_name(Some("main")), None);
    assert_eq!(game_from_thread_name(None), None);
}

#[test]
fn every_nth_position_is_written_with_its_game() {
    let path = scratch("every_nth.txt");
    let _ = std::fs::remove_file(&path);
    let log = LeafLog::create(&path, 3).unwrap();
    let boards: Vec<Board> = [
        "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
        "4k3/8/8/8/8/8/8/3RK3 w - - 0 1",
        "4k3/8/8/8/8/8/8/3BK3 w - - 0 1",
        "4k3/8/8/8/8/8/8/3NK3 w - - 0 1",
        "4k3/8/8/8/8/8/8/2NNK3 w - - 0 1",
        "4k3/8/8/8/8/8/8/2BBK3 b - - 0 1",
    ]
    .iter()
    .map(|f| Board::new_from_fen(f))
    .collect();
    for (i, b) in boards.iter().enumerate() {
        log.record(Some(i), b).unwrap();
    }
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text}");
    for (line, i) in lines.iter().zip([2usize, 5]) {
        let (game, fen) = line.split_once(' ').unwrap();
        assert_eq!(game, i.to_string());
        assert_eq!(Board::new_from_fen(fen).to_fen(), boards[i].to_fen());
    }
}

#[test]
fn a_position_with_no_game_is_marked() {
    let path = scratch("no_game.txt");
    let _ = std::fs::remove_file(&path);
    let log = LeafLog::create(&path, 1).unwrap();
    log.record(None, &Board::new()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("- "), "{text}");
}

#[test]
fn a_corpus_line_may_start_with_its_game() {
    use kingfisher::search::leaf_log::parse_corpus_line;
    let fen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1";
    assert_eq!(parse_corpus_line(&format!("12 {fen}")), (Some(12), fen));
    assert_eq!(parse_corpus_line(&format!("- {fen}")), (None, fen));
    assert_eq!(parse_corpus_line(fen), (None, fen));
}
