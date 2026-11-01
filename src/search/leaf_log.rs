//! Logs every Nth position at which self-play runs the leaf mate search, with
//! the game it came from, so that `mate_bench` can time the search on the
//! positions it actually sees and split them by game. Off unless
//! `KF_LOG_MATE_LEAVES=<path>:<N>` is set.

use crate::board::Board;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Parses `<path>:<N>`, N ≥ 1.
pub fn parse_spec(spec: &str) -> Result<(PathBuf, u64), String> {
    let (path, every) = spec
        .rsplit_once(':')
        .ok_or_else(|| format!("expected <path>:<N>, got {spec:?}"))?;
    if path.is_empty() {
        return Err(format!("no path in {spec:?}"));
    }
    let every: u64 = every
        .parse()
        .map_err(|_| format!("N must be a whole number in {spec:?}"))?;
    if every == 0 {
        return Err(format!("N must be at least 1 in {spec:?}"));
    }
    Ok((PathBuf::from(path), every))
}

/// The game index of a self-play thread named `kf-game-N`.
pub fn game_from_thread_name(name: Option<&str>) -> Option<usize> {
    name?.strip_prefix("kf-game-")?.parse().ok()
}

/// Splits a corpus line into its game (a number, or `-` for none) and FEN.
/// A line holding only a FEN has no game.
pub fn parse_corpus_line(line: &str) -> (Option<usize>, &str) {
    let line = line.trim();
    match line.split_once(' ') {
        Some(("-", fen)) => (None, fen),
        Some((first, fen)) => match first.parse() {
            Ok(game) => (Some(game), fen),
            Err(_) => (None, line),
        },
        None => (None, line),
    }
}

pub struct LeafLog {
    every: u64,
    seen: AtomicU64,
    file: Mutex<File>,
}

impl LeafLog {
    /// Creates (or truncates) the log at `path`, keeping every `every`th position.
    pub fn create(path: &Path, every: u64) -> std::io::Result<Self> {
        Ok(Self {
            every: every.max(1),
            seen: AtomicU64::new(0),
            file: Mutex::new(File::create(path)?),
        })
    }

    /// Counts one position, and writes `game fen` if it is an Nth one.
    pub fn record(&self, game: Option<usize>, board: &Board) -> std::io::Result<()> {
        let n = self.seen.fetch_add(1, Ordering::Relaxed) + 1;
        if !n.is_multiple_of(self.every) {
            return Ok(());
        }
        let game = game.map_or_else(|| "-".to_string(), |g| g.to_string());
        let fen = board.to_fen().unwrap_or_default();
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        writeln!(file, "{game} {fen}")
    }
}

/// The log named by `KF_LOG_MATE_LEAVES`, if set. Panics on a malformed value.
pub fn global() -> Option<&'static LeafLog> {
    static LOG: OnceLock<Option<LeafLog>> = OnceLock::new();
    LOG.get_or_init(|| {
        let spec = std::env::var("KF_LOG_MATE_LEAVES").ok()?;
        let (path, every) = parse_spec(&spec).unwrap_or_else(|e| panic!("KF_LOG_MATE_LEAVES: {e}"));
        Some(LeafLog::create(&path, every).unwrap_or_else(|e| panic!("KF_LOG_MATE_LEAVES: {e}")))
    })
    .as_ref()
}

/// Records `board` in the global log, if any, with this thread's game.
pub fn record_leaf(board: &Board) {
    if let Some(log) = global() {
        let game = game_from_thread_name(std::thread::current().name());
        if let Err(e) = log.record(game, board) {
            eprintln!("KF_LOG_MATE_LEAVES: {e}");
        }
    }
}
