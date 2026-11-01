//! Syzygy endgame tables: the exact result (win, draw or loss with perfect play)
//! of every position with few enough pieces, read from `.rtbw` files.
//!
//! Only the win/draw/loss (WDL) tables are used. Search treats a covered
//! position as solved, and self-play ends a game when it reaches one, so the
//! distance-to-zeroing tables needed to play such endings out are not loaded.

use crate::board::Board;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Position};
use shakmaty_syzygy::Wdl;
use std::sync::Arc;

/// A directory of Syzygy WDL tables, shared between threads.
#[derive(Clone)]
pub struct Tablebase {
    tables: Arc<shakmaty_syzygy::Tablebase<Chess>>,
    max_pieces: u32,
}

impl std::fmt::Debug for Tablebase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tablebase(up to {} pieces)", self.max_pieces)
    }
}

impl Tablebase {
    /// Loads the tables in `dir`. An error if it holds no WDL table.
    pub fn open(dir: &str) -> Result<Self, String> {
        let mut tables = shakmaty_syzygy::Tablebase::new();
        tables
            .add_directory(dir)
            .map_err(|e| format!("cannot read Syzygy tables in {dir}: {e}"))?;
        // A table's name lists its pieces, e.g. KRvK.rtbw has three.
        let max_pieces = std::fs::read_dir(dir)
            .map_err(|e| format!("cannot read {dir}: {e}"))?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|n| {
                n.strip_suffix(".rtbw")
                    .map(|s| s.replace('v', "").len() as u32)
            })
            .max()
            .ok_or_else(|| format!("no Syzygy WDL tables (.rtbw) in {dir}"))?;
        Ok(Self {
            tables: Arc::new(tables),
            max_pieces,
        })
    }

    /// The most pieces (kings included) of any loaded table.
    pub fn max_pieces(&self) -> u32 {
        self.max_pieces
    }

    /// The exact value of `board` for the side to move (+1 win, 0 draw, −1
    /// loss), or None if the tables do not cover it: too many pieces, or
    /// castling rights left.
    pub fn probe_value(&self, board: &Board) -> Option<f32> {
        let c = &board.castling_rights;
        let can_castle =
            c.white_kingside || c.white_queenside || c.black_kingside || c.black_queenside;
        if board.get_all_occupancy().count_ones() > self.max_pieces || can_castle {
            return None;
        }
        let fen: Fen = board.to_fen()?.parse().ok()?;
        let pos: Chess = fen
            .position(CastlingMode::Standard)
            .or_else(|e| e.ignore_invalid_ep_square().map_err(drop))
            .ok()?;
        if pos.is_insufficient_material() {
            return Some(0.0);
        }
        self.tables.probe_wdl(&pos).ok().map(wdl_to_value)
    }

    /// The result of the game from White's side (+1 White wins, 0 draw, −1
    /// Black wins) if the tables cover `board`.
    pub fn adjudicate(&self, board: &Board) -> Option<f32> {
        let v = self.probe_value(board)?;
        Some(if board.w_to_move { v } else { -v })
    }
}

/// +1 for a win, −1 for a loss, and 0 for a draw, including wins and losses
/// that the fifty-move rule turns into draws, since self-play applies the rule.
pub fn wdl_to_value(wdl: Wdl) -> f32 {
    match wdl {
        Wdl::Win => 1.0,
        Wdl::Loss => -1.0,
        Wdl::CursedWin | Wdl::Draw | Wdl::BlessedLoss => 0.0,
    }
}

/// Exact results used without the tiers, from the command line of `self_play`
/// and `evaluate_models`: `--mate-in-1` gives a leaf whose side to move can
/// mate in one its exact value, and `--syzygy <dir>` loads endgame tables that
/// value the leaves they cover and end games that reach them.
#[derive(Debug, Clone, Default)]
pub struct ExactEndings {
    pub mate_in_1: bool,
    pub tablebase: Option<Tablebase>,
}

impl ExactEndings {
    /// From `args` (the program's arguments). An error if `--syzygy` has no
    /// directory or the directory holds no tables.
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let tablebase = match args.iter().position(|a| a == "--syzygy") {
            Some(i) => {
                let dir = args.get(i + 1).ok_or("--syzygy needs a directory")?;
                Some(Tablebase::open(dir)?)
            }
            None => None,
        };
        Ok(Self {
            mate_in_1: args.iter().any(|a| a == "--mate-in-1"),
            tablebase,
        })
    }

    /// `config` with these options set.
    pub fn apply(
        &self,
        config: crate::mcts::tactical_mcts::TacticalMctsConfig,
    ) -> crate::mcts::tactical_mcts::TacticalMctsConfig {
        crate::mcts::tactical_mcts::TacticalMctsConfig {
            leaf_mate_in_1: self.mate_in_1,
            tablebase: self.tablebase.clone(),
            ..config
        }
    }

    /// One line for a program's settings printout.
    pub fn describe(&self) -> String {
        format!(
            "mate in one {}, tablebase {}",
            if self.mate_in_1 { "on" } else { "off" },
            match &self.tablebase {
                Some(t) => format!("up to {} pieces", t.max_pieces()),
                None => "off".into(),
            }
        )
    }
}
