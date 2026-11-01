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

/// Why `ExactEndings::game_result` ended a game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    Mate,
    Tablebase,
}

/// How far to look for a forced mate by the side to move: mates in up to
/// `exhaustive` moves are searched over every legal move, and longer ones,
/// up to `depth`, over checking moves plus at most `quiet` non-checking
/// attacker moves per line. Mate in one is complete either way, since every
/// mating move gives check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MateSearch {
    pub depth: i32,
    pub exhaustive: i32,
    pub quiet: i32,
}

impl MateSearch {
    pub const OFF: Self = Self {
        depth: 0,
        exhaustive: 0,
        quiet: 0,
    };

    /// The command-line default: checking moves only, mates in up to five.
    pub const DEFAULT: Self = Self::checks(5);

    /// Every legal move searched, for mates in up to `n`.
    pub const fn exhaustive(n: i32) -> Self {
        Self {
            depth: n,
            exhaustive: n,
            quiet: 0,
        }
    }

    /// Checking moves only, for mates in up to `n`.
    pub const fn checks(n: i32) -> Self {
        Self {
            depth: n,
            exhaustive: 0,
            quiet: 0,
        }
    }

    pub fn is_on(self) -> bool {
        self.depth > 0
    }

    /// True if the side to move at `board` has a mate this search finds.
    pub fn finds_mate(self, board: &Board, move_gen: &crate::move_generation::MoveGen) -> bool {
        self.is_on() && self.search(board, move_gen).0
    }

    /// Whether this search finds a mate, and the number of positions searched.
    pub fn search(self, board: &Board, move_gen: &crate::move_generation::MoveGen) -> (bool, i32) {
        crate::search::mate_search::has_forced_mate_quiet(
            board,
            move_gen,
            self.depth,
            self.exhaustive,
            self.quiet,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExactEndings {
    pub mate: MateSearch,
    pub tablebase: Option<Tablebase>,
}

/// The number after `flag` in `args`, if the flag is there.
fn number_after(args: &[String], flag: &str) -> Result<Option<i32>, String> {
    match args.iter().position(|a| a == flag) {
        None => Ok(None),
        Some(i) => args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .map(Some)
            .ok_or_else(|| format!("{flag} needs a number of moves")),
    }
}

impl ExactEndings {
    /// From `args` (the program's arguments): `--mate-in N` searches every
    /// move for mates in up to N (`--mate-in-1` is `--mate-in 1`),
    /// `--mate-checks N` checking moves only up to N, `--mate-quiet M` allows
    /// M quiet moves per line in that search, and `--no-mate-search` turns it
    /// off. With none of these, `MateSearch::DEFAULT`. `--syzygy DIR` loads
    /// the tables. An error if a number or directory is missing, or the
    /// directory holds no tables.
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let tablebase = match args.iter().position(|a| a == "--syzygy") {
            Some(i) => {
                let dir = args.get(i + 1).ok_or("--syzygy needs a directory")?;
                Some(Tablebase::open(dir)?)
            }
            None => None,
        };
        let has = |flag: &str| args.iter().any(|a| a == flag);
        let quiet = number_after(args, "--mate-quiet")?.unwrap_or(0);
        let mut exhaustive = number_after(args, "--mate-in")?;
        if has("--mate-in-1") {
            exhaustive = Some(exhaustive.unwrap_or(0).max(1));
        }
        let checks = number_after(args, "--mate-checks")?;
        let mate = if has("--no-mate-search") {
            MateSearch::OFF
        } else if exhaustive.is_none() && checks.is_none() {
            MateSearch {
                quiet,
                ..MateSearch::DEFAULT
            }
        } else {
            let exhaustive = exhaustive.unwrap_or(0);
            MateSearch {
                depth: exhaustive.max(checks.unwrap_or(0)),
                exhaustive,
                quiet,
            }
        };
        Ok(Self { mate, tablebase })
    }

    /// `config` with these options set.
    pub fn apply(
        &self,
        config: crate::mcts::tactical_mcts::TacticalMctsConfig,
    ) -> crate::mcts::tactical_mcts::TacticalMctsConfig {
        crate::mcts::tactical_mcts::TacticalMctsConfig {
            leaf_mate: self.mate,
            tablebase: self.tablebase.clone(),
            ..config
        }
    }

    /// The result from White's side (+1, 0 or −1) if a game at `board` should
    /// end now: the side to move has a mate `self.mate` finds, or the tables cover the
    /// position (not in King of the Hill, where reaching the centre also wins).
    pub fn game_result(
        &self,
        board: &Board,
        move_gen: &crate::move_generation::MoveGen,
        koth: bool,
    ) -> Option<(f32, Ending)> {
        let white = |v: f32| if board.w_to_move { v } else { -v };
        if let Some(v) = self
            .tablebase
            .as_ref()
            .filter(|_| !koth)
            .and_then(|tb| tb.probe_value(board))
        {
            return Some((white(v), Ending::Tablebase));
        }
        if self.mate.finds_mate(board, move_gen) {
            return Some((white(1.0), Ending::Mate));
        }
        None
    }

    /// One line for a program's settings printout.
    pub fn describe(&self) -> String {
        format!(
            "mate search {}, tablebase {}",
            match self.mate {
                MateSearch { depth: 0, .. } => "off".to_string(),
                MateSearch {
                    depth, exhaustive, ..
                } if depth == exhaustive => format!("every move to mate in {depth}"),
                MateSearch {
                    depth,
                    exhaustive,
                    quiet,
                } => {
                    let mut d = format!("checking moves to mate in {depth}");
                    if quiet > 0 {
                        let s = if quiet == 1 { "" } else { "s" };
                        d += &format!(" with {quiet} quiet move{s} per line");
                    }
                    if exhaustive > 0 {
                        d += &format!(", every move to mate in {exhaustive}");
                    }
                    d
                }
            },
            match &self.tablebase {
                Some(t) => format!("up to {} pieces", t.max_pieces()),
                None => "off".into(),
            }
        )
    }
}
