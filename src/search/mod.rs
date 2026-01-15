//! Classical search algorithms for the chess engine.
//!
//! This module provides traditional game-tree search algorithms that form the foundation
//! of the engine's tactical analysis. These algorithms are integrated with the MCTS system
//! through the three-tier architecture:
//!
//! - **Tier 1 (Safety Gate)**: Uses [`mate_search`] for exhaustive forced-win detection
//! - **Tier 2 (Tactical Grafting)**: Uses [`quiescence_search_tactical`] for tactical evaluation
//! - **Tier 3 (Strategic Guidance)**: Classical search provides move ordering hints
//!
//! # Module Structure
//!
//! - [`alpha_beta`]: Full-width alpha-beta search with pruning enhancements (NMP, LMR, killers)
//! - [`mate_search`]: Parallel portfolio mate search with multiple strategies
//! - [`quiescence_search`]: Captures-only search to avoid horizon effect
//! - [`iterative_deepening`]: Iterative deepening with aspiration windows
//! - [`history`]: History heuristic table for move ordering
//! - [`see`]: Static Exchange Evaluation for capture analysis
//! - [`koth`]: King of the Hill variant win detection
//!
//! # Usage Example
//!
//! ```ignore
//! use kingfisher::search::{mate_search, quiescence_search};
//! use kingfisher::boardstack::BoardStack;
//! use kingfisher::move_generation::MoveGen;
//!
//! let mut board = BoardStack::new();
//! let move_gen = MoveGen::new();
//!
//! // Check for forced mate up to depth 4
//! let (score, best_move, nodes) = mate_search(&mut board, &move_gen, 4, false);
//! if score >= 1_000_000 {
//!     println!("Mate found: {:?}", best_move);
//! }
//! ```
//!
//! # Score Conventions
//!
//! - Evaluation scores are in centipawns (100 = 1 pawn advantage)
//! - Mate scores: `1_000_000 + depth` for delivering mate, `-1_000_000 - depth` for being mated
//! - Draw scores: `0`

pub mod alpha_beta;
pub mod quiescence;
pub mod iterative_deepening;
pub mod history;
pub mod mate_search;
mod koth;
mod see;

pub use history::{HistoryTable, MAX_PLY};
pub use mate_search::mate_search;
pub use quiescence::{quiescence_search, quiescence_search_tactical};
pub use iterative_deepening::iterative_deepening_ab_search;
pub use koth::koth_center_in_3;
pub use see::see;
