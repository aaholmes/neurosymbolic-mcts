//! Classical Search Algorithms
//!
//! This module contains the implementation of standard chess search algorithms
//! used by the engine's "SimpleAgent" and as components of the hybrid "HumanlikeAgent".
//!
//! Key components:
//! - **Alpha-Beta Pruning**: The core minimax search optimized with alpha-beta pruning.
//! - **Iterative Deepening**: Time-managed search that progressively deepens.
//! - **Mate Search**: Specialized exhaustive search for finding forced mates.
//! - **Quiescence Search**: Tactical resolution to ensure stable leaf evaluations.
//! - **SEE (Static Exchange Evaluation)**: Heuristic to prune bad captures.

mod alpha_beta;
mod history;
mod iterative_deepening;
mod mate_search;
mod quiescence;
mod see;
pub mod koth;

pub use alpha_beta::alpha_beta_search;
pub use history::{HistoryTable, MAX_PLY};
pub use iterative_deepening::{aspiration_window_ab_search, iterative_deepening_ab_search};
pub use mate_search::mate_search;
pub use quiescence::quiescence_search;
pub use see::see;
pub use koth::koth_center_in_3;
