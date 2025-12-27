//! # Kingfisher Chess Engine Library
//!
//! A sophisticated chess engine featuring innovative **Tactical-First MCTS with Lazy Policy Evaluation**.
//! This library combines classical chess programming techniques with cutting-edge AI methods,
//! implementing a three-tier search prioritization system that follows chess principles while
//! maintaining computational efficiency.
//!
//! ## Key Innovations
//!
//! ### Tactical-First MCTS Architecture
//! - **Tier 1**: Mate search first (exhaustive forced-win analysis)
//! - **Tier 2**: Tactical moves prioritized (captures, checks, forks using classical heuristics)
//! - **Tier 3**: Lazy neural policy evaluation (deferred until after tactical exploration)
//!
//! This approach implements the chess principle of "examine all checks, captures, and threats"
//! while substantially reducing neural network computational overhead.
//!
//! ## Core Modules
//!
//! ### Search & AI
//! - **`mcts`** - Tactical-first MCTS with lazy policy evaluation
//! - **`search`** - Classical alpha-beta with enhancements
//! - **`neural_net`** - Neural network policy integration
//! - **`agent`** - Engine interface and move selection
//!
//! ### Chess Foundation
//! - **`board`**, **`board_utils`**, **`boardstack`** - Board representation
//! - **`move_generation`**, **`magic_bitboard`** - Move generation with magic bitboards
//! - **`eval`**, **`eval_constants`** - Pesto-style tapered evaluation
//! - **`transposition`** - Transposition table for caching
//!
//! ### Training & Analysis
//! - **`training`** - Neural network training pipeline
//! - **`benchmarks`** - Comprehensive strength testing
//! - **`tuning`** - Texel tuning for evaluation optimization
//! - **`egtb`** - Endgame tablebase integration
//!
//! ### Infrastructure
//! - **`uci`** - Universal Chess Interface protocol
//! - **`move_types`**, **`piece_types`** - Core chess types
//! - **`bits`**, **`hash`**, **`utils`** - Utility functions

pub mod agent;
pub mod arena;
pub mod benchmarks;
pub mod bits;
pub mod board;
pub mod board_utils;
pub mod boardstack;
pub mod eval;
pub mod eval_constants;
pub mod egtb;
pub mod hash;
pub mod magic_bitboard;
pub mod magic_constants;
pub mod make_move;
pub mod mcts;
pub mod move_generation;
pub mod move_types;
pub mod neural_net;
pub mod piece_types;
pub mod search;
pub mod tensor;
pub mod training;
pub mod transposition;
pub mod tuning;
pub mod uci;
pub mod utils;
