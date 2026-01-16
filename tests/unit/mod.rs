mod board_tests;
mod move_generation_tests;
mod tensor_tests;
mod node_tests;
mod selection_tests;
mod graphviz_tests;
mod search_logger_tests;
mod tactical_mcts_tests;
mod mate_search_tests;
mod quiescence_tests;
mod see_tests;
mod koth_tests;
mod alpha_beta_tests;
mod iterative_deepening_tests;
mod history_tests;
mod transposition_tests;
mod hash_tests;

// Re-export common utilities
mod common {
    include!("../common/mod.rs");
}
