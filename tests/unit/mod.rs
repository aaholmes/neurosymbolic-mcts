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

// Re-export common utilities
mod common {
    include!("../common/mod.rs");
}
