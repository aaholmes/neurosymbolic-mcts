mod board_tests;
mod move_generation_tests;
mod tensor_tests;
mod node_tests;
mod selection_tests;

// Re-export common utilities
mod common {
    include!("../common/mod.rs");
}
