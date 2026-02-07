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
mod board_utils_tests;
mod bits_tests;
mod eval_tests;
mod boardstack_tests;
mod simulation_tests;
mod make_move_tests;
mod selection_optimization_tests;
mod tree_reuse_tests;
mod gives_check_tests;
mod self_play_loop_tests;
mod incremental_hash_tests;
mod legality_check_tests;
mod training_diversity_tests;
mod agent_tests;
mod utils_tests;
mod magic_bitboard_tests;
mod tensor_extended_tests;
mod training_tests;
mod evaluate_models_tests;

// Re-export common utilities
mod common {
    include!("../common/mod.rs");
}
