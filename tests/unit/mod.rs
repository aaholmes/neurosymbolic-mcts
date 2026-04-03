mod agent_tests;
mod alpha_beta_tests;
mod bits_tests;
mod board_encoding_tests;
mod board_tests;
mod board_utils_tests;
mod boardstack_tests;
mod eval_tests;
mod ext_qsearch_tests;
mod evaluate_models_tests;
mod experiment_config_tests;
mod experiment_metrics_tests;
mod gives_check_tests;
mod graphviz_tests;
mod hash_tests;
mod history_tests;
mod incremental_hash_tests;
mod inference_server_tests;
mod iterative_deepening_tests;
mod koth_tests;
mod legality_check_tests;
mod magic_bitboard_tests;
mod make_move_tests;
mod mate_search_tests;
mod move_generation_tests;
mod neural_mcts_tests;
mod nn_counter_tests;
mod node_tests;
mod pesto_qsearch_tests;
mod position_classifier_tests;
mod quiescence_tests;
mod search_logger_tests;
mod see_tests;
mod selection_optimization_tests;
mod selection_tests;
mod self_play_loop_tests;
mod simulation_tests;
mod tactical_detection_tests;
mod tactical_mcts_tests;
mod tensor_extended_tests;
mod tensor_tests;
mod training_diversity_tests;
mod training_pipeline_tests;
mod training_tests;
mod transposition_tests;
mod tree_reuse_tests;
mod utils_tests;
mod value_target_tests;

// Re-export common utilities
mod common {
    include!("../common/mod.rs");
}

