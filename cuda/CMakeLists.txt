cmake_minimum_required(VERSION 3.18)
project(caissawary_cuda CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Auto-detect GPU architecture; override with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Enable relocatable device code for cross-TU __device__ calls and extern __device__ variables
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Main static library
add_library(caissawary_cuda STATIC
    tree_store.cu
    movegen.cu
    apply_move.cu
    quiescence.cu
    quick_checks.cu
    mcts_kernel.cu
    nn_weights.cu
    nn_ops.cu
    nn_forward.cu
    block_ops.cu
    block_forward.cu
    transformer_weights.cu
    transformer_ops.cu
    transformer_forward.cu
    selfplay.cu
)
target_include_directories(caissawary_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(caissawary_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Phase 1 tests
add_executable(test_tree_store test/test_tree_store.cu)
target_link_libraries(test_tree_store caissawary_cuda)
set_target_properties(test_tree_store PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Phase 2 tests
add_executable(test_movegen test/test_movegen.cu)
target_link_libraries(test_movegen caissawary_cuda)
set_target_properties(test_movegen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Phase 3 tests: quiescence search + quick checks
add_executable(test_quiescence test/test_quiescence.cu)
target_link_libraries(test_quiescence caissawary_cuda)
set_target_properties(test_quiescence PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_quick_checks test/test_quick_checks.cu)
target_link_libraries(test_quick_checks caissawary_cuda)
set_target_properties(test_quick_checks PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Phase 3 tests: MCTS kernel
add_executable(test_mcts_kernel test/test_mcts_kernel.cu)
target_link_libraries(test_mcts_kernel caissawary_cuda)
set_target_properties(test_mcts_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# NN ops tests
add_executable(test_nn_ops test/test_nn_ops.cu)
target_link_libraries(test_nn_ops caissawary_cuda)
set_target_properties(test_nn_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Block ops tests (256-thread inference)
add_executable(test_block_ops test/test_block_ops.cu)
target_link_libraries(test_block_ops caissawary_cuda)
set_target_properties(test_block_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Profiling: inference latency with real weights
add_executable(test_profile_latency test/test_profile_latency.cu)
target_link_libraries(test_profile_latency caissawary_cuda)
set_target_properties(test_profile_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Multi-tree eval: N independent MCTS trees running in parallel
add_executable(test_multi_tree_eval test/test_multi_tree_eval.cu)
target_link_libraries(test_multi_tree_eval caissawary_cuda)
set_target_properties(test_multi_tree_eval PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Transformer tests
add_executable(test_transformer test/test_transformer.cu)
target_link_libraries(test_transformer caissawary_cuda)
set_target_properties(test_transformer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Self-play test
add_executable(test_selfplay test/test_selfplay.cu)
target_link_libraries(test_selfplay caissawary_cuda)
set_target_properties(test_selfplay PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Self-play standalone executable
add_executable(selfplay selfplay_main.cu)
target_link_libraries(selfplay caissawary_cuda)
set_target_properties(selfplay PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Attention debug test
add_executable(test_attn_debug test/test_attn_debug.cu)
target_link_libraries(test_attn_debug caissawary_cuda)
set_target_properties(test_attn_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Transformer vs PyTorch regression tests
add_executable(test_transformer_vs_pytorch test/test_transformer_vs_pytorch.cu)
target_link_libraries(test_transformer_vs_pytorch caissawary_cuda)
set_target_properties(test_transformer_vs_pytorch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Layer-by-layer timing profile
add_executable(test_layer_timing test/test_layer_timing.cu)
target_link_libraries(test_layer_timing caissawary_cuda)
set_target_properties(test_layer_timing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
