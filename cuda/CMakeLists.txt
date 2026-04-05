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
