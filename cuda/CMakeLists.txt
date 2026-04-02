cmake_minimum_required(VERSION 3.18)
project(caissawary_cuda CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Auto-detect GPU architecture; override with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Main static library
add_library(caissawary_cuda STATIC
    tree_store.cu
    movegen.cu
    apply_move.cu
)
target_include_directories(caissawary_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Phase 1 tests
add_executable(test_tree_store test/test_tree_store.cu)
target_link_libraries(test_tree_store caissawary_cuda)

# Phase 2 tests
add_executable(test_movegen test/test_movegen.cu)
target_link_libraries(test_movegen caissawary_cuda)
