cmake_minimum_required(VERSION 3.31)
project(matrix_benchmark_gpu CUDA)

set(CMAKE_CUDA_STANDARD 98)

add_executable(matrix_benchmark_gpu main.cu)

set_target_properties(matrix_benchmark_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
