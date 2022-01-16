cmake_minimum_required(VERSION 3.12)
project(FinalProject CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(FinalProject utils.cu image.cu timer.cu solution.cu sequential_solution.cu parallel_solution_baseline.cu parallel_solution_v2.cu parallel_solution_v3.cu parallel_solution_v4.cu main.cu)

set_target_properties(FinalProject PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
