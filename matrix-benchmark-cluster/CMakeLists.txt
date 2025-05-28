cmake_minimum_required(VERSION 3.31)

# Tell CMake to use mpicc before defining the project
set(CMAKE_C_COMPILER mpicc CACHE STRING "MPI C compiler")

project(matrix_benchmark_cluster C)

set(CMAKE_C_STANDARD 99)

# Find MPI
find_package(MPI REQUIRED)
if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found!")
endif()

# Find PAPI
find_path(PAPI_INCLUDE_DIR papi.h)
find_library(PAPI_LIBRARY papi)

if (NOT PAPI_INCLUDE_DIR OR NOT PAPI_LIBRARY)
    message(FATAL_ERROR "PAPI library not found.")
endif()

# Include headers
include_directories(${MPI_INCLUDE_PATH})
include_directories(${PAPI_INCLUDE_DIR})

# Define executable
add_executable(matrix_benchmark_cluster main.c)

# Link MPI and PAPI
target_link_libraries(matrix_benchmark_cluster ${MPI_C_LIBRARIES} ${PAPI_LIBRARY})