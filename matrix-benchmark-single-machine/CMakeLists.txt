cmake_minimum_required(VERSION 3.31)
project(matrix-benchmark-single-machine C)

set(CMAKE_C_STANDARD 99)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find PAPI using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAPI REQUIRED papi)

# Add your executable
add_executable(matrix-benchmark-single-machine main.c)

# Include PAPI headers
target_include_directories(matrix-benchmark-single-machine PRIVATE ${PAPI_INCLUDE_DIRS})

# Link all necessary libraries
target_link_libraries(matrix-benchmark-single-machine
        PRIVATE
        ${PAPI_LIBRARIES}
        OpenMP::OpenMP_C
)
