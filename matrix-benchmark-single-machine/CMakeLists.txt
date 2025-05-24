cmake_minimum_required(VERSION 3.30)
project(MatrixMult C)

set(CMAKE_C_STANDARD 99)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find PAPI using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAPI REQUIRED papi)

# Add your executable
add_executable(MatrixMult main.c)

target_include_directories(MatrixMult PRIVATE ${PAPI_INCLUDE_DIRS})
# Link OpenMP to your executable
target_link_libraries(MatrixMult PRIVATE ${PAPI_LIBRARIES} OpenMP::OpenMP_C m)