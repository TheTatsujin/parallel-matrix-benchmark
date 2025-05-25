# How to compile

# Cluster setup

1. Make sure cmake finds the PAPI library
2. Build the project

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=ninja/linux/x64/ninja -G Ninja -S Projects/parallel-matrix-benchmark/matrix-benchmark-cluster -B Projects/parallel-matrix-benchmark/matrix-benchmark-cluster/cmake-build-debug
cmake --build Projects/parallel-matrix-benchmark/matrix-benchmark-cluster/cmake-build-debug --target matrix_benchmark_cluster -j 4
```
