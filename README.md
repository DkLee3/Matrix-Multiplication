#  Matrix Multiplication Optimization
## Goal
1. Optimize `matrix_multiply_optimized()`
2. Optimize `matrix_multiply_simd()` with simd
3. Optimize `matrix_multiply_optimized()` with OpenMP or pthread?

## Usage

Only file is `matrix.cpp`. Compiled & run.
```console
make matrix && ./matrix
```

Output like
```console
SIMD Matrix multiplication took 0.36 seconds
Unoptimized Matrix multiplication took 7.65 seconds
SUCCESS: Both optimized and unoptimized gave same results
```


The input size defaults to `1024`, and can be changed by changing line 1 (or something like below)
    #define SIZE 1024
