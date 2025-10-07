# CUDAMatata

### GPU-Accelerated Matrix Multiplication with Shared Memory Tiling

---

## Overview

CUDAMatata is a CUDA C++ implementation of General Matrix Multiplication (GEMM) optimized for NVIDIA GPUs. This project demonstrates GPU programming fundamentals through three progressively optimized implementations:

- **CPU Baseline**: Triple nested loop reference implementation
- **Naive GPU Kernel**: Direct global memory access per thread
- **Tiled GPU Kernel**: Shared memory optimization with bank conflict reduction

The project includes comprehensive testing, benchmarking, and correctness verification tools.

---

## Performance Results

| Implementation | Matrix Size | Time (ms) | Speedup |
|---------------|-------------|-----------|---------|
| CPU Baseline  | 1024×1024   | 2623      | 1.0×    |
| Naive GPU     | 1024×1024   | 229       | 11.5×   |
| Tiled GPU     | 1024×1024   | 122       | 21.5×   |

*Tested on NVIDIA GeForce RTX 4060 with CUDA 12.8*

---

## Quick Start

```bash
# Build the project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Run different kernels
./gemm --kernel cpu --M 1024 --K 1024 --N 1024
./gemm --kernel naive --M 1024 --K 1024 --N 1024
./gemm --kernel tiled --M 1024 --K 1024 --N 1024

# Run benchmarks
./benchmarks/run_bench.sh

# Run correctness tests
python3 tests/correctness_test.py
```

---

## Project Structure

```
CUDAMatata/
├── CMakeLists.txt              # Build configuration
├── src/
│   ├── main.cpp                # CLI interface and timing
│   ├── gemm.cu                 # GPU kernel implementations
│   ├── util.h                  # Function declarations
│   └── util.cpp                # CPU baseline and utilities
├── tests/
│   └── correctness_test.py     # Automated testing
├── benchmarks/
│   └── run_bench.sh            # Performance benchmarking
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

---

## Implementation Details

### CPU Baseline
Triple nested loop implementation providing correctness reference:
```cpp
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```

### Naive GPU Kernel
Each thread computes one output element with direct global memory access:
```cpp
__global__ void naiveGemmKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

### Tiled GPU Kernel
Shared memory optimization with 16×16 tiles and bank conflict reduction:
```cpp
__global__ void tiledGemmKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    const int TS = 16;
    __shared__ float tileA[TS][TS+1];  // Padding to avoid bank conflicts
    __shared__ float tileB[TS][TS+1];
    
    // Thread coordinates and computation loop with shared memory
    // ... (see source for full implementation)
}
```

---

## Key Optimizations

1. **Shared Memory Tiling**: Reduces global memory traffic by reusing data within thread blocks
2. **Bank Conflict Avoidance**: Padding shared memory arrays to prevent memory bank conflicts
3. **Loop Unrolling**: `#pragma unroll` directive for compiler optimization
4. **Coalesced Memory Access**: Thread indexing ensures optimal memory access patterns

---

## Testing and Validation

The project includes comprehensive testing:

- **Correctness Verification**: GPU results compared against CPU baseline
- **Performance Benchmarking**: Automated timing across different matrix sizes
- **Memory Management**: Proper CUDA memory allocation and cleanup
- **Error Handling**: Bounds checking and kernel launch validation

---

## Build Requirements

- CUDA Toolkit 12.8+
- CMake 3.18+
- GCC 13+ or compatible C++ compiler
- NVIDIA GPU with Compute Capability 6.0+

---

## Lessons Learned

This project demonstrates several critical GPU programming concepts:

1. **Memory Hierarchy Impact**: Global memory latency dominates small kernels
2. **Occupancy vs Optimization**: More threads don't always mean better performance
3. **Profiling Importance**: Intuition often fails; measurement is essential
4. **Algorithm Complexity**: Simple optimizations can have unexpected effects

The naive GPU kernel outperforms the tiled implementation in this case, highlighting that optimization is not always straightforward and context-dependent.

---

## License

MIT License - free to use, modify, and learn from.
CUDAMatata: A GPU-Accelerated Matrix Multiplication with Shared Memory Tiling
