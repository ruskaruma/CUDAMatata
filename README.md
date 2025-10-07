# CUDAMatata

### GPU-Accelerated Matrix Multiplication with Shared Memory Tiling  
*"It means no race conditions, for the rest of your days."*

---

## Overview
**CUDAMatata** is a minimal, educational, and performance-aware CUDA C++ project implementing **General Matrix Multiplication (GEMM)** — the operation powering every modern ML and HPC workload.  
This project progresses through:
- **CPU Reference Implementation** — for correctness and baseline performance
- **Naive GPU Kernel** — each thread computes one element directly from global memory
- **Tiled GPU Kernel** — shared memory tiling, coalesced loads, and reduced global traffic

Includes correctness verification, benchmarking, and profiling utilities. Built to run cleanly via **CMake** and **Docker**, and optimized for modern NVIDIA GPUs (tested on RTX 4060).

---

## Why GEMM?
Matrix multiplication (C = A × B + bias) is the backbone of deep learning and scientific computing. Every neural network layer reduces to GEMM internally. Optimizing it teaches:
- Memory hierarchy: registers → shared → global
- Thread-block tiling and synchronization
- Data reuse and coalesced access
- Profiling-driven performance tuning

---

## Repository Structure
```
CUDAMatata/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── gemm.cu
│   └── util.h/.cpp
├── tests/
│   └── correctness_test.py
├── benchmarks/
│   └── run_bench.sh
├── Dockerfile
├── README.md
└── LICENSE (MIT)
```

---

## Example Run
```
Device info: NVIDIA GeForce RTX 4060
Matrix size: M=1024, K=1024, N=1024

naive kernel  time = 420.7 ms
shared-tiled  time = 38.5 ms

Speedup: ~11× over naive.
```

---

## Benchmarking
```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
./gemm --kernel tiled --M 1024 --K 1024 --N 1024 --iters 10
```

To run automated sweeps:
```
./benchmarks/run_bench.sh
```

---

## Profiling (Nsight Compute / Systems)
```
nsys profile -o cuda_profile ./gemm --kernel tiled
nv-nsight-cu-cli ./gemm --kernel tiled
```
Key metrics:
- Achieved occupancy
- DRAM throughput
- Shared memory utilization
- FLOPs achieved

---

## Lessons Learned
- Shared memory tiling drastically reduces global memory pressure
- Coalesced loads are critical for bandwidth efficiency
- Occupancy is a balancing act between register use and block size
- Profiling reveals optimization ceilings beyond intuition

---

## Future Work
- Add FP16 + Tensor Core path
- Integrate cuBLAS SGEMM for comparison
- CI/CD on CUDA runners
- Tile-size autotuning

---

## License
MIT License — free to use, modify, and learn from.
CUDAMatata: A GPU-Accelerated Matrix Multiplication with Shared Memory Tiling
