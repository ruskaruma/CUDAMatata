#!/bin/bash

echo "CUDAMatata Benchmark Suite"
echo "========================="

cd build

echo "Matrix Size,CPU (ms),Naive GPU (ms),Tiled GPU (ms),Speedup"
echo "512,512,512"
./gemm --kernel cpu --M 512 --K 512 --N 512 --iters 3
./gemm --kernel naive --M 512 --K 512 --N 512 --iters 3
./gemm --kernel tiled --M 512 --K 512 --N 512 --iters 3

echo "1024,1024,1024"
./gemm --kernel cpu --M 1024 --K 1024 --N 1024 --iters 1
./gemm --kernel naive --M 1024 --K 1024 --N 1024 --iters 1
./gemm --kernel tiled --M 1024 --K 1024 --N 1024 --iters 1

echo "Benchmark complete!"
