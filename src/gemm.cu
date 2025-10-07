#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <iostream>
#include "util.h"

__global__ void naiveGemmKernel(const float* A, const float* B, float* C, int M, int K, int N)
{
}

__global__ void tiledGemmKernel(const float* A, const float* B, float* C, int M, int K, int N)
{
}

#else

#endif
