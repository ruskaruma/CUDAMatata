#include <cuda_runtime.h>
#include <iostream>
#include "util.h"
__global__ void naiveGemmKernel(const float* A, const float* B, float* C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
__global__ void tiledGemmKernel(const float* A, const float* B, float* C, int M, int K, int N)
{
    const int TS = 16;
    __shared__ float tileA[TS][TS];
    __shared__ float tileB[TS][TS];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    float sum = 0.0f;
    for (int tile = 0; tile < (K + TS - 1) / TS; ++tile)
    {
        int k = tile * TS + threadIdx.x;
        if (row < M && k < K)
        {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + k];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        k = tile * TS + threadIdx.y;
        if (k < K && col < N)
        {
            tileB[threadIdx.y][threadIdx.x] = B[k * N + col];
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < TS; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
void runNaiveGemm(const float* h_A, const float* h_B, float* h_C, int M, int K, int N)
{
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);  //16x16=256 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    naiveGemmKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
void runTiledGemm(const float* h_A, const float* h_B, float* h_C, int M, int K, int N)
{
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    tiledGemmKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
