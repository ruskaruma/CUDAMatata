#pragma once
#include <cstdio>
#include <vector>
#include <chrono>

void cpuGemm(const float* A, const float* B, float* C, int M, int K, int N, float bias);

bool approxEqual(const float* A, const float* B, int size, float tol = 1e-4);

void generateRandomData(float* data, int size);

class CpuTimer {
public:
    CpuTimer() : startTime(std::chrono::high_resolution_clock::now()) {}
    
    float elapsedMs() {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        return duration.count() / 1000.0f;
    }
    
private:
    std::chrono::high_resolution_clock::time_point startTime;
};

void runNaiveGemm(const float* h_A, const float* h_B, float* h_C, int M, int K, int N);
void runTiledGemm(const float* h_A, const float* h_B, float* h_C, int M, int K, int N);
