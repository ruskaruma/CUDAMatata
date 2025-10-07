#include "util.h"
#include <vector>
#include <random>

void cpuGemm(const float* A, const float* B, float* C, int M, int K, int N, float bias)
{

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc + bias; // add bias term
        }
    }
}

// check if two arrays are approximately equal
bool approxEqual(const float* A, const float* B, int size, float tol)
{
    for (int i = 0; i < size; ++i)
    {
        float diff = A[i] - B[i];
        if (diff < 0) diff = -diff;
        if (diff > tol) return false;
    }
    return true;
}

//random test data
void generateRandomData(float* data, int size)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}
