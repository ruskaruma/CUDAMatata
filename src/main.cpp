#include <iostream>
#include <string>
#include <vector>
#include "util.h"

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --kernel <type>     kernel type: cpu, naive, tiled (default: cpu)\n";
    std::cout << "  --M <size>          matrix A rows (default: 1024)\n";
    std::cout << "  --K <size>          matrix A cols / B rows (default: 1024)\n";
    std::cout << "  --N <size>          matrix B cols (default: 1024)\n";
    std::cout << "  --iters <count>     iterations for timing (default: 1)\n";
    std::cout << "  --help              show this help\n";
}

int main(int argc, char** argv)
{
    std::string kernelType = "cpu";
    int M = 1024, K = 1024, N = 1024;
    int iters = 1;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--kernel" && i + 1 < argc) {
            kernelType = argv[++i];
        } else if (arg == "--M" && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        } else if (arg == "--K" && i + 1 < argc) {
            K = std::stoi(argv[++i]);
        } else if (arg == "--N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        }
    }
    
    std::cout << "CUDAMatata: It means no race conditions, for the rest of your days.\n";
    std::cout << "Matrix size: M=" << M << ", K=" << K << ", N=" << N << "\n";
    std::cout << "Kernel: " << kernelType << "\n";
    
    // generate test matrices
    std::vector<float> A(M * K), B(K * N), C_cpu(M * N), C_gpu(M * N);
    generateRandomData(A.data(), M * K);
    generateRandomData(B.data(), K * N);
    
    if (kernelType == "cpu")
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cpuGemm(A.data(), B.data(), C_cpu.data(), M, K, N, 0.0f);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float avgTimeMs = duration.count() / (1000.0f * iters);
        
        std::cout << "CPU GEMM time: " << avgTimeMs << " ms\n";
        
        float sum = 0.0f;
        for (int i = 0; i < M * N; ++i) {
            sum += C_cpu[i];
        }
        std::cout << "Result sum: " << sum << " (should be non-zero)\n";
    }
    else if (kernelType == "naive")
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            runNaiveGemm(A.data(), B.data(), C_gpu.data(), M, K, N);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float avgTimeMs = duration.count() / (1000.0f * iters);
        
        std::cout << "Naive GPU GEMM time: " << avgTimeMs << " ms\n";
        
        cpuGemm(A.data(), B.data(), C_cpu.data(), M, K, N, 0.0f);
        bool correct = approxEqual(C_cpu.data(), C_gpu.data(), M * N);
        std::cout << "Correctness check: " << (correct ? "PASS" : "FAIL") << "\n";
    }
    else if (kernelType == "tiled")
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            runTiledGemm(A.data(), B.data(), C_gpu.data(), M, K, N);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float avgTimeMs = duration.count() / (1000.0f * iters);
        
        std::cout << "Tiled GPU GEMM time: " << avgTimeMs << " ms\n";
        
        cpuGemm(A.data(), B.data(), C_cpu.data(), M, K, N, 0.0f);
        bool correct = approxEqual(C_cpu.data(), C_gpu.data(), M * N);
        std::cout << "Correctness check: " << (correct ? "PASS" : "FAIL") << "\n";
    }
    else
    {
        std::cout << "Unknown kernel type: " << kernelType << "\n";
        return 1;
    }
    
    return 0;
}
