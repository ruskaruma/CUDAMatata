#!/usr/bin/env python3

import subprocess
import sys
import os

def run_test(matrix_size):
    print(f"Testing matrix size {matrix_size}x{matrix_size}")
    
    #run cpu version
    result_cpu = subprocess.run([
        './gemm', '--kernel', 'cpu', '--M', str(matrix_size), 
        '--K', str(matrix_size), '--N', str(matrix_size), '--iters', '1'
    ], capture_output=True, text=True, cwd='build')
    
    #run gpu versions
    result_naive = subprocess.run([
        './gemm', '--kernel', 'naive', '--M', str(matrix_size), 
        '--K', str(matrix_size), '--N', str(matrix_size), '--iters', '1'
    ], capture_output=True, text=True, cwd='build')
    
    result_tiled = subprocess.run([
        './gemm', '--kernel', 'tiled', '--M', str(matrix_size), 
        '--K', str(matrix_size), '--N', str(matrix_size), '--iters', '1'
    ], capture_output=True, text=True, cwd='build')
    
   #checking if all passed or not
    naive_pass = "PASS" in result_naive.stdout
    tiled_pass = "PASS" in result_tiled.stdout
    
    print(f"  CPU: OK")
    print(f"  Naive GPU: {'PASS' if naive_pass else 'FAIL'}")
    print(f"  Tiled GPU: {'PASS' if tiled_pass else 'FAIL'}")
    
    return naive_pass and tiled_pass

def main():
    print("CUDAMatata Correctness Tests")   
    test_sizes = [256, 512, 1024]
    all_passed = True    
    for size in test_sizes:
        if not run_test(size):
            all_passed = False
        print()
    if all_passed:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
