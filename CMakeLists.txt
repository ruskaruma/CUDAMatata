cmake_minimum_required(VERSION 3.18)
project(CUDAMatata LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

#enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

#main executable with cuda support
add_executable(gemm 
    src/main.cpp 
    src/util.cpp
    src/gemm.cu
)

#set cuda architecture for RTX 4060
set_property(TARGET gemm PROPERTY CUDA_ARCHITECTURES 86)
