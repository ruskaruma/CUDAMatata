cmake_minimum_required(VERSION 3.18)
project(CUDAMatata LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# main executable - cpu only for now
add_executable(gemm 
    src/main.cpp 
    src/util.cpp
)

# todo: add cuda support when nvcc is available
