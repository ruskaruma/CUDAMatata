FROM nvidia/cuda:12.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j

CMD ["./build/gemm", "--kernel", "naive", "--M", "1024", "--K", "1024", "--N", "1024"]
