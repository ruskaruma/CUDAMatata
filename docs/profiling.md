# CUDA Profiling Guide

## Nsight Compute Profiling

### Basic Profiling
```bash
# Profile the tiled kernel
nv-nsight-cu-cli --kernel-name tiledGemmKernel ./gemm --kernel tiled --M 1024 --K 1024 --N 1024

# Profile with metrics
nv-nsight-cu-cli --metrics gpu__time_duration,sm__throughput.avg.pct_of_peak_sustained_elapsed ./gemm --kernel tiled
```

### Nsight Systems Profiling
```bash
# Full application trace
nsys profile -o cuda_profile ./gemm --kernel tiled --M 1024 --K 1024 --N 1024

# View results
nsys-ui cuda_profile.nsys-rep
```

## Key Metrics to Monitor

### Performance Metrics
- **Achieved Occupancy**: Should be >50%
- **DRAM Throughput**: Compare to peak bandwidth
- **Shared Memory Utilization**: Check for bank conflicts
- **FLOPs Achieved**: Compare to theoretical peak

### Memory Metrics
- **Global Memory Load Efficiency**: Should be >80%
- **Shared Memory Bank Conflicts**: Should be minimal
- **L1 Cache Hit Rate**: Higher is better

## Optimization Targets

1. **Occupancy**: Increase active warps per SM
2. **Memory Bandwidth**: Reduce global memory traffic
3. **Compute Utilization**: Maximize ALU usage
4. **Latency Hiding**: Overlap memory and compute

## Common Issues

- **Low Occupancy**: Reduce register usage or increase block size
- **Memory Bottlenecks**: Implement better memory coalescing
- **Bank Conflicts**: Add padding to shared memory arrays
