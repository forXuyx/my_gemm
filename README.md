# 深入浅出矩阵乘法优化

## 吞吐量
```
峰值吞吐量(FLOPS) = CUDA 核心数 × 时钟频率 × 每个 CUDA 核心每个周期的 FLOPS
```
以本repo使用的GPU为例：
NVIDIA GeForce RTX 2070 
CUDA 核心数: 2304
时钟频率: 1410 MHz
每个 CUDA 核心每个周期的 FLOPS: 2
峰值吞吐量(FLOPS) = 2304 × 1410 × 2 = 6.5 TFLOPS = 6500 GFLOPS