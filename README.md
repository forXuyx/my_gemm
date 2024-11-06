# 深入浅出矩阵乘法优化

## 吞吐量
```
峰值吞吐量(FLOPS) = CUDA 核心数 × 时钟频率 × 每个 CUDA 核心每个周期的 FLOPS
```
以本repo使用的GPU为例：
NVIDIA A100-PCIE-40GB 
CUDA 核心数: 6912
时钟频率: 1410 MHz
每个 CUDA 核心每个周期的 FLOPS: 2
峰值吞吐量(FLOPS) = 6912 × 1410 × 2 = 19.46 TFLOPS = 19460 GFLOPS

## 矩阵乘法
矩阵乘法是一种常见的线性代数运算，其计算公式如下：
```
C = A × B
C[i][j] = Σ A[i][k] × B[k][j]
```
