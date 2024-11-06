#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.h"


// gpu版本的矩阵乘法（naive版）
__global__ void naiveGemm (float *a, float *b, float *c, const int M, const int N, const int K) {

    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float sum = 0;
        #pragma unroll
        for (int k = 0; k < K; k ++ ) {
            sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = sum;
    }
}

int main() {

    // 矩阵大小
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    const int outer_repeat = 10, inner_repeat = 1;
    const int M = 512, N = 512, K = 512; // 矩阵大小
    const int BM = 32, BN = 32; // 块大小
    const int TESTNUM = 15; // 测试次数

    // --------------------------------------------
    // cublas版本的矩阵乘法

    printf("\nKernel = cublasGemm\n");

    // 计算误差
    float max_error = testCublasMaxError(M, N, K);
    printf("Max error: %f\n", max_error);

    // 测试性能
    for (int i = 0; i < TESTNUM; i ++ ) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j ++ ) {
            double this_sec = testCublasPerformance(M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = 2.0 * M * N * K / avg_sec / 1024 / 1024 / 1024;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }

    // --------------------------------------------
    // naive版本的矩阵乘法

    printf("\nKernel = naiveGemm\n");
    void (*gpuGemm) (float *, float *, float *, const int, const int, const int) = naiveGemm;

    // 计算误差
    dim3 blk(BM, BN);
    dim3 grid(ceil_div(M, BM), ceil_div(N, BN));
    max_error = testMaxError(gpuGemm, grid, blk, M, N, K);
    printf("Max error: %f\n", max_error);

    // 测试性能
    for (int i = 0; i < TESTNUM; i ++ ) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blk(BM, BN);
        dim3 grid(ceil_div(M, BM), ceil_div(N, BN));

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j ++ ) {
            double this_sec = testPerformance(gpuGemm, grid, blk, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = 2.0 * M * N * K / avg_sec / 1024 / 1024 / 1024;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }

    // --------------------------------------------


    return 0;
}