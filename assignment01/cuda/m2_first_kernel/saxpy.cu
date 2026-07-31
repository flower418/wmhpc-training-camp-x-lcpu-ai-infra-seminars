#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t error = (call);                                 \
        if (error != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",         \
                    cudaGetErrorName(error), __FILE__, __LINE__,    \
                    cudaGetErrorString(error));                     \
            exit(1);                                                \
        }                                                           \
    } while (0)                                                     \

#define CUDA_CHECK_KERNEL()                                         \
    do {                                                            \
        CUDA_CHECK(cudaGetLastError());                             \
        CUDA_CHECK(cudaDeviceSynchronize());                        \
    } while (0)                                                     \

__global__ void saxpy(float* x, float* y, int n) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        y[i] = 2 * x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    size_t bytes = (size_t)n * sizeof(float);
    double SUM = 0;

    if (n == 0) {
        printf("SUM=0\n");
        exit(0);
    }

    float* x = (float*)malloc(bytes);
    float* y = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) {
        x[i] = ((i % 2048) - 1024) * 0.5f;
        y[i] = (i % 1024) - 512;
    }

    float* dx;
    float* dy;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, bytes, cudaMemcpyHostToDevice));

    // 用于给 kernel 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    saxpy<<<4096, 256>>>(dx, dy, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK_KERNEL();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(y, dy, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        SUM += y[i];
    }

    printf("n=%d, time=%fms\n", n, ms);
    printf("SUM=%.0f\n", SUM);
}