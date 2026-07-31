// 问题 0.1：第一个 CUDA 程序（模块 8 的编译实验也用它）。
// 编译运行：make run/m0_env/01_hello
#include "common.h"

__global__ void hello() {
    printf("hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

// kernel<<<gridDim, blockDim>>>(...)
// gridDim 表示启动了几个 block，blockDim 表示一个 block 里有几个 thread
int main() {
    hello<<<4, 8>>>();
    CUDA_CHECK_KERNEL();
    return 0;
}
