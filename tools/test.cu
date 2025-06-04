#include <cstdio>
#include <cuda_runtime.h>

__global__ void testKernel() {
    // Do nothing
}

int main() {
    // Launch kernel with 1 block and 1 thread
    testKernel<<<1, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel to finish and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA kernel executed successfully.\n");
    return 0;
}