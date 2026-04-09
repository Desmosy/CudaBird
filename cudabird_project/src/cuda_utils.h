#ifndef CUDABIRD_CUDA_UTILS_H
#define CUDABIRD_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t cuda_status__ = (call);                                           \
        if (cuda_status__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(cuda_status__));                               \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                           \
    do {                                                                              \
        cudaError_t cuda_status__ = cudaGetLastError();                               \
        if (cuda_status__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(cuda_status__));                     \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

void initialize_curand_states(curandState* d_states, int count, unsigned long long seed);

#endif // CUDABIRD_CUDA_UTILS_H
