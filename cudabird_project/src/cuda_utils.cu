#include "config.h"
#include "cuda_utils.h"

namespace {

__global__ void initialize_curand_states_kernel(curandState* states,
                                                int count,
                                                unsigned long long seed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    curand_init(seed, idx, 0, &states[idx]);
}

} // namespace

void initialize_curand_states(curandState* d_states, int count, unsigned long long seed) {
    const int grid_size = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initialize_curand_states_kernel<<<grid_size, BLOCK_SIZE>>>(d_states, count, seed);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}
