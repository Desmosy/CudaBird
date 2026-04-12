#include "network_kernels.h"
#include "../cuda_utils.h"

#include <curand_kernel.h>

namespace {

__global__ void initialize_population_kernel(NetworkWeights* networks,
                                             curandState* rng_states,
                                             int population_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) {
        return;
    }

    curandState local_state = rng_states[idx];
    NetworkWeights network = {};

    for (int param_idx = 0; param_idx < NETWORK_PARAM_COUNT; ++param_idx) {
        network.params[param_idx] = curand_normal(&local_state) * 0.5f;
    }

    networks[idx] = network;
    rng_states[idx] = local_state;
}

} // namespace

void initialize_population(NetworkWeights* d_networks,
                           curandState* d_rng_states,
                           int population_size) {
    const int grid_size = (population_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initialize_population_kernel<<<grid_size, BLOCK_SIZE>>>(d_networks,
                                                            d_rng_states,
                                                            population_size);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}
