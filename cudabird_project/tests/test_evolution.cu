#include "evolution/evolution.h"
#include "evolution/evolution_kernels.h"
#include "neural_net/network.h"
#include "neural_net/network_kernels.h"
#include "cuda_utils.h"

#include <assert.h>
#include <vector>
#include <stdio.h>

int main(int argc, char** argv) {
    NetworkWeights* d_population = nullptr;
    NetworkWeights* d_next_population = nullptr;
    curandState* d_rng_states = nullptr;
    int* d_ranked_indices = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_population),
                          sizeof(NetworkWeights) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next_population),
                          sizeof(NetworkWeights) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rng_states),
                          sizeof(curandState) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ranked_indices),
                          sizeof(int) * POPULATION_SIZE));

    initialize_curand_states(d_rng_states, POPULATION_SIZE, 999ULL);
    initialize_population(d_population, d_rng_states, POPULATION_SIZE);

    std::vector<int> ranked_indices(POPULATION_SIZE);
    for (int idx = 0; idx < POPULATION_SIZE; ++idx) {
        ranked_indices[idx] = idx;
    }

    CUDA_CHECK(cudaMemcpy(d_ranked_indices,
                          ranked_indices.data(),
                          sizeof(int) * POPULATION_SIZE,
                          cudaMemcpyHostToDevice));
    evolve_population(d_next_population,
                      d_population,
                      d_ranked_indices,
                      d_rng_states,
                      POPULATION_SIZE);

    std::vector<NetworkWeights> host_population(POPULATION_SIZE);
    std::vector<NetworkWeights> host_next_population(POPULATION_SIZE);
    CUDA_CHECK(cudaMemcpy(host_population.data(),
                          d_population,
                          sizeof(NetworkWeights) * POPULATION_SIZE,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_next_population.data(),
                          d_next_population,
                          sizeof(NetworkWeights) * POPULATION_SIZE,
                          cudaMemcpyDeviceToHost));

    for (int elite_idx = 0; elite_idx < ELITE_COUNT; ++elite_idx) {
        for (int param_idx = 0; param_idx < NETWORK_PARAM_COUNT; ++param_idx) {
            assert(host_population[elite_idx].params[param_idx] ==
                   host_next_population[elite_idx].params[param_idx]);
        }
    }

    CUDA_CHECK(cudaFree(d_ranked_indices));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_next_population));
    CUDA_CHECK(cudaFree(d_population));

    printf("test_evolution passed\n");
    return 0;
}
