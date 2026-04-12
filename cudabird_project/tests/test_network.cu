#include "neural_net/network.h"
#include "neural_net/network_kernels.h"
#include "cuda_utils.h"

#include <assert.h>
#include <vector>
#include <stdio.h>

int main(int argc, char** argv) {
    NetworkWeights* d_networks = nullptr;
    curandState* d_rng_states = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_networks),
                          sizeof(NetworkWeights) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rng_states),
                          sizeof(curandState) * POPULATION_SIZE));

    initialize_curand_states(d_rng_states, POPULATION_SIZE, 4321ULL);
    initialize_population(d_networks, d_rng_states, POPULATION_SIZE);

    std::vector<NetworkWeights> host_networks(POPULATION_SIZE);
    CUDA_CHECK(cudaMemcpy(host_networks.data(),
                          d_networks,
                          sizeof(NetworkWeights) * POPULATION_SIZE,
                          cudaMemcpyDeviceToHost));

    int non_zero_params = 0;
    for (int network_idx = 0; network_idx < POPULATION_SIZE; ++network_idx) {
        float inputs[INPUT_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};
        const float output = run_network(host_networks[network_idx], inputs);
        assert(output >= 0.0f);
        assert(output <= 1.0f);

        for (int param_idx = 0; param_idx < NETWORK_PARAM_COUNT; ++param_idx) {
            if (host_networks[network_idx].params[param_idx] != 0.0f) {
                non_zero_params += 1;
                break;
            }
        }
    }

    assert(non_zero_params > 0);

    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_networks));

    printf("test_network passed\n");
    return 0;
}
