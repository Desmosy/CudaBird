#include "game/game_state.h"
#include "game/game_kernels.h"
#include "neural_net/network_kernels.h"
#include "cuda_utils.h"

#include <assert.h>
#include <vector>
#include <stdio.h>

int main(int argc, char** argv) {
    GameState* d_games = nullptr;
    NetworkWeights* d_networks = nullptr;
    curandState* d_rng_states = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_games),
                          sizeof(GameState) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_networks),
                          sizeof(NetworkWeights) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rng_states),
                          sizeof(curandState) * POPULATION_SIZE));

    initialize_curand_states(d_rng_states, POPULATION_SIZE, 1234ULL);
    initialize_population(d_networks, d_rng_states, POPULATION_SIZE);
    reset_games(d_games, d_rng_states, POPULATION_SIZE);
    simulate_generation(d_games, d_networks, d_rng_states, POPULATION_SIZE, 32);

    std::vector<GameState> host_games(POPULATION_SIZE);
    CUDA_CHECK(cudaMemcpy(host_games.data(),
                          d_games,
                          sizeof(GameState) * POPULATION_SIZE,
                          cudaMemcpyDeviceToHost));

    int valid_states = 0;
    for (int idx = 0; idx < POPULATION_SIZE; ++idx) {
        assert(host_games[idx].ticks_alive >= 0);
        assert(host_games[idx].bird_y > -200.0f);
        if (host_games[idx].ticks_alive > 0) {
            valid_states += 1;
        }
    }

    assert(valid_states > 0);

    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_networks));
    CUDA_CHECK(cudaFree(d_games));

    printf("test_game passed\n");
    return 0;
}
