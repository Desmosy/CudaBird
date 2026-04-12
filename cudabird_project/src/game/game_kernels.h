#ifndef CUDABIRD_GAME_KERNELS_H
#define CUDABIRD_GAME_KERNELS_H

#include "game_state.h"
#include "../neural_net/network.h"

#include <curand_kernel.h>

void reset_games(GameState* d_games, curandState* d_rng_states, int population_size);
void simulate_generation(GameState* d_games,
                         const NetworkWeights* d_networks,
                         curandState* d_rng_states,
                         int population_size,
                         int max_ticks);

#endif // CUDABIRD_GAME_KERNELS_H
