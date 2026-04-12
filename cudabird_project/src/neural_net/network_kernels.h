#ifndef CUDABIRD_NETWORK_KERNELS_H
#define CUDABIRD_NETWORK_KERNELS_H

#include "network.h"
#include "../game/game_state.h"
#include <curand_kernel.h>

void initialize_population(NetworkWeights* d_networks,
                           curandState* d_rng_states,
                           int population_size);

#endif // CUDABIRD_NETWORK_KERNELS_H
