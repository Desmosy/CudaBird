#ifndef CUDABIRD_EVOLUTION_KERNELS_H
#define CUDABIRD_EVOLUTION_KERNELS_H

#include "evolution.h"

#include <curand_kernel.h>
#include <vector>

GenerationSummary collect_generation_summary(const GameState* d_games,
                                             int population_size,
                                             int generation,
                                             std::vector<int>* ranked_indices);

void evolve_population(NetworkWeights* d_next_population,
                       const NetworkWeights* d_current_population,
                       const int* d_ranked_indices,
                       curandState* d_rng_states,
                       int population_size);

#endif // CUDABIRD_EVOLUTION_KERNELS_H
