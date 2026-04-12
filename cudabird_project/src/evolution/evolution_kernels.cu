#include "evolution.h"
#include "evolution_kernels.h"
#include "../cuda_utils.h"

#include <algorithm>
#include <curand_kernel.h>
#include <vector>

namespace {

struct RankedGame {
    int index;
    float fitness;
    int score;
};

__global__ void evolve_population_kernel(NetworkWeights* next_population,
                                         const NetworkWeights* current_population,
                                         const int* ranked_indices,
                                         curandState* rng_states,
                                         int population_size,
                                         int elite_count,
                                         int parent_pool_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) {
        return;
    }

    curandState local_rng = rng_states[idx];

    if (idx < elite_count) {
        next_population[idx] = current_population[ranked_indices[idx]];
        rng_states[idx] = local_rng;
        return;
    }

    const int parent_rank = static_cast<int>(curand_uniform(&local_rng) *
                                             static_cast<float>(parent_pool_size));
    const int clamped_parent_rank =
        parent_rank < parent_pool_size ? parent_rank : (parent_pool_size - 1);
    const int parent_index = ranked_indices[clamped_parent_rank];

    NetworkWeights child = current_population[parent_index];
    for (int param_idx = 0; param_idx < NETWORK_PARAM_COUNT; ++param_idx) {
        if (curand_uniform(&local_rng) < MUTATION_RATE) {
            child.params[param_idx] += curand_normal(&local_rng) * MUTATION_SCALE;
        }
    }

    next_population[idx] = child;
    rng_states[idx] = local_rng;
}

} // namespace

GenerationSummary collect_generation_summary(const GameState* d_games,
                                             int population_size,
                                             int generation,
                                             std::vector<int>* ranked_indices) {
    std::vector<GameState> host_games(population_size);
    CUDA_CHECK(cudaMemcpy(host_games.data(),
                          d_games,
                          sizeof(GameState) * population_size,
                          cudaMemcpyDeviceToHost));

    std::vector<RankedGame> ranked_games(population_size);
    float fitness_sum = 0.0f;
    float score_sum = 0.0f;

    for (int idx = 0; idx < population_size; ++idx) {
        ranked_games[idx].index = idx;
        ranked_games[idx].fitness = host_games[idx].fitness;
        ranked_games[idx].score = host_games[idx].score;
        fitness_sum += host_games[idx].fitness;
        score_sum += static_cast<float>(host_games[idx].score);
    }

    std::sort(ranked_games.begin(),
              ranked_games.end(),
              [](const RankedGame& left, const RankedGame& right) {
                  if (left.fitness == right.fitness) {
                      return left.score > right.score;
                  }
                  return left.fitness > right.fitness;
              });

    if (ranked_indices != nullptr) {
        ranked_indices->resize(population_size);
        for (int idx = 0; idx < population_size; ++idx) {
            (*ranked_indices)[idx] = ranked_games[idx].index;
        }
    }

    GenerationSummary summary = {};
    summary.generation = generation;
    summary.best_index = ranked_games.front().index;
    summary.best_score = ranked_games.front().score;
    summary.best_fitness = ranked_games.front().fitness;
    summary.average_fitness = fitness_sum / static_cast<float>(population_size);
    summary.average_score = score_sum / static_cast<float>(population_size);
    return summary;
}

void evolve_population(NetworkWeights* d_next_population,
                       const NetworkWeights* d_current_population,
                       const int* d_ranked_indices,
                       curandState* d_rng_states,
                       int population_size) {
    const int elite_count = ELITE_COUNT < population_size ? ELITE_COUNT : population_size;
    const int parent_pool_size =
        PARENT_POOL_SIZE < population_size ? PARENT_POOL_SIZE : population_size;
    const int grid_size = (population_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    evolve_population_kernel<<<grid_size, BLOCK_SIZE>>>(d_next_population,
                                                        d_current_population,
                                                        d_ranked_indices,
                                                        d_rng_states,
                                                        population_size,
                                                        elite_count,
                                                        parent_pool_size);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}
