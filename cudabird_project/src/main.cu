#include "config.h"
#include "cuda_utils.h"
#include "evolution/evolution.h"
#include "evolution/evolution_kernels.h"
#include "game/game_state.h"
#include "game/game_kernels.h"
#include "neural_net/network.h"
#include "neural_net/network_kernels.h"
#include "visualization/renderer.h"

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

int main(int argc, char** argv) {
    int requested_generations = MAX_GENERATIONS;
    if (argc > 1) {
        requested_generations = std::max(1, atoi(argv[1]));
    }

    printf("CudaBird training run\n");
    printf("Population: %d | generations: %d | max ticks/gen: %d\n",
           POPULATION_SIZE,
           requested_generations,
           MAX_TICKS_PER_GENERATION);

    GameState* d_games = nullptr;
    NetworkWeights* d_population = nullptr;
    NetworkWeights* d_next_population = nullptr;
    curandState* d_rng_states = nullptr;
    int* d_ranked_indices = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_games),
                          sizeof(GameState) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_population),
                          sizeof(NetworkWeights) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next_population),
                          sizeof(NetworkWeights) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rng_states),
                          sizeof(curandState) * POPULATION_SIZE));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ranked_indices),
                          sizeof(int) * POPULATION_SIZE));

    initialize_curand_states(d_rng_states,
                             POPULATION_SIZE,
                             static_cast<unsigned long long>(time(nullptr)));
    initialize_population(d_population, d_rng_states, POPULATION_SIZE);

    Renderer renderer("training_log.csv");
    GenerationSummary best_run = {};
    best_run.best_fitness = -1.0f;

    for (int generation = 0; generation < requested_generations; ++generation) {
        reset_games(d_games, d_rng_states, POPULATION_SIZE);
        simulate_generation(d_games,
                            d_population,
                            d_rng_states,
                            POPULATION_SIZE,
                            MAX_TICKS_PER_GENERATION);

        std::vector<int> ranked_indices;
        const GenerationSummary summary =
            collect_generation_summary(d_games, POPULATION_SIZE, generation, &ranked_indices);

        if (summary.best_fitness > best_run.best_fitness) {
            best_run = summary;
        }

        if (generation % DEFAULT_GENERATION_REPORT_INTERVAL == 0) {
            renderer.render_generation(summary);
        }

        if (generation + 1 < requested_generations) {
            CUDA_CHECK(cudaMemcpy(d_ranked_indices,
                                  ranked_indices.data(),
                                  sizeof(int) * POPULATION_SIZE,
                                  cudaMemcpyHostToDevice));
            evolve_population(d_next_population,
                              d_population,
                              d_ranked_indices,
                              d_rng_states,
                              POPULATION_SIZE);
            std::swap(d_population, d_next_population);
        }
    }

    printf("Best run: generation %d, fitness %.2f, score %d\n",
           best_run.generation,
           best_run.best_fitness,
           best_run.best_score);

    CUDA_CHECK(cudaFree(d_ranked_indices));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_next_population));
    CUDA_CHECK(cudaFree(d_population));
    CUDA_CHECK(cudaFree(d_games));

    return 0;
}
