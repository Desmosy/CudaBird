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
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

namespace {

struct RunOptions {
    int population_size;
    int generations;
    int max_ticks;
    int report_every;
    unsigned long long seed;
};

void print_usage(const char* program_name) {
    printf("Usage: %s [generation_count] [--population N] [--generations N] [--ticks N] "
           "[--report-every N] [--seed N]\n",
           program_name);
}

int parse_positive_int(const char* value, const char* flag_name) {
    const int parsed = atoi(value);
    if (parsed <= 0) {
        fprintf(stderr, "Expected a positive integer for %s, got '%s'\n", flag_name, value);
        exit(EXIT_FAILURE);
    }

    return parsed;
}

RunOptions parse_run_options(int argc, char** argv) {
    RunOptions options = {
        POPULATION_SIZE,
        MAX_GENERATIONS,
        MAX_TICKS_PER_GENERATION,
        DEFAULT_GENERATION_REPORT_INTERVAL,
        static_cast<unsigned long long>(time(nullptr))
    };

    for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
        const char* arg = argv[arg_idx];

        if (strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        }

        if (strcmp(arg, "--generations") == 0) {
            if (arg_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --generations\n");
                exit(EXIT_FAILURE);
            }
            options.generations = parse_positive_int(argv[++arg_idx], "--generations");
            continue;
        }

        if (strcmp(arg, "--population") == 0) {
            if (arg_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --population\n");
                exit(EXIT_FAILURE);
            }
            options.population_size = parse_positive_int(argv[++arg_idx], "--population");
            continue;
        }

        if (strcmp(arg, "--ticks") == 0) {
            if (arg_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --ticks\n");
                exit(EXIT_FAILURE);
            }
            options.max_ticks = parse_positive_int(argv[++arg_idx], "--ticks");
            continue;
        }

        if (strcmp(arg, "--report-every") == 0) {
            if (arg_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --report-every\n");
                exit(EXIT_FAILURE);
            }
            options.report_every = parse_positive_int(argv[++arg_idx], "--report-every");
            continue;
        }

        if (strcmp(arg, "--seed") == 0) {
            if (arg_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --seed\n");
                exit(EXIT_FAILURE);
            }
            options.seed = static_cast<unsigned long long>(strtoull(argv[++arg_idx], nullptr, 10));
            continue;
        }

        if (arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }

        options.generations = parse_positive_int(arg, "generation_count");
    }

    return options;
}

} // namespace

int main(int argc, char** argv) {
    const RunOptions options = parse_run_options(argc, argv);
    const ReplaySettings replay_settings = {
        options.max_ticks,
        DEFAULT_REPLAY_FPS,
        DEFAULT_REPLAY_SCALE,
        options.seed ^
            (static_cast<unsigned long long>(options.generations + options.population_size) *
             1315423911ULL)
    };

    printf("CudaBird training run\n");
    printf("Population: %d | generations: %d | max ticks/gen: %d | report every: %d | seed: %llu\n",
           options.population_size,
           options.generations,
           options.max_ticks,
           options.report_every,
           options.seed);

    GameState* d_games = nullptr;
    NetworkWeights* d_population = nullptr;
    NetworkWeights* d_next_population = nullptr;
    curandState* d_rng_states = nullptr;
    int* d_ranked_indices = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_games),
                          sizeof(GameState) * options.population_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_population),
                          sizeof(NetworkWeights) * options.population_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next_population),
                          sizeof(NetworkWeights) * options.population_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rng_states),
                          sizeof(curandState) * options.population_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ranked_indices),
                          sizeof(int) * options.population_size));

    initialize_curand_states(d_rng_states, options.population_size, options.seed);
    initialize_population(d_population, d_rng_states, options.population_size);

    Renderer renderer("training_log.csv");
    GenerationSummary best_run = {};
    NetworkWeights best_network = {};
    best_run.best_fitness = -1.0f;

    for (int generation = 0; generation < options.generations; ++generation) {
        reset_games(d_games, d_rng_states, options.population_size);
        simulate_generation(d_games,
                            d_population,
                            d_rng_states,
                            options.population_size,
                            options.max_ticks);

        std::vector<int> ranked_indices;
        const GenerationSummary summary =
            collect_generation_summary(d_games,
                                       options.population_size,
                                       generation,
                                       &ranked_indices);

        if (summary.best_fitness > best_run.best_fitness) {
            best_run = summary;
            CUDA_CHECK(cudaMemcpy(&best_network,
                                  d_population + summary.best_index,
                                  sizeof(NetworkWeights),
                                  cudaMemcpyDeviceToHost));
            renderer.save_best_network(best_network, summary);
        }

        if (generation % options.report_every == 0) {
            renderer.render_generation(summary);
        }

        if (generation + 1 < options.generations) {
            CUDA_CHECK(cudaMemcpy(d_ranked_indices,
                                  ranked_indices.data(),
                                  sizeof(int) * options.population_size,
                                  cudaMemcpyHostToDevice));
            evolve_population(d_next_population,
                              d_population,
                              d_ranked_indices,
                              d_rng_states,
                              options.population_size);
            std::swap(d_population, d_next_population);
        }
    }

    printf("Best run: generation %d, fitness %.2f, score %d\n",
           best_run.generation,
           best_run.best_fitness,
           best_run.best_score);
    renderer.render_best_replay(best_network, best_run, replay_settings);

    CUDA_CHECK(cudaFree(d_ranked_indices));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_next_population));
    CUDA_CHECK(cudaFree(d_population));
    CUDA_CHECK(cudaFree(d_games));

    return 0;
}
