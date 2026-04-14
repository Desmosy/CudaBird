#include "config.h"
#include "game/game_state.h"
#include "neural_net/network.h"
#include "evolution/evolution.h"
#include "visualization/renderer.h"

#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
            options.generations = parse_positive_int(argv[++arg_idx], "--generations");
            continue;
        }

        if (strcmp(arg, "--population") == 0) {
            options.population_size = parse_positive_int(argv[++arg_idx], "--population");
            continue;
        }

        if (strcmp(arg, "--ticks") == 0) {
            options.max_ticks = parse_positive_int(argv[++arg_idx], "--ticks");
            continue;
        }

        if (strcmp(arg, "--report-every") == 0) {
            options.report_every = parse_positive_int(argv[++arg_idx], "--report-every");
            continue;
        }

        if (strcmp(arg, "--seed") == 0) {
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

float random_gap_center(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float min_gap_center = PIPE_MARGIN + (PIPE_GAP * 0.5f);
    const float max_gap_center = GAME_HEIGHT - PIPE_MARGIN - (PIPE_GAP * 0.5f);
    return min_gap_center + dist(rng) * (max_gap_center - min_gap_center);
}

int find_next_pipe_index(const GameState& game_state) {
    int best_index = 0;
    float best_x = 1.0e30f;

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        const float pipe_trailing_edge = game_state.pipes[pipe_idx].x + PIPE_WIDTH;
        if (pipe_trailing_edge >= BIRD_X && game_state.pipes[pipe_idx].x < best_x) {
            best_index = pipe_idx;
            best_x = game_state.pipes[pipe_idx].x;
        }
    }

    return best_index;
}

void advance_pipes(GameState* game_state, std::mt19937& rng) {
    float rightmost_x = game_state->pipes[0].x;

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        PipeState* pipe = &game_state->pipes[pipe_idx];
        pipe->x -= PIPE_SPEED;

        if (pipe->x > rightmost_x) {
            rightmost_x = pipe->x;
        }

        if (!pipe->passed && pipe->x + PIPE_WIDTH < BIRD_X) {
            pipe->passed = 1;
            game_state->score += 1;
        }
    }

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        PipeState* pipe = &game_state->pipes[pipe_idx];
        if (pipe->x + PIPE_WIDTH >= 0.0f) {
            continue;
        }

        pipe->x = rightmost_x + PIPE_SPACING;
        pipe->gap_center_y = random_gap_center(rng);
        pipe->passed = 0;
        rightmost_x = pipe->x;
    }
}

int collides_with_pipe(const GameState& game_state) {
    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        const PipeState& pipe = game_state.pipes[pipe_idx];
        const float pipe_left = pipe.x;
        const float pipe_right = pipe.x + PIPE_WIDTH;
        const int overlaps_horizontally =
            (BIRD_X + BIRD_RADIUS) >= pipe_left && (BIRD_X - BIRD_RADIUS) <= pipe_right;

        if (!overlaps_horizontally) {
            continue;
        }

        const float gap_top = pipe.gap_center_y - (PIPE_GAP * 0.5f);
        const float gap_bottom = pipe.gap_center_y + (PIPE_GAP * 0.5f);
        if ((game_state.bird_y - BIRD_RADIUS) < gap_top ||
            (game_state.bird_y + BIRD_RADIUS) > gap_bottom) {
            return 1;
        }
    }

    return 0;
}

void initialize_population_cpu(std::vector<NetworkWeights>& networks, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& net : networks) {
        for (int i = 0; i < NETWORK_PARAM_COUNT; i++) {
            net.params[i] = dist(rng) * 0.5f;
        }
    }
}

void reset_games_cpu(std::vector<GameState>& games, std::mt19937& rng) {
    for (auto& game : games) {
        game.bird_y = BIRD_START_Y;
        game.bird_velocity = 0.0f;
        game.fitness = 0.0f;
        game.alive = 1;
        game.score = 0;
        game.ticks_alive = 0;
        game.next_pipe_index = 0;

        for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
            game.pipes[pipe_idx].x =
                GAME_WIDTH + INITIAL_PIPE_OFFSET + static_cast<float>(pipe_idx) * PIPE_SPACING;
            game.pipes[pipe_idx].gap_center_y = random_gap_center(rng);
            game.pipes[pipe_idx].passed = 0;
        }
    }
}

void simulate_generation_cpu(std::vector<GameState>& games, 
                             const std::vector<NetworkWeights>& networks,
                             std::mt19937& rng,
                             int max_ticks) {
    for (size_t idx = 0; idx < games.size(); ++idx) {
        GameState& game_state = games[idx];
        const NetworkWeights& network = networks[idx];

        for (int tick = 0; tick < max_ticks && game_state.alive; ++tick) {
            game_state.next_pipe_index = find_next_pipe_index(game_state);
            const PipeState& next_pipe = game_state.pipes[game_state.next_pipe_index];

            float inputs[INPUT_SIZE];
            inputs[0] = (game_state.bird_y / GAME_HEIGHT) * 2.0f - 1.0f;
            inputs[1] = game_state.bird_velocity / 10.0f;
            inputs[2] = ((next_pipe.x - BIRD_X) / GAME_WIDTH) * 2.0f - 1.0f;
            inputs[3] = (next_pipe.gap_center_y / GAME_HEIGHT) * 2.0f - 1.0f;

            const float flap_probability = run_network(network, inputs);
            if (flap_probability > FLAP_THRESHOLD) {
                game_state.bird_velocity = FLAP_STRENGTH;
            }

            game_state.bird_velocity += GRAVITY;
            game_state.bird_y += game_state.bird_velocity;
            advance_pipes(&game_state, rng);

            game_state.ticks_alive += 1;
            game_state.fitness = static_cast<float>(game_state.ticks_alive) +
                                 static_cast<float>(game_state.score) * 250.0f;

            const int out_of_bounds =
                (game_state.bird_y - BIRD_RADIUS) < 0.0f ||
                (game_state.bird_y + BIRD_RADIUS) > GAME_HEIGHT;
            if (out_of_bounds || collides_with_pipe(game_state)) {
                game_state.alive = 0;
            }
        }
    }
}

struct RankedGame {
    int index;
    float fitness;
    int score;
};

GenerationSummary collect_summary_cpu(const std::vector<GameState>& games,
                                      int generation,
                                      std::vector<int>& ranked_indices) {
    std::vector<RankedGame> ranked_games(games.size());
    float fitness_sum = 0.0f;
    float score_sum = 0.0f;

    for (size_t idx = 0; idx < games.size(); ++idx) {
        ranked_games[idx].index = idx;
        ranked_games[idx].fitness = games[idx].fitness;
        ranked_games[idx].score = games[idx].score;
        fitness_sum += games[idx].fitness;
        score_sum += static_cast<float>(games[idx].score);
    }

    std::sort(ranked_games.begin(), ranked_games.end(),
              [](const RankedGame& left, const RankedGame& right) {
                  if (left.fitness == right.fitness) {
                      return left.score > right.score;
                  }
                  return left.fitness > right.fitness;
              });

    ranked_indices.resize(games.size());
    for (size_t idx = 0; idx < games.size(); ++idx) {
        ranked_indices[idx] = ranked_games[idx].index;
    }

    GenerationSummary summary = {};
    summary.generation = generation;
    summary.best_index = ranked_games.front().index;
    summary.best_score = ranked_games.front().score;
    summary.best_fitness = ranked_games.front().fitness;
    summary.average_fitness = fitness_sum / static_cast<float>(games.size());
    summary.average_score = score_sum / static_cast<float>(games.size());
    return summary;
}

void evolve_population_cpu(std::vector<NetworkWeights>& next_pop,
                           const std::vector<NetworkWeights>& current_pop,
                           const std::vector<int>& ranked_indices,
                           std::mt19937& rng) {
    const int pop_size = current_pop.size();
    const int elite_count = ELITE_COUNT < pop_size ? ELITE_COUNT : pop_size;
    const int parent_pool_size = PARENT_POOL_SIZE < pop_size ? PARENT_POOL_SIZE : pop_size;

    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    for (int idx = 0; idx < pop_size; ++idx) {
        if (idx < elite_count) {
            next_pop[idx] = current_pop[ranked_indices[idx]];
            continue;
        }

        const int parent_rank = static_cast<int>(unif(rng) * static_cast<float>(parent_pool_size));
        const int clamped_parent_rank = parent_rank < parent_pool_size ? parent_rank : (parent_pool_size - 1);
        const int parent_index = ranked_indices[clamped_parent_rank];

        NetworkWeights child = current_pop[parent_index];
        for (int param_idx = 0; param_idx < NETWORK_PARAM_COUNT; ++param_idx) {
            if (unif(rng) < MUTATION_RATE) {
                child.params[param_idx] += norm(rng) * MUTATION_SCALE;
            }
        }

        next_pop[idx] = child;
    }
}

} // namespace

int main(int argc, char** argv) {
    const RunOptions options = parse_run_options(argc, argv);

    printf("CudaBird CPU Baseline Training Run\n");
    printf("Population: %d | generations: %d | max ticks/gen: %d | report every: %d | seed: %llu\n",
           options.population_size,
           options.generations,
           options.max_ticks,
           options.report_every,
           options.seed);

    std::mt19937 rng(options.seed);
    std::vector<GameState> games(options.population_size);
    std::vector<NetworkWeights> population(options.population_size);
    std::vector<NetworkWeights> next_population(options.population_size);

    initialize_population_cpu(population, rng);

    Renderer renderer("training_log_cpu.csv");
    GenerationSummary best_run = {};
    NetworkWeights best_network = {};
    best_run.best_fitness = -1.0f;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int generation = 0; generation < options.generations; ++generation) {
        reset_games_cpu(games, rng);
        simulate_generation_cpu(games, population, rng, options.max_ticks);

        std::vector<int> ranked_indices;
        const GenerationSummary summary = collect_summary_cpu(games, generation, ranked_indices);

        if (summary.best_fitness > best_run.best_fitness) {
            best_run = summary;
            best_network = population[summary.best_index];
            renderer.save_best_network(best_network, summary);
        }

        if (generation % options.report_every == 0) {
            renderer.render_generation(summary);
        }

        if (generation + 1 < options.generations) {
            evolve_population_cpu(next_population, population, ranked_indices, rng);
            std::swap(population, next_population);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double duration = (end_time.tv_sec - start_time.tv_sec) + 
                      (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("\n=== CPU PERFORMANCE REPORT ===\n");
    printf("Total Execution Time: %.2f seconds\n", duration);
    printf("Time per Generation: %.4f seconds\n", duration / options.generations);
    printf("Speed: %.2f games/sec\n", (options.population_size * options.generations) / duration);
    printf("==============================\n\n");

    printf("Best CPU run: generation %d, fitness %.2f, score %d\n",
           best_run.generation,
           best_run.best_fitness,
           best_run.best_score);

    return 0;
}
