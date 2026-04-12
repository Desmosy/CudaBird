#include "game_kernels.h"
#include "../cuda_utils.h"

#include <curand_kernel.h>

namespace {

__device__ float random_gap_center(curandState* rng_state) {
    const float min_gap_center = PIPE_MARGIN + (PIPE_GAP * 0.5f);
    const float max_gap_center = GAME_HEIGHT - PIPE_MARGIN - (PIPE_GAP * 0.5f);
    return min_gap_center + curand_uniform(rng_state) * (max_gap_center - min_gap_center);
}

__device__ int find_next_pipe_index(const GameState& game_state) {
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

__device__ void advance_pipes(GameState* game_state, curandState* rng_state) {
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
        pipe->gap_center_y = random_gap_center(rng_state);
        pipe->passed = 0;
        rightmost_x = pipe->x;
    }
}

__device__ int collides_with_pipe(const GameState& game_state) {
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

__global__ void reset_games_kernel(GameState* games,
                                   curandState* rng_states,
                                   int population_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) {
        return;
    }

    curandState local_rng = rng_states[idx];
    GameState game_state = {};
    game_state.bird_y = BIRD_START_Y;
    game_state.bird_velocity = 0.0f;
    game_state.fitness = 0.0f;
    game_state.alive = 1;
    game_state.score = 0;
    game_state.ticks_alive = 0;
    game_state.next_pipe_index = 0;

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        game_state.pipes[pipe_idx].x =
            GAME_WIDTH + INITIAL_PIPE_OFFSET + static_cast<float>(pipe_idx) * PIPE_SPACING;
        game_state.pipes[pipe_idx].gap_center_y = random_gap_center(&local_rng);
        game_state.pipes[pipe_idx].passed = 0;
    }

    games[idx] = game_state;
    rng_states[idx] = local_rng;
}

__global__ void simulate_generation_kernel(GameState* games,
                                           const NetworkWeights* networks,
                                           curandState* rng_states,
                                           int population_size,
                                           int max_ticks) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) {
        return;
    }

    GameState game_state = games[idx];
    curandState local_rng = rng_states[idx];

    for (int tick = 0; tick < max_ticks && game_state.alive; ++tick) {
        game_state.next_pipe_index = find_next_pipe_index(game_state);
        const PipeState& next_pipe = game_state.pipes[game_state.next_pipe_index];

        float inputs[INPUT_SIZE];
        inputs[0] = (game_state.bird_y / GAME_HEIGHT) * 2.0f - 1.0f;
        inputs[1] = game_state.bird_velocity / 10.0f;
        inputs[2] = ((next_pipe.x - BIRD_X) / GAME_WIDTH) * 2.0f - 1.0f;
        inputs[3] = (next_pipe.gap_center_y / GAME_HEIGHT) * 2.0f - 1.0f;

        const float flap_probability = run_network(networks[idx], inputs);
        if (flap_probability > FLAP_THRESHOLD) {
            game_state.bird_velocity = FLAP_STRENGTH;
        }

        game_state.bird_velocity += GRAVITY;
        game_state.bird_y += game_state.bird_velocity;
        advance_pipes(&game_state, &local_rng);

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

    games[idx] = game_state;
    rng_states[idx] = local_rng;
}

__global__ void capture_replay_frames_kernel(ReplayFrame* frames,
                                             int* frame_count,
                                             GameState initial_game_state,
                                             curandState initial_rng_state,
                                             NetworkWeights network,
                                             int max_ticks) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    GameState game_state = initial_game_state;
    curandState local_rng = initial_rng_state;
    float flap_probability = 0.0f;
    int recorded_frames = 0;

    frames[recorded_frames].game_state = game_state;
    frames[recorded_frames].flap_probability = flap_probability;
    recorded_frames += 1;

    for (int tick = 0; tick < max_ticks && game_state.alive; ++tick) {
        game_state.next_pipe_index = find_next_pipe_index(game_state);
        const PipeState& next_pipe = game_state.pipes[game_state.next_pipe_index];

        float inputs[INPUT_SIZE];
        inputs[0] = (game_state.bird_y / GAME_HEIGHT) * 2.0f - 1.0f;
        inputs[1] = game_state.bird_velocity / 10.0f;
        inputs[2] = ((next_pipe.x - BIRD_X) / GAME_WIDTH) * 2.0f - 1.0f;
        inputs[3] = (next_pipe.gap_center_y / GAME_HEIGHT) * 2.0f - 1.0f;

        flap_probability = run_network(network, inputs);
        if (flap_probability > FLAP_THRESHOLD) {
            game_state.bird_velocity = FLAP_STRENGTH;
        }

        game_state.bird_velocity += GRAVITY;
        game_state.bird_y += game_state.bird_velocity;
        advance_pipes(&game_state, &local_rng);

        game_state.ticks_alive += 1;
        game_state.fitness = static_cast<float>(game_state.ticks_alive) +
                             static_cast<float>(game_state.score) * 250.0f;

        const int out_of_bounds =
            (game_state.bird_y - BIRD_RADIUS) < 0.0f ||
            (game_state.bird_y + BIRD_RADIUS) > GAME_HEIGHT;
        if (out_of_bounds || collides_with_pipe(game_state)) {
            game_state.alive = 0;
        }

        frames[recorded_frames].game_state = game_state;
        frames[recorded_frames].flap_probability = flap_probability;
        recorded_frames += 1;
    }

    *frame_count = recorded_frames;
}

} // namespace

void reset_games(GameState* d_games, curandState* d_rng_states, int population_size) {
    const int grid_size = (population_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reset_games_kernel<<<grid_size, BLOCK_SIZE>>>(d_games, d_rng_states, population_size);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void simulate_generation(GameState* d_games,
                         const NetworkWeights* d_networks,
                         curandState* d_rng_states,
                         int population_size,
                         int max_ticks) {
    const int grid_size = (population_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    simulate_generation_kernel<<<grid_size, BLOCK_SIZE>>>(d_games,
                                                          d_networks,
                                                          d_rng_states,
                                                          population_size,
                                                          max_ticks);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void capture_replay_frames(const GameState& initial_game_state,
                           const curandState& initial_rng_state,
                           const NetworkWeights& network,
                           int max_ticks,
                           ReplayFrame* h_frames,
                           int* h_frame_count) {
    ReplayFrame* d_frames = nullptr;
    int* d_frame_count = nullptr;
    const size_t frame_capacity = static_cast<size_t>(max_ticks + 1);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_frames),
                          sizeof(ReplayFrame) * frame_capacity));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_frame_count), sizeof(int)));

    capture_replay_frames_kernel<<<1, 1>>>(d_frames,
                                           d_frame_count,
                                           initial_game_state,
                                           initial_rng_state,
                                           network,
                                           max_ticks);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_frame_count,
                          d_frame_count,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_frames,
                          d_frames,
                          sizeof(ReplayFrame) * static_cast<size_t>(*h_frame_count),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_frame_count));
    CUDA_CHECK(cudaFree(d_frames));
}
