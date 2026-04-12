#ifndef CUDABIRD_GAME_STATE_H
#define CUDABIRD_GAME_STATE_H

#include "../config.h"

struct PipeState {
    float x;
    float gap_center_y;
    int passed;
};

struct GameState {
    float bird_y;
    float bird_velocity;
    float fitness;
    int alive;
    int score;
    int ticks_alive;
    int next_pipe_index;
    PipeState pipes[MAX_PIPES];
};

#endif // CUDABIRD_GAME_STATE_H
