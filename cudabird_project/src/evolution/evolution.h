#ifndef CUDABIRD_EVOLUTION_H
#define CUDABIRD_EVOLUTION_H

#include "../neural_net/network.h"
#include "../game/game_state.h"

struct GenerationSummary {
    int generation;
    int best_index;
    int best_score;
    float best_fitness;
    float average_fitness;
    float average_score;
};

#endif // CUDABIRD_EVOLUTION_H
