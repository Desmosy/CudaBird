#ifndef CUDABIRD_NETWORK_H
#define CUDABIRD_NETWORK_H

#include "../config.h"

#include <cuda_runtime.h>
#include <math.h>

struct NetworkWeights {
    float params[NETWORK_PARAM_COUNT];
};

__host__ __device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ inline float run_network(const NetworkWeights& network,
                                             const float inputs[INPUT_SIZE]) {
    float hidden[HIDDEN_SIZE];

    for (int hidden_idx = 0; hidden_idx < HIDDEN_SIZE; ++hidden_idx) {
        float activation = network.params[INPUT_TO_HIDDEN_PARAM_COUNT + hidden_idx];

        for (int input_idx = 0; input_idx < INPUT_SIZE; ++input_idx) {
            const int weight_idx = hidden_idx * INPUT_SIZE + input_idx;
            activation += network.params[weight_idx] * inputs[input_idx];
        }

        hidden[hidden_idx] = tanhf(activation);
    }

    float output = network.params[INPUT_TO_HIDDEN_PARAM_COUNT + HIDDEN_BIAS_PARAM_COUNT +
                                  HIDDEN_TO_OUTPUT_PARAM_COUNT];

    for (int hidden_idx = 0; hidden_idx < HIDDEN_SIZE; ++hidden_idx) {
        const int weight_idx = INPUT_TO_HIDDEN_PARAM_COUNT + HIDDEN_BIAS_PARAM_COUNT +
                               hidden_idx;
        output += network.params[weight_idx] * hidden[hidden_idx];
    }

    return sigmoidf(output);
}

#endif // CUDABIRD_NETWORK_H
