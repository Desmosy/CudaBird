#ifndef CUDABIRD_CONFIG_H
#define CUDABIRD_CONFIG_H
#define POPULATION_SIZE     4096    // Number of birds (games) per generation
#define MAX_GENERATIONS     500     // Max evolutionary generations
#define MAX_PIPES           10      // Max pipes on screen at once
#define GAME_WIDTH          288     // Virtual game width
#define GAME_HEIGHT         512     // Virtual game height
#define PIPE_GAP            100     // Vertical gap between top/bottom pipe
#define PIPE_SPEED          2.0f    // Horizontal speed of pipes
#define PIPE_SPACING        200     // Horizontal distance between pipes
#define GRAVITY             0.5f    // Downward acceleration
#define FLAP_STRENGTH      -8.0f   // Upward velocity on flap
#define BIRD_RADIUS         12.0f   // Collision radius of bird
#define INPUT_SIZE          4       // Inputs: bird_y, bird_velocity, pipe_dist_x, pipe_gap_y
#define HIDDEN_SIZE         8       // Hidden layer neurons
#define OUTPUT_SIZE         1       // Output: flap probability
#define FLAP_THRESHOLD      0.5f   // If output > threshold, bird flaps
#define MUTATION_RATE       0.1f    // Probability of mutating a weight
#define MUTATION_SCALE      0.3f    // Stddev of Gaussian mutation noise
#define ELITE_COUNT         64      // Top birds copied unchanged
#define TOURNAMENT_SIZE     4       // Tournament selection size
#define BLOCK_SIZE          256     // Threads per block for most kernels
#endif // CUDABIRD_CONFIG_H
