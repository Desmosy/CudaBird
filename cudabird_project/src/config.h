#ifndef CUDABIRD_CONFIG_H
#define CUDABIRD_CONFIG_H

#define POPULATION_SIZE             4096
#define MAX_GENERATIONS             500
#define MAX_TICKS_PER_GENERATION    2048
#define MAX_PIPES                   10

#define GAME_WIDTH                  288.0f
#define GAME_HEIGHT                 512.0f
#define BIRD_X                      72.0f
#define BIRD_START_Y                (GAME_HEIGHT * 0.5f)
#define BIRD_RADIUS                 12.0f

#define PIPE_WIDTH                  52.0f
#define PIPE_GAP                    110.0f
#define PIPE_SPEED                  2.0f
#define PIPE_SPACING                180.0f
#define INITIAL_PIPE_OFFSET         120.0f
#define PIPE_MARGIN                 56.0f

#define GRAVITY                     0.5f
#define FLAP_STRENGTH               -8.0f
#define FLAP_THRESHOLD              0.5f

#define INPUT_SIZE                  4
#define HIDDEN_SIZE                 8
#define OUTPUT_SIZE                 1

#define INPUT_TO_HIDDEN_PARAM_COUNT (INPUT_SIZE * HIDDEN_SIZE)
#define HIDDEN_BIAS_PARAM_COUNT     (HIDDEN_SIZE)
#define HIDDEN_TO_OUTPUT_PARAM_COUNT (HIDDEN_SIZE * OUTPUT_SIZE)
#define OUTPUT_BIAS_PARAM_COUNT     (OUTPUT_SIZE)
#define NETWORK_PARAM_COUNT         (INPUT_TO_HIDDEN_PARAM_COUNT + HIDDEN_BIAS_PARAM_COUNT + HIDDEN_TO_OUTPUT_PARAM_COUNT + OUTPUT_BIAS_PARAM_COUNT)

#define MUTATION_RATE               0.1f
#define MUTATION_SCALE              0.3f
#define ELITE_COUNT                 64
#define PARENT_POOL_SIZE            256

#define BLOCK_SIZE                  256
#define DEFAULT_GENERATION_REPORT_INTERVAL 1

#endif // CUDABIRD_CONFIG_H
