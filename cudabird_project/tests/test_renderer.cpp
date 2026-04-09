#include "visualization/renderer.h"

#include <assert.h>
#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

#ifdef CUDABIRD_HAVE_OPENCV
    Renderer renderer("test_training_log.csv");

    NetworkWeights network = {};
    GenerationSummary summary = {};
    summary.generation = 7;
    summary.best_score = 1;
    summary.best_fitness = 128.0f;

    ReplaySettings settings = {
        48,
        12,
        1,
        123ULL
    };

    renderer.render_best_replay(network, summary, settings);

    struct stat latest_stat = {};
    struct stat generation_stat = {};
    assert(stat("outputs/best_replay_latest.avi", &latest_stat) == 0);
    assert(stat("outputs/best_replay_gen_0007.avi", &generation_stat) == 0);
    assert(latest_stat.st_size > 0);
    assert(generation_stat.st_size > 0);
#endif

    printf("test_renderer passed\n");
    return 0;
}
