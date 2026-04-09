#include "renderer.h"

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/stat.h>

namespace {

void ensure_directory_exists(const std::string& path) {
    if (path.empty()) {
        return;
    }

    if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
        return;
    }

    std::fprintf(stderr, "Warning: unable to create checkpoint directory '%s'\n", path.c_str());
}

} // namespace

Renderer::Renderer(const std::string& log_path)
    : checkpoint_dir_("outputs"),
      log_file_(log_path.c_str()) {
    ensure_directory_exists(checkpoint_dir_);

    if (log_file_.is_open()) {
        log_file_ << "generation,best_fitness,average_fitness,best_score,average_score\n";
    }
}

Renderer::~Renderer() = default;

void Renderer::render_generation(const GenerationSummary& summary) {
    std::printf("Generation %4d | best fitness %8.2f | avg fitness %8.2f | "
                "best score %4d | avg score %6.2f\n",
                summary.generation,
                summary.best_fitness,
                summary.average_fitness,
                summary.best_score,
                summary.average_score);

    if (log_file_.is_open()) {
        log_file_ << summary.generation << ','
                  << summary.best_fitness << ','
                  << summary.average_fitness << ','
                  << summary.best_score << ','
                  << summary.average_score << '\n';
        log_file_.flush();
    }
}

void Renderer::save_best_network(const NetworkWeights& network,
                                 const GenerationSummary& summary) {
    ensure_directory_exists(checkpoint_dir_);

    const std::string latest_path = checkpoint_dir_ + "/best_network_latest.txt";
    char generation_path[128];
    std::snprintf(generation_path,
                  sizeof(generation_path),
                  "%s/best_network_gen_%04d.txt",
                  checkpoint_dir_.c_str(),
                  summary.generation);

    const std::string header =
        "generation=" + std::to_string(summary.generation) +
        "\nbest_score=" + std::to_string(summary.best_score) +
        "\nbest_fitness=" + std::to_string(summary.best_fitness) +
        "\nparameters=" + std::to_string(NETWORK_PARAM_COUNT) + "\n";

    std::ofstream latest_file(latest_path.c_str());
    std::ofstream generation_file(generation_path);
    if (!latest_file.is_open() || !generation_file.is_open()) {
        std::fprintf(stderr, "Warning: unable to write network checkpoint\n");
        return;
    }

    latest_file << header;
    generation_file << header;

    for (int param_idx = 0; param_idx < NETWORK_PARAM_COUNT; ++param_idx) {
        latest_file << network.params[param_idx] << '\n';
        generation_file << network.params[param_idx] << '\n';
    }
}
