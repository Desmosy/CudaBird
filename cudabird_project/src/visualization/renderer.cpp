#include "renderer.h"

#include <cstdio>
#include <stdio.h>
#include <fstream>

Renderer::Renderer(const std::string& log_path) : log_file_(log_path.c_str()) {
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
