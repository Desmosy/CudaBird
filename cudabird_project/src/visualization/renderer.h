#ifndef CUDABIRD_RENDERER_H
#define CUDABIRD_RENDERER_H

#include "../evolution/evolution.h"
#include "../game/game_state.h"
#include "../neural_net/network.h"

#include <fstream>
#include <string>

class Renderer {
  public:
    explicit Renderer(const std::string& log_path);
    ~Renderer();

    void render_generation(const GenerationSummary& summary);

  private:
    std::ofstream log_file_;
};

#endif // CUDABIRD_RENDERER_H
