#ifndef CUDABIRD_RENDERER_H
#define CUDABIRD_RENDERER_H

#include "../evolution/evolution.h"
#include "../game/game_state.h"
#include "../neural_net/network.h"

#include <fstream>
#include <string>
#include <vector>

struct ReplaySettings {
    int max_ticks;
    int fps;
    int scale;
    int hold_frames;
};

class Renderer {
  public:
    explicit Renderer(const std::string& log_path);
    ~Renderer();

    void render_generation(const GenerationSummary& summary);
    void save_best_network(const NetworkWeights& network, const GenerationSummary& summary);
    void render_best_replay(const std::vector<ReplayFrame>& frames,
                            const GenerationSummary& summary,
                            const ReplaySettings& settings);

  private:
    std::string checkpoint_dir_;
    std::ofstream log_file_;
};

#endif // CUDABIRD_RENDERER_H
