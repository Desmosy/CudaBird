#include "renderer.h"

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <sys/stat.h>

#ifdef CUDABIRD_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

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

float replay_random_gap_center(std::mt19937* rng) {
    const float min_gap_center = PIPE_MARGIN + (PIPE_GAP * 0.5f);
    const float max_gap_center = GAME_HEIGHT - PIPE_MARGIN - (PIPE_GAP * 0.5f);
    std::uniform_real_distribution<float> distribution(min_gap_center, max_gap_center);
    return distribution(*rng);
}

int replay_find_next_pipe_index(const GameState& game_state) {
    int best_index = 0;
    float best_x = 1.0e30f;

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        const float pipe_trailing_edge = game_state.pipes[pipe_idx].x + PIPE_WIDTH;
        if (pipe_trailing_edge >= BIRD_X && game_state.pipes[pipe_idx].x < best_x) {
            best_index = pipe_idx;
            best_x = game_state.pipes[pipe_idx].x;
        }
    }

    return best_index;
}

void initialize_replay_game(GameState* game_state, std::mt19937* rng) {
    *game_state = {};
    game_state->bird_y = BIRD_START_Y;
    game_state->bird_velocity = 0.0f;
    game_state->fitness = 0.0f;
    game_state->alive = 1;
    game_state->score = 0;
    game_state->ticks_alive = 0;
    game_state->next_pipe_index = 0;

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        game_state->pipes[pipe_idx].x =
            GAME_WIDTH + INITIAL_PIPE_OFFSET + static_cast<float>(pipe_idx) * PIPE_SPACING;
        game_state->pipes[pipe_idx].gap_center_y = replay_random_gap_center(rng);
        game_state->pipes[pipe_idx].passed = 0;
    }
}

void advance_replay_pipes(GameState* game_state, std::mt19937* rng) {
    float rightmost_x = game_state->pipes[0].x;

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        PipeState* pipe = &game_state->pipes[pipe_idx];
        pipe->x -= PIPE_SPEED;

        if (pipe->x > rightmost_x) {
            rightmost_x = pipe->x;
        }

        if (!pipe->passed && pipe->x + PIPE_WIDTH < BIRD_X) {
            pipe->passed = 1;
            game_state->score += 1;
        }
    }

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        PipeState* pipe = &game_state->pipes[pipe_idx];
        if (pipe->x + PIPE_WIDTH >= 0.0f) {
            continue;
        }

        pipe->x = rightmost_x + PIPE_SPACING;
        pipe->gap_center_y = replay_random_gap_center(rng);
        pipe->passed = 0;
        rightmost_x = pipe->x;
    }
}

bool replay_collides_with_pipe(const GameState& game_state) {
    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        const PipeState& pipe = game_state.pipes[pipe_idx];
        const float pipe_left = pipe.x;
        const float pipe_right = pipe.x + PIPE_WIDTH;
        const bool overlaps_horizontally =
            (BIRD_X + BIRD_RADIUS) >= pipe_left && (BIRD_X - BIRD_RADIUS) <= pipe_right;

        if (!overlaps_horizontally) {
            continue;
        }

        const float gap_top = pipe.gap_center_y - (PIPE_GAP * 0.5f);
        const float gap_bottom = pipe.gap_center_y + (PIPE_GAP * 0.5f);
        if ((game_state.bird_y - BIRD_RADIUS) < gap_top ||
            (game_state.bird_y + BIRD_RADIUS) > gap_bottom) {
            return true;
        }
    }

    return false;
}

void simulate_replay_tick(GameState* game_state,
                          const NetworkWeights& network,
                          std::mt19937* rng,
                          float* flap_probability) {
    if (!game_state->alive) {
        return;
    }

    game_state->next_pipe_index = replay_find_next_pipe_index(*game_state);
    const PipeState& next_pipe = game_state->pipes[game_state->next_pipe_index];

    float inputs[INPUT_SIZE];
    inputs[0] = (game_state->bird_y / GAME_HEIGHT) * 2.0f - 1.0f;
    inputs[1] = game_state->bird_velocity / 10.0f;
    inputs[2] = ((next_pipe.x - BIRD_X) / GAME_WIDTH) * 2.0f - 1.0f;
    inputs[3] = (next_pipe.gap_center_y / GAME_HEIGHT) * 2.0f - 1.0f;

    *flap_probability = run_network(network, inputs);
    if (*flap_probability > FLAP_THRESHOLD) {
        game_state->bird_velocity = FLAP_STRENGTH;
    }

    game_state->bird_velocity += GRAVITY;
    game_state->bird_y += game_state->bird_velocity;
    advance_replay_pipes(game_state, rng);

    game_state->ticks_alive += 1;
    game_state->fitness = static_cast<float>(game_state->ticks_alive) +
                          static_cast<float>(game_state->score) * 250.0f;

    const bool out_of_bounds =
        (game_state->bird_y - BIRD_RADIUS) < 0.0f ||
        (game_state->bird_y + BIRD_RADIUS) > GAME_HEIGHT;
    if (out_of_bounds || replay_collides_with_pipe(*game_state)) {
        game_state->alive = 0;
    }
}

std::string build_generation_checkpoint_path(const std::string& directory, int generation) {
    char generation_path[128];
    std::snprintf(generation_path,
                  sizeof(generation_path),
                  "%s/best_network_gen_%04d.txt",
                  directory.c_str(),
                  generation);
    return std::string(generation_path);
}

std::string build_generation_replay_path(const std::string& directory, int generation) {
    char generation_path[128];
    std::snprintf(generation_path,
                  sizeof(generation_path),
                  "%s/best_replay_gen_%04d.avi",
                  directory.c_str(),
                  generation);
    return std::string(generation_path);
}

#ifdef CUDABIRD_HAVE_OPENCV

int scaled_pixels(float value, int scale) {
    return static_cast<int>(value * static_cast<float>(scale));
}

bool open_video_writer(cv::VideoWriter* writer,
                       const std::string& path,
                       int width,
                       int height,
                       int fps) {
    const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    if (writer->open(path, fourcc, static_cast<double>(fps), cv::Size(width, height))) {
        return true;
    }

    std::fprintf(stderr, "Warning: unable to open replay video '%s'\n", path.c_str());
    return false;
}

void draw_pipe(cv::Mat* frame, const PipeState& pipe, int scale) {
    const int left = scaled_pixels(pipe.x, scale);
    const int right = scaled_pixels(pipe.x + PIPE_WIDTH, scale);
    const int gap_top = scaled_pixels(pipe.gap_center_y - (PIPE_GAP * 0.5f), scale);
    const int gap_bottom = scaled_pixels(pipe.gap_center_y + (PIPE_GAP * 0.5f), scale);
    const int frame_height = frame->rows;

    cv::rectangle(*frame,
                  cv::Point(left, 0),
                  cv::Point(right, gap_top),
                  cv::Scalar(71, 153, 66),
                  cv::FILLED);
    cv::rectangle(*frame,
                  cv::Point(left, gap_bottom),
                  cv::Point(right, frame_height),
                  cv::Scalar(71, 153, 66),
                  cv::FILLED);

    cv::rectangle(*frame,
                  cv::Point(left, 0),
                  cv::Point(right, gap_top),
                  cv::Scalar(37, 94, 35),
                  scale);
    cv::rectangle(*frame,
                  cv::Point(left, gap_bottom),
                  cv::Point(right, frame_height),
                  cv::Scalar(37, 94, 35),
                  scale);
}

void draw_replay_frame(cv::Mat* frame,
                       const GameState& game_state,
                       const GenerationSummary& summary,
                       int tick,
                       int max_ticks,
                       float flap_probability,
                       int scale) {
    frame->setTo(cv::Scalar(248, 221, 148));

    for (int pipe_idx = 0; pipe_idx < MAX_PIPES; ++pipe_idx) {
        draw_pipe(frame, game_state.pipes[pipe_idx], scale);
    }

    const int bird_x = scaled_pixels(BIRD_X, scale);
    const int bird_y = scaled_pixels(game_state.bird_y, scale);
    const int bird_radius = scaled_pixels(BIRD_RADIUS, scale);
    cv::circle(*frame,
               cv::Point(bird_x, bird_y),
               bird_radius,
               cv::Scalar(67, 181, 255),
               cv::FILLED);
    cv::circle(*frame,
               cv::Point(bird_x, bird_y),
               bird_radius,
               cv::Scalar(24, 94, 179),
               scale);

    const int panel_height = scaled_pixels(92.0f, scale);
    cv::rectangle(*frame,
                  cv::Point(0, 0),
                  cv::Point(frame->cols, panel_height),
                  cv::Scalar(40, 34, 26),
                  cv::FILLED);

    const double font_scale = 0.38 * static_cast<double>(scale);
    const int line_height = 24 * scale;
    char flap_text[32];
    std::snprintf(flap_text, sizeof(flap_text), "%.3f", flap_probability);

    cv::putText(*frame,
                "Best bird replay",
                cv::Point(12 * scale, 24 * scale),
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                cv::Scalar(255, 255, 255),
                1 + (scale / 2),
                cv::LINE_AA);
    cv::putText(*frame,
                "generation " + std::to_string(summary.generation) +
                    " | best score " + std::to_string(summary.best_score) +
                    " | replay score " + std::to_string(game_state.score),
                cv::Point(12 * scale, 24 * scale + line_height),
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                cv::Scalar(228, 228, 228),
                1,
                cv::LINE_AA);
    cv::putText(*frame,
                "tick " + std::to_string(tick) + "/" + std::to_string(max_ticks) +
                    " | fitness " + std::to_string(static_cast<int>(game_state.fitness)) +
                    " | flap " + flap_text,
                cv::Point(12 * scale, 24 * scale + (2 * line_height)),
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                cv::Scalar(210, 210, 210),
                1,
                cv::LINE_AA);

    if (!game_state.alive) {
        cv::putText(*frame,
                    "Replay ended",
                    cv::Point(12 * scale, frame->rows - (20 * scale)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.48 * static_cast<double>(scale),
                    cv::Scalar(44, 44, 255),
                    1 + (scale / 2),
                    cv::LINE_AA);
    }
}

#endif

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
    const std::string generation_path =
        build_generation_checkpoint_path(checkpoint_dir_, summary.generation);

    const std::string header =
        "generation=" + std::to_string(summary.generation) +
        "\nbest_score=" + std::to_string(summary.best_score) +
        "\nbest_fitness=" + std::to_string(summary.best_fitness) +
        "\nparameters=" + std::to_string(NETWORK_PARAM_COUNT) + "\n";

    std::ofstream latest_file(latest_path.c_str());
    std::ofstream generation_file(generation_path.c_str());
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

void Renderer::render_best_replay(const NetworkWeights& network,
                                  const GenerationSummary& summary,
                                  const ReplaySettings& settings) {
    ensure_directory_exists(checkpoint_dir_);

#ifndef CUDABIRD_HAVE_OPENCV
    (void)network;
    (void)summary;
    (void)settings;
    std::fprintf(stderr, "Replay rendering skipped because OpenCV support is unavailable\n");
#else
    const int frame_width = scaled_pixels(GAME_WIDTH, settings.scale);
    const int frame_height = scaled_pixels(GAME_HEIGHT, settings.scale);
    const std::string latest_path = checkpoint_dir_ + "/best_replay_latest.avi";
    const std::string generation_path =
        build_generation_replay_path(checkpoint_dir_, summary.generation);

    cv::VideoWriter latest_writer;
    cv::VideoWriter generation_writer;
    const bool latest_open =
        open_video_writer(&latest_writer, latest_path, frame_width, frame_height, settings.fps);
    const bool generation_open =
        open_video_writer(&generation_writer, generation_path, frame_width, frame_height, settings.fps);

    if (!latest_open && !generation_open) {
        return;
    }

    auto write_frame = [&](const cv::Mat& frame) {
        if (latest_open) {
            latest_writer.write(frame);
        }
        if (generation_open) {
            generation_writer.write(frame);
        }
    };

    std::mt19937 replay_rng(static_cast<std::mt19937::result_type>(settings.seed));
    GameState game_state = {};
    initialize_replay_game(&game_state, &replay_rng);

    cv::Mat frame(frame_height, frame_width, CV_8UC3);
    float flap_probability = 0.0f;
    int rendered_ticks = 0;

    draw_replay_frame(&frame,
                      game_state,
                      summary,
                      rendered_ticks,
                      settings.max_ticks,
                      flap_probability,
                      settings.scale);
    write_frame(frame);

    while (rendered_ticks < settings.max_ticks && game_state.alive) {
        simulate_replay_tick(&game_state, network, &replay_rng, &flap_probability);
        rendered_ticks += 1;

        draw_replay_frame(&frame,
                          game_state,
                          summary,
                          rendered_ticks,
                          settings.max_ticks,
                          flap_probability,
                          settings.scale);
        write_frame(frame);
    }

    for (int hold_frame = 0; hold_frame < DEFAULT_REPLAY_HOLD_FRAMES; ++hold_frame) {
        write_frame(frame);
    }

    latest_writer.release();
    generation_writer.release();

    std::printf("Saved replay video to %s\n", latest_path.c_str());
#endif
}
