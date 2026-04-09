#include "renderer.h"

#include <cerrno>
#include <cstdio>
#include <fstream>
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
                       const ReplayFrame& replay_frame,
                       const GenerationSummary& summary,
                       int tick,
                       int max_ticks,
                       int scale) {
    const GameState& game_state = replay_frame.game_state;
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
    std::snprintf(flap_text, sizeof(flap_text), "%.3f", replay_frame.flap_probability);

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

void Renderer::render_best_replay(const std::vector<ReplayFrame>& frames,
                                  const GenerationSummary& summary,
                                  const ReplaySettings& settings) {
    ensure_directory_exists(checkpoint_dir_);

#ifndef CUDABIRD_HAVE_OPENCV
    (void)frames;
    (void)summary;
    (void)settings;
    std::fprintf(stderr, "Replay rendering skipped because OpenCV support is unavailable\n");
#else
    if (frames.empty()) {
        std::fprintf(stderr, "Replay rendering skipped because no frames were captured\n");
        return;
    }

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

    cv::Mat frame(frame_height, frame_width, CV_8UC3);
    const int max_ticks = settings.max_ticks > 0 ? settings.max_ticks : static_cast<int>(frames.size() - 1);

    for (size_t frame_idx = 0; frame_idx < frames.size(); ++frame_idx) {
        draw_replay_frame(&frame,
                          frames[frame_idx],
                          summary,
                          static_cast<int>(frame_idx),
                          max_ticks,
                          settings.scale);
        write_frame(frame);
    }

    for (int hold_frame = 0; hold_frame < settings.hold_frames; ++hold_frame) {
        write_frame(frame);
    }

    latest_writer.release();
    generation_writer.release();

    std::printf("Saved replay video to %s\n", latest_path.c_str());
#endif
}
