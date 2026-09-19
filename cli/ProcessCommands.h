#pragma once

#include <CLI/CLI.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace dlssvid::cli {

// `dlssvid process` (the whole pipeline over the pass cache, or --passthrough) and `dlssvid bench`
// (the same run into a scratch folder, timings per stage) — stage 8, ТЗ §7 / §9.
class ProcessCommands {
public:
    void Register(CLI::App& app);
    int Dispatch();

    struct ProcessArgs {
        std::string input, output, passes, project;
        std::string stages;                // comma list: enable only these (order is canonical)
        std::vector<std::string> params;   // stage.key=value
        std::string depthBackend, flowBackend, upscaleBackend, nrBackend, fgBackend;
        double scale = 0.0;
        int multiplier = 0;
        bool noSkipExisting = false;
        bool disableUnavailable = false;
        std::string codec = "h264_nvenc";
        std::vector<std::string> codecOptions;
        std::string hwaccel = "none";
        int64_t frames = -1;
        bool warp = false;
        bool passthrough = false;
        bool noGpuRoundTrip = false;
        std::string json;
    };
    struct BenchArgs {
        std::string input, passes, output;
        std::string stages;
        std::vector<std::string> params;
        int64_t frames = 30;
        bool warp = false;
        bool keep = false;
        std::string codec = "h264_nvenc";
        std::string json;
    };

private:
    CLI::App* process_ = nullptr;
    CLI::App* bench_ = nullptr;
    ProcessArgs pa_;
    BenchArgs ba_;
};

}  // namespace dlssvid::cli
