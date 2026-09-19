#pragma once

#include <CLI/CLI.hpp>
#include <cstdint>
#include <string>

namespace dlssvid::cli {

// `dlssvid fg` (Frame Generation -> color_fg; `--check` for the diagnostics only) — stage 7, ТЗ §4 / §7.
class FgCommands {
public:
    void Register(CLI::App& app);
    int Dispatch();  // exit code when the subcommand was parsed, -1 otherwise

    struct FgArgs {
        std::string input, output;
        std::string colorDir, depthDir, mvDir;
        bool noAuto = false;
        std::string backend = "dlssg";
        int multiplier = 2;
        std::string model = "rife49";
        bool fp32 = false;
        std::string modelsDir;
        std::string backbufferFormat = "rgba16f";
        std::string format = "exr", video, codec = "h264_nvenc";
        int64_t frames = -1;
        bool warp = false;
        std::string dllDir;
        bool skipDriverCheck = false;
        bool check = false;
        std::string checkSize = "1280x720";
        std::string json;
        int64_t crashAfter = -1;
    };

private:
    CLI::App* fg_ = nullptr;
    FgArgs fa_;
};

}  // namespace dlssvid::cli
