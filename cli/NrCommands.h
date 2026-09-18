#pragma once

#include <CLI/CLI.hpp>
#include <cstdint>
#include <string>

namespace dlssvid::cli {

// `dlssvid nr` (Neural Rendering -> color_nr, `--check` for the diagnostics only) and
// `dlssvid nr-patch` (dlssnr-patcher wrapper) — stage 6, ТЗ §4 / §7.
class NrCommands {
public:
    void Register(CLI::App& app);
    int Dispatch();  // exit code when one of the subcommands was parsed, -1 otherwise

    struct NrArgs {
        std::string input, output;
        std::string colorDir, depthDir, mvDir, masksDir;
        bool noAuto = false;  // do not pick color_sr / depth_dlss / mv_dlss / mask_* under -o automatically
        std::string backend = "ngx";
        float intensity = 1.f;
        std::string style = "natural";
        int preset = 3;
        float localTone = 1.f, localStructure = 1.f, skinStructure = -1.f;
        bool autoMask = false;
        int passes = 1;
        double modelScale = 1.0;
        float transfer = 1.f, maxRatio = 4.f;
        bool noGuides = false;
        float temporal = 0.f, temporalThreshold = 0.1f, skinBlend = 1.f;
        std::string tonemap = "passthrough", inputTransfer = "srgb";
        float exposure = 1.f;
        std::string format = "exr", video, codec = "h264_nvenc";
        int64_t frames = -1;
        bool warp = false;
        std::string dllDir, paramsBlock = "capability";
        bool skipDriverCheck = false;
        bool check = false;
        std::string checkSize = "1280x720";
        std::string json;
    };
    struct PatchArgs {
        std::string input, output, patcher, cudaBin, python, arch = "all";
        bool dryRun = false;
    };

private:
    CLI::App* nr_ = nullptr;
    CLI::App* patch_ = nullptr;
    NrArgs na_;
    PatchArgs pa_;
};

}  // namespace dlssvid::cli
