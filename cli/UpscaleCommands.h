#pragma once

#include <CLI/CLI.hpp>
#include <cstdint>
#include <string>

namespace dlssvid::cli {

// `dlssvid upscale` (stage 5: color_sr through rtxvsr | dlss | nis | bicubic) and `dlssvid compare`
// (PSNR / SSIM between two videos or pass folders — the A/B tool of docs/benchmarks.md).
class UpscaleCommands {
public:
    void Register(CLI::App& app);
    int Dispatch();

    struct UpscaleArgs {
        std::string input, output;
        std::string backend = "rtxvsr";
        double scale = 2.0;
        std::string target;       // WxH
        std::string depthDir, mvDir;
        float sharpness = 0.5f;
        std::string preset = "default";
        bool artifactReductionOnly = false;
        bool noJitter = false;
        float jitterSign = 1.f;
        bool noFallback = false;
        std::string format = "exr";
        std::string video, codec = "h264_nvenc";
        int64_t frames = -1;
        bool warp = false;
        std::string dllDir;
    };
    struct CompareArgs {
        std::string ref, test;
        int64_t frames = -1;
        int64_t offset = 0;      // test frame index offset relative to the reference
        std::string json;        // per-frame results
        bool quiet = false;
    };

private:
    UpscaleArgs ua_;
    CompareArgs ca_;
    CLI::App* upscale_ = nullptr;
    CLI::App* compare_ = nullptr;
};

}  // namespace dlssvid::cli
