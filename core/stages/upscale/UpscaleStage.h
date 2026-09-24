#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "convert/ColorConvert.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "passes/PassSequence.h"
#include "pipeline/IStage.h"
#include "stages/upscale/BicubicUpscaler.h"
#include "stages/upscale/IUpscaler.h"

namespace dlssvid {

// Pipeline stage "Upscale" (ТЗ §3–4): colour (+ depth_dlss / mv_dlss for DLSS) -> `color_sr` at the
// target resolution through an IUpscaler backend. Depth / MV come from the GPU cache slot when the
// depth and flow stages run in the same pipeline, or from pass folders. Writes the pass folder,
// uploads the result into the cache slot and can encode a preview video.
struct UpscaleStageOptions {
    std::string backend = "dlss";          // dlss (default) | nis | bicubic
    bool allowFallback = true;             // backend unavailable -> nis with a warning (ТЗ §4: a missing SDK must not crash)
    double scale = 2.0;                    // x1.5 / x2 / x3
    uint32_t targetWidth = 0, targetHeight = 0;  // explicit target instead of `scale`
    uint32_t maxWidth = 3840, maxHeight = 2160;  // v1 cap (ТЗ §3)
    float sharpness = 0.5f;                // nis
    std::string preset = "default";        // dlss render preset hint
    bool artifactReductionOnly = false;    // no scaling (nis sharpen)
    bool useJitter = true;                 // dlss: jitter emulation (HACK)
    float jitterSign = 1.f;                // dlss: sign of the reported jitter
    std::filesystem::path dllDir;          // nvngx_dlss.dll folder override
    std::filesystem::path outputDir;       // pass root: <outputDir>/color_sr (empty: no files)
    FileFormat format = FileFormat::Exr;   // exr (half) | png (16-bit)
    std::filesystem::path depthDir, mvDir; // depth_dlss / mv_dlss pass folders (optional)
    bool useCachePasses = true;            // take depth_dlss / mv_dlss from the cache slot when present
    bool uploadToGpu = true;               // keep color_sr in the cache slot
    std::filesystem::path videoOut;        // optional preview video (no audio)
    std::string videoCodec = "h264_nvenc";
    ColorInfo color;
    Rational fps;
    std::string sourceFile, sourceHash;
    std::function<void(int64_t frame, double ms)> onFrame;
};

class UpscaleStage final : public IStage {
public:
    explicit UpscaleStage(UpscaleStageOptions options, std::unique_ptr<IUpscaler> upscaler = nullptr);
    ~UpscaleStage() override;

    std::string_view Name() const override { return "upscale"; }
    void Init(const StageConfig& config, D3D12Device& device) override;
    void Process(FrameContext& ctx) override;
    void Finish() override;
    void Shutdown() override;

    struct Stats {
        int64_t frames = 0;
        double msSum = 0.0;  // GPU submit + wait per frame
        double MeanMs() const { return frames ? msSum / static_cast<double>(frames) : 0.0; }
        uint32_t inputWidth = 0, inputHeight = 0, outputWidth = 0, outputHeight = 0;
        std::string backend;  // backend actually used
        bool fellBack = false;
        int jitterPhases = 0;
        bool depthUsed = false, mvUsed = false;
    };
    const Stats& GetStats() const { return stats_; }
    const IUpscaler* Upscaler() const { return upscaler_.get(); }
    std::filesystem::path OutDir() const { return options_.outputDir / "color_sr"; }

private:
    void Setup(uint32_t w, uint32_t h);
    void ApplyParams(const nlohmann::json& params);

    UpscaleStageOptions options_;
    std::unique_ptr<IUpscaler> upscaler_;
    std::unique_ptr<IUpscaler> injected_;
    D3D12Device* device_ = nullptr;
    std::unique_ptr<Resampler> resampler_;
    ComPtr<ID3D12Resource> input_, jittered_, output_;
    uint32_t inW_ = 0, inH_ = 0, outW_ = 0, outH_ = 0;
    int phases_ = 8;
    bool first_ = true;
    std::optional<PassReader> depthReader_, mvReader_;
    std::unique_ptr<PassWriter> writer_;
    std::unique_ptr<VideoEncoder> encoder_;
    Stats stats_;
    bool initialized_ = false;
};

struct UpscaleRunResult {
    UpscaleStage::Stats stats;
    std::filesystem::path outDir;
    nlohmann::json upscaler;
};
UpscaleRunResult RunUpscale(VideoDecoder& decoder, D3D12Device& device, UpscaleStageOptions options, int64_t maxFrames = -1,
                            const std::function<void(int64_t)>& progress = {},
           const nlohmann::json& params = nlohmann::json::object());

}  // namespace dlssvid
