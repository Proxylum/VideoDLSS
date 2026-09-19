#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "convert/ColorConvert.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "passes/PassSequence.h"
#include "pipeline/IStage.h"
#include "stages/fg/IFrameGenerator.h"

namespace dlssvid {

// Pipeline stage "Frame Generation" (ТЗ §4, §8): colour (color_nr / color_sr / the decoded frame) + depth_dlss /
// mv_dlss -> `color_fg` with `multiplier` x frames: real frame i at index i * multiplier, generated frames between.
// The manifest carries fps * multiplier; the optional preview video runs at that rate. The GUI runs this stage as
// the `dlssvid fg` process (TaskQueue), so a crash inside the backend never reaches the GUI (ТЗ §9).
struct FgStageOptions {
    std::string backend = "dlssg";       // dlssg | rife | blend
    int multiplier = 2;                  // 2 | 3 | 4
    std::string backbufferFormat = "rgba16f";
    std::string model = "rife49";
    bool fp16 = true;
    std::string modelsDir;
    std::filesystem::path dllDir;
    bool requireDriver = true;
    std::filesystem::path colorDir;      // colour pass folder (empty: slot color_nr / color_sr when present, else the frame)
    std::filesystem::path depthDir, mvDir;
    bool useCachePasses = true;
    std::filesystem::path outputDir;     // pass root: <outputDir>/color_fg (empty: no files)
    FileFormat format = FileFormat::Exr;
    std::filesystem::path videoOut;
    std::string videoCodec = "h264_nvenc";
    bool disableWhenUnavailable = false;
    int64_t crashAfter = -1;             // development: terminate the process after N real frames (isolation test)
    ColorInfo color;
    Rational fps;
    std::string sourceFile, sourceHash;
    std::function<void(int64_t frame, double ms)> onFrame;
};

class FgStage final : public IStage {
public:
    explicit FgStage(FgStageOptions options, std::unique_ptr<IFrameGenerator> generator = nullptr);
    ~FgStage() override;

    std::string_view Name() const override { return "fg"; }
    void Init(const StageConfig& config, D3D12Device& device) override;
    void Process(FrameContext& ctx) override;
    void Finish() override;
    void Shutdown() override;

    struct Stats {
        int64_t frames = 0;     // real frames seen
        int64_t generated = 0;  // frames produced by the backend
        double msSum = 0.0;     // backend time per real frame
        double MeanMs() const { return frames ? msSum / static_cast<double>(frames) : 0.0; }
        uint32_t width = 0, height = 0;
        std::string backend, colorSource;
        int multiplier = 0;
        bool depthUsed = false, mvUsed = false;
        bool disabled = false;
        std::string disabledReason;
    };
    const Stats& GetStats() const { return stats_; }
    const IFrameGenerator* Generator() const { return generator_.get(); }
    const FgDiagnostics* Diagnostics() const { return diag_ ? &*diag_ : nullptr; }
    std::filesystem::path OutDir() const { return options_.outputDir / "color_fg"; }

private:
    void ApplyParams(const nlohmann::json& params);
    void Setup(uint32_t w, uint32_t h);
    void Disable(const std::string& reason);

    FgStageOptions options_;
    std::unique_ptr<IFrameGenerator> generator_, injected_;
    std::optional<FgDiagnostics> diag_;
    D3D12Device* device_ = nullptr;
    ComPtr<ID3D12Resource> texA_, texB_;
    ID3D12Resource* curTex_ = nullptr;
    ID3D12Resource* prevTex_ = nullptr;
    PassImage cpuCur_, cpuPrev_;
    uint32_t inW_ = 0, inH_ = 0;
    bool setup_ = false, disabled_ = false, warnedDisabled_ = false;
    std::optional<PassReader> colorReader_, depthReader_, mvReader_;
    std::unique_ptr<PassWriter> writer_;
    std::unique_ptr<VideoEncoder> encoder_;
    std::vector<PassImage> generated_;
    Stats stats_;
    bool initialized_ = false;
};

struct FgRunResult {
    FgStage::Stats stats;
    std::filesystem::path outDir;
    nlohmann::json backend, diagnostics;
};
FgRunResult RunFg(VideoDecoder& decoder, D3D12Device& device, FgStageOptions options, int64_t maxFrames = -1, const std::function<void(int64_t)>& progress = {});

}  // namespace dlssvid
