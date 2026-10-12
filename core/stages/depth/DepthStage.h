#pragma once

#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

#include "convert/ColorConvert.h"
#include "passes/Manifest.h"
#include "passes/PassSequence.h"
#include "pipeline/IStage.h"
#include "stages/depth/DepthPostProcess.h"
#include "stages/depth/IDepthEstimator.h"

namespace dlssvid {

// Temporal alignment requested for the depth (ТЗ §2, TASK-0038 audit depth-7). auto: scale over the window for metric
// models (the shift would break the metres), scale + shift for relative ones; the explicit modes are kept for any model.
enum class DepthStabilize { Auto, None, Scale, ScaleShift };
DepthStabilize ParseDepthStabilize(const std::string& s);  // auto | none | scale | scale_shift; throws otherwise
TemporalStabilizer::Mode ResolveStabilizeMode(DepthStabilize requested, bool metric);
const char* StabilizeModeName(TemporalStabilizer::Mode mode);  // none | scale | scale_shift
const char* DepthStabilizeName(DepthStabilize s);              // auto | none | scale | scale_shift

// Pipeline stage "Depth" (ТЗ §4): collects colour frames, runs the estimator in windows,
// post-processes (holes, temporal alignment over a centred window of ±N frames - the output is delayed by N frames),
// writes depth_raw (+ depth_dlss) pass folders and uploads the depth into the GPU frame cache slot when the slot still
// holds that frame (with the delay it often does not; the stages of `process` read the pass folders).
struct DepthStageOptions {
    std::string backend = "da3";
    DepthEstimatorConfig estimator;
    std::filesystem::path outputDir;  // pass root: <outputDir>/depth_raw, <outputDir>/depth_dlss
    bool writeDlss = true;
    DepthParams dlss;                 // near/far (metric) or relative mapping
    DepthStabilize stabilize = DepthStabilize::Auto;
    int stabilizeWindow = 8;          // radius N: frame t is aligned over t-N..t+N, the output is delayed by N frames
    bool fillHoles = true;
    FileFormat format = FileFormat::Exr;
    ColorInfo color;                  // YUV -> RGB for the estimator input
    Rational fps;
    std::string sourceFile, sourceHash;
    bool uploadToGpu = true;
    std::filesystem::path mvDir;      // mv_dlss pass folder: TAE with motion compensation (stage 3)
    std::function<void(int64_t frame, double tae)> onFrame;
};

class DepthStage final : public IStage {
public:
    explicit DepthStage(DepthStageOptions options, std::unique_ptr<IDepthEstimator> estimator = nullptr);
    ~DepthStage() override;

    std::string_view Name() const override { return "depth"; }
    void Init(const StageConfig& config, D3D12Device& device) override;
    void Process(FrameContext& ctx) override;
    void Finish() override;
    void Shutdown() override;

    struct Stats {
        int64_t frames = 0;
        double taeSum = 0;
        int64_t taeCount = 0;
        double MeanTae() const { return taeCount ? taeSum / taeCount : 0.0; }
        int64_t holesFilled = 0;
        int windows = 0;
        float scaleMin = 1.f, scaleMax = 1.f;  // the alignment applied over the clip
        float shiftMin = 0.f, shiftMax = 0.f;
    };
    const Stats& GetStats() const { return stats_; }
    const IDepthEstimator& Estimator() const { return *estimator_; }
    const TemporalStabilizer& Stabilizer() const { return stabilizer_; }
    std::filesystem::path RawDir() const { return options_.outputDir / "depth_raw"; }
    std::filesystem::path DlssDir() const { return options_.outputDir / "depth_dlss"; }

private:
    struct Pending {
        int64_t index;
        PassImage rgb;
        GpuFrameCache* cache;
        D3D12Device* device;
    };
    void RunWindow(bool flush);
    void Emit(int64_t index, PassImage depth);
    void Drain();  // frames whose alignment window is complete
    void Finalize(int64_t index, PassImage depth, const ScaleShift& applied);

    DepthStageOptions options_;
    std::unique_ptr<IDepthEstimator> estimator_;
    std::deque<Pending> pending_;
    std::deque<std::pair<int64_t, PassImage>> overlapPrev_;  // previous window's tail for blending
    std::unique_ptr<PassWriter> rawWriter_, dlssWriter_;
    TemporalStabilizer stabilizer_;
    std::optional<PassImage> prevDepth_;
    std::optional<PassReader> mvReader_;
    GpuFrameCache* cache_ = nullptr;  // the pipeline's cache (one per pipeline): the upload of a delayed frame
    D3D12Device* device_ = nullptr;
    Stats stats_;
    bool initialized_ = false;
};

// Convenience for the CLI and tests: decode -> DepthStage. Returns stage stats.
struct DepthRunResult {
    DepthStage::Stats stats;
    std::filesystem::path rawDir, dlssDir;
    nlohmann::json estimator;
    std::string stabilize;  // the alignment applied: none | scale | scale_shift
};
class VideoDecoder;
class D3D12Device;
DepthRunResult RunDepth(VideoDecoder& decoder, D3D12Device& device, DepthStageOptions options, int64_t maxFrames = -1,
                        const std::function<void(int64_t)>& progress = {});

}  // namespace dlssvid
