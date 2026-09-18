#pragma once

#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

#include "convert/ColorConvert.h"
#include "convert/MvConvert.h"
#include "gpu/GpuFrame.h"
#include "passes/PassSequence.h"
#include "pipeline/IStage.h"
#include "stages/flow/IFlowEstimator.h"

namespace dlssvid {

// Pipeline stage "Motion vectors" (ТЗ §4): forward flow between consecutive frames -> mv_raw
// at frame t (one frame of latency), mv_dlss for frame t+1 (backward, DLSS convention, scaled
// to the target resolution, occlusions resolved with depth when available), warp-PSNR stats.
struct FlowStageOptions {
    std::string backend = "ofa";
    FlowEstimatorConfig estimator;
    std::filesystem::path outputDir;  // <outputDir>/mv_raw, <outputDir>/mv_dlss
    bool writeDlss = true;
    uint32_t targetWidth = 0, targetHeight = 0;  // mv_dlss resolution (0 = source)
    MvConvertOptions convert;         // dilation etc.
    std::filesystem::path depthDir;   // depth_raw pass folder for occlusion handling (optional)
    bool useCacheDepth = true;        // otherwise take depth_raw from the GPU cache slot if a depth stage ran
    FileFormat format = FileFormat::Exr;
    ColorInfo color;
    Rational fps;
    std::string sourceFile, sourceHash;
    bool uploadToGpu = true;
    bool computeWarpPsnr = true;
    std::function<void(int64_t frame, double warpPsnr)> onFrame;
};

class FlowStage final : public IStage {
public:
    explicit FlowStage(FlowStageOptions options, std::unique_ptr<IFlowEstimator> estimator = nullptr);
    ~FlowStage() override;

    std::string_view Name() const override { return "flow"; }
    void Init(const StageConfig& config, D3D12Device& device) override;
    void Process(FrameContext& ctx) override;
    void Finish() override;
    void Shutdown() override;

    struct Stats {
        int64_t frames = 0;
        int64_t pairs = 0;
        double psnrSum = 0;
        int64_t psnrCount = 0;
        double MeanWarpPsnr() const { return psnrCount ? psnrSum / psnrCount : 0.0; }
        double minWarpPsnr = 1e9;
        double meanMagnitude = 0;  // mean |flow| px over all pairs
    };
    const Stats& GetStats() const { return stats_; }
    const IFlowEstimator& Estimator() const { return *estimator_; }
    std::filesystem::path RawDir() const { return options_.outputDir / "mv_raw"; }
    std::filesystem::path DlssDir() const { return options_.outputDir / "mv_dlss"; }

private:
    struct Held {
        int64_t index = -1;
        PassImage rgb;
        std::optional<PassImage> depth;
        GpuFrameCache* cache = nullptr;
        // GPU frames are only valid until the next decode: OFA consumes them immediately, so
        // the held frame keeps a copy of the NV12 planes in device memory owned by the stage.
        GpuFrame gpu;
    };
    void EmitPair(Held& prev, Held& cur);
    void EmitLast(Held& last);
    std::optional<PassImage> DepthFor(int64_t index, GpuFrameCache* cache) const;
    void EnsureWriters(uint32_t w, uint32_t h);

    FlowStageOptions options_;
    std::unique_ptr<IFlowEstimator> estimator_;
    std::optional<Held> prev_;
    std::unique_ptr<PassWriter> rawWriter_, dlssWriter_;
    std::optional<PassReader> depthReader_;
    Stats stats_;
    bool initialized_ = false;
    void* gpuCopy_[2] = {nullptr, nullptr};  // device buffers for the held NV12 frame (double buffered)
    size_t gpuCopyBytes_ = 0;
    int gpuCopySlot_ = 0;
};

struct FlowRunResult {
    FlowStage::Stats stats;
    std::filesystem::path rawDir, dlssDir;
    nlohmann::json estimator;
};
class VideoDecoder;
class D3D12Device;
FlowRunResult RunFlow(VideoDecoder& decoder, D3D12Device& device, FlowStageOptions options, int64_t maxFrames = -1,
                      const std::function<void(int64_t)>& progress = {});

}  // namespace dlssvid
