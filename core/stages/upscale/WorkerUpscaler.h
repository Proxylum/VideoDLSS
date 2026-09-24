#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "passes/PassImage.h"
#include "stages/upscale/IUpscaler.h"

namespace dlssvid {

class Subprocess;

// Client of sr_worker/worker.py: video super-resolution in PyTorch — RealBasicVSR (Apache-2.0), the "maximum quality
// for video" mode of stage 5: the model propagates features across a window of frames (bidirectional, aligned by
// SPyNet flow), so the whole window is sent at once and the stage overlaps consecutive windows. Frames travel as .npz
// files through a scratch folder, control is JSON lines on the worker's stdin/stdout (the depth worker's protocol).
// The worker resizes the model's x4 output to the target size itself (NativeScale() == 0).
class WorkerUpscaler final : public IUpscaler {
public:
    static constexpr const char* kDefaultModel = "realbasicvsr";
    static UpscalerAvailability Available();

    WorkerUpscaler();
    ~WorkerUpscaler() override;

    std::string_view Name() const override { return "worker"; }
    void Init(D3D12Device& device, const UpscalerConfig& config) override;
    void Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) override;
    bool ProcessesOnCpu() const override { return true; }
    int NativeScale() const override { return 0; }  // the worker delivers the target size
    int WindowSize() const override { return window_; }
    int WindowOverlap() const override { return overlap_; }
    void EvaluateCpuWindow(const std::vector<const PassImage*>& rgbaIn, uint32_t targetW, uint32_t targetH, std::vector<PassImage>& rgbaOut) override;
    nlohmann::json Describe() const override;
    void Shutdown() override;

    static std::filesystem::path FindProjectRoot();
    static std::filesystem::path FindPython(const UpscalerConfig& config, const std::filesystem::path& projectRoot);

private:
    nlohmann::json Call(const nlohmann::json& request, uint32_t timeoutMs);

    UpscalerConfig config_;
    std::string backend_ = "realbasicvsr", modelId_;
    std::unique_ptr<Subprocess> proc_;
    std::filesystem::path scratch_;
    nlohmann::json info_;
    int window_ = 15, overlap_ = 3, scale_ = 4;
    int64_t seq_ = 0;
};

}  // namespace dlssvid
