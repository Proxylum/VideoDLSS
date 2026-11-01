#pragma once

#include <filesystem>
#include <memory>

#include "stages/depth/IDepthEstimator.h"

namespace dlssvid {

class Subprocess;

// Client of depth_worker/worker.py: the PyTorch reference path (DA3, VDA) and the home of
// ICDepth when its code is released. Frames travel as .npz files through a scratch folder;
// control is JSON lines on the worker's stdin/stdout (see depth_worker/README.md).
class WorkerDepthEstimator final : public IDepthEstimator {
public:
    // `backend` is the worker-side backend: "da3" | "vda" | "icdepth" | "stub".
    explicit WorkerDepthEstimator(std::string backend);
    ~WorkerDepthEstimator() override;

    std::string Name() const override { return "worker:" + backend_; }
    bool IsMetric() const override { return metric_; }
    int WindowSize() const override { return window_; }
    int WindowOverlap() const override { return overlap_; }
    void Init(const DepthEstimatorConfig& config) override;
    void Estimate(const std::vector<const PassImage*>& rgb, std::vector<PassImage>& depthOut) override;
    void Shutdown() override;
    nlohmann::json Describe() const override;

    // Locates python: config.extra["python"], DLSSVID_PYTHON, models/export/.venv, then "python".
    static std::filesystem::path FindPython(const DepthEstimatorConfig& config, const std::filesystem::path& projectRoot);
    static std::filesystem::path FindProjectRoot();
    // The worker's models_dir (registry.json + cache/): config.modelsDir, else the registry's folder (DLSSVID_MODELS_DIR
    // first, as the TensorRT path and the fingerprints) - not <project root>/models.
    static std::filesystem::path FindModelsDir(const DepthEstimatorConfig& config);

private:
    nlohmann::json Call(const nlohmann::json& request, uint32_t timeoutMs);

    std::string backend_;
    DepthEstimatorConfig config_;
    std::unique_ptr<Subprocess> proc_;
    std::filesystem::path scratch_;
    bool metric_ = false;
    int window_ = 1, overlap_ = 0;
    nlohmann::json info_;
    int64_t seq_ = 0;
};

}  // namespace dlssvid
