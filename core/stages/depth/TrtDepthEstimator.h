#pragma once

#include <memory>

#include "stages/depth/DepthPreprocess.h"
#include "stages/depth/IDepthEstimator.h"

namespace dlssvid {

class TrtEngine;
class ModelRegistry;
struct ModelEntry;

// TensorRT backend for ONNX depth models (Depth Anything 3 metric/mono, Video Depth Anything).
// Registry params: input_size, window (1 for DA3, 32 for VDA), overlap, metric, output
// ("depth" | "disparity"), input ("image"), output_name, mean/std. The ONNX file is produced
// by models/export for the exact input geometry; when it is missing the estimator runs the
// export script in models/export/.venv (PyTorch reference path) once.
class TrtDepthEstimator final : public IDepthEstimator {
public:
    // `family` = "da3" | "vda": selects the registry default model.
    explicit TrtDepthEstimator(std::string family);
    ~TrtDepthEstimator() override;

    std::string Name() const override { return family_; }
    bool IsMetric() const override { return metric_; }
    int WindowSize() const override { return window_; }
    int WindowOverlap() const override { return overlap_; }
    void Init(const DepthEstimatorConfig& config) override;
    void Estimate(const std::vector<const PassImage*>& rgb, std::vector<PassImage>& depthOut) override;
    void Shutdown() override;
    nlohmann::json Describe() const override;

    static std::string DefaultModel(const std::string& family);

private:
    void EnsureEngine(uint32_t srcWidth, uint32_t srcHeight);
    std::filesystem::path OnnxFor(const ModelInputSize& size);

    std::string family_;
    DepthEstimatorConfig config_;
    std::unique_ptr<ModelRegistry> registry_;
    std::string modelId_, modelFile_, license_;
    bool metric_ = false;
    bool disparity_ = false;
    int window_ = 1, overlap_ = 0;
    std::string inputName_ = "image", outputName_ = "depth", skyName_;
    float skyThreshold_ = 0.3f, skyQuantile_ = 0.99f;
    bool hasSky_ = false;
    std::vector<float> skyHost_;
    Normalization norm_;
    ModelInputSize inputSize_;
    uint32_t srcW_ = 0, srcH_ = 0;
    std::unique_ptr<TrtEngine> engine_;
    std::vector<float> inputHost_, outputHost_;
    int guidedRadius_ = 8;
};

}  // namespace dlssvid
