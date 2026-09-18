#pragma once

#include <filesystem>
#include <memory>

#include "stages/flow/IFlowEstimator.h"

namespace dlssvid {

class TrtEngine;
class ModelRegistry;

// SEA-RAFT (or any two-image flow ONNX) through TensorRT. Frames are resized to the model
// geometry (source resolution capped at maxRes on the shorter side, rounded to a multiple
// of 8) and the vectors are scaled back to source pixels.
class TrtFlowEstimator final : public IFlowEstimator {
public:
    explicit TrtFlowEstimator(std::string family = "searaft");
    ~TrtFlowEstimator() override;
    std::string Name() const override { return family_; }
    void Init(const FlowEstimatorConfig& config, uint32_t width, uint32_t height) override;
    void Estimate(const FlowInput& a, const FlowInput& b, PassImage& flowOut, PassImage* confidenceOut) override;
    void Shutdown() override;
    nlohmann::json Describe() const override;

    static std::string DefaultModel(const std::string& family);

private:
    std::filesystem::path OnnxFor(uint32_t w, uint32_t h);

    std::string family_;
    FlowEstimatorConfig config_;
    std::unique_ptr<ModelRegistry> registry_;
    std::string modelId_, license_;
    std::vector<std::string> inputNames_{"image1", "image2"};
    std::string outputName_ = "flow";
    float inputScale_ = 255.f;
    int multiple_ = 8;
    uint32_t srcW_ = 0, srcH_ = 0, modelW_ = 0, modelH_ = 0;
    std::unique_ptr<TrtEngine> engine_;
    std::vector<float> host1_, host2_, outHost_;
};

}  // namespace dlssvid
