#pragma once

#include <memory>

#include "stages/flow/IFlowEstimator.h"

namespace dlssvid {

// NVIDIA Optical Flow Accelerator (nvofapi64.dll from the driver, CUDA interface). Consumes
// NV12 GPU frames straight from NVDEC (no CPU copy) or ABGR8 uploaded from CPU RGB. Output:
// forward flow at the source resolution (S10.5 fixed point -> float px, grid upsampled) and
// a confidence map derived from the cost buffer.
class OfaFlowEstimator final : public IFlowEstimator {
public:
    OfaFlowEstimator();
    ~OfaFlowEstimator() override;
    std::string Name() const override { return "ofa"; }
    bool WantsGpuFrames() const override { return true; }
    void Init(const FlowEstimatorConfig& config, uint32_t width, uint32_t height) override;
    void Estimate(const FlowInput& a, const FlowInput& b, PassImage& flowOut, PassImage* confidenceOut) override;
    void Shutdown() override;
    nlohmann::json Describe() const override;

    // nvofapi64.dll present and an optical-flow session can be created on this GPU.
    static bool Available(std::string* reason = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dlssvid
