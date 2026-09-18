#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "gpu/GpuFrame.h"
#include "passes/PassImage.h"

namespace dlssvid {

struct FlowEstimatorConfig {
    std::string modelId;      // registry id for model backends (searaft)
    int maxRes = 720;         // shorter side cap for the model input (searaft); OFA runs at source res
    bool fp16 = true;
    std::string modelsDir;
    nlohmann::json extra = nlohmann::json::object();  // ofa: perf_level (slow|medium|fast), grid (1|2|4); stub: dx, dy
};

// One frame as the estimator may consume it: CPU RGB ([0,1] float, R G B) and/or the GPU
// NV12 frame from NVDEC. Backends use what they can (OFA prefers the GPU frame: no CPU copy).
struct FlowInput {
    const PassImage* rgb = nullptr;
    const GpuFrame* gpu = nullptr;
};

// Forward optical flow a -> b in pixels of the source resolution: docs/conventions.md §3
// (`mv_raw`, channels u, v, F32). Optional per-pixel confidence in [0,1].
class IFlowEstimator {
public:
    virtual ~IFlowEstimator() = default;
    virtual std::string Name() const = 0;
    virtual void Init(const FlowEstimatorConfig& config, uint32_t width, uint32_t height) = 0;
    virtual void Estimate(const FlowInput& a, const FlowInput& b, PassImage& flowOut, PassImage* confidenceOut = nullptr) = 0;
    virtual void Shutdown() {}
    virtual nlohmann::json Describe() const { return {{"backend", Name()}}; }
    virtual bool WantsGpuFrames() const { return false; }
};

// Factory: "ofa" | "searaft" | "stub".
std::unique_ptr<IFlowEstimator> CreateFlowEstimator(const std::string& backend);
std::vector<std::string> FlowBackends();

}  // namespace dlssvid
