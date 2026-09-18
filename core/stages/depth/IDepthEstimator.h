#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "passes/PassImage.h"

namespace dlssvid {

// Common configuration for every depth backend (ТЗ §2: the interface is abstract so the
// backend can change without touching the pipeline).
struct DepthEstimatorConfig {
    std::string modelId;      // registry id, e.g. "da3metric-large"; backend default when empty
    int inputSize = 518;      // model input on the shorter side (multiple of 14 for DINOv2)
    int maxInputRes = 1080;   // frames larger than this (shorter side) are downscaled before the model
    bool fp16 = true;
    std::string modelsDir;    // models/ folder (registry.json + cache); default = auto
    nlohmann::json extra = nlohmann::json::object();
};

// Backend contract: RGB frames in ([0,1] float, R,G,B channels, source resolution) ->
// depth_raw out (F32 "Z", source resolution, larger = farther, metres when IsMetric()).
// A backend that needs a window asks for WindowSize() frames at a time; the stage feeds
// windows with WindowOverlap() frames of overlap and blends the overlap.
class IDepthEstimator {
public:
    virtual ~IDepthEstimator() = default;
    virtual std::string Name() const = 0;
    virtual bool IsMetric() const = 0;
    virtual int WindowSize() const { return 1; }
    virtual int WindowOverlap() const { return 0; }
    virtual void Init(const DepthEstimatorConfig& config) = 0;
    virtual void Estimate(const std::vector<const PassImage*>& rgb, std::vector<PassImage>& depthOut) = 0;
    virtual void Shutdown() {}
    virtual nlohmann::json Describe() const { return {{"backend", Name()}, {"metric", IsMetric()}}; }
};

// Factory: "stub" | "da3" | "vda" | "worker" (+ "worker:<backend>" to pick the worker backend).
std::unique_ptr<IDepthEstimator> CreateDepthEstimator(const std::string& backend);
std::vector<std::string> DepthBackends();

}  // namespace dlssvid
