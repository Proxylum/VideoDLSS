#pragma once

#include "stages/depth/IDepthEstimator.h"

namespace dlssvid {

// Deterministic synthetic backend for tests and pipeline plumbing: depth = 1 + 4 * luminance
// metres, window of 1. Also exposes a "windowed" variant for exercising overlap blending.
class StubDepthEstimator final : public IDepthEstimator {
public:
    explicit StubDepthEstimator(int window = 1, int overlap = 0) : window_(window), overlap_(overlap) {}
    std::string Name() const override { return "stub"; }
    bool IsMetric() const override { return true; }
    int WindowSize() const override { return window_; }
    int WindowOverlap() const override { return overlap_; }
    void Init(const DepthEstimatorConfig&) override { calls_ = 0; }
    void Estimate(const std::vector<const PassImage*>& rgb, std::vector<PassImage>& depthOut) override;
    int Calls() const { return calls_; }

    static PassImage Expected(const PassImage& rgb);  // what Estimate() produces for one frame

private:
    int window_, overlap_, calls_ = 0;
};

}  // namespace dlssvid
