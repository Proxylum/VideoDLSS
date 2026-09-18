#pragma once

#include "stages/flow/IFlowEstimator.h"

namespace dlssvid {

// Constant flow (extra.dx, extra.dy; default 0) — exercises the stage plumbing in tests.
class StubFlowEstimator final : public IFlowEstimator {
public:
    std::string Name() const override { return "stub"; }
    void Init(const FlowEstimatorConfig& config, uint32_t, uint32_t) override {
        dx_ = config.extra.value("dx", 0.f);
        dy_ = config.extra.value("dy", 0.f);
        calls_ = 0;
    }
    void Estimate(const FlowInput& a, const FlowInput&, PassImage& flowOut, PassImage* confidenceOut) override;
    int Calls() const { return calls_; }

private:
    float dx_ = 0.f, dy_ = 0.f;
    int calls_ = 0;
};

}  // namespace dlssvid
