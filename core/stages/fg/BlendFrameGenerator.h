#pragma once

#include "stages/fg/IFrameGenerator.h"

namespace dlssvid {

// Naive baseline: the k-th intermediate frame is lerp(prev, cur, k / multiplier) on the CPU. Deterministic, no
// GPU — the stage, indexing, multipliers and the CLI are tested with it; it is also the "frame blending" reference
// point in docs/benchmarks.md.
class BlendFrameGenerator final : public IFrameGenerator {
public:
    std::string_view Name() const override { return "blend"; }
    void Init(D3D12Device& device, const FgConfig& config) override;
    void Generate(const FgInputs& in, std::vector<PassImage>& out) override;
    const FgDiagnostics& Diagnostics() const override { return diag_; }
    nlohmann::json Describe() const override;
    int Calls() const { return calls_; }
    bool LastReset() const { return lastReset_; }

private:
    FgConfig config_;
    FgDiagnostics diag_;
    int calls_ = 0;
    bool lastReset_ = false;
};

// RGB image (any float / integer type) -> RGB F32 in [0, 1] (integers scaled), width x height.
PassImage ToRgbF32(const PassImage& img);

}  // namespace dlssvid
