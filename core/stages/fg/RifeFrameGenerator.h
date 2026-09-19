#pragma once

#include <memory>

#include "stages/fg/IFrameGenerator.h"

namespace dlssvid {

// RIFE 4.x (hzwer/Practical-RIFE, MIT) through TensorRT: the ONNX exports of yuvraj108c/rife-onnx from the model
// registry (`rife49` | `rife48` | `rife47`), one fixed-shape engine per padded frame size (multiples of 32), inputs
// img0 / img1 NCHW RGB [0, 1] and `timestep` = k / multiplier. The baseline of ТЗ §8 for the DLSS-G A/B.
class RifeFrameGenerator final : public IFrameGenerator {
public:
    RifeFrameGenerator();
    ~RifeFrameGenerator() override;

    std::string_view Name() const override { return "rife"; }
    static FgAvailability Available();
    void Init(D3D12Device& device, const FgConfig& config) override;
    void Generate(const FgInputs& in, std::vector<PassImage>& out) override;
    const FgDiagnostics& Diagnostics() const override { return diag_; }
    nlohmann::json Describe() const override;
    void Shutdown() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    FgConfig config_;
    FgDiagnostics diag_;
};

}  // namespace dlssvid
