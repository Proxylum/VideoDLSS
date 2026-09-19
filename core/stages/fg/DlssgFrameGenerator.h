#pragma once

#include <memory>

#include "stages/fg/IFrameGenerator.h"

namespace dlssvid {

// DLSS Frame Generation through the NGX API of the DLSS SDK (NVSDK_NGX_Feature_FrameGeneration, "DLSS-FG
// Programming Guide" 310.7): no swapchain, no Streamline. Inputs: the current frame as the backbuffer (RGBA16F,
// or RGBA8 when the runtime refuses 16F), depth_dlss and mv_dlss as guides at their own resolution; output: the
// frame between the previous and the current backbuffer (the runtime keeps the previous one), `multiplier - 1`
// evaluates per pair for multi-frame generation. Camera constants are synthetic (HACK, docs/architecture.md).
class DlssgFrameGenerator final : public IFrameGenerator {
public:
    DlssgFrameGenerator();
    ~DlssgFrameGenerator() override;

    std::string_view Name() const override { return "dlssg"; }
    static FgAvailability Available(const std::filesystem::path& dllDir = {});
    void Init(D3D12Device& device, const FgConfig& config) override;
    void Generate(const FgInputs& in, std::vector<PassImage>& out) override;
    bool UsesGuides() const override { return true; }
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
