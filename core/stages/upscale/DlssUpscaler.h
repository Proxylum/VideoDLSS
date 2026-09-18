#pragma once

#include <memory>

#include "stages/upscale/IUpscaler.h"

namespace dlssvid {

// DLSS Super Resolution through the NGX SDK (D3D12). The input is a decoded video frame, so the
// camera jitter DLSS relies on is emulated by the stage (sub-pixel resample, HACK — ТЗ §3) and
// reported through UpscaleInputs::jitter. Requires an RTX GPU, a driver with DLSS and
// nvngx_dlss.dll next to the executable (bin/nvidia/, docs/dll-setup.md). Built only with the
// DLSS SDK (DLSSVID_WITH_DLSS); otherwise Available() explains how to enable it.
class DlssUpscaler final : public IUpscaler {
public:
    DlssUpscaler();
    ~DlssUpscaler() override;
    static UpscalerAvailability Available(const std::filesystem::path& dllDir = {});

    std::string_view Name() const override { return "dlss"; }
    void Init(D3D12Device& device, const UpscalerConfig& config) override;
    void Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) override;
    bool WantsJitter() const override { return config_.useJitter; }
    bool WantsDepthAndMv() const override { return true; }
    nlohmann::json Describe() const override;
    void Shutdown() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    UpscalerConfig config_;
};

}  // namespace dlssvid
