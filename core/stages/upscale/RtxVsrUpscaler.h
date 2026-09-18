#pragma once

#include "stages/upscale/IUpscaler.h"

namespace dlssvid {

// RTX Video Super Resolution (RTX Video SDK 1.1) — the default upscaler of ТЗ §3. The SDK is
// distributed through developer.nvidia.com behind an NVIDIA account and is not available on the
// development machine (TASK-0011), so this is the integration point only: Available() explains
// what to do and Init() fails with the same message; the stage then falls back to `nis`.
// TODO(stage 5b): real integration behind RTX_VIDEO_SDK_ROOT (D3D12 path, artifact-reduction flag).
class RtxVsrUpscaler final : public IUpscaler {
public:
    static UpscalerAvailability Available();
    std::string_view Name() const override { return "rtxvsr"; }
    void Init(D3D12Device& device, const UpscalerConfig& config) override;
    void Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) override;
    nlohmann::json Describe() const override;
};

}  // namespace dlssvid
