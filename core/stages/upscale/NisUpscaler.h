#pragma once

#include <memory>
#include <vector>

#include "gpu/ComputeKernel.h"
#include "stages/upscale/IUpscaler.h"

namespace dlssvid {

// NVIDIA Image Scaling (NIS 1.0.3, MIT — third_party/nis): NVScaler for x1..x2 per pass (chained
// twice above x2), NVSharpen when no scaling is requested («artifact reduction only» analogue).
// Spatial only, deterministic, runs on WARP — the always-available backend and the fallback when
// DLSS is not usable.
class NisUpscaler final : public IUpscaler {
public:
    std::string_view Name() const override { return "nis"; }
    void Init(D3D12Device& device, const UpscalerConfig& config) override;
    void Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) override;
    nlohmann::json Describe() const override;
    void Shutdown() override;

private:
    struct Pass {
        uint32_t inW = 0, inH = 0, outW = 0, outH = 0;
        bool scaler = true;
        alignas(256) unsigned char config[256];  // NISConfig (opaque here: the NIS header is included in the .cpp)
    };
    std::vector<Pass> passes_;
    std::unique_ptr<ComputeKernel> scale_, sharpen_;
    ComPtr<ID3D12Resource> coefScale_, coefUsm_, intermediate_;
    UpscalerConfig config_;
};

}  // namespace dlssvid
