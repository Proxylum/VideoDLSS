#pragma once

#include <memory>

#include "stages/nr/INrBackend.h"

namespace dlssvid {

// DLSS 5 Neural Rendering through NGX Feature 18 (`nvngx_dlssnr.dll`), after ComfyUI-DLSS5-NR and
// OptiScaler_DLSSNR (docs/plans/06-nr.md): the NGX core from the driver (gpu/Ngx), the snippet loaded
// from bin/nvidia/, every call into it made from the forwarder module `nvngx.dll_dlssvid.dll` (HACK:
// the snippet checks its caller's module name), parameters DLSSNR.* on the core's parameter block.
// Init logs the GPU architecture, the DLL path + SHA-256 and the CreateFeature(18) result (ТЗ §4)
// and checks the driver (>= 616.56). Guides: depth_dlss (reverse-Z) and mv_dlss (px, backward) as
// sub-rects at their own resolution; without motion vectors the model runs in still-image mode
// (history reset every frame).
class NgxNrBackend final : public INrBackend {
public:
    NgxNrBackend();
    ~NgxNrBackend() override;

    std::string_view Name() const override { return "ngx"; }
    static NrAvailability Available(const std::filesystem::path& dllDir = {});
    void Init(D3D12Device& device, const NrConfig& config) override;
    void Evaluate(ID3D12GraphicsCommandList* cl, const NrInputs& in, ID3D12Resource* output) override;
    bool UsesGuides() const override { return config_.useGuides; }
    const NrDiagnostics& Diagnostics() const override { return diag_; }
    nlohmann::json Describe() const override;
    void Shutdown() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    NrConfig config_;
    NrDiagnostics diag_;
};

}  // namespace dlssvid
