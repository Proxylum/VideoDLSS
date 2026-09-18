#pragma once

#include <memory>

#include "gpu/ComputeKernel.h"
#include "stages/nr/INrBackend.h"

namespace dlssvid {

// Deterministic stand-in for the model (StubNr.hlsl): local contrast + tone scaled by `intensity`.
// Runs on WARP; exercises the stage, resolve, masks, temporal filter, pass count and model resolution
// in tests. Never used as an automatic fallback.
class StubNrBackend final : public INrBackend {
public:
    std::string_view Name() const override { return "stub"; }
    void Init(D3D12Device& device, const NrConfig& config) override;
    void Evaluate(ID3D12GraphicsCommandList* cl, const NrInputs& in, ID3D12Resource* output) override;
    bool UsesGuides() const override { return config_.useGuides; }  // accepted (and ignored by the shader) so the stage's guide plumbing is testable
    const NrDiagnostics& Diagnostics() const override { return diag_; }
    nlohmann::json Describe() const override;
    void Shutdown() override { kernel_.reset(); }
    int Calls() const { return calls_; }
    bool LastHadDepth() const { return lastDepth_; }
    bool LastHadMv() const { return lastMv_; }
    bool LastReset() const { return lastReset_; }

private:
    std::unique_ptr<ComputeKernel> kernel_;
    NrConfig config_;
    NrDiagnostics diag_;
    int calls_ = 0;
    bool lastDepth_ = false, lastMv_ = false, lastReset_ = false;
};

}  // namespace dlssvid
