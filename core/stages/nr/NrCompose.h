#pragma once

#include <cstdint>
#include <memory>

#include "gpu/ComputeKernel.h"

namespace dlssvid {

// Resolve parameters (NrCompose.hlsl, docs/architecture.md stage 6).
struct NrResolveParams {
    uint32_t width = 0, height = 0;          // output (= original) size
    uint32_t workWidth = 0, workHeight = 0;  // model / proxy size
    float transfer = 1.f;                    // ratio transfer strength (mode 1)
    float maxRatio = 4.f;                    // ratio guard (mode 1)
    float temporal = 0.f;                    // 0 = off, else blend weight with the previous frame
    float temporalThreshold = 0.1f;          // colour difference at which the temporal blend is fully rejected
    float skinBlend = 1.f;                   // edit strength inside the face / skin masks
    uint32_t mvWidth = 0, mvHeight = 0;      // guide size of `mv` (for the px scale)
};

struct NrResolveInputs {
    ID3D12Resource* original = nullptr;  // RGBA16F width x height (tone-mapped frame)
    ID3D12Resource* model = nullptr;     // RGBA16F work size
    ID3D12Resource* proxy = nullptr;     // RGBA16F work size (what the model saw); nullptr when work == full size
    ID3D12Resource* prev = nullptr;      // RGBA16F width x height, previous resolved frame (temporal), optional
    ID3D12Resource* mv = nullptr;        // RG32F mv_dlss (temporal), optional
    ID3D12Resource* protect = nullptr;   // R32F [0,1], optional
    ID3D12Resource* skin = nullptr;      // R32F [0,1], optional
};

class NrCompose {
public:
    explicit NrCompose(D3D12Device& device);
    ~NrCompose();
    // All resources in COMMON before and after; `output` (RGBA16F width x height) needs ALLOW_UNORDERED_ACCESS.
    void Resolve(ID3D12GraphicsCommandList* cl, const NrResolveInputs& in, ID3D12Resource* output, const NrResolveParams& p);

private:
    std::unique_ptr<ComputeKernel> kernel_;
};

}  // namespace dlssvid
