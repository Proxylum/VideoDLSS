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
    float gain = 1.f;                        // nr.intensity: the edit strength after the model (PR-7), >= 0; > 1 extrapolates
    uint32_t mvWidth = 0, mvHeight = 0;      // guide size of `mv` (for the px scale)
    uint32_t lowpassDivisor = 0;             // the control mask's G / B split: d in px; 0 = NrLowpassDivisor(width, height)
};

struct NrResolveInputs {
    ID3D12Resource* original = nullptr;  // RGBA16F width x height (tone-mapped frame)
    ID3D12Resource* model = nullptr;     // RGBA16F work size
    ID3D12Resource* proxy = nullptr;     // RGBA16F work size (what the model saw); nullptr when work == full size
    ID3D12Resource* prev = nullptr;      // RGBA16F width x height, previous resolved frame (temporal), optional
    ID3D12Resource* mv = nullptr;        // RG32F mv_dlss (temporal), optional
    ID3D12Resource* protect = nullptr;   // R32F [0,1], optional
    ID3D12Resource* skin = nullptr;      // R32F [0,1], optional
    // PR-8: mask_control at the frame size - R16_UNORM (grey: (v, 1, 1)) or R16G16B16A16_UNORM (R, G, B); nullptr: the default
    // variant, the resolve of PR-7 bit for bit
    ID3D12Resource* control = nullptr;
    bool controlRgb = false;    // control is RGBA (R, G, B), else one grey channel
    bool controlSplit = false;  // G differs from B somewhere: the low band of the edit is computed (needs controlRgb)
};

class NrCompose {
public:
    explicit NrCompose(D3D12Device& device);
    ~NrCompose();
    // All resources in COMMON before and after; `output` (RGBA16F width x height) needs ALLOW_UNORDERED_ACCESS. With a control mask
    // the mask variant runs (its kernels are made at the first such call); with a split one the low band of the edit first
    // (CSEditBox -> CSEditGauss into an RGBA32F grid), in the same command list.
    void Resolve(ID3D12GraphicsCommandList* cl, const NrResolveInputs& in, ID3D12Resource* output, const NrResolveParams& p);
    uint32_t LowpassRuns() const { return lowpassRuns_; }   // resolves that computed the low band (tests)
    ID3D12Resource* LowBand() const { return low_.Get(); }  // the last low band, RGBA32F NrLowpassGrid; nullptr before a split (tests)

private:
    void ResolveMasked(ID3D12GraphicsCommandList* cl, const NrResolveInputs& in, ID3D12Resource* output, const NrResolveParams& p, const void* base);
    D3D12Device& device_;
    std::unique_ptr<ComputeKernel> kernel_, maskKernel_, boxKernel_, gaussKernel_;
    ComPtr<ID3D12Resource> box_, low_;  // the low band grid: the box means, then the Gauss
    uint32_t lowW_ = 0, lowH_ = 0, lowpassRuns_ = 0;
};

}  // namespace dlssvid
