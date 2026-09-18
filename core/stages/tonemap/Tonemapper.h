#pragma once

#include <memory>
#include <optional>
#include <string>

#include "gpu/ComputeKernel.h"

namespace dlssvid {

// Tonemap options (ТЗ §4 «Tonemap»: HDR / linear colour -> SDR sRGB; passthrough for SDR sources).
struct TonemapOptions {
    std::string curve = "passthrough";    // passthrough | aces | reinhard
    std::string inputTransfer = "srgb";   // srgb (decoded video) | linear | pq | hlg
    float exposure = 1.f;                 // multiplies linear light
    bool IsIdentity() const { return curve == "passthrough" && inputTransfer == "srgb" && exposure == 1.f; }
};
std::optional<int> TonemapCurveId(const std::string& s);
std::optional<int> TonemapTransferId(const std::string& s);
// CPU reference of the shader for one value (tests and documentation): input in the given transfer -> sRGB [0,1].
float TonemapReference(float value, const TonemapOptions& options);

// Compute tonemapper (Tonemap.hlsl): RGBA16F -> RGBA16F display-referred sRGB [0, 1], alpha = 1. Runs on WARP.
class Tonemapper {
public:
    explicit Tonemapper(D3D12Device& device);
    ~Tonemapper();
    // Both resources in COMMON before and after; `dst` needs D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
    void Run(ID3D12GraphicsCommandList* cl, ID3D12Resource* src, ID3D12Resource* dst, uint32_t width, uint32_t height, const TonemapOptions& options);

private:
    std::unique_ptr<ComputeKernel> kernel_;
};

}  // namespace dlssvid
