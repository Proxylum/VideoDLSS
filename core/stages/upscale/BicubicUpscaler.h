#pragma once

#include <memory>

#include "gpu/ComputeKernel.h"
#include "stages/upscale/IUpscaler.h"

namespace dlssvid {

// GPU resampler (Resample.hlsl): bilinear or Catmull-Rom, any in/out size, optional sub-pixel
// sampling shift. Used for the jitter emulation and as the `bicubic` baseline.
class Resampler {
public:
    enum class Filter { Bilinear = 0, CatmullRom = 1 };
    explicit Resampler(D3D12Device& device);
    ~Resampler();
    // dst pixel (x, y) = src sampled at ((x + 0.5) * src/dst + shift) input px. Both resources in COMMON
    // before and after; `dst` needs D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
    void Run(ID3D12GraphicsCommandList* cl, ID3D12Resource* src, uint32_t srcW, uint32_t srcH, ID3D12Resource* dst, uint32_t dstW, uint32_t dstH,
             float shiftX = 0.f, float shiftY = 0.f, Filter filter = Filter::CatmullRom);

private:
    std::unique_ptr<ComputeKernel> kernel_;
};

// Naive spatial upscaler (Catmull-Rom): deterministic, runs on WARP; the baseline every other
// backend is compared against in docs/benchmarks.md.
class BicubicUpscaler final : public IUpscaler {
public:
    std::string_view Name() const override { return "bicubic"; }
    void Init(D3D12Device& device, const UpscalerConfig& config) override;
    void Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) override;
    nlohmann::json Describe() const override;
    void Shutdown() override { resampler_.reset(); }

private:
    std::unique_ptr<Resampler> resampler_;
    UpscalerConfig config_;
};

}  // namespace dlssvid
