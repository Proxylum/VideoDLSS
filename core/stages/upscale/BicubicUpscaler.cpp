#include "stages/upscale/BicubicUpscaler.h"

#include "stages/upscale/shaders/Resample_cs.h"
#include "util/Error.h"

namespace dlssvid {

namespace {
struct ResampleConstants {
    float inSize[2];
    float outSize[2];
    float shift[2];
    uint32_t filter;
    uint32_t pad;
};
}  // namespace

Resampler::Resampler(D3D12Device& device) {
    ComputeKernel::Desc d;
    d.bytecode = g_ResampleCS;
    d.bytecodeSize = sizeof(g_ResampleCS);
    d.srvCount = 1;
    d.uavCount = 1;
    d.maxDispatches = 16;
    d.constantSlotBytes = 256;
    kernel_ = std::make_unique<ComputeKernel>(device, d);
}

Resampler::~Resampler() = default;

void Resampler::Run(ID3D12GraphicsCommandList* cl, ID3D12Resource* src, uint32_t srcW, uint32_t srcH, ID3D12Resource* dst, uint32_t dstW, uint32_t dstH, float shiftX,
                    float shiftY, Filter filter) {
    if (!src || !dst) Throw("Resampler::Run: null resource");
    ResampleConstants c{};
    c.inSize[0] = static_cast<float>(srcW);
    c.inSize[1] = static_cast<float>(srcH);
    c.outSize[0] = static_cast<float>(dstW);
    c.outSize[1] = static_cast<float>(dstH);
    c.shift[0] = shiftX;
    c.shift[1] = shiftY;
    c.filter = static_cast<uint32_t>(filter);
    ComputeKernel::Transition(cl, src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ComputeKernel::Transition(cl, dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    kernel_->Dispatch(cl, &c, sizeof(c), {src}, {dst}, (dstW + 7) / 8, (dstH + 7) / 8);
    ComputeKernel::Transition(cl, src, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    ComputeKernel::Transition(cl, dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
}

void BicubicUpscaler::Init(D3D12Device& device, const UpscalerConfig& config) {
    config_ = config;
    resampler_ = std::make_unique<Resampler>(device);
}

void BicubicUpscaler::Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) {
    if (!resampler_) Throw("BicubicUpscaler: not initialised");
    resampler_->Run(cl, in.color, config_.inputWidth, config_.inputHeight, output, config_.outputWidth, config_.outputHeight, 0.f, 0.f, Resampler::Filter::CatmullRom);
}

nlohmann::json BicubicUpscaler::Describe() const {
    return {{"backend", "bicubic"}, {"filter", "catmull-rom"}, {"input", {config_.inputWidth, config_.inputHeight}}, {"output", {config_.outputWidth, config_.outputHeight}}};
}

}  // namespace dlssvid
