#include "stages/upscale/NisUpscaler.h"

#include <cmath>
#include <cstring>

#include "NIS_Config.h"
#include "stages/upscale/shaders/Nis_scale_cs.h"
#include "stages/upscale/shaders/Nis_sharpen_cs.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace {
constexpr uint32_t kBlockWidth = 32, kBlockHeight = 24;  // must match Nis.hlsl
static_assert(sizeof(NISConfig) <= 256, "NISConfig grew beyond the constant slot");

// The kernels load the 8 taps of a phase as two float4 texels: coef[int2(0|1, phase)] (NIS_Scaler.h),
// so the table is a (kFilterSize / 4) x kPhaseCount RGBA32F texture, one row per phase.
ComPtr<ID3D12Resource> CoefficientTexture(D3D12Device& device, const float (*table)[kFilterSize]) {
    static_assert(kFilterSize == 8, "NIS filter size changed");
    ComPtr<ID3D12Resource> tex = device.CreateTexture2D(static_cast<uint32_t>(kFilterSize / 4), static_cast<uint32_t>(kPhaseCount), DXGI_FORMAT_R32G32B32A32_FLOAT);
    device.UploadTexture2D(tex.Get(), reinterpret_cast<const uint8_t*>(table), kFilterSize * sizeof(float));
    return tex;
}
}  // namespace

void NisUpscaler::Init(D3D12Device& device, const UpscalerConfig& config) {
    Shutdown();
    config_ = config;
    const uint32_t inW = config.inputWidth, inH = config.inputHeight;
    const uint32_t outW = config.outputWidth, outH = config.outputHeight;
    if (!inW || !inH || !outW || !outH) Throw("nis: empty input or output size");
    if (outW < inW || outH < inH) Throw("nis: downscaling is not supported (output must be >= input)");

    // Plan the passes: NVScaler handles ratios 1..2 per pass; larger ratios are chained (x2 first).
    passes_.clear();
    const bool sharpenOnly = config.artifactReductionOnly || (outW == inW && outH == inH);
    if (sharpenOnly) {
        Pass p;
        p.inW = p.outW = inW;
        p.inH = p.outH = inH;
        p.scaler = false;
        NISConfig c{};
        if (!NVSharpenUpdateConfig(c, config.sharpness, 0, 0, inW, inH, inW, inH, 0, 0)) Throw("nis: NVSharpenUpdateConfig failed");
        std::memcpy(p.config, &c, sizeof(c));
        passes_.push_back(p);
    } else {
        uint32_t curW = inW, curH = inH;
        while (true) {
            const double ratio = static_cast<double>(outW) / curW;
            const bool last = ratio <= 2.0 + 1e-6;
            Pass p;
            p.inW = curW;
            p.inH = curH;
            p.outW = last ? outW : curW * 2;
            p.outH = last ? outH : curH * 2;
            p.scaler = true;
            NISConfig c{};
            if (!NVScalerUpdateConfig(c, config.sharpness, 0, 0, p.inW, p.inH, p.inW, p.inH, 0, 0, p.outW, p.outH, p.outW, p.outH))
                Throw("nis: NVScalerUpdateConfig rejected " + std::to_string(p.inW) + "x" + std::to_string(p.inH) + " -> " + std::to_string(p.outW) + "x" +
                      std::to_string(p.outH));
            std::memcpy(p.config, &c, sizeof(c));
            passes_.push_back(p);
            if (last) break;
            curW = p.outW;
            curH = p.outH;
            if (passes_.size() > 4) Throw("nis: scale factor too large");
        }
    }
    if (passes_.size() > 1) {
        const Pass& first = passes_.front();
        intermediate_ = device.CreateTexture2D(first.outW, first.outH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
    coefScale_ = CoefficientTexture(device, coef_scale);
    coefUsm_ = CoefficientTexture(device, coef_usm);

    ComputeKernel::Desc d;
    d.maxDispatches = 8;
    d.constantSlotBytes = 256;
    d.uavCount = 1;
    d.bytecode = g_NisScaleCS;
    d.bytecodeSize = sizeof(g_NisScaleCS);
    d.srvCount = 3;
    scale_ = std::make_unique<ComputeKernel>(device, d);
    d.bytecode = g_NisSharpenCS;
    d.bytecodeSize = sizeof(g_NisSharpenCS);
    d.srvCount = 1;
    sharpen_ = std::make_unique<ComputeKernel>(device, d);
    Log()->info("nis: {}x{} -> {}x{} in {} pass(es), sharpness {:.2f}{}", inW, inH, outW, outH, passes_.size(), config.sharpness, sharpenOnly ? " (sharpen only)" : "");
}

void NisUpscaler::Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) {
    if (passes_.empty() || !scale_) Throw("nis: not initialised");
    ID3D12Resource* src = in.color;
    for (size_t i = 0; i < passes_.size(); ++i) {
        const Pass& p = passes_[i];
        ID3D12Resource* dst = i + 1 == passes_.size() ? output : intermediate_.Get();
        ComputeKernel::Transition(cl, src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ComputeKernel::Transition(cl, dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (p.scaler) {
            ComputeKernel::Transition(cl, coefScale_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ComputeKernel::Transition(cl, coefUsm_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            scale_->Dispatch(cl, p.config, sizeof(NISConfig), {src, coefScale_.Get(), coefUsm_.Get()}, {dst}, (p.outW + kBlockWidth - 1) / kBlockWidth,
                             (p.outH + kBlockHeight - 1) / kBlockHeight);
            ComputeKernel::Transition(cl, coefScale_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            ComputeKernel::Transition(cl, coefUsm_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        } else {
            sharpen_->Dispatch(cl, p.config, sizeof(NISConfig), {src}, {dst}, (p.outW + kBlockWidth - 1) / kBlockWidth, (p.outH + kBlockHeight - 1) / kBlockHeight);
        }
        ComputeKernel::Transition(cl, src, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        ComputeKernel::Transition(cl, dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        src = dst;
    }
}

nlohmann::json NisUpscaler::Describe() const {
    nlohmann::json j = {{"backend", "nis"}, {"version", "1.0.3"}, {"sharpness", config_.sharpness}, {"passes", passes_.size()}};
    if (!passes_.empty()) j["mode"] = passes_.front().scaler ? "scaler" : "sharpen";
    return j;
}

void NisUpscaler::Shutdown() {
    passes_.clear();
    scale_.reset();
    sharpen_.reset();
    coefScale_.Reset();
    coefUsm_.Reset();
    intermediate_.Reset();
}

}  // namespace dlssvid
