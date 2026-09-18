#include "stages/tonemap/Tonemapper.h"

#include <algorithm>
#include <cmath>

#include "stages/tonemap/shaders/Tonemap_cs.h"
#include "util/Error.h"

namespace dlssvid {

namespace {
struct Constants {
    uint32_t width, height;
    uint32_t curve;
    uint32_t transfer;
    float exposure;
    float pad[3];
};

float SrgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }
float LinearToSrgb(float c) {
    c = std::clamp(c, 0.f, 1.f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}
float PqToLinear(float n) {
    const float m1 = 0.1593017578125f, m2 = 78.84375f, c1 = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;
    const float p = std::pow(std::max(n, 0.f), 1.f / m2);
    const float nits = 10000.f * std::pow(std::max(p - c1, 0.f) / std::max(c2 - c3 * p, 1e-6f), 1.f / m1);
    return nits / 203.f;
}
float HlgToLinear(float e) {
    const float a = 0.17883277f, b = 0.28466892f, c = 0.55991073f;
    const float scene = e <= 0.5f ? e * e / 3.f : (std::exp((e - c) / a) + b) / 12.f;
    return scene * (1000.f / 203.f);
}
float Aces(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.f, 1.f);
}
}  // namespace

std::optional<int> TonemapCurveId(const std::string& s) {
    if (s == "passthrough") return 0;
    if (s == "aces") return 1;
    if (s == "reinhard") return 2;
    return std::nullopt;
}

std::optional<int> TonemapTransferId(const std::string& s) {
    if (s == "srgb") return 0;
    if (s == "linear") return 1;
    if (s == "pq") return 2;
    if (s == "hlg") return 3;
    return std::nullopt;
}

float TonemapReference(float value, const TonemapOptions& o) {
    const auto curve = TonemapCurveId(o.curve);
    const auto transfer = TonemapTransferId(o.inputTransfer);
    if (!curve) Throw("tonemap: --tonemap must be passthrough | aces | reinhard (got '" + o.curve + "')");
    if (!transfer) Throw("tonemap: --input-transfer must be srgb | linear | pq | hlg (got '" + o.inputTransfer + "')");
    if (o.IsIdentity()) return std::clamp(value, 0.f, 1.f);
    float lin;
    switch (*transfer) {
        case 0: lin = SrgbToLinear(std::clamp(value, 0.f, 1.f)); break;
        case 2: lin = PqToLinear(std::clamp(value, 0.f, 1.f)); break;
        case 3: lin = HlgToLinear(std::clamp(value, 0.f, 1.f)); break;
        default: lin = std::max(value, 0.f); break;
    }
    lin *= o.exposure;
    float mapped = lin;
    if (*curve == 1) mapped = Aces(lin);
    else if (*curve == 2) mapped = lin / (1.f + lin);
    return LinearToSrgb(mapped);
}

Tonemapper::Tonemapper(D3D12Device& device) {
    ComputeKernel::Desc d;
    d.bytecode = g_TonemapCS;
    d.bytecodeSize = sizeof(g_TonemapCS);
    d.srvCount = 1;
    d.uavCount = 1;
    d.maxDispatches = 16;
    kernel_ = std::make_unique<ComputeKernel>(device, d);
}

Tonemapper::~Tonemapper() = default;

void Tonemapper::Run(ID3D12GraphicsCommandList* cl, ID3D12Resource* src, ID3D12Resource* dst, uint32_t width, uint32_t height, const TonemapOptions& o) {
    const auto curve = TonemapCurveId(o.curve);
    const auto transfer = TonemapTransferId(o.inputTransfer);
    if (!curve) Throw("tonemap: --tonemap must be passthrough | aces | reinhard (got '" + o.curve + "')");
    if (!transfer) Throw("tonemap: --input-transfer must be srgb | linear | pq | hlg (got '" + o.inputTransfer + "')");
    Constants c{width, height, static_cast<uint32_t>(*curve), static_cast<uint32_t>(*transfer), o.exposure, {0, 0, 0}};
    ComputeKernel::Transition(cl, src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ComputeKernel::Transition(cl, dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    kernel_->Dispatch(cl, &c, sizeof(c), {src}, {dst}, (width + 7) / 8, (height + 7) / 8);
    ComputeKernel::Transition(cl, src, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    ComputeKernel::Transition(cl, dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
}

}  // namespace dlssvid
