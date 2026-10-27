#include "stages/nr/NrCompose.h"

#include <cmath>
#include <vector>

#include "stages/nr/NrLowpass.h"
#include "stages/nr/shaders/NrComposeMask_cs.h"
#include "stages/nr/shaders/NrCompose_cs.h"
#include "stages/nr/shaders/NrEditBox_cs.h"
#include "stages/nr/shaders/NrEditGauss_cs.h"
#include "util/Error.h"

namespace dlssvid {

namespace {
struct Constants {
    uint32_t width, height;
    uint32_t workWidth, workHeight;
    uint32_t mode;
    float transfer, maxRatio, temporal, threshold, skinBlend;
    float mvScaleX, mvScaleY;
    uint32_t flags;
    float gain;
    uint32_t lowSizePacked, lowDiv;  // the mask variant's low band grid (PR-8); 0 in the default variant, as the padding was
};
static_assert(sizeof(Constants) == 64, "NrCompose.hlsl cbuffer: 4 registers");

std::unique_ptr<ComputeKernel> MakeKernel(D3D12Device& device, const void* bytecode, size_t size, uint32_t srvs) {
    ComputeKernel::Desc d;
    d.bytecode = bytecode;
    d.bytecodeSize = size;
    d.srvCount = srvs;
    d.uavCount = 1;
    d.maxDispatches = 16;
    return std::make_unique<ComputeKernel>(device, d);
}
}  // namespace

NrCompose::NrCompose(D3D12Device& device) : device_(device) {
    ComputeKernel::Desc d;
    d.bytecode = g_NrComposeCS;
    d.bytecodeSize = sizeof(g_NrComposeCS);
    d.srvCount = 7;
    d.uavCount = 1;
    d.maxDispatches = 16;
    kernel_ = std::make_unique<ComputeKernel>(device, d);
}

NrCompose::~NrCompose() = default;

void NrCompose::Resolve(ID3D12GraphicsCommandList* cl, const NrResolveInputs& in, ID3D12Resource* output, const NrResolveParams& p) {
    if (!in.original || !in.model || !output) Throw("nr resolve: original, model and output are required");
    if (!std::isfinite(p.gain) || p.gain < 0.f) Throw("nr resolve: the edit strength must be >= 0");
    const bool reduced = in.proxy && (p.workWidth != p.width || p.workHeight != p.height);
    Constants c{};
    c.width = p.width;
    c.height = p.height;
    c.workWidth = reduced ? p.workWidth : p.width;
    c.workHeight = reduced ? p.workHeight : p.height;
    c.mode = reduced ? 1u : 0u;
    c.transfer = p.transfer;
    c.maxRatio = p.maxRatio < 1.f ? 1.f : p.maxRatio;
    c.temporal = p.temporal;
    c.threshold = p.temporalThreshold;
    c.skinBlend = p.skinBlend;
    c.gain = p.gain;
    c.mvScaleX = p.mvWidth ? static_cast<float>(p.width) / static_cast<float>(p.mvWidth) : 1.f;
    c.mvScaleY = p.mvHeight ? static_cast<float>(p.height) / static_cast<float>(p.mvHeight) : 1.f;
    c.flags = (in.prev ? 1u : 0u) | (in.mv ? 2u : 0u) | (in.protect ? 4u : 0u) | (in.skin ? 8u : 0u);
    if (in.control) return ResolveMasked(cl, in, output, p, &c);

    std::vector<ID3D12Resource*> srvs{in.original, in.model, reduced ? in.proxy : nullptr, in.prev, in.mv, in.protect, in.skin};
    for (ID3D12Resource* r : srvs)
        if (r) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    kernel_->Dispatch(cl, &c, sizeof(c), srvs, {output}, (p.width + 7) / 8, (p.height + 7) / 8);
    for (ID3D12Resource* r : srvs)
        if (r) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
}

// The variant with the control mask (PR-8): t7 the mask, t8 the low band of a split one, computed first in the same list.
void NrCompose::ResolveMasked(ID3D12GraphicsCommandList* cl, const NrResolveInputs& in, ID3D12Resource* output, const NrResolveParams& p, const void* base) {
    Constants c = *static_cast<const Constants*>(base);
    const D3D12_RESOURCE_DESC cd = in.control->GetDesc();
    if (cd.Width != p.width || cd.Height != p.height) Throw("nr resolve: the control mask must be the frame size");
    if (in.controlSplit && !in.controlRgb) Throw("nr resolve: only a colour control mask splits the edit");
    c.flags |= (in.controlRgb ? 16u : 0u) | (in.controlSplit ? 32u : 0u);
    const bool reduced = c.mode == 1u;
    if (!maskKernel_) maskKernel_ = MakeKernel(device_, g_NrComposeMaskCS, sizeof(g_NrComposeMaskCS), 9);

    std::vector<ID3D12Resource*> srvs{in.original, in.model, reduced ? in.proxy : nullptr, in.prev, in.mv, in.protect, in.skin, in.control, nullptr};
    for (ID3D12Resource* r : srvs)
        if (r) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (in.controlSplit) {
        const NrLowGrid g = NrLowpassGrid(p.width, p.height, p.lowpassDivisor ? p.lowpassDivisor : NrLowpassDivisor(p.width, p.height));
        if (g.width > 0xFFFF || g.height > 0xFFFF) Throw("nr resolve: the low band grid is too large");
        c.lowSizePacked = g.width | (g.height << 16);
        c.lowDiv = g.div;
        if (!box_ || lowW_ != g.width || lowH_ != g.height) {
            box_ = device_.CreateTexture2D(g.width, g.height, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            low_ = device_.CreateTexture2D(g.width, g.height, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            lowW_ = g.width;
            lowH_ = g.height;
        }
        if (!boxKernel_) boxKernel_ = MakeKernel(device_, g_NrEditBoxCS, sizeof(g_NrEditBoxCS), 3);
        if (!gaussKernel_) gaussKernel_ = MakeKernel(device_, g_NrEditGaussCS, sizeof(g_NrEditGaussCS), 9);
        const uint32_t gx = (g.width + 7) / 8, gy = (g.height + 7) / 8;
        ComputeKernel::Transition(cl, box_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        boxKernel_->Dispatch(cl, &c, sizeof(c), {in.original, in.model, reduced ? in.proxy : nullptr}, {box_.Get()}, gx, gy);
        ComputeKernel::Transition(cl, box_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ComputeKernel::Transition(cl, low_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gaussKernel_->Dispatch(cl, &c, sizeof(c), {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, box_.Get()}, {low_.Get()}, gx, gy);
        ComputeKernel::Transition(cl, low_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        srvs[8] = low_.Get();
        ++lowpassRuns_;
    }
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    maskKernel_->Dispatch(cl, &c, sizeof(c), srvs, {output}, (p.width + 7) / 8, (p.height + 7) / 8);
    for (ID3D12Resource* r : srvs)
        if (r) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (in.controlSplit) ComputeKernel::Transition(cl, box_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
}

}  // namespace dlssvid
