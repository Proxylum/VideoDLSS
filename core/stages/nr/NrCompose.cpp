#include "stages/nr/NrCompose.h"

#include <vector>

#include "stages/nr/shaders/NrCompose_cs.h"
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
    uint32_t pad[3];
};
}  // namespace

NrCompose::NrCompose(D3D12Device& device) {
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
    c.mvScaleX = p.mvWidth ? static_cast<float>(p.width) / static_cast<float>(p.mvWidth) : 1.f;
    c.mvScaleY = p.mvHeight ? static_cast<float>(p.height) / static_cast<float>(p.mvHeight) : 1.f;
    c.flags = (in.prev ? 1u : 0u) | (in.mv ? 2u : 0u) | (in.protect ? 4u : 0u) | (in.skin ? 8u : 0u);

    std::vector<ID3D12Resource*> srvs{in.original, in.model, reduced ? in.proxy : nullptr, in.prev, in.mv, in.protect, in.skin};
    for (ID3D12Resource* r : srvs)
        if (r) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    kernel_->Dispatch(cl, &c, sizeof(c), srvs, {output}, (p.width + 7) / 8, (p.height + 7) / 8);
    for (ID3D12Resource* r : srvs)
        if (r) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
}

}  // namespace dlssvid
