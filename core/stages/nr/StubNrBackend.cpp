#include "stages/nr/StubNrBackend.h"

#include "stages/nr/shaders/StubNr_cs.h"
#include "util/Error.h"

namespace dlssvid {

namespace {
struct Constants {
    uint32_t width, height;
    float intensity, localTone, localStructure;
    float pad[3];
};
}  // namespace

void StubNrBackend::Init(D3D12Device& device, const NrConfig& config) {
    config_ = config;
    ComputeKernel::Desc d;
    d.bytecode = g_StubNrCS;
    d.bytecodeSize = sizeof(g_StubNrCS);
    d.srvCount = 1;
    d.uavCount = 1;
    d.maxDispatches = 16;
    kernel_ = std::make_unique<ComputeKernel>(device, d);
    calls_ = 0;
    diag_ = {};
    diag_.gpu = device.AdapterName();
    diag_.architecture = GpuArchitectureFromName(device.AdapterName());
    diag_.driver = NvidiaDriverFromUmd(device.UmdDriverVersion());
    diag_.driverOk = diag_.driver.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor);
    diag_.initResult = diag_.createResult = "stub";
    diag_.ok = true;
}

void StubNrBackend::Evaluate(ID3D12GraphicsCommandList* cl, const NrInputs& in, ID3D12Resource* output) {
    if (!kernel_) Throw("stub nr: not initialised");
    if (!in.color || !output) Throw("stub nr: colour input and output are required");
    Constants c{config_.width, config_.height, config_.intensity, config_.localTone, config_.localStructure, {0, 0, 0}};
    ComputeKernel::Transition(cl, in.color, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    kernel_->Dispatch(cl, &c, sizeof(c), {in.color}, {output}, (config_.width + 7) / 8, (config_.height + 7) / 8);
    ComputeKernel::Transition(cl, in.color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    ++calls_;
    lastDepth_ = in.depth != nullptr;
    lastMv_ = in.mv != nullptr;
    lastReset_ = in.reset;
}

nlohmann::json StubNrBackend::Describe() const {
    return {{"backend", "stub"}, {"intensity", config_.intensity}, {"local_tone", config_.localTone}, {"local_structure", config_.localStructure}, {"guides", config_.useGuides}};
}

}  // namespace dlssvid
