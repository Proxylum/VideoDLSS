#include "stages/nr/StubNrBackend.h"

#include <chrono>

#include "stages/nr/shaders/StubNr_cs.h"
#include "util/Error.h"

namespace dlssvid {

namespace {
struct Constants {
    uint32_t width, height;
    float localTone, localStructure;
};
}  // namespace

void StubNrBackend::Init(D3D12Device& device, const NrConfig& config) {
    config_ = config;
    diag_ = {};
    NrMvScale(config.mvScale, 1, 1, 0, 0);  // unit | auto, else throws - as the ngx backend
    ComputeKernel::Desc d;
    d.bytecode = g_StubNrCS;
    d.bytecodeSize = sizeof(g_StubNrCS);
    d.srvCount = 1;
    d.uavCount = 1;
    d.maxDispatches = 16;
    // the timings and memory reading of the ngx backend, so `nr --check` prints the same lines for either: the stub's init is
    // its compute kernel, there is no feature to create (0 ms), its VRAM is what the kernel's buffers take (0 or ? on WARP)
    const auto t0 = std::chrono::steady_clock::now();
    const auto memBefore = device.LocalVideoMemory();
    device_ = &device;
    kernel_ = std::make_unique<ComputeKernel>(device, d);
    diag_.initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    diag_.createMs = 0.0;
    diag_.vramInitBytes = VideoMemoryGrowth(memBefore, device.LocalVideoMemory());
    diag_.vramFeatureBytes = memBefore ? 0 : -1;
    if (memBefore) diag_.vramBudgetBytes = static_cast<int64_t>(memBefore->budget);
    calls_ = 0;
    diag_.gpu = device.AdapterName();
    diag_.architecture = GpuArchitectureFromName(device.AdapterName());
    diag_.driver = NvidiaDriverFromUmd(device.UmdDriverVersion());
    diag_.driverOk = diag_.driver.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor);
    diag_.initResult = diag_.createResult = "stub";
    diag_.ok = true;
}

NrRecreation StubNrBackend::RecreateFeature() {
    if (!kernel_ || !device_) Throw("stub nr: not initialised");
    NrRecreation rec;
    rec.result = rec.releaseResult = "stub";
    rec.ok = true;
    rec.releaseMs = rec.createMs = 0.0;
    if (const auto m = device_->LocalVideoMemory()) rec.vramAfterRelease = rec.vramAfterCreate = static_cast<int64_t>(m->currentUsage);
    return rec;
}

void StubNrBackend::Evaluate(ID3D12GraphicsCommandList* cl, const NrInputs& in, ID3D12Resource* output) {
    if (!kernel_) Throw("stub nr: not initialised");
    if (!in.color || !output) Throw("stub nr: colour input and output are required");
    Constants c{config_.width, config_.height, config_.localTone, config_.localStructure};
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
    return {{"backend", "stub"}, {"local_tone", config_.localTone}, {"local_structure", config_.localStructure}, {"guides", config_.useGuides}};
}

}  // namespace dlssvid
