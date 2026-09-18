#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "gpu/D3D12Device.h"

namespace dlssvid {

// A compute shader with a fixed root layout: b0 = root constant buffer, t0..tN = SRV table,
// u0..uM = UAV table, s0 = linear clamp sampler, s1 = point clamp sampler. Descriptors and
// constants come from rings so Dispatch() can be recorded many times into one command list
// (the rings are sized for `maxDispatches` per ExecuteAndWait).
class ComputeKernel {
public:
    struct Desc {
        const void* bytecode = nullptr;
        size_t bytecodeSize = 0;
        uint32_t srvCount = 1;
        uint32_t uavCount = 1;
        uint32_t maxDispatches = 64;
        uint32_t constantSlotBytes = 512;  // >= sizeof(constants), multiple of 256
    };

    ComputeKernel(D3D12Device& device, const Desc& desc);
    ~ComputeKernel();
    ComputeKernel(const ComputeKernel&) = delete;
    ComputeKernel& operator=(const ComputeKernel&) = delete;

    // Binds `constants`, the SRVs (nullptr = null descriptor), the UAVs and dispatches. Resources must
    // already be in the states the shader needs (inputs NON_PIXEL_SHADER_RESOURCE, outputs UNORDERED_ACCESS).
    void Dispatch(ID3D12GraphicsCommandList* cl, const void* constants, size_t constantBytes, const std::vector<ID3D12Resource*>& srvs,
                  const std::vector<ID3D12Resource*>& uavs, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ = 1);

    // Resource barrier helpers.
    static void Transition(ID3D12GraphicsCommandList* cl, ID3D12Resource* res, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to);
    static void UavBarrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* res);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dlssvid
