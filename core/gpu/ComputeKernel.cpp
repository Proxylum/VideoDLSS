#include "gpu/ComputeKernel.h"

#include <cstring>
#include <directx/d3dx12.h>

#include "util/Error.h"

namespace dlssvid {

struct ComputeKernel::Impl {
    D3D12Device& device;
    Desc desc;
    ComPtr<ID3D12RootSignature> rootSig;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12DescriptorHeap> heap;
    uint32_t inc = 0;
    uint32_t descriptorsPerDispatch = 0;
    uint32_t cursor = 0;
    ComPtr<ID3D12Resource> constants;
    uint8_t* constantsMapped = nullptr;
    uint32_t constantCursor = 0;

    explicit Impl(D3D12Device& d, const Desc& de) : device(d), desc(de) {}

    void WriteSrv(D3D12_CPU_DESCRIPTOR_HANDLE handle, ID3D12Resource* res) {
        if (res) {
            device.Get()->CreateShaderResourceView(res, nullptr, handle);
            return;
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC null{};
        null.Format = DXGI_FORMAT_R32_FLOAT;
        null.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        null.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        null.Texture2D.MipLevels = 1;
        device.Get()->CreateShaderResourceView(nullptr, &null, handle);
    }
    void WriteUav(D3D12_CPU_DESCRIPTOR_HANDLE handle, ID3D12Resource* res) {
        if (res) {
            device.Get()->CreateUnorderedAccessView(res, nullptr, nullptr, handle);
            return;
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC null{};
        null.Format = DXGI_FORMAT_R32_FLOAT;
        null.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device.Get()->CreateUnorderedAccessView(nullptr, nullptr, &null, handle);
    }
};

ComputeKernel::ComputeKernel(D3D12Device& device, const Desc& desc) : impl_(std::make_unique<Impl>(device, desc)) {
    Impl& im = *impl_;
    ID3D12Device* dev = device.Get();

    CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, std::max(1u, desc.srvCount), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, std::max(1u, desc.uavCount), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    CD3DX12_ROOT_PARAMETER1 params[3];
    params[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE);
    params[1].InitAsDescriptorTable(1, &ranges[0]);
    params[2].InitAsDescriptorTable(1, &ranges[1]);
    CD3DX12_STATIC_SAMPLER_DESC samplers[2];
    samplers[0].Init(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
    samplers[1].Init(1, D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rs;
    rs.Init_1_1(3, params, 2, samplers, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> blob, error;
    const HRESULT hr = D3DX12SerializeVersionedRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1_1, &blob, &error);
    if (FAILED(hr)) Throw(std::string("root signature: ") + (error ? static_cast<const char*>(error->GetBufferPointer()) : "?"));
    CheckHr(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&im.rootSig)), "CreateRootSignature(compute)");

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = im.rootSig.Get();
    pso.CS = {desc.bytecode, desc.bytecodeSize};
    CheckHr(dev->CreateComputePipelineState(&pso, IID_PPV_ARGS(&im.pso)), "CreateComputePipelineState");

    im.descriptorsPerDispatch = std::max(1u, desc.srvCount) + std::max(1u, desc.uavCount);
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = im.descriptorsPerDispatch * std::max(1u, desc.maxDispatches);
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CheckHr(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&im.heap)), "CreateDescriptorHeap(compute)");
    im.inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    const uint64_t constantBytes = static_cast<uint64_t>(desc.constantSlotBytes) * std::max(1u, desc.maxDispatches);
    im.constants = device.CreateBuffer(constantBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    const CD3DX12_RANGE noRead(0, 0);
    CheckHr(im.constants->Map(0, &noRead, reinterpret_cast<void**>(&im.constantsMapped)), "Map(compute constants)");
}

ComputeKernel::~ComputeKernel() {
    if (impl_ && impl_->constants && impl_->constantsMapped) impl_->constants->Unmap(0, nullptr);
}

void ComputeKernel::Dispatch(ID3D12GraphicsCommandList* cl, const void* constants, size_t constantBytes, const std::vector<ID3D12Resource*>& srvs,
                             const std::vector<ID3D12Resource*>& uavs, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) {
    Impl& im = *impl_;
    if (constantBytes > im.desc.constantSlotBytes) Throw("ComputeKernel: constants larger than the slot");
    const uint32_t slot = im.cursor++ % std::max(1u, im.desc.maxDispatches);
    const uint32_t base = slot * im.descriptorsPerDispatch;
    CD3DX12_CPU_DESCRIPTOR_HANDLE cpu(im.heap->GetCPUDescriptorHandleForHeapStart(), static_cast<INT>(base), im.inc);
    CD3DX12_GPU_DESCRIPTOR_HANDLE gpu(im.heap->GetGPUDescriptorHandleForHeapStart(), static_cast<INT>(base), im.inc);
    const uint32_t srvCount = std::max(1u, im.desc.srvCount), uavCount = std::max(1u, im.desc.uavCount);
    for (uint32_t i = 0; i < srvCount; ++i) im.WriteSrv(CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu, static_cast<INT>(i), im.inc), i < srvs.size() ? srvs[i] : nullptr);
    for (uint32_t i = 0; i < uavCount; ++i)
        im.WriteUav(CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu, static_cast<INT>(srvCount + i), im.inc), i < uavs.size() ? uavs[i] : nullptr);

    const uint32_t cslot = im.constantCursor++ % std::max(1u, im.desc.maxDispatches);
    uint8_t* dst = im.constantsMapped + static_cast<size_t>(cslot) * im.desc.constantSlotBytes;
    if (constants && constantBytes) std::memcpy(dst, constants, constantBytes);
    const D3D12_GPU_VIRTUAL_ADDRESS cbv = im.constants->GetGPUVirtualAddress() + static_cast<uint64_t>(cslot) * im.desc.constantSlotBytes;

    ID3D12DescriptorHeap* heaps[] = {im.heap.Get()};
    cl->SetDescriptorHeaps(1, heaps);
    cl->SetComputeRootSignature(im.rootSig.Get());
    cl->SetPipelineState(im.pso.Get());
    cl->SetComputeRootConstantBufferView(0, cbv);
    cl->SetComputeRootDescriptorTable(1, gpu);
    cl->SetComputeRootDescriptorTable(2, CD3DX12_GPU_DESCRIPTOR_HANDLE(gpu, static_cast<INT>(srvCount), im.inc));
    cl->Dispatch(std::max(1u, groupsX), std::max(1u, groupsY), std::max(1u, groupsZ));
}

void ComputeKernel::Transition(ID3D12GraphicsCommandList* cl, ID3D12Resource* res, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    if (!res || from == to) return;
    const auto b = CD3DX12_RESOURCE_BARRIER::Transition(res, from, to);
    cl->ResourceBarrier(1, &b);
}

void ComputeKernel::UavBarrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* res) {
    const auto b = CD3DX12_RESOURCE_BARRIER::UAV(res);
    cl->ResourceBarrier(1, &b);
}

}  // namespace dlssvid
