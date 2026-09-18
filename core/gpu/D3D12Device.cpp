#include "gpu/D3D12Device.h"

#include <directx/d3dx12.h>

#include <cstring>

#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

std::string Narrow(const wchar_t* w) {
    if (!w) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n > 0 ? n - 1 : 0), '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

}  // namespace

D3D12Device::D3D12Device(const Options& options) {
    UINT factoryFlags = 0;
    if (options.debugLayer) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            Log()->info("D3D12 debug layer enabled");
        } else {
            Log()->warn("D3D12 debug layer requested but unavailable (install Graphics Tools)");
        }
    }

    ComPtr<IDXGIFactory6> factory;
    CheckHr(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    ComPtr<IDXGIAdapter1> adapter;
    PickAdapter(options, factory, adapter);

    CheckHr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_)), "D3D12CreateDevice");

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CheckHr(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)), "CreateCommandQueue");
    CheckHr(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)),
            "CreateCommandAllocator");
    CheckHr(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                                       IID_PPV_ARGS(&list_)),
            "CreateCommandList");
    CheckHr(list_->Close(), "CommandList::Close");
    CheckHr(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "CreateFence");
    fenceEvent_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent_) Throw("CreateEvent failed");

    Log()->info("D3D12 device: {}{} (VRAM {} MiB)", adapterName_, isWarp_ ? " [WARP]" : "",
                dedicatedVideoMemory_ / (1024 * 1024));
}

D3D12Device::~D3D12Device() {
    try {
        WaitIdle();
    } catch (...) {
    }
    if (fenceEvent_) CloseHandle(fenceEvent_);
}

void D3D12Device::PickAdapter(const Options& options, ComPtr<IDXGIFactory6>& factory, ComPtr<IDXGIAdapter1>& adapter) {
    auto describe = [&](IDXGIAdapter1* a) {
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        adapterName_ = Narrow(d.Description);
        adapterLuid_ = d.AdapterLuid;
        vendorId_ = d.VendorId;
        dedicatedVideoMemory_ = d.DedicatedVideoMemory;
        isWarp_ = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
    };

    if (!options.useWarp) {
        for (UINT i = 0;
             SUCCEEDED(factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
             ++i) {
            DXGI_ADAPTER_DESC1 d{};
            adapter->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                describe(adapter.Get());
                return;
            }
        }
        Log()->warn("no hardware D3D12 adapter found, falling back to WARP");
    }
    CheckHr(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
    describe(adapter.Get());
    isWarp_ = true;
}

ComPtr<ID3D12Resource> D3D12Device::CreateBuffer(uint64_t size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES initialState,
                                                 D3D12_RESOURCE_FLAGS flags, D3D12_HEAP_FLAGS heapFlags) {
    const CD3DX12_HEAP_PROPERTIES props(heap);
    const CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size, flags);
    ComPtr<ID3D12Resource> res;
    CheckHr(device_->CreateCommittedResource(&props, heapFlags, &desc, initialState, nullptr, IID_PPV_ARGS(&res)),
            "CreateCommittedResource(buffer)");
    return res;
}

ComPtr<ID3D12Resource> D3D12Device::CreateTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format,
                                                    D3D12_RESOURCE_FLAGS flags) {
    const CD3DX12_HEAP_PROPERTIES props(D3D12_HEAP_TYPE_DEFAULT);
    const CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, flags);
    ComPtr<ID3D12Resource> res;
    CheckHr(device_->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             IID_PPV_ARGS(&res)),
            "CreateCommittedResource(texture2d)");
    return res;
}

void D3D12Device::ExecuteAndWait(const std::function<void(ID3D12GraphicsCommandList*)>& record) {
    CheckHr(allocator_->Reset(), "CommandAllocator::Reset");
    CheckHr(list_->Reset(allocator_.Get(), nullptr), "CommandList::Reset");
    record(list_.Get());
    CheckHr(list_->Close(), "CommandList::Close");
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    WaitIdle();
}

void D3D12Device::WaitIdle() {
    const uint64_t value = ++fenceValue_;
    CheckHr(queue_->Signal(fence_.Get(), value), "CommandQueue::Signal");
    if (fence_->GetCompletedValue() < value) {
        CheckHr(fence_->SetEventOnCompletion(value, fenceEvent_), "Fence::SetEventOnCompletion");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
}

void D3D12Device::UploadBuffer(ID3D12Resource* dst, const void* data, uint64_t size) {
    ComPtr<ID3D12Resource> staging = CreateBuffer(size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const CD3DX12_RANGE noRead(0, 0);
    CheckHr(staging->Map(0, &noRead, &mapped), "Map(upload)");
    std::memcpy(mapped, data, static_cast<size_t>(size));
    staging->Unmap(0, nullptr);

    ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        const auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->ResourceBarrier(1, &toCopy);
        cl->CopyBufferRegion(dst, 0, staging.Get(), 0, size);
        const auto toCommon = CD3DX12_RESOURCE_BARRIER::Transition(dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        cl->ResourceBarrier(1, &toCommon);
    });
}

std::vector<uint8_t> D3D12Device::ReadbackBuffer(ID3D12Resource* src, uint64_t size) {
    ComPtr<ID3D12Resource> staging = CreateBuffer(size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        const auto toSrc = CD3DX12_RESOURCE_BARRIER::Transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->ResourceBarrier(1, &toSrc);
        cl->CopyBufferRegion(staging.Get(), 0, src, 0, size);
        const auto toCommon = CD3DX12_RESOURCE_BARRIER::Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        cl->ResourceBarrier(1, &toCommon);
    });
    std::vector<uint8_t> out(static_cast<size_t>(size));
    void* mapped = nullptr;
    const CD3DX12_RANGE readRange(0, static_cast<SIZE_T>(size));
    CheckHr(staging->Map(0, &readRange, &mapped), "Map(readback)");
    std::memcpy(out.data(), mapped, out.size());
    const CD3DX12_RANGE noWrite(0, 0);
    staging->Unmap(0, &noWrite);
    return out;
}

void D3D12Device::UploadTexture2D(ID3D12Resource* dst, const uint8_t* data, size_t srcRowPitch) {
    const D3D12_RESOURCE_DESC desc = dst->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT rows = 0;
    UINT64 rowSize = 0, total = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &rows, &rowSize, &total);

    ComPtr<ID3D12Resource> staging = CreateBuffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    uint8_t* mapped = nullptr;
    const CD3DX12_RANGE noRead(0, 0);
    CheckHr(staging->Map(0, &noRead, reinterpret_cast<void**>(&mapped)), "Map(upload)");
    for (UINT r = 0; r < rows; ++r) {
        std::memcpy(mapped + fp.Offset + static_cast<size_t>(r) * fp.Footprint.RowPitch, data + r * srcRowPitch,
                    static_cast<size_t>(rowSize));
    }
    staging->Unmap(0, nullptr);

    ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        const auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->ResourceBarrier(1, &toCopy);
        const CD3DX12_TEXTURE_COPY_LOCATION dstLoc(dst, 0);
        const CD3DX12_TEXTURE_COPY_LOCATION srcLoc(staging.Get(), fp);
        cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        const auto toCommon = CD3DX12_RESOURCE_BARRIER::Transition(dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        cl->ResourceBarrier(1, &toCommon);
    });
}

std::vector<uint8_t> D3D12Device::ReadbackTexture2D(ID3D12Resource* src, size_t& rowPitchOut) {
    const D3D12_RESOURCE_DESC desc = src->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT rows = 0;
    UINT64 rowSize = 0, total = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &rows, &rowSize, &total);

    ComPtr<ID3D12Resource> staging = CreateBuffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        const auto toSrc = CD3DX12_RESOURCE_BARRIER::Transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->ResourceBarrier(1, &toSrc);
        const CD3DX12_TEXTURE_COPY_LOCATION dstLoc(staging.Get(), fp);
        const CD3DX12_TEXTURE_COPY_LOCATION srcLoc(src, 0);
        cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        const auto toCommon = CD3DX12_RESOURCE_BARRIER::Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        cl->ResourceBarrier(1, &toCommon);
    });

    rowPitchOut = static_cast<size_t>(rowSize);
    std::vector<uint8_t> out(static_cast<size_t>(rowSize) * rows);
    uint8_t* mapped = nullptr;
    const CD3DX12_RANGE readRange(0, static_cast<SIZE_T>(total));
    CheckHr(staging->Map(0, &readRange, reinterpret_cast<void**>(&mapped)), "Map(readback)");
    for (UINT r = 0; r < rows; ++r) {
        std::memcpy(out.data() + static_cast<size_t>(r) * rowSize,
                    mapped + fp.Offset + static_cast<size_t>(r) * fp.Footprint.RowPitch, static_cast<size_t>(rowSize));
    }
    const CD3DX12_RANGE noWrite(0, 0);
    staging->Unmap(0, &noWrite);
    return out;
}

std::vector<uint8_t> D3D12Device::ReadbackTexel(ID3D12Resource* src, uint32_t x, uint32_t y, size_t& bytesPerTexelOut) {
    D3D12_RESOURCE_DESC desc = src->GetDesc();
    // Footprint of a 1x1 texture of the same format gives the texel size and the padded row.
    D3D12_RESOURCE_DESC one = desc;
    one.Width = 1;
    one.Height = 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT rows = 0;
    UINT64 rowSize = 0, total = 0;
    device_->GetCopyableFootprints(&one, 0, 1, 0, &fp, &rows, &rowSize, &total);
    bytesPerTexelOut = static_cast<size_t>(rowSize);
    ComPtr<ID3D12Resource> staging = CreateBuffer(std::max<UINT64>(total, 256), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        const auto toSrc = CD3DX12_RESOURCE_BARRIER::Transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->ResourceBarrier(1, &toSrc);
        const CD3DX12_TEXTURE_COPY_LOCATION dstLoc(staging.Get(), fp);
        const CD3DX12_TEXTURE_COPY_LOCATION srcLoc(src, 0);
        const D3D12_BOX box{x, y, 0, x + 1, y + 1, 1};
        cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &box);
        const auto toCommon = CD3DX12_RESOURCE_BARRIER::Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        cl->ResourceBarrier(1, &toCommon);
    });
    std::vector<uint8_t> out(bytesPerTexelOut);
    void* mapped = nullptr;
    const CD3DX12_RANGE readRange(0, static_cast<SIZE_T>(total));
    CheckHr(staging->Map(0, &readRange, &mapped), "Map(readback texel)");
    std::memcpy(out.data(), static_cast<const uint8_t*>(mapped) + fp.Offset, out.size());
    const CD3DX12_RANGE noWrite(0, 0);
    staging->Unmap(0, &noWrite);
    return out;
}

}  // namespace dlssvid
