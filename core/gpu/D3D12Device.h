#pragma once

#include <directx/d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dlssvid {

using Microsoft::WRL::ComPtr;

// One D3D12 device shared by the whole pipeline and (later) the viewport.
// Stage 0 exposes only what the frame cache and tests need: resource creation and
// synchronous upload / readback through a direct queue.
class D3D12Device {
public:
    struct Options {
        bool useWarp = false;     // software adapter (deterministic, no GPU needed)
        bool debugLayer = false;  // D3D12 debug layer if the Graphics Tools are installed
    };

    explicit D3D12Device(const Options& options = {});
    ~D3D12Device();

    D3D12Device(const D3D12Device&) = delete;
    D3D12Device& operator=(const D3D12Device&) = delete;

    ID3D12Device* Get() const { return device_.Get(); }
    ID3D12CommandQueue* Queue() const { return queue_.Get(); }
    const std::string& AdapterName() const { return adapterName_; }
    LUID AdapterLuid() const { return adapterLuid_; }
    bool IsWarp() const { return isWarp_; }
    bool IsNvidia() const { return vendorId_ == 0x10DE; }
    uint64_t DedicatedVideoMemory() const { return dedicatedVideoMemory_; }
    // DXGI user-mode driver version (IDXGIAdapter::CheckInterfaceSupport): product.version.subversion.build
    // packed as a LARGE_INTEGER; 0 when unknown (WARP). NVIDIA: 32.0.15.9186 -> driver 591.86 (NvidiaDriverFromUmd).
    uint64_t UmdDriverVersion() const { return umdVersion_; }

    ComPtr<ID3D12Resource> CreateBuffer(uint64_t size, D3D12_HEAP_TYPE heap,
                                        D3D12_RESOURCE_STATES initialState,
                                        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
                                        D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);
    ComPtr<ID3D12Resource> CreateTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format,
                                           D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);

    // Synchronous helpers: record, execute and wait.
    void UploadBuffer(ID3D12Resource* dst, const void* data, uint64_t size);
    std::vector<uint8_t> ReadbackBuffer(ID3D12Resource* src, uint64_t size);
    void UploadTexture2D(ID3D12Resource* dst, const uint8_t* data, size_t srcRowPitch);
    // Returns tightly packed rows (rowPitchOut = bytes per row in the returned vector).
    std::vector<uint8_t> ReadbackTexture2D(ID3D12Resource* src, size_t& rowPitchOut);
    // One texel of a 2D texture (pixel probe): bytes of the texel in the texture's format.
    std::vector<uint8_t> ReadbackTexel(ID3D12Resource* src, uint32_t x, uint32_t y, size_t& bytesPerTexelOut);

    void ExecuteAndWait(const std::function<void(ID3D12GraphicsCommandList*)>& record);
    void WaitIdle();

private:
    void PickAdapter(const Options& options, ComPtr<IDXGIFactory6>& factory, ComPtr<IDXGIAdapter1>& adapter);

    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_ = nullptr;
    uint64_t fenceValue_ = 0;

    std::string adapterName_;
    LUID adapterLuid_{};
    uint32_t vendorId_ = 0;
    uint64_t dedicatedVideoMemory_ = 0;
    uint64_t umdVersion_ = 0;
    bool isWarp_ = false;
};

}  // namespace dlssvid
