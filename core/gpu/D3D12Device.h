#pragma once

#include <directx/d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace dlssvid {

using Microsoft::WRL::ComPtr;

// DXGI video memory of THIS process on the adapter (IDXGIAdapter3::QueryVideoMemoryInfo, local segment group): per process,
// so a GPU shared with other work (the CI runner) does not skew it - cudaMemGetInfo (TrtEngine) would.
struct VideoMemoryInfo {
    uint64_t currentUsage = 0, budget = 0;
};
// Growth of CurrentUsage between two readings, bytes >= 0 (a release in between reads as 0); -1 when either is missing.
int64_t VideoMemoryGrowth(const std::optional<VideoMemoryInfo>& before, const std::optional<VideoMemoryInfo>& after);

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
    // This process's local video memory on the adapter (nr --check: what the NR feature takes); nullopt when the adapter
    // cannot tell (WARP may not)
    std::optional<VideoMemoryInfo> LocalVideoMemory() const;

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

    // Texture uploads split in two (TASK-0040: the viewport uploaded a 1080p frame in 2.5-3 ms on its render thread,
    // four of them per jump in a 2x2 grid): StageTexture2D - any thread - makes the texture and an upload buffer with
    // the rows in the texture's copy layout; CopyStagedTextures - the thread that owns the queue - copies a batch of
    // them in one command list and one wait.
    struct StagedTexture {
        ComPtr<ID3D12Resource> texture;  // default heap, COMMON state
        ComPtr<ID3D12Resource> upload;   // released after the copy
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    };
    // `reuse`: a texture of the same size and format to write into instead of a new one (the caller's pool).
    StagedTexture StageTexture2D(uint32_t width, uint32_t height, DXGI_FORMAT format, const uint8_t* data, size_t srcRowPitch,
                                 ComPtr<ID3D12Resource> reuse = nullptr);
    void CopyStagedTextures(std::vector<StagedTexture*> textures);

    void ExecuteAndWait(const std::function<void(ID3D12GraphicsCommandList*)>& record);
    void WaitIdle();

private:
    void PickAdapter(const Options& options, ComPtr<IDXGIFactory6>& factory, ComPtr<IDXGIAdapter1>& adapter);

    ComPtr<ID3D12Device> device_;
    ComPtr<IDXGIAdapter3> adapter3_;  // DXGI 1.4 QueryVideoMemoryInfo; null where unsupported
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_ = nullptr;
    uint64_t fenceValue_ = 0;

    // UploadTexture2D: one upload buffer, grown when a larger texture comes and mapped for good (TASK-0040: a new 16.6 MB
    // buffer per 1080p RGBA16F frame cost the viewport ~2 ms per upload). The device is used from one thread at a time.
    ComPtr<ID3D12Resource> uploadStaging_;
    uint64_t uploadStagingBytes_ = 0;
    uint8_t* uploadMapped_ = nullptr;
    // StageTexture2D: upload buffers back from CopyStagedTextures, mapped for good (a new 16 MB upload buffer per frame
    // cost ~2 ms of allocation and page zeroing on the loader thread)
    struct PooledUpload {
        ComPtr<ID3D12Resource> buffer;
        uint64_t bytes = 0;
        uint8_t* mapped = nullptr;
    };
    std::mutex uploadPoolMutex_;
    std::vector<PooledUpload> uploadPool_;
    std::map<ID3D12Resource*, uint8_t*> uploadMappedOf_;  // mapped pointers of buffers handed out (under uploadPoolMutex_)

    std::string adapterName_;
    LUID adapterLuid_{};
    uint32_t vendorId_ = 0;
    uint64_t dedicatedVideoMemory_ = 0;
    uint64_t umdVersion_ = 0;
    bool isWarp_ = false;
};

}  // namespace dlssvid
