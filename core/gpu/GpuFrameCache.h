#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gpu/D3D12Device.h"
#include "gpu/GpuFrame.h"
#include "passes/PassImage.h"
#include "pipeline/Frame.h"

namespace dlssvid {

// Ring buffer of GPU frame slots (ТЗ §4 «Кэш кадров»). Stage 0 stores each frame as one
// tightly packed D3D12 buffer (same byte layout as CpuFrame::data). Later stages will add
// per-pass textures (colour RGBA16F, depth R32F, MV RG16F) to the same slot structure.
class GpuFrameCache {
public:
    struct PassTexture {
        ComPtr<ID3D12Resource> texture;
        uint32_t width = 0, height = 0;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
        std::vector<std::string> channels;
        PixelType type = PixelType::F32;
    };
    enum class Layout { Yuv420p, Nv12 };  // byte layout inside `buffer`
    struct Slot {
        int64_t frameIndex = -1;
        FrameDesc desc;
        ComPtr<ID3D12Resource> buffer;  // default heap, ByteSize() bytes (D3D12_HEAP_FLAG_SHARED when CUDA is attached)
        Layout layout = Layout::Yuv420p;
        void* cudaPtr = nullptr;         // mapped CUDA pointer of `buffer` when CUDA is attached
        void* cudaMemory = nullptr;      // cudaExternalMemory_t
        std::map<std::string, PassTexture> passes;  // "depth_raw" -> R32F, "mv_raw" -> RG32F, colour -> RGBA16F
    };

    // Pass textures (stage 2+): 1 channel -> R32_FLOAT, 2 -> R32G32_FLOAT, 3/4 -> R16G16B16A16_FLOAT.
    void UploadPass(Slot& slot, const std::string& name, const PassImage& img);
    PassImage DownloadPass(const Slot& slot, const std::string& name) const;

    GpuFrameCache(D3D12Device& device, uint32_t slots);
    ~GpuFrameCache();

    // With CUDA attached, slot buffers live on shared heaps and are imported into CUDA so
    // NVDEC frames can be copied in device-to-device (UploadFromGpuFrame).
    void AttachCuda(class CudaInterop* cuda);
    bool CudaAttached() const { return cuda_ != nullptr; }
    void UploadFromGpuFrame(Slot& slot, const GpuFrame& frame);

    uint32_t SlotCount() const { return static_cast<uint32_t>(slots_.size()); }

    // Slot for a frame index (ring: index % slots). Reallocates the buffer if the size changes.
    Slot& Acquire(int64_t frameIndex, const FrameDesc& desc);
    Slot* Find(int64_t frameIndex);

    void Upload(Slot& slot, const CpuFrame& frame);
    void Download(const Slot& slot, CpuFrame& frame);

private:
    void ReleaseCuda(Slot& slot);

    D3D12Device& device_;
    std::vector<Slot> slots_;
    class CudaInterop* cuda_ = nullptr;
};

}  // namespace dlssvid
