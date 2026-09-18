#pragma once

#include <cstdint>
#include <vector>

#include "gpu/D3D12Device.h"
#include "pipeline/Frame.h"

namespace dlssvid {

// Ring buffer of GPU frame slots (ТЗ §4 «Кэш кадров»). Stage 0 stores each frame as one
// tightly packed D3D12 buffer (same byte layout as CpuFrame::data). Later stages will add
// per-pass textures (colour RGBA16F, depth R32F, MV RG16F) to the same slot structure.
class GpuFrameCache {
public:
    struct Slot {
        int64_t frameIndex = -1;
        FrameDesc desc;
        ComPtr<ID3D12Resource> buffer;  // default heap, ByteSize() bytes
    };

    GpuFrameCache(D3D12Device& device, uint32_t slots);

    uint32_t SlotCount() const { return static_cast<uint32_t>(slots_.size()); }

    // Slot for a frame index (ring: index % slots). Reallocates the buffer if the size changes.
    Slot& Acquire(int64_t frameIndex, const FrameDesc& desc);
    Slot* Find(int64_t frameIndex);

    void Upload(Slot& slot, const CpuFrame& frame);
    void Download(const Slot& slot, CpuFrame& frame);

private:
    D3D12Device& device_;
    std::vector<Slot> slots_;
};

}  // namespace dlssvid
