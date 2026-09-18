#include "gpu/GpuFrameCache.h"

#include "util/Error.h"

namespace dlssvid {

GpuFrameCache::GpuFrameCache(D3D12Device& device, uint32_t slots) : device_(device), slots_(slots == 0 ? 1 : slots) {}

GpuFrameCache::Slot& GpuFrameCache::Acquire(int64_t frameIndex, const FrameDesc& desc) {
    if (frameIndex < 0) Throw("GpuFrameCache::Acquire: negative frame index");
    Slot& slot = slots_[static_cast<size_t>(frameIndex % slots_.size())];
    if (!slot.buffer || slot.desc != desc) {
        slot.desc = desc;
        slot.buffer = device_.CreateBuffer(desc.ByteSize(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    }
    slot.frameIndex = frameIndex;
    return slot;
}

GpuFrameCache::Slot* GpuFrameCache::Find(int64_t frameIndex) {
    if (frameIndex < 0) return nullptr;
    Slot& slot = slots_[static_cast<size_t>(frameIndex % slots_.size())];
    return slot.frameIndex == frameIndex && slot.buffer ? &slot : nullptr;
}

void GpuFrameCache::Upload(Slot& slot, const CpuFrame& frame) {
    if (slot.desc != frame.desc) Throw("GpuFrameCache::Upload: frame/slot format mismatch");
    device_.UploadBuffer(slot.buffer.Get(), frame.data.data(), frame.data.size());
}

void GpuFrameCache::Download(const Slot& slot, CpuFrame& frame) {
    frame.Allocate(slot.desc);
    frame.data = device_.ReadbackBuffer(slot.buffer.Get(), slot.desc.ByteSize());
}

}  // namespace dlssvid
