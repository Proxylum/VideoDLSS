#include "gpu/GpuFrameCache.h"

#include "util/Error.h"

#ifdef DLSSVID_WITH_CUDA
#include "gpu/CudaInterop.h"
#endif

namespace dlssvid {

GpuFrameCache::GpuFrameCache(D3D12Device& device, uint32_t slots) : device_(device), slots_(slots == 0 ? 1 : slots) {}

GpuFrameCache::~GpuFrameCache() {
    for (Slot& s : slots_) ReleaseCuda(s);
}

void GpuFrameCache::ReleaseCuda(Slot& slot) {
#ifdef DLSSVID_WITH_CUDA
    if (cuda_ && (slot.cudaPtr || slot.cudaMemory)) {
        CudaInterop::ImportedBuffer b;
        b.devicePtr = slot.cudaPtr;
        b.memory = static_cast<cudaExternalMemory_t>(slot.cudaMemory);
        b.size = slot.desc.ByteSize();
        cuda_->Release(b);
    }
#endif
    slot.cudaPtr = nullptr;
    slot.cudaMemory = nullptr;
}

void GpuFrameCache::AttachCuda(CudaInterop* cuda) {
    for (Slot& s : slots_) {
        ReleaseCuda(s);
        s.buffer.Reset();  // re-created on a shared heap at the next Acquire
    }
    cuda_ = cuda;
}

GpuFrameCache::Slot& GpuFrameCache::Acquire(int64_t frameIndex, const FrameDesc& desc) {
    if (frameIndex < 0) Throw("GpuFrameCache::Acquire: negative frame index");
    Slot& slot = slots_[static_cast<size_t>(frameIndex % slots_.size())];
    if (!slot.buffer || slot.desc != desc) {
        ReleaseCuda(slot);
        slot.desc = desc;
        slot.buffer = device_.CreateBuffer(desc.ByteSize(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_NONE,
                                           cuda_ ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE);
#ifdef DLSSVID_WITH_CUDA
        if (cuda_) {
            CudaInterop::ImportedBuffer b = cuda_->ImportBuffer(slot.buffer.Get(), desc.ByteSize());
            slot.cudaPtr = b.devicePtr;
            slot.cudaMemory = b.memory;
        }
#endif
    }
    slot.frameIndex = frameIndex;
    slot.layout = Layout::Yuv420p;
    return slot;
}

void GpuFrameCache::UploadFromGpuFrame(Slot& slot, const GpuFrame& frame) {
#ifdef DLSSVID_WITH_CUDA
    if (!cuda_ || !slot.cudaPtr) Throw("GpuFrameCache::UploadFromGpuFrame: CUDA not attached");
    if (frame.width != slot.desc.width || frame.height != slot.desc.height) Throw("GpuFrameCache::UploadFromGpuFrame: size mismatch");
    cuda_->CopyNv12(slot.cudaPtr, slot.desc.ByteSize(), frame);
    slot.layout = Layout::Nv12;
#else
    (void)slot;
    (void)frame;
    Throw("GpuFrameCache::UploadFromGpuFrame: built without CUDA");
#endif
}

GpuFrameCache::Slot* GpuFrameCache::Find(int64_t frameIndex) {
    if (frameIndex < 0) return nullptr;
    Slot& slot = slots_[static_cast<size_t>(frameIndex % slots_.size())];
    return slot.frameIndex == frameIndex && slot.buffer ? &slot : nullptr;
}

void GpuFrameCache::Upload(Slot& slot, const CpuFrame& frame) {
    if (slot.desc != frame.desc) Throw("GpuFrameCache::Upload: frame/slot format mismatch");
    device_.UploadBuffer(slot.buffer.Get(), frame.data.data(), frame.data.size());
    slot.layout = Layout::Yuv420p;
}

void GpuFrameCache::Download(const Slot& slot, CpuFrame& frame) {
    frame.Allocate(slot.desc);
    std::vector<uint8_t> bytes = device_.ReadbackBuffer(slot.buffer.Get(), slot.desc.ByteSize());
    if (slot.layout == Layout::Nv12) {
        const uint32_t w = slot.desc.width, h = slot.desc.height;
        Nv12ToYuv420p(bytes.data(), static_cast<int>(w), bytes.data() + static_cast<size_t>(w) * h, static_cast<int>(w), w, h, frame);
    } else {
        frame.data = std::move(bytes);
    }
}

namespace {
// Texture layout for a pass image: (format, channels stored, sample type stored).
struct TexLayout {
    DXGI_FORMAT format;
    size_t channels;
    PixelType type;
};
TexLayout LayoutFor(const PassImage& img) {
    switch (img.channels.size()) {
        case 1: return {DXGI_FORMAT_R32_FLOAT, 1, PixelType::F32};
        case 2: return {DXGI_FORMAT_R32G32_FLOAT, 2, PixelType::F32};
        default: return {DXGI_FORMAT_R16G16B16A16_FLOAT, 4, PixelType::F16};
    }
}
}  // namespace

void GpuFrameCache::UploadPass(Slot& slot, const std::string& name, const PassImage& img) {
    if (img.Empty()) Throw("GpuFrameCache::UploadPass: empty image");
    const TexLayout lay = LayoutFor(img);
    PassTexture& pt = slot.passes[name];
    if (!pt.texture || pt.width != img.width || pt.height != img.height || pt.format != lay.format) {
        pt.texture = device_.CreateTexture2D(img.width, img.height, lay.format);
        pt.width = img.width;
        pt.height = img.height;
        pt.format = lay.format;
    }
    pt.channels = img.channels;
    pt.type = img.type;
    // Repack into the texture layout (pad to 4 channels for colour, convert sample type).
    PassImage staged;
    std::vector<std::string> ch(lay.channels);
    for (size_t c = 0; c < lay.channels; ++c) ch[c] = c < img.channels.size() ? img.channels[c] : (c == 3 ? "A" : "pad");
    staged.Allocate(img.width, img.height, lay.type, ch);
    for (uint32_t y = 0; y < img.height; ++y)
        for (uint32_t x = 0; x < img.width; ++x)
            for (size_t c = 0; c < lay.channels; ++c) staged.Set(x, y, c, c < img.channels.size() ? img.Get(x, y, c) : (c == 3 ? 1.f : 0.f));
    device_.UploadTexture2D(pt.texture.Get(), staged.data.data(), staged.RowBytes());
}

PassImage GpuFrameCache::DownloadPass(const Slot& slot, const std::string& name) const {
    auto it = slot.passes.find(name);
    if (it == slot.passes.end() || !it->second.texture) Throw("GpuFrameCache::DownloadPass: no pass '" + name + "' in slot");
    const PassTexture& pt = it->second;
    size_t pitch = 0;
    std::vector<uint8_t> bytes = device_.ReadbackTexture2D(pt.texture.Get(), pitch);
    const size_t stored = pt.format == DXGI_FORMAT_R32_FLOAT ? 1 : pt.format == DXGI_FORMAT_R32G32_FLOAT ? 2 : 4;
    const PixelType storedType = pt.format == DXGI_FORMAT_R16G16B16A16_FLOAT ? PixelType::F16 : PixelType::F32;
    PassImage staged;
    std::vector<std::string> ch(stored);
    for (size_t c = 0; c < stored; ++c) ch[c] = "c" + std::to_string(c);
    staged.Allocate(pt.width, pt.height, storedType, ch);
    staged.data = std::move(bytes);
    PassImage out;
    out.Allocate(pt.width, pt.height, pt.type, pt.channels);
    for (uint32_t y = 0; y < pt.height; ++y)
        for (uint32_t x = 0; x < pt.width; ++x)
            for (size_t c = 0; c < pt.channels.size(); ++c) out.Set(x, y, c, staged.Get(x, y, c));
    return out;
}

}  // namespace dlssvid
