#pragma once

#include <cstdint>

namespace dlssvid {

// A decoded frame that stays on the GPU (NVDEC output): NV12 in CUDA device memory, in the
// primary CUDA context of the D3D12 adapter's device. Valid until the decoder produces the
// next frame.
struct GpuFrame {
    uint32_t width = 0;
    uint32_t height = 0;
    uintptr_t y = 0;      // CUdeviceptr of the luma plane
    uintptr_t uv = 0;     // CUdeviceptr of the interleaved chroma plane (height/2 rows)
    size_t pitch = 0;     // bytes per row (same for both planes)
    int64_t index = -1;
    bool Valid() const { return y != 0 && uv != 0 && width && height; }
};

}  // namespace dlssvid
