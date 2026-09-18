#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace dlssvid {

// Stage 0 keeps decoded frames in their native 8-bit 4:2:0 planar layout so that the
// passthrough path is bit-exact with FFmpeg. Later stages convert to RGBA16F on the GPU.
enum class PixelFormat : uint8_t { Yuv420p = 0 };

std::string_view ToString(PixelFormat f);

struct FrameDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    PixelFormat format = PixelFormat::Yuv420p;

    size_t PlaneCount() const;
    size_t PlaneWidth(size_t plane) const;   // bytes per row (tightly packed)
    size_t PlaneHeight(size_t plane) const;  // rows
    size_t PlaneSize(size_t plane) const;    // bytes
    size_t PlaneOffset(size_t plane) const;  // byte offset inside CpuFrame::data
    size_t ByteSize() const;                 // all planes, tightly packed

    bool operator==(const FrameDesc&) const = default;
};

struct CpuFrame {
    FrameDesc desc;
    int64_t index = -1;  // 0-based decode order
    int64_t pts = 0;     // in the source stream time base
    std::vector<uint8_t> data;

    void Allocate(const FrameDesc& d);
    uint8_t* Plane(size_t plane) { return data.data() + desc.PlaneOffset(plane); }
    const uint8_t* Plane(size_t plane) const { return data.data() + desc.PlaneOffset(plane); }
};

// Copy three planar 4:2:0 planes (arbitrary pitches) into a tightly packed frame. Bit-exact.
void CopyPlanar420(const uint8_t* const src[3], const int pitch[3], uint32_t width, uint32_t height, CpuFrame& out);

// De-interleave NV12 (Y + interleaved UV) into tightly packed YUV420P. Bit-exact.
void Nv12ToYuv420p(const uint8_t* y, int yPitch, const uint8_t* uv, int uvPitch, uint32_t width, uint32_t height,
                   CpuFrame& out);

}  // namespace dlssvid
