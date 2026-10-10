#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dlssvid {

// Decoded frames keep their native planar YUV layout so that the passthrough path is bit-exact with FFmpeg and
// nothing is lost before the stages convert to RGBA16F on the GPU. The 8-bit formats hold one byte per sample; the
// 16-bit ones hold little-endian 16-bit words with `FrameDesc::bitDepth` significant bits, LSB-aligned (10-bit video:
// 0 .. 1023), the layout FFmpeg calls yuv420p10le / yuv422p10le / yuv444p10le.
enum class PixelFormat : uint8_t { Yuv420p = 0, Yuv422p, Yuv444p, Yuv420p16, Yuv422p16, Yuv444p16 };

std::string_view ToString(PixelFormat f);
// Chroma subsampling as the usual three digits: "420" | "422" | "444".
std::string_view ChromaName(PixelFormat f);
// Chroma shifts: 4:2:0 -> (1, 1), 4:2:2 -> (1, 0), 4:4:4 -> (0, 0).
int ChromaShiftX(PixelFormat f);
int ChromaShiftY(PixelFormat f);
// Bytes per sample: 1 for the 8-bit formats, 2 for the 16-bit ones.
size_t BytesPerSample(PixelFormat f);
// The format for a chroma subsampling and a bit depth (8 -> the 8-bit formats; 10 | 12 | 16 -> the 16-bit ones).
PixelFormat PixelFormatFor(std::string_view chroma, int bitDepth);

struct FrameDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    PixelFormat format = PixelFormat::Yuv420p;
    uint8_t bitDepth = 8;  // significant bits per sample: 8 for the 8-bit formats, 10 | 12 | 16 for the 16-bit ones

    size_t PlaneCount() const;
    size_t PlaneWidth(size_t plane) const;   // bytes per row (tightly packed)
    size_t PlaneCols(size_t plane) const;    // samples per row
    size_t PlaneHeight(size_t plane) const;  // rows
    size_t PlaneSize(size_t plane) const;    // bytes
    size_t PlaneOffset(size_t plane) const;  // byte offset inside CpuFrame::data
    size_t ByteSize() const;                 // all planes, tightly packed
    bool HighBitDepth() const { return BytesPerSample(format) == 2; }
    // "yuv420p", "yuv444p10", "yuv422p12", "yuv420p16": the format with its bit depth (FFmpeg's spelling without "le").
    std::string Name() const;
    // "8-bit 4:2:0", "10-bit 4:4:4": for logs and reports.
    std::string Describe() const;

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
    uint16_t* Plane16(size_t plane) { return reinterpret_cast<uint16_t*>(Plane(plane)); }
    const uint16_t* Plane16(size_t plane) const { return reinterpret_cast<const uint16_t*>(Plane(plane)); }
};

// Copy three planar 4:2:0 planes (arbitrary pitches) into a tightly packed frame. Bit-exact.
void CopyPlanar420(const uint8_t* const src[3], const int pitch[3], uint32_t width, uint32_t height, CpuFrame& out);

// Copy three planes of any planar format (arbitrary pitches) into a tightly packed frame of `desc`. For the 16-bit
// formats `shiftRight` moves MSB-aligned samples (P010-like layouts, NVDEC's 16-bit output) down to LSB alignment.
void CopyPlanar(const uint8_t* const src[3], const int pitch[3], const FrameDesc& desc, CpuFrame& out, int shiftRight = 0);

// De-interleave NV12 (Y + interleaved UV) into tightly packed YUV420P. Bit-exact.
void Nv12ToYuv420p(const uint8_t* y, int yPitch, const uint8_t* uv, int uvPitch, uint32_t width, uint32_t height,
                   CpuFrame& out);

// De-interleave a 16-bit semi-planar frame (P010 / P016 / P210 / P216 / P410 / P416: Y + interleaved UV words) into
// the tightly packed planar 16-bit format of `desc`, shifting MSB-aligned samples down by `shiftRight`.
void SemiPlanar16ToPlanar(const uint8_t* y, int yPitch, const uint8_t* uv, int uvPitch, const FrameDesc& desc, CpuFrame& out, int shiftRight);

}  // namespace dlssvid
