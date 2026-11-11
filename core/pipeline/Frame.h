#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <string_view>
#include <vector>

namespace dlssvid {

// Decoded frames keep their native planar YUV layout so that the passthrough path is bit-exact with FFmpeg (up to the
// permutation of the samples that shows a rotated stream upright, OrientFrame) and nothing is lost before the stages convert to
// RGBA16F on the GPU. The 8-bit formats hold one byte per sample; the
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

// How a stored frame is shown (TASK-0072: a phone video kept landscape with a display matrix): turned clockwise by `rotate`
// degrees (0 | 90 | 180 | 270), then mirrored left-right when `hflip`. The decoder applies it: the stages, the viewport and the
// result see the frame as players show it.
struct DisplayOrientation {
    uint16_t rotate = 0;
    bool hflip = false;

    bool Identity() const { return rotate == 0 && !hflip; }
    bool SwapsAxes() const { return rotate == 90 || rotate == 270; }
    // The orientation that undoes this one: a turn back; a mirrored orientation is its own inverse.
    DisplayOrientation Inverse() const { return hflip ? *this : DisplayOrientation{static_cast<uint16_t>((360 - rotate) % 360), false}; }
    // "" | "r90" | "r180" | "r270" | "h" | "r90h" | "r180h" | "r270h": in a pass's source format (fingerprints) and the settings note.
    std::string Token() const;
    // "rotated 90 deg clockwise", "mirrored", "rotated 180 deg and mirrored": for logs and reports; "" for the identity.
    std::string Describe() const;

    bool operator==(const DisplayOrientation&) const = default;
};

// The orientation a token names ("" is the identity); nullopt for anything else.
std::optional<DisplayOrientation> OrientationFromToken(std::string_view token);

// The proportions of a pixel (TASK-0073: an anamorphic source - DVD 720x480 at 32:27 is shown 16:9): width : height of one sample. 0:1
// (and 1:1) is square. The stages work on the stored grid; `process` stretches its result to square pixels when it encodes it.
struct PixelAspect {
    uint32_t num = 0;
    uint32_t den = 1;

    bool Square() const { return num == 0 || num == den; }
    double Value() const { return Square() ? 1.0 : static_cast<double>(num) / den; }
    PixelAspect Inverse() const { return Square() ? PixelAspect{} : PixelAspect{den, num}; }
    // "" (square) | "32:27": in logs, the settings note
    std::string Token() const;
    // The frame a w x h grid of these pixels is shown as, in square pixels, nothing narrowed: wider than 1 - the width grows, narrower -
    // the height grows (720x480 at 8:9 -> 720x540); even sides. The grid itself when square.
    std::pair<uint32_t, uint32_t> DisplaySize(uint32_t w, uint32_t h) const;

    bool operator==(const PixelAspect& o) const { return Square() ? o.Square() : num == o.num && den == o.den; }
};

// num:den reduced; 0, negative or 1:1 - square; outside 1/4 .. 4 (a broken file: 1000:1) - square, `why` says so.
PixelAspect PixelAspectFrom(int64_t num, int64_t den, std::string* why = nullptr);

// The layout of a stored frame `desc` once shown with `o`: the sides swapped for 90 / 270. The chroma grid must map onto itself for
// the samples to be permuted exactly; where it does not - 4:2:2 turned by 90 / 270 (4:4:0 is not a format here), or an odd side of a
// subsampled axis that `o` reverses - the frame becomes 4:4:4 at the same bit depth, its chroma repeated (still exact).
FrameDesc OrientedDesc(const FrameDesc& desc, DisplayOrientation o);

struct CpuFrame;
// `out` = `in` shown with `o` (layout OrientedDesc(in.desc, o)); index and pts carried over.
void OrientFrame(const CpuFrame& in, DisplayOrientation o, CpuFrame& out);

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
