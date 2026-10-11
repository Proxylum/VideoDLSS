#include "pipeline/Frame.h"

#include <cstring>

#include "util/Error.h"

namespace dlssvid {

std::string_view ToString(PixelFormat f) {
    switch (f) {
        case PixelFormat::Yuv420p: return "yuv420p";
        case PixelFormat::Yuv422p: return "yuv422p";
        case PixelFormat::Yuv444p: return "yuv444p";
        case PixelFormat::Yuv420p16: return "yuv420p16";
        case PixelFormat::Yuv422p16: return "yuv422p16";
        case PixelFormat::Yuv444p16: return "yuv444p16";
    }
    return "unknown";
}

std::string_view ChromaName(PixelFormat f) {
    switch (f) {
        case PixelFormat::Yuv420p:
        case PixelFormat::Yuv420p16: return "420";
        case PixelFormat::Yuv422p:
        case PixelFormat::Yuv422p16: return "422";
        case PixelFormat::Yuv444p:
        case PixelFormat::Yuv444p16: return "444";
    }
    return "420";
}

int ChromaShiftX(PixelFormat f) { return ChromaName(f) == "444" ? 0 : 1; }
int ChromaShiftY(PixelFormat f) { return ChromaName(f) == "420" ? 1 : 0; }

size_t BytesPerSample(PixelFormat f) {
    switch (f) {
        case PixelFormat::Yuv420p16:
        case PixelFormat::Yuv422p16:
        case PixelFormat::Yuv444p16: return 2;
        default: return 1;
    }
}

PixelFormat PixelFormatFor(std::string_view chroma, int bitDepth) {
    const bool wide = bitDepth > 8;
    if (chroma == "444") return wide ? PixelFormat::Yuv444p16 : PixelFormat::Yuv444p;
    if (chroma == "422") return wide ? PixelFormat::Yuv422p16 : PixelFormat::Yuv422p;
    if (chroma == "420") return wide ? PixelFormat::Yuv420p16 : PixelFormat::Yuv420p;
    Throw("PixelFormatFor: chroma must be 420 | 422 | 444 (got '" + std::string(chroma) + "')");
}

size_t FrameDesc::PlaneCount() const { return 3; }

size_t FrameDesc::PlaneCols(size_t plane) const {
    if (plane == 0) return width;
    return (width + (1u << ChromaShiftX(format)) - 1) >> ChromaShiftX(format);
}

size_t FrameDesc::PlaneWidth(size_t plane) const { return PlaneCols(plane) * BytesPerSample(format); }

size_t FrameDesc::PlaneHeight(size_t plane) const {
    if (plane == 0) return height;
    return (height + (1u << ChromaShiftY(format)) - 1) >> ChromaShiftY(format);
}

size_t FrameDesc::PlaneSize(size_t plane) const { return PlaneWidth(plane) * PlaneHeight(plane); }

size_t FrameDesc::PlaneOffset(size_t plane) const {
    size_t off = 0;
    for (size_t p = 0; p < plane; ++p) off += PlaneSize(p);
    return off;
}

size_t FrameDesc::ByteSize() const { return PlaneOffset(PlaneCount()); }

std::string FrameDesc::Name() const {
    std::string n = "yuv" + std::string(ChromaName(format)) + "p";
    if (HighBitDepth()) n += std::to_string(bitDepth);
    return n;
}

std::string FrameDesc::Describe() const {
    const std::string_view c = ChromaName(format);
    return std::to_string(HighBitDepth() ? bitDepth : 8) + "-bit " + std::string(1, c[0]) + ":" + std::string(1, c[1]) + ":" + std::string(1, c[2]);
}

void CpuFrame::Allocate(const FrameDesc& d) {
    if (d.width == 0 || d.height == 0) Throw("CpuFrame::Allocate: zero-sized frame");
    if (d.HighBitDepth() ? (d.bitDepth != 10 && d.bitDepth != 12 && d.bitDepth != 16) : d.bitDepth != 8)
        Throw("CpuFrame::Allocate: bit depth " + std::to_string(d.bitDepth) + " does not fit " + std::string(ToString(d.format)));
    desc = d;
    data.resize(d.ByteSize());
}

void CopyPlanar420(const uint8_t* const src[3], const int pitch[3], uint32_t width, uint32_t height, CpuFrame& out) {
    out.Allocate(FrameDesc{width, height, PixelFormat::Yuv420p});
    for (size_t p = 0; p < 3; ++p) {
        const size_t rowBytes = out.desc.PlaneWidth(p);
        const size_t rows = out.desc.PlaneHeight(p);
        uint8_t* dst = out.Plane(p);
        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(dst + r * rowBytes, src[p] + static_cast<ptrdiff_t>(r) * pitch[p], rowBytes);
        }
    }
}

void CopyPlanar(const uint8_t* const src[3], const int pitch[3], const FrameDesc& desc, CpuFrame& out, int shiftRight) {
    out.Allocate(desc);
    for (size_t p = 0; p < 3; ++p) {
        const size_t rowBytes = desc.PlaneWidth(p);
        const size_t rows = desc.PlaneHeight(p);
        if (!desc.HighBitDepth() || shiftRight == 0) {
            uint8_t* dst = out.Plane(p);
            for (size_t r = 0; r < rows; ++r) std::memcpy(dst + r * rowBytes, src[p] + static_cast<ptrdiff_t>(r) * pitch[p], rowBytes);
            continue;
        }
        const size_t cols = desc.PlaneCols(p);
        uint16_t* dst = out.Plane16(p);
        for (size_t r = 0; r < rows; ++r) {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(src[p] + static_cast<ptrdiff_t>(r) * pitch[p]);
            uint16_t* d = dst + r * cols;
            for (size_t c = 0; c < cols; ++c) d[c] = static_cast<uint16_t>(s[c] >> shiftRight);
        }
    }
}

void Nv12ToYuv420p(const uint8_t* y, int yPitch, const uint8_t* uv, int uvPitch, uint32_t width, uint32_t height,
                   CpuFrame& out) {
    out.Allocate(FrameDesc{width, height, PixelFormat::Yuv420p});
    // Y plane
    uint8_t* dstY = out.Plane(0);
    for (uint32_t r = 0; r < height; ++r) std::memcpy(dstY + static_cast<size_t>(r) * width, y + static_cast<ptrdiff_t>(r) * yPitch, width);
    // UV plane -> U, V
    const size_t cw = out.desc.PlaneWidth(1);
    const size_t ch = out.desc.PlaneHeight(1);
    uint8_t* dstU = out.Plane(1);
    uint8_t* dstV = out.Plane(2);
    for (size_t r = 0; r < ch; ++r) {
        const uint8_t* srcRow = uv + static_cast<ptrdiff_t>(r) * uvPitch;
        for (size_t c = 0; c < cw; ++c) {
            dstU[r * cw + c] = srcRow[2 * c];
            dstV[r * cw + c] = srcRow[2 * c + 1];
        }
    }
}

void SemiPlanar16ToPlanar(const uint8_t* y, int yPitch, const uint8_t* uv, int uvPitch, const FrameDesc& desc, CpuFrame& out, int shiftRight) {
    if (!desc.HighBitDepth()) Throw("SemiPlanar16ToPlanar: the target must be a 16-bit format");
    out.Allocate(desc);
    const uint32_t width = desc.width, height = desc.height;
    uint16_t* dstY = out.Plane16(0);
    for (uint32_t r = 0; r < height; ++r) {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(y + static_cast<ptrdiff_t>(r) * yPitch);
        uint16_t* d = dstY + static_cast<size_t>(r) * width;
        for (uint32_t c = 0; c < width; ++c) d[c] = static_cast<uint16_t>(s[c] >> shiftRight);
    }
    const size_t cw = desc.PlaneCols(1), ch = desc.PlaneHeight(1);
    uint16_t* dstU = out.Plane16(1);
    uint16_t* dstV = out.Plane16(2);
    for (size_t r = 0; r < ch; ++r) {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(uv + static_cast<ptrdiff_t>(r) * uvPitch);
        for (size_t c = 0; c < cw; ++c) {
            dstU[r * cw + c] = static_cast<uint16_t>(s[2 * c] >> shiftRight);
            dstV[r * cw + c] = static_cast<uint16_t>(s[2 * c + 1] >> shiftRight);
        }
    }
}

}  // namespace dlssvid
