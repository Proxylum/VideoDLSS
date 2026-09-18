#include "pipeline/Frame.h"

#include <cstring>

#include "util/Error.h"

namespace dlssvid {

std::string_view ToString(PixelFormat f) {
    switch (f) {
        case PixelFormat::Yuv420p: return "yuv420p";
    }
    return "unknown";
}

size_t FrameDesc::PlaneCount() const { return 3; }

size_t FrameDesc::PlaneWidth(size_t plane) const {
    if (plane == 0) return width;
    return (width + 1) / 2;
}

size_t FrameDesc::PlaneHeight(size_t plane) const {
    if (plane == 0) return height;
    return (height + 1) / 2;
}

size_t FrameDesc::PlaneSize(size_t plane) const { return PlaneWidth(plane) * PlaneHeight(plane); }

size_t FrameDesc::PlaneOffset(size_t plane) const {
    size_t off = 0;
    for (size_t p = 0; p < plane; ++p) off += PlaneSize(p);
    return off;
}

size_t FrameDesc::ByteSize() const { return PlaneOffset(PlaneCount()); }

void CpuFrame::Allocate(const FrameDesc& d) {
    if (d.width == 0 || d.height == 0) Throw("CpuFrame::Allocate: zero-sized frame");
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

}  // namespace dlssvid
