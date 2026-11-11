#include "pipeline/Frame.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <numeric>

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

std::string DisplayOrientation::Token() const {
    if (Identity()) return {};
    return (rotate ? "r" + std::to_string(rotate) : std::string()) + (hflip ? "h" : "");
}

std::string DisplayOrientation::Describe() const {
    if (Identity()) return {};
    std::string s = rotate ? "rotated " + std::to_string(rotate) + " deg" + (rotate == 180 ? "" : " clockwise") : std::string();
    if (hflip) s += rotate ? " and mirrored" : "mirrored";
    return s;
}

std::optional<DisplayOrientation> OrientationFromToken(std::string_view token) {
    for (const int r : {0, 90, 180, 270})
        for (const bool h : {false, true})
            if (const DisplayOrientation o{static_cast<uint16_t>(r), h}; o.Token() == token) return o;
    return std::nullopt;
}

std::string PixelAspect::Token() const { return Square() ? std::string() : std::to_string(num) + ":" + std::to_string(den); }

std::string PixelAspect::Stamp() const { return num == 0 ? std::string() : Square() ? std::string("1:1") : Token(); }

bool PixelAspect::MovesAPixel(uint32_t w, uint32_t h) const {
    if (Square()) return false;
    const double v = Value();
    return v > 1.0 ? static_cast<double>(w) * (v - 1.0) >= 0.5 : static_cast<double>(h) * (1.0 / v - 1.0) >= 0.5;
}

std::optional<PixelAspect> PixelAspectFromStamp(std::string_view stamp) {
    const size_t colon = stamp.find(':');
    if (colon == std::string_view::npos) return std::nullopt;
    int64_t num = 0, den = 0;
    const std::string_view a = stamp.substr(0, colon), b = stamp.substr(colon + 1);
    const auto ra = std::from_chars(a.data(), a.data() + a.size(), num);
    const auto rb = std::from_chars(b.data(), b.data() + b.size(), den);
    if (ra.ec != std::errc() || ra.ptr != a.data() + a.size() || rb.ec != std::errc() || rb.ptr != b.data() + b.size() || num <= 0 || den <= 0)
        return std::nullopt;
    if (num == den) return PixelAspect{1, 1};
    const PixelAspect p = PixelAspectFrom(num, den);
    if (p.Square()) return std::nullopt;  // not plausible
    return p;
}

std::pair<uint32_t, uint32_t> PixelAspect::DisplaySize(uint32_t w, uint32_t h) const {
    if (Square() || !w || !h) return {w, h};  // square, or a size not known yet: as is
    auto even = [](double v) { return std::max<uint32_t>(2u, static_cast<uint32_t>(std::lround(v / 2.0)) * 2u); };
    if (num > den) return {even(static_cast<double>(w) * num / den), h};
    return {w, even(static_cast<double>(h) * den / num)};
}

PixelAspect PixelAspectFrom(int64_t num, int64_t den, std::string* why) {
    if (num <= 0 || den <= 0) return {};
    const int64_t g = std::gcd(num, den);
    num /= g;
    den /= g;
    if (num == den) return {};
    const double v = static_cast<double>(num) / static_cast<double>(den);
    if (v < 0.25 || v > 4.0 || num > 65535 || den > 65535) {
        if (why) *why = "a pixel aspect of " + std::to_string(num) + ":" + std::to_string(den) + " is not plausible: the pixels are taken as square";
        return {};
    }
    return {static_cast<uint32_t>(num), static_cast<uint32_t>(den)};
}

namespace {

// The stored luma sample shown at (X, Y): sx = ax*X + bx*Y + cx, sy = ay*X + by*Y + cy, for a stored W x H frame. Shown = mirrored if
// hflip (turned clockwise by rotate (stored)):
//   0: (X, Y)          h: (W-1-X, Y)
//   90: (Y, H-1-X)     90h: (Y, X)             (FFmpeg's transpose clock / cclock_flip)
//   180: (W-1-X, H-1-Y)  180h: (X, H-1-Y)      (hflip+vflip / vflip)
//   270: (W-1-Y, X)    270h: (W-1-Y, H-1-X)    (cclock / clock_flip)
struct LumaMap {
    int ax, bx, cx, ay, by, cy;
};

LumaMap MapFor(DisplayOrientation o, int w, int h) {
    switch (o.rotate) {
        case 90: return o.hflip ? LumaMap{0, 1, 0, 1, 0, 0} : LumaMap{0, 1, 0, -1, 0, h - 1};
        case 180: return o.hflip ? LumaMap{1, 0, 0, 0, -1, h - 1} : LumaMap{-1, 0, w - 1, 0, -1, h - 1};
        case 270: return o.hflip ? LumaMap{0, -1, w - 1, -1, 0, h - 1} : LumaMap{0, -1, w - 1, 1, 0, 0};
        default: return o.hflip ? LumaMap{-1, 0, w - 1, 0, 1, 0} : LumaMap{1, 0, 0, 0, 1, 0};
    }
}

bool ReversesStoredX(DisplayOrientation o) {
    const LumaMap m = MapFor(o, 2, 2);
    return m.ax < 0 || m.bx < 0;
}

bool ReversesStoredY(DisplayOrientation o) {
    const LumaMap m = MapFor(o, 2, 2);
    return m.ay < 0 || m.by < 0;
}

// One plane: each shown sample takes the stored sample under the top-left luma of its block (shifts: the plane's chroma shifts in the
// shown and the stored layout, 0 for luma). In tiles, so that a turn reads the stored plane a band at a time.
template <typename T>
void OrientPlane(const T* src, size_t srcCols, T* dst, size_t dstCols, size_t dstRows, const LumaMap& m, int dsx, int dsy, int ssx, int ssy) {
    constexpr size_t kTile = 32;
    for (size_t ty = 0; ty < dstRows; ty += kTile)
        for (size_t tx = 0; tx < dstCols; tx += kTile) {
            const size_t ye = std::min(dstRows, ty + kTile), xe = std::min(dstCols, tx + kTile);
            for (size_t y = ty; y < ye; ++y) {
                const int ly = static_cast<int>(y) << dsy;
                T* d = dst + y * dstCols;
                for (size_t x = tx; x < xe; ++x) {
                    const int lx = static_cast<int>(x) << dsx;
                    const int sx = (m.ax * lx + m.bx * ly + m.cx) >> ssx;
                    const int sy = (m.ay * lx + m.by * ly + m.cy) >> ssy;
                    d[x] = src[static_cast<size_t>(sy) * srcCols + static_cast<size_t>(sx)];
                }
            }
        }
}

}  // namespace

FrameDesc OrientedDesc(const FrameDesc& desc, DisplayOrientation o) {
    if (o.Identity()) return desc;
    FrameDesc r = desc;
    if (o.SwapsAxes()) std::swap(r.width, r.height);
    const int sx = ChromaShiftX(desc.format), sy = ChromaShiftY(desc.format);
    const bool exact = !(o.SwapsAxes() && sx != sy) && !(ReversesStoredX(o) && sx && (desc.width & 1u)) && !(ReversesStoredY(o) && sy && (desc.height & 1u));
    if (!exact) r.format = desc.HighBitDepth() ? PixelFormat::Yuv444p16 : PixelFormat::Yuv444p;
    return r;
}

void OrientFrame(const CpuFrame& in, DisplayOrientation o, CpuFrame& out) {
    const FrameDesc& s = in.desc;
    out.Allocate(OrientedDesc(s, o));
    out.index = in.index;
    out.pts = in.pts;
    const LumaMap m = MapFor(o, static_cast<int>(s.width), static_cast<int>(s.height));
    for (size_t p = 0; p < 3; ++p) {
        const int dsx = p ? ChromaShiftX(out.desc.format) : 0, dsy = p ? ChromaShiftY(out.desc.format) : 0;
        const int ssx = p ? ChromaShiftX(s.format) : 0, ssy = p ? ChromaShiftY(s.format) : 0;
        if (s.HighBitDepth())
            OrientPlane(in.Plane16(p), s.PlaneCols(p), out.Plane16(p), out.desc.PlaneCols(p), out.desc.PlaneHeight(p), m, dsx, dsy, ssx, ssy);
        else
            OrientPlane(in.Plane(p), s.PlaneCols(p), out.Plane(p), out.desc.PlaneCols(p), out.desc.PlaneHeight(p), m, dsx, dsy, ssx, ssy);
    }
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
