#include "convert/ColorConvert.h"

#include <immintrin.h>
#include <intrin.h>

#include "util/Half.h"

#include <algorithm>
#include <cmath>

#include "util/Error.h"

namespace dlssvid {

ColorInfo ColorInfoFromStream(const VideoStreamInfo& info) {
    ColorInfo c;
    // AVCOL_SPC_*: 1 = BT.709, 5/6 = BT.601 (625/525); AVCOL_RANGE_JPEG = 2.
    if (info.colorSpace == 1) c.matrix = ColorMatrix::Bt709;
    else if (info.colorSpace == 5 || info.colorSpace == 6) c.matrix = ColorMatrix::Bt601;
    else c.matrix = info.height >= 720 ? ColorMatrix::Bt709 : ColorMatrix::Bt601;
    c.range = info.colorRange == 2 ? ColorRange::Full : ColorRange::Limited;
    return c;
}

std::string_view ToString(ColorMatrix m) { return m == ColorMatrix::Bt709 ? "bt709" : "bt601"; }
std::string_view ToString(ColorRange r) { return r == ColorRange::Full ? "full" : "limited"; }

PassImage Yuv420pToRgb(const CpuFrame& frame, const ColorInfo& info, PixelType outType) {
    if (frame.desc.format != PixelFormat::Yuv420p) Throw("Yuv420pToRgb: frame is not yuv420p");
    const uint32_t w = frame.desc.width, h = frame.desc.height;
    PassImage out = MakePassImage(PassKind::ColorSource, w, h, outType);

    const bool full = info.range == ColorRange::Full;
    const float yScale = full ? 1.f / 255.f : 1.f / 219.f;
    const float yOff = full ? 0.f : 16.f;
    const float cScale = full ? 1.f / 255.f : 1.f / 224.f;
    // Y'CbCr -> R'G'B' coefficients
    const float kr = info.matrix == ColorMatrix::Bt709 ? 0.2126f : 0.299f;
    const float kb = info.matrix == ColorMatrix::Bt709 ? 0.0722f : 0.114f;
    const float kg = 1.f - kr - kb;
    const float crR = 2.f * (1.f - kr);
    const float cbB = 2.f * (1.f - kb);
    const float cbG = -2.f * kb * (1.f - kb) / kg;
    const float crG = -2.f * kr * (1.f - kr) / kg;

    const uint8_t* Y = frame.Plane(0);
    const uint8_t* U = frame.Plane(1);
    const uint8_t* V = frame.Plane(2);
    const size_t cw = frame.desc.PlaneWidth(1);
    const float scaleOut = outType == PixelType::U16 ? 65535.f : (outType == PixelType::U8 ? 255.f : 1.f);

    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float yy = (static_cast<float>(Y[static_cast<size_t>(y) * w + x]) - yOff) * yScale;
            const size_t ci = static_cast<size_t>(y / 2) * cw + x / 2;
            const float cb = (static_cast<float>(U[ci]) - 128.f) * cScale;
            const float cr = (static_cast<float>(V[ci]) - 128.f) * cScale;
            const float r = std::clamp(yy + crR * cr, 0.f, 1.f);
            const float g = std::clamp(yy + cbG * cb + crG * cr, 0.f, 1.f);
            const float b = std::clamp(yy + cbB * cb, 0.f, 1.f);
            out.Set(x, y, 0, r * scaleOut);
            out.Set(x, y, 1, g * scaleOut);
            out.Set(x, y, 2, b * scaleOut);
        }
    return out;
}

namespace {

// 4 floats -> 4 halves (round to nearest even), F16C or the portable routine.
inline void StoreHalf4(uint16_t* d, float r, float g, float b, float a, bool f16c) {
    if (f16c) {
        const __m128i h = _mm_cvtps_ph(_mm_set_ps(a, b, g, r), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(d), h);
    } else {
        d[0] = FloatToHalf(r);
        d[1] = FloatToHalf(g);
        d[2] = FloatToHalf(b);
        d[3] = FloatToHalf(a);
    }
}

}  // namespace

bool HasF16C() {
    static const bool has = [] {
        int r[4] = {0, 0, 0, 0};
        __cpuid(r, 1);
        return (r[2] & (1 << 29)) != 0;
    }();
    return has;
}

PassImage Yuv420pToRgba16f(const CpuFrame& frame, const ColorInfo& info) {
    if (frame.desc.format != PixelFormat::Yuv420p) Throw("Yuv420pToRgba16f: frame is not yuv420p");
    const uint32_t w = frame.desc.width, h = frame.desc.height;
    PassImage out;
    out.Allocate(w, h, PixelType::F16, {"R", "G", "B", "A"});

    const bool full = info.range == ColorRange::Full;
    const float yScale = full ? 1.f / 255.f : 1.f / 219.f;
    const float yOff = full ? 0.f : 16.f;
    const float cScale = full ? 1.f / 255.f : 1.f / 224.f;
    const float kr = info.matrix == ColorMatrix::Bt709 ? 0.2126f : 0.299f;
    const float kb = info.matrix == ColorMatrix::Bt709 ? 0.0722f : 0.114f;
    const float kg = 1.f - kr - kb;
    const float crR = 2.f * (1.f - kr);
    const float cbB = 2.f * (1.f - kb);
    const float cbG = -2.f * kb * (1.f - kb) / kg;
    const float crG = -2.f * kr * (1.f - kr) / kg;

    const uint8_t* Y = frame.Plane(0);
    const uint8_t* U = frame.Plane(1);
    const uint8_t* V = frame.Plane(2);
    const size_t cw = frame.desc.PlaneWidth(1);
    const bool f16c = HasF16C();
    uint16_t* dst = out.As<uint16_t>();
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* yr = Y + static_cast<size_t>(y) * w;
        const uint8_t* ur = U + static_cast<size_t>(y / 2) * cw;
        const uint8_t* vr = V + static_cast<size_t>(y / 2) * cw;
        uint16_t* d = dst + static_cast<size_t>(y) * w * 4;
        for (uint32_t x = 0; x < w; ++x, d += 4) {
            const float yy = (static_cast<float>(yr[x]) - yOff) * yScale;
            const float cb = (static_cast<float>(ur[x / 2]) - 128.f) * cScale;
            const float cr = (static_cast<float>(vr[x / 2]) - 128.f) * cScale;
            const float r = std::clamp(yy + crR * cr, 0.f, 1.f);
            const float g = std::clamp(yy + cbG * cb + crG * cr, 0.f, 1.f);
            const float b = std::clamp(yy + cbB * cb, 0.f, 1.f);
            StoreHalf4(d, r, g, b, 1.f, f16c);
        }
    }
    return out;
}

void RgbToYuv420p(const PassImage& rgb, const ColorInfo& info, CpuFrame& out) {
    if (rgb.channels.size() < 3) Throw("RgbToYuv420p: needs an RGB image");
    const uint32_t w = rgb.width, h = rgb.height;
    out.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    const bool full = info.range == ColorRange::Full;
    const float kr = info.matrix == ColorMatrix::Bt709 ? 0.2126f : 0.299f;
    const float kb = info.matrix == ColorMatrix::Bt709 ? 0.0722f : 0.114f;
    const float kg = 1.f - kr - kb;
    const float scale = rgb.type == PixelType::U8 ? 1.f / 255.f : rgb.type == PixelType::U16 ? 1.f / 65535.f : 1.f;
    const float yScale = full ? 255.f : 219.f, yOff = full ? 0.f : 16.f, cScale = full ? 255.f : 224.f;
    uint8_t* Y = out.Plane(0);
    uint8_t* U = out.Plane(1);
    uint8_t* V = out.Plane(2);
    const size_t cw = out.desc.PlaneWidth(1), ch = out.desc.PlaneHeight(1);
    std::vector<float> cb(static_cast<size_t>(w) * h), cr(static_cast<size_t>(w) * h);
    auto q = [](float v) { return static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L)); };
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float r = std::clamp(rgb.Get(x, y, 0) * scale, 0.f, 1.f);
            const float g = std::clamp(rgb.Get(x, y, 1) * scale, 0.f, 1.f);
            const float b = std::clamp(rgb.Get(x, y, 2) * scale, 0.f, 1.f);
            const float yy = kr * r + kg * g + kb * b;
            Y[static_cast<size_t>(y) * w + x] = q(yOff + yScale * yy);
            cb[static_cast<size_t>(y) * w + x] = (b - yy) / (2.f * (1.f - kb));
            cr[static_cast<size_t>(y) * w + x] = (r - yy) / (2.f * (1.f - kr));
        }
    for (size_t cy = 0; cy < ch; ++cy)
        for (size_t cx = 0; cx < cw; ++cx) {
            float sb = 0.f, sr = 0.f;
            int n = 0;
            for (uint32_t dy = 0; dy < 2; ++dy)
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    const uint32_t x = std::min<uint32_t>(static_cast<uint32_t>(cx * 2 + dx), w - 1);
                    const uint32_t y = std::min<uint32_t>(static_cast<uint32_t>(cy * 2 + dy), h - 1);
                    sb += cb[static_cast<size_t>(y) * w + x];
                    sr += cr[static_cast<size_t>(y) * w + x];
                    ++n;
                }
            U[cy * cw + cx] = q(128.f + cScale * sb / static_cast<float>(n));
            V[cy * cw + cx] = q(128.f + cScale * sr / static_cast<float>(n));
        }
}

PassImage ToRgba16f(const PassImage& img) {
    const uint32_t w = img.width, h = img.height;
    const size_t nc = std::max<size_t>(1, img.channels.size());
    PassImage out;
    out.Allocate(w, h, PixelType::F16, {"R", "G", "B", "A"});
    uint16_t* d = out.As<uint16_t>();
    const size_t n = static_cast<size_t>(w) * h;
    if (img.type == PixelType::F16) {
        const uint16_t* s = img.As<uint16_t>();
        for (size_t i = 0; i < n; ++i, d += 4, s += nc) {
            d[0] = s[0];
            d[1] = s[nc > 1 ? 1 : 0];
            d[2] = s[nc > 2 ? 2 : 0];
            d[3] = 0x3C00;  // 1.0
        }
        return out;
    }
    const bool f16c = HasF16C();
    const float scale = img.type == PixelType::U8 ? 1.f / 255.f : img.type == PixelType::U16 ? 1.f / 65535.f : 1.f;
    auto convert = [&](auto* s) {
        for (size_t i = 0; i < n; ++i, d += 4, s += nc) {
            const float r = static_cast<float>(s[0]) * scale;
            const float g = static_cast<float>(s[nc > 1 ? 1 : 0]) * scale;
            const float b = static_cast<float>(s[nc > 2 ? 2 : 0]) * scale;
            StoreHalf4(d, r, g, b, 1.f, f16c);
        }
    };
    switch (img.type) {
        case PixelType::U8: convert(img.As<uint8_t>()); break;
        case PixelType::U16: convert(img.As<uint16_t>()); break;
        default: convert(img.As<float>()); break;
    }
    return out;
}

}  // namespace dlssvid
