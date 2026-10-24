#include "convert/ColorConvert.h"

#include <immintrin.h>
#include <intrin.h>

#include "util/Half.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

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

namespace {

// Y'CbCr <-> R'G'B' constants for a matrix, a range and a bit depth. For 8 bits these are the numbers the 8-bit
// converter always used (16 / 219 / 128 / 224); limited range scales with the depth (SMPTE / ITU: 64..940 at 10 bits).
struct YuvCoeffs {
    float yOff, yScale, cOff, cScale;  // (Y - yOff) * yScale, (C - cOff) * cScale
    float yCode, cCode;                // the inverse: code values per unit (219 / 224 at 8 bits limited, maxv full)
    float kr, kb, kg;
    float crR, cbB, cbG, crG;
    float maxv;  // the code value of 1.0 in full range (255, 1023, 4095, 65535)
};

YuvCoeffs Coeffs(const ColorInfo& info, int bitDepth) {
    YuvCoeffs k;
    const bool full = info.range == ColorRange::Full;
    const int extra = bitDepth - 8;
    k.maxv = static_cast<float>((1u << bitDepth) - 1u);
    k.yCode = full ? k.maxv : static_cast<float>(219 << extra);
    k.cCode = full ? k.maxv : static_cast<float>(224 << extra);
    k.yScale = 1.f / k.yCode;
    k.yOff = full ? 0.f : static_cast<float>(16 << extra);
    k.cOff = static_cast<float>(128 << extra);
    k.cScale = 1.f / k.cCode;
    k.kr = info.matrix == ColorMatrix::Bt709 ? 0.2126f : 0.299f;
    k.kb = info.matrix == ColorMatrix::Bt709 ? 0.0722f : 0.114f;
    k.kg = 1.f - k.kr - k.kb;
    k.crR = 2.f * (1.f - k.kr);
    k.cbB = 2.f * (1.f - k.kb);
    k.cbG = -2.f * k.kb * (1.f - k.kb) / k.kg;
    k.crG = -2.f * k.kr * (1.f - k.kr) / k.kg;
    return k;
}

// Walks every pixel of a planar frame, replicating chroma per block, and hands `emit(x, y, r, g, b)` the clamped
// display-referred RGB in [0, 1]. The arithmetic is the one the 8-bit 4:2:0 converter always used.
template <typename T, typename Emit>
void ForEachRgb(const CpuFrame& frame, const YuvCoeffs& k, Emit emit, uint32_t y0 = 0, uint32_t y1 = UINT32_MAX) {
    const uint32_t w = frame.desc.width, h = std::min(frame.desc.height, y1);
    const int sx = ChromaShiftX(frame.desc.format), sy = ChromaShiftY(frame.desc.format);
    const T* Y = reinterpret_cast<const T*>(frame.Plane(0));
    const T* U = reinterpret_cast<const T*>(frame.Plane(1));
    const T* V = reinterpret_cast<const T*>(frame.Plane(2));
    const size_t cw = frame.desc.PlaneCols(1);
    for (uint32_t y = y0; y < h; ++y) {
        const T* yr = Y + static_cast<size_t>(y) * w;
        const T* ur = U + static_cast<size_t>(y >> sy) * cw;
        const T* vr = V + static_cast<size_t>(y >> sy) * cw;
        for (uint32_t x = 0; x < w; ++x) {
            const float yy = (static_cast<float>(yr[x]) - k.yOff) * k.yScale;
            const float cb = (static_cast<float>(ur[x >> sx]) - k.cOff) * k.cScale;
            const float cr = (static_cast<float>(vr[x >> sx]) - k.cOff) * k.cScale;
            const float r = std::clamp(yy + k.crR * cr, 0.f, 1.f);
            const float g = std::clamp(yy + k.cbG * cb + k.crG * cr, 0.f, 1.f);
            const float b = std::clamp(yy + k.cbB * cb, 0.f, 1.f);
            emit(x, y, r, g, b);
        }
    }
}

template <typename Emit>
void ForEachRgbOf(const CpuFrame& frame, const ColorInfo& info, Emit emit, uint32_t y0 = 0, uint32_t y1 = UINT32_MAX) {
    if (frame.data.size() < frame.desc.ByteSize()) Throw("YuvToRgb: frame has no data");
    const YuvCoeffs k = Coeffs(info, frame.desc.HighBitDepth() ? frame.desc.bitDepth : 8);
    if (frame.desc.HighBitDepth()) ForEachRgb<uint16_t>(frame, k, emit, y0, y1);
    else ForEachRgb<uint8_t>(frame, k, emit, y0, y1);
}

// 4 floats -> 4 halves (round to nearest even), F16C or the portable routine.
inline void StoreHalf4(uint16_t* d, float r, float g, float b, float a, bool f16c) {
    if (f16c) {
        const __m128i h = _mm_cvtps_ph(_mm_set_ps(a, b, g, r), _MM_FROUND_TO_NEAREST_INT);  // imm8 0..7: vcvtps2ph raises no exceptions anyway
        _mm_storel_epi64(reinterpret_cast<__m128i*>(d), h);
    } else {
        d[0] = FloatToHalf(r);
        d[1] = FloatToHalf(g);
        d[2] = FloatToHalf(b);
        d[3] = FloatToHalf(a);
    }
}

}  // namespace

PassImage YuvToRgb(const CpuFrame& frame, const ColorInfo& info, PixelType outType) {
    const uint32_t w = frame.desc.width, h = frame.desc.height;
    PassImage out = MakePassImage(PassKind::ColorSource, w, h, outType);
    const float scaleOut = outType == PixelType::U16 ? 65535.f : (outType == PixelType::U8 ? 255.f : 1.f);
    ForEachRgbOf(frame, info, [&](uint32_t x, uint32_t y, float r, float g, float b) {
        out.Set(x, y, 0, r * scaleOut);
        out.Set(x, y, 1, g * scaleOut);
        out.Set(x, y, 2, b * scaleOut);
    });
    return out;
}

bool HasF16C() {
    static const bool has = [] {
        int r[4] = {0, 0, 0, 0};
        __cpuid(r, 1);
        return (r[2] & (1 << 29)) != 0;
    }();
    return has;
}

PassImage YuvToRgba16f(const CpuFrame& frame, const ColorInfo& info, int threads) {
    const uint32_t w = frame.desc.width, h = frame.desc.height;
    if (frame.data.size() < frame.desc.ByteSize()) Throw("YuvToRgb: frame has no data");
    PassImage out;
    out.Allocate(w, h, PixelType::F16, {"R", "G", "B", "A"});
    const bool f16c = HasF16C();
    uint16_t* dst = out.As<uint16_t>();
    auto rows = [&](uint32_t y0, uint32_t y1) {
        ForEachRgbOf(frame, info, [&](uint32_t x, uint32_t y, float r, float g, float b) {
            StoreHalf4(dst + (static_cast<size_t>(y) * w + x) * 4, r, g, b, 1.f, f16c);
        }, y0, y1);
    };
    // bands of rows on several threads (TASK-0040: 8-12 ms single-threaded at 1080p, a third of the scrubbing budget);
    // every pixel is computed alone, so the result is the same bit for bit whatever the count
    int n = threads > 0 ? threads : (h >= 256 ? std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 2, 1, 8) : 1);
    n = std::clamp(n, 1, static_cast<int>(std::max(1u, h)));
    if (n == 1) {
        rows(0, h);
        return out;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(n - 1));
    const auto band = [&](int i) { return static_cast<uint32_t>(static_cast<uint64_t>(h) * static_cast<uint64_t>(i) / static_cast<uint64_t>(n)); };
    for (int i = 1; i < n; ++i) pool.emplace_back(rows, band(i), band(i + 1));
    rows(0, band(1));
    for (auto& t : pool) t.join();
    return out;
}

namespace {

template <typename T>
void RgbToYuvPlanar(const PassImage& rgb, const YuvCoeffs& k, CpuFrame& out) {
    const uint32_t w = rgb.width, h = rgb.height;
    const int sx = ChromaShiftX(out.desc.format), sy = ChromaShiftY(out.desc.format);
    const float scale = rgb.type == PixelType::U8 ? 1.f / 255.f : rgb.type == PixelType::U16 ? 1.f / 65535.f : 1.f;
    // the code value ranges: full range maps 1.0 to maxv, limited to 219 / 224 steps above the offsets
    const float yScale = k.yCode, yOff = k.yOff, cScale = k.cCode, cOff = k.cOff;
    T* Y = reinterpret_cast<T*>(out.Plane(0));
    T* U = reinterpret_cast<T*>(out.Plane(1));
    T* V = reinterpret_cast<T*>(out.Plane(2));
    const size_t cw = out.desc.PlaneCols(1), ch = out.desc.PlaneHeight(1);
    const long maxv = static_cast<long>(k.maxv);
    std::vector<float> cb(static_cast<size_t>(w) * h), cr(static_cast<size_t>(w) * h);
    auto q = [maxv](float v) { return static_cast<T>(std::clamp(std::lround(v), 0L, maxv)); };
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float r = std::clamp(rgb.Get(x, y, 0) * scale, 0.f, 1.f);
            const float g = std::clamp(rgb.Get(x, y, 1) * scale, 0.f, 1.f);
            const float b = std::clamp(rgb.Get(x, y, 2) * scale, 0.f, 1.f);
            const float yy = k.kr * r + k.kg * g + k.kb * b;
            Y[static_cast<size_t>(y) * w + x] = q(yOff + yScale * yy);
            cb[static_cast<size_t>(y) * w + x] = (b - yy) / (2.f * (1.f - k.kb));
            cr[static_cast<size_t>(y) * w + x] = (r - yy) / (2.f * (1.f - k.kr));
        }
    const uint32_t bw = 1u << sx, bh = 1u << sy;
    for (size_t cy = 0; cy < ch; ++cy)
        for (size_t cx = 0; cx < cw; ++cx) {
            float sb = 0.f, sr = 0.f;
            int n = 0;
            for (uint32_t dy = 0; dy < bh; ++dy)
                for (uint32_t dx = 0; dx < bw; ++dx) {
                    const uint32_t x = std::min<uint32_t>(static_cast<uint32_t>(cx * bw + dx), w - 1);
                    const uint32_t y = std::min<uint32_t>(static_cast<uint32_t>(cy * bh + dy), h - 1);
                    sb += cb[static_cast<size_t>(y) * w + x];
                    sr += cr[static_cast<size_t>(y) * w + x];
                    ++n;
                }
            U[cy * cw + cx] = q(cOff + cScale * sb / static_cast<float>(n));
            V[cy * cw + cx] = q(cOff + cScale * sr / static_cast<float>(n));
        }
}

}  // namespace

void RgbToYuv(const PassImage& rgb, const ColorInfo& info, FrameDesc target, CpuFrame& out) {
    if (rgb.channels.size() < 3) Throw("RgbToYuv: needs an RGB image");
    target.width = rgb.width;
    target.height = rgb.height;
    if (!target.HighBitDepth()) target.bitDepth = 8;
    out.Allocate(target);
    const YuvCoeffs k = Coeffs(info, target.HighBitDepth() ? target.bitDepth : 8);
    if (target.HighBitDepth()) RgbToYuvPlanar<uint16_t>(rgb, k, out);
    else RgbToYuvPlanar<uint8_t>(rgb, k, out);
}

void RgbToYuv420p(const PassImage& rgb, const ColorInfo& info, CpuFrame& out) { RgbToYuv(rgb, info, FrameDesc{rgb.width, rgb.height, PixelFormat::Yuv420p}, out); }

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
