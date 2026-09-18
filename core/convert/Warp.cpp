#include "convert/Warp.h"

#include <cmath>

#include "util/Error.h"

namespace dlssvid {

WarpResult WarpBackward(const PassImage& src, const PassImage& mv) {
    if (mv.channels.size() != 2) Throw("WarpBackward: mv must have 2 channels");
    if (mv.width != src.width || mv.height != src.height) Throw("WarpBackward: mv/src size mismatch");
    const size_t C = src.channels.size();
    WarpResult r;
    r.image.Allocate(src.width, src.height, PixelType::F32, src.channels);
    r.mask.Allocate(src.width, src.height, PixelType::F32, {"A"});
    size_t valid = 0;
    for (uint32_t y = 0; y < src.height; ++y)
        for (uint32_t x = 0; x < src.width; ++x) {
            const float sx = static_cast<float>(x) + mv.Get(x, y, 0);
            const float sy = static_cast<float>(y) + mv.Get(x, y, 1);
            const bool inside = std::isfinite(sx) && std::isfinite(sy) && sx >= 0.f && sy >= 0.f && sx <= static_cast<float>(src.width - 1) &&
                                sy <= static_cast<float>(src.height - 1);
            if (!inside) {
                for (size_t c = 0; c < C; ++c) r.image.Set(x, y, c, src.Get(x, y, c));
                r.mask.Set(x, y, 0, 0.f);
                continue;
            }
            const uint32_t x0 = static_cast<uint32_t>(sx), y0 = static_cast<uint32_t>(sy);
            const uint32_t x1 = std::min(x0 + 1, src.width - 1), y1 = std::min(y0 + 1, src.height - 1);
            const float fx = sx - static_cast<float>(x0), fy = sy - static_cast<float>(y0);
            for (size_t c = 0; c < C; ++c) {
                const float v = (1 - fy) * ((1 - fx) * src.Get(x0, y0, c) + fx * src.Get(x1, y0, c)) +
                                fy * ((1 - fx) * src.Get(x0, y1, c) + fx * src.Get(x1, y1, c));
                r.image.Set(x, y, c, v);
            }
            r.mask.Set(x, y, 0, 1.f);
            ++valid;
        }
    r.validFraction = static_cast<double>(valid) / (static_cast<double>(src.width) * src.height);
    return r;
}

double Psnr(const PassImage& a, const PassImage& b, float peak, const PassImage* mask) {
    if (a.width != b.width || a.height != b.height || a.channels.size() != b.channels.size()) Throw("Psnr: layout mismatch");
    double se = 0;
    size_t n = 0;
    for (uint32_t y = 0; y < a.height; ++y)
        for (uint32_t x = 0; x < a.width; ++x) {
            if (mask && mask->Get(x, y, 0) <= 0.f) continue;
            for (size_t c = 0; c < a.channels.size(); ++c) {
                const double d = static_cast<double>(a.Get(x, y, c)) - b.Get(x, y, c);
                se += d * d;
                ++n;
            }
        }
    if (n == 0) return 0.0;
    const double mse = se / static_cast<double>(n);
    if (mse <= 0) return 99.0;
    return 10.0 * std::log10(static_cast<double>(peak) * peak / mse);
}

double WarpPsnr(const PassImage& prev, const PassImage& cur, const PassImage& mvBackward, float peak) {
    const WarpResult w = WarpBackward(prev, mvBackward);
    const PassImage curF = cur.type == PixelType::F32 ? cur : cur.ConvertTo(PixelType::F32);
    return Psnr(w.image, curF, peak, &w.mask);
}

double TemporalAlignmentErrorWarped(const PassImage& prevDepth, const PassImage& curDepth, const PassImage& mvBackward) {
    const WarpResult w = WarpBackward(prevDepth, mvBackward);
    double diff = 0, mean = 0;
    size_t n = 0;
    for (uint32_t y = 0; y < curDepth.height; ++y)
        for (uint32_t x = 0; x < curDepth.width; ++x) {
            if (w.mask.Get(x, y, 0) <= 0.f) continue;
            const float a = w.image.Get(x, y, 0), b = curDepth.Get(x, y, 0);
            if (!std::isfinite(a) || !std::isfinite(b) || a <= 0.f || b <= 0.f) continue;
            diff += std::fabs(static_cast<double>(b) - a);
            mean += a;
            ++n;
        }
    if (n == 0 || mean <= 0) return 0.0;
    return (diff / n) / (mean / n);
}

}  // namespace dlssvid
