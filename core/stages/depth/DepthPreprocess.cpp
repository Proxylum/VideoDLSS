#include "stages/depth/DepthPreprocess.h"

#include <algorithm>
#include <cmath>

#include "util/Error.h"

namespace dlssvid {

ModelInputSize ComputeModelInputSize(uint32_t srcWidth, uint32_t srcHeight, int inputSize, int multiple, double maxAspect) {
    if (srcWidth == 0 || srcHeight == 0 || inputSize <= 0 || multiple <= 0) Throw("ComputeModelInputSize: bad arguments");
    const double ratio = static_cast<double>(std::max(srcWidth, srcHeight)) / std::min(srcWidth, srcHeight);
    double shortSide = inputSize;
    if (ratio > maxAspect) shortSide = inputSize * maxAspect / ratio;  // very wide/tall: keep the token budget
    shortSide = std::round(shortSide / multiple) * multiple;
    if (shortSide < multiple) shortSide = multiple;
    const double scale = shortSide / std::min(srcWidth, srcHeight);
    double longSide = std::max(srcWidth, srcHeight) * scale;
    longSide = std::round(longSide / multiple) * multiple;
    if (longSide < shortSide) longSide = shortSide;
    ModelInputSize s;
    if (srcWidth >= srcHeight) {
        s.width = static_cast<uint32_t>(longSide);
        s.height = static_cast<uint32_t>(shortSide);
    } else {
        s.width = static_cast<uint32_t>(shortSide);
        s.height = static_cast<uint32_t>(longSide);
    }
    return s;
}

PassImage ResizeBilinear(const PassImage& src, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) Throw("ResizeBilinear: zero target size");
    const size_t C = src.channels.size();
    PassImage out;
    out.Allocate(width, height, PixelType::F32, src.channels);
    if (width == src.width && height == src.height) {
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                for (size_t c = 0; c < C; ++c) out.Set(x, y, c, src.Get(x, y, c));
        return out;
    }
    const float sx = static_cast<float>(src.width) / width, sy = static_cast<float>(src.height) / height;
    float* o = out.As<float>();
    for (uint32_t y = 0; y < height; ++y) {
        const float fy = std::clamp((y + 0.5f) * sy - 0.5f, 0.f, static_cast<float>(src.height - 1));
        const uint32_t y0 = static_cast<uint32_t>(fy), y1 = std::min(y0 + 1, src.height - 1);
        const float wy = fy - y0;
        for (uint32_t x = 0; x < width; ++x) {
            const float fx = std::clamp((x + 0.5f) * sx - 0.5f, 0.f, static_cast<float>(src.width - 1));
            const uint32_t x0 = static_cast<uint32_t>(fx), x1 = std::min(x0 + 1, src.width - 1);
            const float wx = fx - x0;
            for (size_t c = 0; c < C; ++c) {
                const float v = (1 - wy) * ((1 - wx) * src.Get(x0, y0, c) + wx * src.Get(x1, y0, c)) +
                                wy * ((1 - wx) * src.Get(x0, y1, c) + wx * src.Get(x1, y1, c));
                o[(static_cast<size_t>(y) * width + x) * C + c] = v;
            }
        }
    }
    return out;
}

PassImage ResizeArea(const PassImage& src, uint32_t width, uint32_t height) {
    if (width >= src.width || height >= src.height) return ResizeBilinear(src, width, height);
    const size_t C = src.channels.size();
    PassImage out;
    out.Allocate(width, height, PixelType::F32, src.channels);
    float* o = out.As<float>();
    const double sx = static_cast<double>(src.width) / width, sy = static_cast<double>(src.height) / height;
    for (uint32_t y = 0; y < height; ++y) {
        const uint32_t y0 = static_cast<uint32_t>(y * sy), y1 = std::max(y0 + 1, static_cast<uint32_t>((y + 1) * sy));
        for (uint32_t x = 0; x < width; ++x) {
            const uint32_t x0 = static_cast<uint32_t>(x * sx), x1 = std::max(x0 + 1, static_cast<uint32_t>((x + 1) * sx));
            for (size_t c = 0; c < C; ++c) {
                double sum = 0;
                size_t n = 0;
                for (uint32_t yy = y0; yy < std::min(y1, src.height); ++yy)
                    for (uint32_t xx = x0; xx < std::min(x1, src.width); ++xx) {
                        sum += src.Get(xx, yy, c);
                        ++n;
                    }
                o[(static_cast<size_t>(y) * width + x) * C + c] = n ? static_cast<float>(sum / n) : 0.f;
            }
        }
    }
    return out;
}

void ToNchwNormalized(const PassImage& rgb, const Normalization& norm, std::vector<float>& out) {
    if (rgb.channels.size() < 3) Throw("ToNchwNormalized: need RGB");
    const size_t plane = static_cast<size_t>(rgb.width) * rgb.height;
    const size_t base = out.size();
    out.resize(base + 3 * plane);
    for (size_t c = 0; c < 3; ++c) {
        float* dst = out.data() + base + c * plane;
        for (uint32_t y = 0; y < rgb.height; ++y)
            for (uint32_t x = 0; x < rgb.width; ++x) dst[static_cast<size_t>(y) * rgb.width + x] = (rgb.Get(x, y, c) - norm.mean[c]) / norm.std[c];
    }
}

PassImage Luminance(const PassImage& rgb) {
    if (rgb.channels.size() < 3) Throw("Luminance: need RGB");
    PassImage out;
    out.Allocate(rgb.width, rgb.height, PixelType::F32, {"Y"});
    float* o = out.As<float>();
    for (uint32_t y = 0; y < rgb.height; ++y)
        for (uint32_t x = 0; x < rgb.width; ++x)
            o[static_cast<size_t>(y) * rgb.width + x] = 0.2126f * rgb.Get(x, y, 0) + 0.7152f * rgb.Get(x, y, 1) + 0.0722f * rgb.Get(x, y, 2);
    return out;
}

namespace {

// Box-filter mean via integral image (double precision accumulation); edges use the clipped window.
std::vector<float> BoxMean(const float* src, uint32_t w, uint32_t h, int r) {
    std::vector<double> integral(static_cast<size_t>(w + 1) * (h + 1), 0.0);
    for (uint32_t y = 0; y < h; ++y) {
        double row = 0;
        for (uint32_t x = 0; x < w; ++x) {
            row += src[static_cast<size_t>(y) * w + x];
            integral[static_cast<size_t>(y + 1) * (w + 1) + x + 1] = integral[static_cast<size_t>(y) * (w + 1) + x + 1] + row;
        }
    }
    std::vector<float> out(static_cast<size_t>(w) * h);
    for (uint32_t y = 0; y < h; ++y) {
        const long y0 = std::max<long>(0, static_cast<long>(y) - r), y1 = std::min<long>(h - 1, static_cast<long>(y) + r);
        for (uint32_t x = 0; x < w; ++x) {
            const long x0 = std::max<long>(0, static_cast<long>(x) - r), x1 = std::min<long>(w - 1, static_cast<long>(x) + r);
            const double s = integral[static_cast<size_t>(y1 + 1) * (w + 1) + x1 + 1] - integral[static_cast<size_t>(y0) * (w + 1) + x1 + 1] -
                             integral[static_cast<size_t>(y1 + 1) * (w + 1) + x0] + integral[static_cast<size_t>(y0) * (w + 1) + x0];
            out[static_cast<size_t>(y) * w + x] = static_cast<float>(s / ((y1 - y0 + 1) * (x1 - x0 + 1)));
        }
    }
    return out;
}

}  // namespace

PassImage GuidedFilter(const PassImage& src, const PassImage& guide, int radius, float eps) {
    if (src.width != guide.width || src.height != guide.height) Throw("GuidedFilter: size mismatch");
    if (src.channels.size() != 1 || guide.channels.size() != 1) Throw("GuidedFilter: single-channel images expected");
    const uint32_t w = src.width, h = src.height;
    const size_t n = static_cast<size_t>(w) * h;
    std::vector<float> p(n), I(n), Ip(n), II(n);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            p[i] = src.Get(x, y, 0);
            I[i] = guide.Get(x, y, 0);
            Ip[i] = I[i] * p[i];
            II[i] = I[i] * I[i];
        }
    const auto meanI = BoxMean(I.data(), w, h, radius), meanP = BoxMean(p.data(), w, h, radius);
    const auto meanIp = BoxMean(Ip.data(), w, h, radius), meanII = BoxMean(II.data(), w, h, radius);
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; ++i) {
        const float varI = meanII[i] - meanI[i] * meanI[i];
        const float covIp = meanIp[i] - meanI[i] * meanP[i];
        a[i] = covIp / (varI + eps);
        b[i] = meanP[i] - a[i] * meanI[i];
    }
    const auto meanA = BoxMean(a.data(), w, h, radius), meanB = BoxMean(b.data(), w, h, radius);
    PassImage out;
    out.Allocate(w, h, PixelType::F32, src.channels);
    float* o = out.As<float>();
    for (size_t i = 0; i < n; ++i) o[i] = meanA[i] * I[i] + meanB[i];
    return out;
}

PassImage UpsampleDepthGuided(const PassImage& depthLow, const PassImage& rgbFull, int radius, float eps) {
    PassImage up = ResizeBilinear(depthLow, rgbFull.width, rgbFull.height);
    if (radius <= 0) return up;
    const PassImage luma = Luminance(rgbFull);
    PassImage out = GuidedFilter(up, luma, radius, eps);
    out.channels = depthLow.channels;
    return out;
}

}  // namespace dlssvid
