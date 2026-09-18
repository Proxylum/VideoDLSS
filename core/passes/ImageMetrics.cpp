#include "passes/ImageMetrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "util/Error.h"

namespace dlssvid {

namespace {

float Scale(PixelType t) { return t == PixelType::U8 ? 1.f / 255.f : t == PixelType::U16 ? 1.f / 65535.f : 1.f; }

// Separable Gaussian blur of an F32 single-channel plane (window 11, sigma 1.5), edges clamped.
std::vector<float> Gaussian(const std::vector<float>& src, uint32_t w, uint32_t h) {
    constexpr int kRadius = 5;
    float k[2 * kRadius + 1];
    float sum = 0.f;
    for (int i = -kRadius; i <= kRadius; ++i) {
        k[i + kRadius] = std::exp(-static_cast<float>(i * i) / (2.f * 1.5f * 1.5f));
        sum += k[i + kRadius];
    }
    for (float& v : k) v /= sum;
    std::vector<float> tmp(src.size()), out(src.size());
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int i = -kRadius; i <= kRadius; ++i) {
                const int xx = std::clamp<int>(static_cast<int>(x) + i, 0, static_cast<int>(w) - 1);
                acc += src[y * w + xx] * k[i + kRadius];
            }
            tmp[y * w + x] = acc;
        }
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int i = -kRadius; i <= kRadius; ++i) {
                const int yy = std::clamp<int>(static_cast<int>(y) + i, 0, static_cast<int>(h) - 1);
                acc += tmp[yy * w + x] * k[i + kRadius];
            }
            out[y * w + x] = acc;
        }
    return out;
}

}  // namespace

PassImage LumaOf(const PassImage& img) {
    PassImage out;
    out.Allocate(img.width, img.height, PixelType::F32, {"Y"});
    const float s = Scale(img.type);
    const size_t nc = img.channels.size();
    float* d = out.As<float>();
    for (uint32_t y = 0; y < img.height; ++y)
        for (uint32_t x = 0; x < img.width; ++x) {
            float v;
            if (nc >= 3) {
                v = 0.2126f * img.Get(x, y, 0) + 0.7152f * img.Get(x, y, 1) + 0.0722f * img.Get(x, y, 2);
            } else {
                v = img.Get(x, y, 0);
            }
            d[static_cast<size_t>(y) * img.width + x] = v * s;
        }
    return out;
}

ImageMetrics CompareImages(const PassImage& reference, const PassImage& test) {
    if (reference.width != test.width || reference.height != test.height) Throw("CompareImages: sizes differ");
    if (reference.Empty() || test.Empty()) Throw("CompareImages: empty image");
    const uint32_t w = reference.width, h = reference.height;
    const size_t n = static_cast<size_t>(w) * h;
    ImageMetrics m;

    // RGB PSNR over min(channels, 3) colour samples
    const size_t nc = std::min<size_t>(3, std::min(reference.channels.size(), test.channels.size()));
    const float sr = Scale(reference.type), st = Scale(test.type);
    double mseRgb = 0.0;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < nc; ++c) {
                const double d = static_cast<double>(reference.Get(x, y, c)) * sr - static_cast<double>(test.Get(x, y, c)) * st;
                mseRgb += d * d;
            }
    mseRgb /= static_cast<double>(n * nc);
    m.psnrRgb = mseRgb > 0.0 ? 10.0 * std::log10(1.0 / mseRgb) : std::numeric_limits<double>::infinity();

    // luma
    const PassImage ya = LumaOf(reference), yb = LumaOf(test);
    const float* a = ya.As<float>();
    const float* b = yb.As<float>();
    double mseY = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        mseY += d * d;
    }
    mseY /= static_cast<double>(n);
    m.mseY = mseY;
    m.psnrY = mseY > 0.0 ? 10.0 * std::log10(1.0 / mseY) : std::numeric_limits<double>::infinity();

    // SSIM (Wang et al. 2004) on luma with Gaussian statistics
    std::vector<float> A(a, a + n), B(b, b + n), AA(n), BB(n), AB(n);
    for (size_t i = 0; i < n; ++i) {
        AA[i] = A[i] * A[i];
        BB[i] = B[i] * B[i];
        AB[i] = A[i] * B[i];
    }
    const std::vector<float> muA = Gaussian(A, w, h), muB = Gaussian(B, w, h), eAA = Gaussian(AA, w, h), eBB = Gaussian(BB, w, h), eAB = Gaussian(AB, w, h);
    constexpr double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
    double ssim = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double ma = muA[i], mb = muB[i];
        const double va = eAA[i] - ma * ma, vb = eBB[i] - mb * mb, cov = eAB[i] - ma * mb;
        ssim += ((2.0 * ma * mb + C1) * (2.0 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
    }
    m.ssimY = ssim / static_cast<double>(n);
    return m;
}

}  // namespace dlssvid
