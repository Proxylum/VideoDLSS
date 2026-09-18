#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "convert/MvConvert.h"
#include "convert/Warp.h"

using namespace dlssvid;

namespace {
// Smooth image so bilinear sampling of an integer shift is exact.
PassImage Image(uint32_t w, uint32_t h, int shiftX = 0, int shiftY = 0) {
    PassImage img = MakePassImage(PassKind::ColorSource, w, h, PixelType::F32);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float sx = static_cast<float>(static_cast<int>(x) - shiftX), sy = static_cast<float>(static_cast<int>(y) - shiftY);
            img.Set(x, y, 0, 0.5f + 0.4f * std::sin(sx * 0.21f) * std::cos(sy * 0.17f));
            img.Set(x, y, 1, 0.5f + 0.3f * std::cos(sx * 0.13f + sy * 0.11f));
            img.Set(x, y, 2, 0.5f + 0.2f * std::sin((sx + sy) * 0.09f));
        }
    return img;
}
PassImage ConstMv(uint32_t w, uint32_t h, float u, float v) {
    PassImage m = MakePassImage(PassKind::MvDlss, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            m.Set(x, y, 0, u);
            m.Set(x, y, 1, v);
        }
    return m;
}
}  // namespace

TEST_CASE("WarpBackward reconstructs a shifted frame exactly and masks the border", "[warp]") {
    const uint32_t w = 64, h = 40;
    const PassImage prev = Image(w, h);
    const PassImage cur = Image(w, h, 3, -2);  // content moved by (+3, -2): cur(p) = prev(p - (3,-2))
    // backward vector from cur to prev is (-3, +2)
    const WarpResult r = WarpBackward(prev, ConstMv(w, h, -3.f, 2.f));
    CHECK(r.image.channels == prev.channels);
    CHECK(r.validFraction > 0.85);
    CHECK(r.mask.Get(0, 0, 0) == 0.f);   // x - 3 < 0
    CHECK(r.mask.Get(10, 10, 0) == 1.f);
    for (uint32_t y = 4; y < h - 4; ++y)
        for (uint32_t x = 4; x < w - 4; ++x)
            for (size_t c = 0; c < 3; ++c) CHECK(std::fabs(r.image.Get(x, y, c) - cur.Get(x, y, c)) < 1e-5f);
    CHECK(WarpPsnr(prev, cur, ConstMv(w, h, -3.f, 2.f)) > 60.0);
    CHECK(WarpPsnr(prev, cur, ConstMv(w, h, 0.f, 0.f)) < 30.0);
    CHECK_THROWS(WarpBackward(prev, MakePassImage(PassKind::DepthRaw, w, h)));
}

TEST_CASE("Psnr basics", "[warp]") {
    const PassImage a = Image(16, 16);
    CHECK(Psnr(a, a) == 99.0);
    PassImage b = a;
    for (uint32_t y = 0; y < 16; ++y)
        for (uint32_t x = 0; x < 16; ++x)
            for (size_t c = 0; c < 3; ++c) b.Set(x, y, c, a.Get(x, y, c) + 0.1f);
    CHECK(std::fabs(Psnr(a, b) - 20.0) < 1e-4);  // mse = 0.01 -> 20 dB (float rounding)
    PassImage mask;
    mask.Allocate(16, 16, PixelType::F32, {"A"});
    for (auto& v : mask.data) v = 0;
    CHECK(Psnr(a, b, 1.f, &mask) == 0.0);  // nothing valid
}

TEST_CASE("Forward flow -> mv_dlss -> warp closes the loop", "[warp][mv]") {
    const uint32_t w = 48, h = 32;
    const PassImage prev = Image(w, h), cur = Image(w, h, 2, 1);
    PassImage flow = MakePassImage(PassKind::MvRaw, w, h);  // forward flow prev->cur: (+2, +1)
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            flow.Set(x, y, 0, 2.f);
            flow.Set(x, y, 1, 1.f);
        }
    MvConvertOptions opt;
    opt.dilateRadius = 0;
    const PassImage mv = ForwardFlowToBackwardMv(flow, nullptr, opt);
    CHECK(mv.Get(10, 10, 0) == -2.f);
    CHECK(WarpPsnr(prev, cur, mv) > 60.0);
}

TEST_CASE("TemporalAlignmentErrorWarped compensates motion", "[warp][depth]") {
    const uint32_t w = 40, h = 30;
    PassImage prevD = MakePassImage(PassKind::DepthRaw, w, h), curD = MakePassImage(PassKind::DepthRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            prevD.Set(x, y, 0, 1.f + 0.05f * x);
            curD.Set(x, y, 0, 1.f + 0.05f * (static_cast<int>(x) - 4));  // content shifted right by 4
        }
    const PassImage mv = ConstMv(w, h, -4.f, 0.f);
    CHECK(TemporalAlignmentErrorWarped(prevD, curD, mv) < 1e-5);
    CHECK(TemporalAlignmentErrorWarped(prevD, curD, ConstMv(w, h, 0.f, 0.f)) > 0.05);
}
