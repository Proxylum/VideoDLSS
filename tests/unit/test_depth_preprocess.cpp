#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "stages/depth/DepthPreprocess.h"

using namespace dlssvid;

namespace {
PassImage Rgb(uint32_t w, uint32_t h, float r, float g, float b) {
    PassImage img = MakePassImage(PassKind::ColorSource, w, h, PixelType::F32);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            img.Set(x, y, 0, r);
            img.Set(x, y, 1, g);
            img.Set(x, y, 2, b);
        }
    return img;
}
}  // namespace

TEST_CASE("Model input size follows the Depth Anything rule", "[depth][preprocess]") {
    ModelInputSize s = ComputeModelInputSize(1920, 1080, 518);
    CHECK(s.height == 518);
    CHECK(s.width == 924);  // 518 * 16/9 = 920.9 -> 66 * 14
    s = ComputeModelInputSize(1080, 1920, 518);
    CHECK(s.width == 518);
    CHECK(s.height == 924);
    s = ComputeModelInputSize(1000, 1000, 518);
    CHECK(s.width == 518);
    CHECK(s.height == 518);
    // ultra-wide clamps the token budget: shorter side shrinks
    s = ComputeModelInputSize(4000, 1000, 518);
    CHECK(s.height < 518);
    CHECK(s.height % 14 == 0);
    CHECK(s.width % 14 == 0);
    CHECK_THROWS(ComputeModelInputSize(0, 10, 518));
}

TEST_CASE("ResizeBilinear and ResizeArea preserve constants and interpolate", "[depth][preprocess]") {
    const PassImage flat = Rgb(20, 10, 0.25f, 0.5f, 0.75f);
    const PassImage up = ResizeBilinear(flat, 37, 23);
    CHECK(up.width == 37);
    CHECK(up.type == PixelType::F32);
    CHECK(std::fabs(up.Get(36, 22, 2) - 0.75f) < 1e-6f);
    const PassImage down = ResizeArea(flat, 7, 3);
    CHECK(std::fabs(down.Get(3, 1, 1) - 0.5f) < 1e-6f);
    // gradient: midpoint of a 2-pixel ramp is the average
    PassImage ramp;
    ramp.Allocate(2, 1, PixelType::F32, {"Z"});
    ramp.Set(0, 0, 0, 0.f);
    ramp.Set(1, 0, 0, 1.f);
    const PassImage r3 = ResizeBilinear(ramp, 3, 1);
    CHECK(std::fabs(r3.Get(1, 0, 0) - 0.5f) < 1e-6f);
    CHECK_THROWS(ResizeBilinear(ramp, 0, 1));
}

TEST_CASE("ToNchwNormalized lays out planes and normalises", "[depth][preprocess]") {
    const PassImage img = Rgb(3, 2, 0.485f, 0.456f, 0.406f);  // == ImageNet mean -> zeros
    std::vector<float> t;
    ToNchwNormalized(img, Normalization{}, t);
    REQUIRE(t.size() == 3 * 6);
    for (float v : t) CHECK(std::fabs(v) < 1e-5f);
    const PassImage white = Rgb(3, 2, 1.f, 1.f, 1.f);
    t.clear();
    ToNchwNormalized(white, Normalization{}, t);
    CHECK(std::fabs(t[0] - (1.f - 0.485f) / 0.229f) < 1e-5f);   // R plane first
    CHECK(std::fabs(t[6] - (1.f - 0.456f) / 0.224f) < 1e-5f);   // G plane
    CHECK(std::fabs(t[12] - (1.f - 0.406f) / 0.225f) < 1e-5f);  // B plane
}

TEST_CASE("GuidedFilter keeps edges of the guide and smooths noise", "[depth][preprocess]") {
    const uint32_t w = 40, h = 20;
    PassImage guide, src;
    guide.Allocate(w, h, PixelType::F32, {"Y"});
    src.Allocate(w, h, PixelType::F32, {"Z"});
    uint32_t seed = 7;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const bool right = x >= 20;
            guide.Set(x, y, 0, right ? 1.f : 0.f);
            seed = seed * 1664525u + 1013904223u;
            const float noise = ((seed >> 16) % 100) / 1000.f - 0.05f;  // +-0.05
            src.Set(x, y, 0, (right ? 5.f : 1.f) + noise);
        }
    const PassImage out = GuidedFilter(src, guide, 4, 1e-3f);
    // edge preserved: left/right means differ, noise reduced
    double l = 0, r = 0, ln = 0, rn = 0;
    for (uint32_t y = 4; y < h - 4; ++y) {
        l += out.Get(5, y, 0);
        r += out.Get(34, y, 0);
        ln += std::fabs(out.Get(5, y, 0) - 1.f);
        rn += std::fabs(out.Get(34, y, 0) - 5.f);
    }
    const double n = h - 8;
    CHECK(std::fabs(l / n - 1.0) < 0.05);
    CHECK(std::fabs(r / n - 5.0) < 0.05);
    CHECK(ln / n < 0.02);
    CHECK(rn / n < 0.02);
    CHECK(std::fabs(out.Get(19, 10, 0) - 1.f) < 0.3f);  // pixel just left of the edge stays near 1
    CHECK(std::fabs(out.Get(20, 10, 0) - 5.f) < 0.3f);  // just right stays near 5
}

TEST_CASE("UpsampleDepthGuided returns full-resolution single-channel depth", "[depth][preprocess]") {
    PassImage low;
    low.Allocate(4, 2, PixelType::F32, {"Z"});
    for (uint32_t y = 0; y < 2; ++y)
        for (uint32_t x = 0; x < 4; ++x) low.Set(x, y, 0, 1.f + x);
    const PassImage rgb = Rgb(16, 8, 0.3f, 0.3f, 0.3f);
    const PassImage up = UpsampleDepthGuided(low, rgb, 2);
    CHECK(up.width == 16);
    CHECK(up.height == 8);
    CHECK(up.channels == std::vector<std::string>{"Z"});
    CHECK(up.Get(0, 4, 0) < up.Get(15, 4, 0));
    const PassImage plain = UpsampleDepthGuided(low, rgb, 0);
    CHECK(std::fabs(plain.Get(15, 4, 0) - 4.f) < 1e-5f);
}
