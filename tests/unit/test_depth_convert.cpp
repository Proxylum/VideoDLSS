#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

#include "convert/DepthConvert.h"

using namespace dlssvid;

TEST_CASE("reverse-Z: near maps to 1, far to 0, monotonic decreasing", "[convert][depth]") {
    const float n = 0.1f, f = 1000.f;
    CHECK(LinearToReverseZ(n, n, f) == 1.f);
    CHECK(LinearToReverseZ(f, n, f) == 0.f);
    float prev = 2.f;
    for (float z = n; z <= f; z *= 1.3f) {
        const float d = LinearToReverseZ(z, n, f);
        CHECK(d <= prev);
        CHECK(d >= 0.f);
        CHECK(d <= 1.f);
        prev = d;
    }
    // out of range / invalid
    CHECK(LinearToReverseZ(0.f, n, f) == 0.f);
    CHECK(LinearToReverseZ(-5.f, n, f) == 0.f);
    CHECK(LinearToReverseZ(std::numeric_limits<float>::quiet_NaN(), n, f) == 0.f);
    CHECK(LinearToReverseZ(std::numeric_limits<float>::infinity(), n, f) == 0.f);
    CHECK(LinearToReverseZ(0.01f, n, f) == 1.f);   // clamped to near
    CHECK(LinearToReverseZ(5000.f, n, f) == 0.f);  // clamped to far
    CHECK_THROWS(LinearToReverseZ(1.f, 0.f, 10.f));
    CHECK_THROWS(LinearToReverseZ(1.f, 10.f, 1.f));
}

TEST_CASE("reverse-Z: invertible within float precision", "[convert][depth]") {
    const float n = 0.25f, f = 250.f;
    for (float z = n; z <= f; z *= 1.05f) {
        const float back = ReverseZToLinear(LinearToReverseZ(z, n, f), n, f);
        CHECK(std::fabs(back - z) <= z * 2e-5f);
    }
    CHECK(ReverseZToLinear(1.f, n, f) == n);
    CHECK(ReverseZToLinear(0.f, n, f) == f);
    CHECK(ReverseZToLinear(std::numeric_limits<float>::quiet_NaN(), n, f) == f);
    // half-depth check from the formula: z = 2*near*far/(near+far) gives d = 0.5
    const float zHalf = 2 * n * f / (n + f);
    CHECK(std::fabs(LinearToReverseZ(zHalf, n, f) - 0.5f) < 1e-6f);
}

TEST_CASE("depth_raw <-> depth_dlss on images (metric)", "[convert][depth]") {
    PassImage raw = MakePassImage(PassKind::DepthRaw, 16, 8);
    for (uint32_t y = 0; y < 8; ++y)
        for (uint32_t x = 0; x < 16; ++x) raw.Set(x, y, 0, 0.5f + x * 3.f + y * 0.25f);
    raw.Set(0, 0, 0, std::numeric_limits<float>::quiet_NaN());
    DepthParams p;
    p.zNear = 0.1f;
    p.zFar = 100.f;
    const PassImage dlss = DepthRawToDlss(raw, p);
    CHECK(dlss.channels == std::vector<std::string>{"Z"});
    CHECK(dlss.type == PixelType::F32);
    CHECK(dlss.Get(0, 0, 0) == 0.f);  // NaN -> far
    CHECK(dlss.Get(1, 0, 0) > dlss.Get(2, 0, 0));  // nearer = larger
    const PassImage back = DepthDlssToRaw(dlss, p);
    for (uint32_t y = 0; y < 8; ++y)
        for (uint32_t x = 1; x < 16; ++x) CHECK(std::fabs(back.Get(x, y, 0) - raw.Get(x, y, 0)) <= raw.Get(x, y, 0) * 2e-5f);
    CHECK(back.Get(0, 0, 0) == 100.f);
}

TEST_CASE("depth_raw <-> depth_dlss on images (relative, range recorded)", "[convert][depth]") {
    PassImage raw = MakePassImage(PassKind::DepthRaw, 10, 4);
    for (uint32_t y = 0; y < 4; ++y)
        for (uint32_t x = 0; x < 10; ++x) raw.Set(x, y, 0, 3.f + x * 0.7f);  // arbitrary units, 3..9.3
    DepthParams p;
    p.relative = true;
    p.zNear = 1.f;
    p.zFar = 50.f;
    const PassImage dlss = DepthRawToDlss(raw, p);
    CHECK(p.minValue == 3.f);
    CHECK(std::fabs(p.maxValue - 9.3f) < 1e-5f);
    CHECK(dlss.Get(0, 0, 0) == 1.f);  // min raw -> near -> 1
    CHECK(dlss.Get(9, 0, 0) == 0.f);  // max raw -> far -> 0
    const PassImage back = DepthDlssToRaw(dlss, p);
    for (uint32_t x = 0; x < 10; ++x) CHECK(std::fabs(back.Get(x, 1, 0) - raw.Get(x, 1, 0)) < 2e-4f);
    // a flat relative image maps to the near plane without dividing by zero
    PassImage flat = MakePassImage(PassKind::DepthRaw, 2, 2);
    for (auto& v : flat.data) v = 0;
    DepthParams pf;
    pf.relative = true;
    CHECK(DepthRawToDlss(flat, pf).Get(1, 1, 0) == 1.f);
    CHECK(ComputeDepthRange(flat).valid);
}
