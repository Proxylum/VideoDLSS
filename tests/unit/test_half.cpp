#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

#include "util/Half.h"

using namespace dlssvid;

TEST_CASE("half: exact values round-trip", "[half]") {
    for (float v : {0.f, -0.f, 1.f, -1.f, 0.5f, 2.f, 1024.f, 65504.f, -65504.f, 0.00006103515625f /*min normal*/, 5.9604645e-8f /*min subnormal*/}) {
        const uint16_t h = FloatToHalf(v);
        CHECK(HalfToFloat(h) == v);
    }
}

TEST_CASE("half: known encodings", "[half]") {
    CHECK(FloatToHalf(1.f) == 0x3C00);
    CHECK(FloatToHalf(-2.f) == 0xC000);
    CHECK(FloatToHalf(0.f) == 0x0000);
    CHECK(FloatToHalf(65504.f) == 0x7BFF);
    CHECK(FloatToHalf(1e6f) == 0x7C00);  // overflow -> +inf
    CHECK(HalfToFloat(0x7C00) == std::numeric_limits<float>::infinity());
    CHECK(std::isnan(HalfToFloat(0x7E00)));
    CHECK(std::isnan(HalfToFloat(FloatToHalf(std::numeric_limits<float>::quiet_NaN()))));
}

TEST_CASE("half: round to nearest even and precision", "[half]") {
    // 1 + 2^-11 is exactly halfway between 1 and 1+2^-10 -> rounds to even (1.0)
    CHECK(HalfToFloat(FloatToHalf(1.f + std::ldexp(1.f, -11))) == 1.f);
    // 1 + 3*2^-11 -> rounds up to 1 + 2^-9
    CHECK(HalfToFloat(FloatToHalf(1.f + 3 * std::ldexp(1.f, -11))) == 1.f + std::ldexp(1.f, -9));
    // relative error of a random-ish set stays within half precision (2^-11)
    for (int i = 1; i < 2000; ++i) {
        const float v = 0.001f * i * (i % 2 ? 1.f : -1.f) * 37.7f;
        const float back = HalfToFloat(FloatToHalf(v));
        CHECK(std::fabs(back - v) <= std::fabs(v) * std::ldexp(1.f, -11) + 1e-7f);
    }
}
