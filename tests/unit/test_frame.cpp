#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <vector>

#include "pipeline/Frame.h"

using namespace dlssvid;

TEST_CASE("FrameDesc yuv420p plane geometry", "[frame]") {
    FrameDesc d{1920, 1080, PixelFormat::Yuv420p};
    CHECK(d.PlaneCount() == 3);
    CHECK(d.PlaneWidth(0) == 1920);
    CHECK(d.PlaneHeight(0) == 1080);
    CHECK(d.PlaneWidth(1) == 960);
    CHECK(d.PlaneHeight(2) == 540);
    CHECK(d.PlaneOffset(0) == 0);
    CHECK(d.PlaneOffset(1) == 1920u * 1080u);
    CHECK(d.PlaneOffset(2) == 1920u * 1080u + 960u * 540u);
    CHECK(d.ByteSize() == 1920u * 1080u * 3 / 2);
}

TEST_CASE("FrameDesc odd dimensions round chroma up", "[frame]") {
    FrameDesc d{7, 5, PixelFormat::Yuv420p};
    CHECK(d.PlaneWidth(1) == 4);
    CHECK(d.PlaneHeight(1) == 3);
    CHECK(d.ByteSize() == 7u * 5u + 2u * 4u * 3u);
}

TEST_CASE("CpuFrame allocates and addresses planes", "[frame]") {
    CpuFrame f;
    f.Allocate(FrameDesc{4, 2, PixelFormat::Yuv420p});
    REQUIRE(f.data.size() == 8u + 2u + 2u);
    CHECK(f.Plane(1) == f.data.data() + 8);
    CHECK(f.Plane(2) == f.data.data() + 10);
    CHECK_THROWS(f.Allocate(FrameDesc{0, 2, PixelFormat::Yuv420p}));
}

TEST_CASE("CopyPlanar420 strips padding and is bit-exact", "[frame]") {
    const uint32_t w = 6, h = 4;
    // Source planes with padded pitches (row bytes > width).
    std::vector<uint8_t> y(16 * h), u(16 * 2), v(16 * 2);
    std::iota(y.begin(), y.end(), uint8_t{0});
    std::iota(u.begin(), u.end(), uint8_t{100});
    std::iota(v.begin(), v.end(), uint8_t{200});
    const uint8_t* const src[3] = {y.data(), u.data(), v.data()};
    const int pitch[3] = {16, 16, 16};

    CpuFrame f;
    CopyPlanar420(src, pitch, w, h, f);
    REQUIRE(f.desc == FrameDesc{w, h, PixelFormat::Yuv420p});
    for (uint32_t r = 0; r < h; ++r)
        for (uint32_t c = 0; c < w; ++c) CHECK(f.Plane(0)[r * w + c] == y[r * 16 + c]);
    for (uint32_t r = 0; r < 2; ++r)
        for (uint32_t c = 0; c < 3; ++c) {
            CHECK(f.Plane(1)[r * 3 + c] == u[r * 16 + c]);
            CHECK(f.Plane(2)[r * 3 + c] == v[r * 16 + c]);
        }
}

TEST_CASE("Nv12ToYuv420p de-interleaves chroma bit-exactly", "[frame]") {
    const uint32_t w = 4, h = 2;
    const uint8_t y[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    // one chroma row: U0 V0 U1 V1 (pitch 8, only 4 used)
    const uint8_t uv[8] = {10, 20, 11, 21, 0, 0, 0, 0};
    CpuFrame f;
    Nv12ToYuv420p(y, 4, uv, 8, w, h, f);
    REQUIRE(f.data.size() == 8u + 2u + 2u);
    CHECK(std::vector<uint8_t>(f.Plane(0), f.Plane(0) + 8) == std::vector<uint8_t>(y, y + 8));
    CHECK(f.Plane(1)[0] == 10);
    CHECK(f.Plane(1)[1] == 11);
    CHECK(f.Plane(2)[0] == 20);
    CHECK(f.Plane(2)[1] == 21);
}
