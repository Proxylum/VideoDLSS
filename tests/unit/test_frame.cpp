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

// TASK-0034: frames keep the source's bit depth and chroma layout.
TEST_CASE("FrameDesc: 4:2:2 / 4:4:4 layouts and 16-bit formats with a bit depth", "[frame]") {
    const FrameDesc d422{1920, 1080, PixelFormat::Yuv422p16, 10};
    CHECK(d422.PlaneCols(1) == 960);
    CHECK(d422.PlaneHeight(1) == 1080);
    CHECK(d422.PlaneWidth(1) == 1920);  // bytes: two per sample
    CHECK(d422.ByteSize() == 1920u * 1080u * 2u * 2u);
    CHECK(d422.Name() == "yuv422p10");
    CHECK(d422.Describe() == "10-bit 4:2:2");
    CHECK(d422.HighBitDepth());
    const FrameDesc d444{7, 5, PixelFormat::Yuv444p, 8};
    CHECK(d444.PlaneCols(2) == 7);
    CHECK(d444.PlaneHeight(2) == 5);
    CHECK(d444.ByteSize() == 7u * 5u * 3u);
    CHECK(d444.Name() == "yuv444p");
    CHECK(d444.Describe() == "8-bit 4:4:4");
    CHECK_FALSE(d444.HighBitDepth());
    const FrameDesc d420{7, 5, PixelFormat::Yuv420p16, 12};
    CHECK(d420.PlaneCols(1) == 4);
    CHECK(d420.PlaneHeight(1) == 3);
    CHECK(d420.PlaneOffset(1) == 7u * 5u * 2u);
    CHECK(d420.Name() == "yuv420p12");
    CHECK(FrameDesc{4, 2, PixelFormat::Yuv420p}.Name() == "yuv420p");  // the legacy layout keeps its name
    CHECK(ChromaName(PixelFormat::Yuv422p) == "422");
    CHECK(ChromaShiftX(PixelFormat::Yuv444p16) == 0);
    CHECK(ChromaShiftY(PixelFormat::Yuv422p16) == 0);
    CHECK(ChromaShiftX(PixelFormat::Yuv422p16) == 1);
    CHECK(ChromaShiftY(PixelFormat::Yuv420p16) == 1);
    CHECK(BytesPerSample(PixelFormat::Yuv420p) == 1);
    CHECK(BytesPerSample(PixelFormat::Yuv444p16) == 2);
    CHECK(PixelFormatFor("444", 10) == PixelFormat::Yuv444p16);
    CHECK(PixelFormatFor("422", 8) == PixelFormat::Yuv422p);
    CHECK(PixelFormatFor("420", 16) == PixelFormat::Yuv420p16);
    CHECK_THROWS(PixelFormatFor("411", 8));
    CHECK(ToString(PixelFormat::Yuv444p16) == "yuv444p16");
    CpuFrame f;
    CHECK_THROWS(f.Allocate(FrameDesc{4, 2, PixelFormat::Yuv420p16, 8}));  // a 16-bit format carries 10 / 12 / 16 bits
    CHECK_THROWS(f.Allocate(FrameDesc{4, 2, PixelFormat::Yuv420p, 10}));
    f.Allocate(FrameDesc{4, 2, PixelFormat::Yuv444p16, 16});
    CHECK(f.data.size() == 4u * 2u * 3u * 2u);
    CHECK(f.Plane16(1) == reinterpret_cast<uint16_t*>(f.data.data()) + 8);
}

TEST_CASE("CopyPlanar keeps 16-bit planes and shifts MSB-aligned words down", "[frame]") {
    const uint32_t w = 4, h = 2;
    // 4:2:2 10-bit stored as MSB-aligned words (value << 6, what NVDEC hands out), pitch 16 bytes = 8 words (padded)
    std::vector<uint16_t> y(8 * h), u(8 * h), v(8 * h);
    for (size_t i = 0; i < y.size(); ++i) {  // 10-bit values (< 1024) in the top bits of the words
        y[i] = static_cast<uint16_t>((i * 60 + 1) << 6);
        u[i] = static_cast<uint16_t>((i * 7) << 6);
        v[i] = static_cast<uint16_t>((1000 - i) << 6);
    }
    const uint8_t* const src[3] = {reinterpret_cast<const uint8_t*>(y.data()), reinterpret_cast<const uint8_t*>(u.data()), reinterpret_cast<const uint8_t*>(v.data())};
    const int pitch[3] = {16, 16, 16};
    CpuFrame f;
    CopyPlanar(src, pitch, FrameDesc{w, h, PixelFormat::Yuv422p16, 10}, f, 6);
    REQUIRE(f.desc == FrameDesc{w, h, PixelFormat::Yuv422p16, 10});
    for (uint32_t r = 0; r < h; ++r)
        for (uint32_t c = 0; c < w; ++c) CHECK(f.Plane16(0)[r * w + c] == (r * 8 + c) * 60 + 1);
    for (uint32_t r = 0; r < h; ++r)
        for (uint32_t c = 0; c < 2; ++c) {
            CHECK(f.Plane16(1)[r * 2 + c] == (r * 8 + c) * 7);
            CHECK(f.Plane16(2)[r * 2 + c] == 1000 - (r * 8 + c));
        }
    // the same words as LSB-aligned 16-bit samples: no shift, a bit-exact copy
    CpuFrame g;
    CopyPlanar(src, pitch, FrameDesc{w, h, PixelFormat::Yuv422p16, 16}, g, 0);
    CHECK(g.Plane16(0)[5] == y[8 + 1]);
    // 8-bit 4:4:4 planes with padding
    std::vector<uint8_t> a(8 * h);
    std::iota(a.begin(), a.end(), uint8_t{0});
    const uint8_t* const src8[3] = {a.data(), a.data(), a.data()};
    const int pitch8[3] = {8, 8, 8};
    CpuFrame e;
    CopyPlanar(src8, pitch8, FrameDesc{w, h, PixelFormat::Yuv444p, 8}, e);
    CHECK(e.data.size() == 24);
    CHECK(e.Plane(2)[w + 3] == a[8 + 3]);
}

TEST_CASE("SemiPlanar16ToPlanar de-interleaves P010 chroma and shifts to 10 bits", "[frame]") {
    const uint32_t w = 4, h = 2;
    const uint16_t y[8] = {1 << 6, 2 << 6, 3 << 6, 4 << 6, 5 << 6, 6 << 6, 7 << 6, 8 << 6};
    const uint16_t uv[8] = {10 << 6, 20 << 6, 11 << 6, 21 << 6, 0, 0, 0, 0};  // one chroma row: U0 V0 U1 V1 (pitch 16 bytes, 8 used)
    CpuFrame f;
    SemiPlanar16ToPlanar(reinterpret_cast<const uint8_t*>(y), 8, reinterpret_cast<const uint8_t*>(uv), 16, FrameDesc{w, h, PixelFormat::Yuv420p16, 10}, f, 6);
    REQUIRE(f.data.size() == (8u + 2u + 2u) * 2u);
    for (int i = 0; i < 8; ++i) CHECK(f.Plane16(0)[i] == i + 1);
    CHECK(f.Plane16(1)[0] == 10);
    CHECK(f.Plane16(1)[1] == 11);
    CHECK(f.Plane16(2)[0] == 20);
    CHECK(f.Plane16(2)[1] == 21);
    CHECK_THROWS(SemiPlanar16ToPlanar(reinterpret_cast<const uint8_t*>(y), 8, reinterpret_cast<const uint8_t*>(uv), 16, FrameDesc{w, h, PixelFormat::Yuv420p, 8}, f, 0));
}
