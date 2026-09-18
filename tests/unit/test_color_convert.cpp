#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "convert/ColorConvert.h"

using namespace dlssvid;

namespace {
CpuFrame Frame(uint32_t w, uint32_t h, uint8_t y, uint8_t u, uint8_t v) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    std::fill(f.Plane(0), f.Plane(0) + f.desc.PlaneSize(0), y);
    std::fill(f.Plane(1), f.Plane(1) + f.desc.PlaneSize(1), u);
    std::fill(f.Plane(2), f.Plane(2) + f.desc.PlaneSize(2), v);
    return f;
}
}  // namespace

TEST_CASE("YUV -> RGB: limited-range white, black and grey (BT.709)", "[convert][color]") {
    ColorInfo ci{ColorMatrix::Bt709, ColorRange::Limited};
    const PassImage white = Yuv420pToRgb(Frame(4, 2, 235, 128, 128), ci, PixelType::F32);
    CHECK(white.channels == std::vector<std::string>{"R", "G", "B"});
    for (size_t c = 0; c < 3; ++c) CHECK(std::fabs(white.Get(1, 1, c) - 1.f) < 1e-6f);
    const PassImage black = Yuv420pToRgb(Frame(4, 2, 16, 128, 128), ci, PixelType::F32);
    for (size_t c = 0; c < 3; ++c) CHECK(black.Get(0, 0, c) == 0.f);
    const PassImage grey = Yuv420pToRgb(Frame(4, 2, 126, 128, 128), ci, PixelType::F32);  // (126-16)/219
    for (size_t c = 0; c < 3; ++c) CHECK(std::fabs(grey.Get(3, 1, c) - 110.f / 219.f) < 1e-6f);
    // super-white clamps
    const PassImage sw = Yuv420pToRgb(Frame(2, 2, 255, 128, 128), ci, PixelType::F32);
    CHECK(sw.Get(0, 0, 0) == 1.f);
}

TEST_CASE("YUV -> RGB: full range and BT.601 red", "[convert][color]") {
    ColorInfo full{ColorMatrix::Bt709, ColorRange::Full};
    const PassImage w = Yuv420pToRgb(Frame(2, 2, 255, 128, 128), full, PixelType::F32);
    CHECK(std::fabs(w.Get(0, 0, 1) - 1.f) < 1e-6f);
    // BT.601 limited: pure red (R=1,G=0,B=0) is Y=81, Cb=90, Cr=240
    ColorInfo c601{ColorMatrix::Bt601, ColorRange::Limited};
    const PassImage red = Yuv420pToRgb(Frame(2, 2, 81, 90, 240), c601, PixelType::F32);
    CHECK(std::fabs(red.Get(0, 0, 0) - 1.f) < 0.01f);
    CHECK(red.Get(0, 0, 1) < 0.01f);
    CHECK(red.Get(0, 0, 2) < 0.01f);
    // BT.709 limited: pure blue is Y=32, Cb=240, Cr=118
    ColorInfo c709{ColorMatrix::Bt709, ColorRange::Limited};
    const PassImage blue = Yuv420pToRgb(Frame(2, 2, 32, 240, 118), c709, PixelType::F32);
    CHECK(blue.Get(1, 1, 2) > 0.98f);
    CHECK(blue.Get(1, 1, 0) < 0.02f);
}

TEST_CASE("YUV -> RGB: output types and chroma replication", "[convert][color]") {
    ColorInfo ci{ColorMatrix::Bt709, ColorRange::Limited};
    CpuFrame f = Frame(4, 2, 126, 128, 128);  // mid grey so a chroma shift is visible without clamping
    f.Plane(1)[0] = 200;  // first 2x2 block gets a chroma shift
    const PassImage u16 = Yuv420pToRgb(f, ci, PixelType::U16);
    CHECK(u16.type == PixelType::U16);
    CHECK(u16.Get(3, 1, 0) == std::round(110.f / 219.f * 65535.f));
    // the chroma shift affects pixels (0..1, 0..1) identically, not (2, 0)
    CHECK(u16.Get(0, 0, 2) == u16.Get(1, 1, 2));
    CHECK(u16.Get(0, 0, 2) != u16.Get(2, 0, 2));
    const PassImage f16 = Yuv420pToRgb(f, ci, PixelType::F16);
    CHECK(f16.type == PixelType::F16);
    CHECK(std::fabs(f16.Get(3, 1, 1) - 110.f / 219.f) < 1e-3f);  // half precision of the grey level
}

TEST_CASE("ColorInfoFromStream uses metadata or resolution", "[convert][color]") {
    VideoStreamInfo i;
    i.width = 1920;
    i.height = 1080;
    i.colorSpace = 2;
    CHECK(ColorInfoFromStream(i).matrix == ColorMatrix::Bt709);
    i.height = 480;
    CHECK(ColorInfoFromStream(i).matrix == ColorMatrix::Bt601);
    i.colorSpace = 1;
    CHECK(ColorInfoFromStream(i).matrix == ColorMatrix::Bt709);
    i.colorRange = 2;
    CHECK(ColorInfoFromStream(i).range == ColorRange::Full);
    CHECK(ToString(ColorMatrix::Bt601) == "bt601");
}
