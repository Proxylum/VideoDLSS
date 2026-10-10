#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

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
    const PassImage white = YuvToRgb(Frame(4, 2, 235, 128, 128), ci, PixelType::F32);
    CHECK(white.channels == std::vector<std::string>{"R", "G", "B"});
    for (size_t c = 0; c < 3; ++c) CHECK(std::fabs(white.Get(1, 1, c) - 1.f) < 1e-6f);
    const PassImage black = YuvToRgb(Frame(4, 2, 16, 128, 128), ci, PixelType::F32);
    for (size_t c = 0; c < 3; ++c) CHECK(black.Get(0, 0, c) == 0.f);
    const PassImage grey = YuvToRgb(Frame(4, 2, 126, 128, 128), ci, PixelType::F32);  // (126-16)/219
    for (size_t c = 0; c < 3; ++c) CHECK(std::fabs(grey.Get(3, 1, c) - 110.f / 219.f) < 1e-6f);
    // super-white clamps
    const PassImage sw = YuvToRgb(Frame(2, 2, 255, 128, 128), ci, PixelType::F32);
    CHECK(sw.Get(0, 0, 0) == 1.f);
}

TEST_CASE("YUV -> RGB: full range and BT.601 red", "[convert][color]") {
    ColorInfo full{ColorMatrix::Bt709, ColorRange::Full};
    const PassImage w = YuvToRgb(Frame(2, 2, 255, 128, 128), full, PixelType::F32);
    CHECK(std::fabs(w.Get(0, 0, 1) - 1.f) < 1e-6f);
    // BT.601 limited: pure red (R=1,G=0,B=0) is Y=81, Cb=90, Cr=240
    ColorInfo c601{ColorMatrix::Bt601, ColorRange::Limited};
    const PassImage red = YuvToRgb(Frame(2, 2, 81, 90, 240), c601, PixelType::F32);
    CHECK(std::fabs(red.Get(0, 0, 0) - 1.f) < 0.01f);
    CHECK(red.Get(0, 0, 1) < 0.01f);
    CHECK(red.Get(0, 0, 2) < 0.01f);
    // BT.709 limited: pure blue is Y=32, Cb=240, Cr=118
    ColorInfo c709{ColorMatrix::Bt709, ColorRange::Limited};
    const PassImage blue = YuvToRgb(Frame(2, 2, 32, 240, 118), c709, PixelType::F32);
    CHECK(blue.Get(1, 1, 2) > 0.98f);
    CHECK(blue.Get(1, 1, 0) < 0.02f);
}

TEST_CASE("YUV -> RGB: output types and chroma replication", "[convert][color]") {
    ColorInfo ci{ColorMatrix::Bt709, ColorRange::Limited};
    CpuFrame f = Frame(4, 2, 126, 128, 128);  // mid grey so a chroma shift is visible without clamping
    f.Plane(1)[0] = 200;  // first 2x2 block gets a chroma shift
    const PassImage u16 = YuvToRgb(f, ci, PixelType::U16);
    CHECK(u16.type == PixelType::U16);
    CHECK(u16.Get(3, 1, 0) == std::round(110.f / 219.f * 65535.f));
    // the chroma shift affects pixels (0..1, 0..1) identically, not (2, 0)
    CHECK(u16.Get(0, 0, 2) == u16.Get(1, 1, 2));
    CHECK(u16.Get(0, 0, 2) != u16.Get(2, 0, 2));
    const PassImage f16 = YuvToRgb(f, ci, PixelType::F16);
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

// ---- TASK-0034: any bit depth and chroma layout; the 8-bit 4:2:0 path stays bit-identical ----------------------------

namespace {

// The 8-bit 4:2:0 converters as they were before the formats work: the generic ones must match them bit for bit
// (the golden references were made with these).
PassImage RefYuv420pToRgb(const CpuFrame& frame, const ColorInfo& info, PixelType outType) {
    const uint32_t w = frame.desc.width, h = frame.desc.height;
    PassImage out = MakePassImage(PassKind::ColorSource, w, h, outType);
    const bool full = info.range == ColorRange::Full;
    const float yScale = full ? 1.f / 255.f : 1.f / 219.f;
    const float yOff = full ? 0.f : 16.f;
    const float cScale = full ? 1.f / 255.f : 1.f / 224.f;
    const float kr = info.matrix == ColorMatrix::Bt709 ? 0.2126f : 0.299f;
    const float kb = info.matrix == ColorMatrix::Bt709 ? 0.0722f : 0.114f;
    const float kg = 1.f - kr - kb;
    const float crR = 2.f * (1.f - kr), cbB = 2.f * (1.f - kb), cbG = -2.f * kb * (1.f - kb) / kg, crG = -2.f * kr * (1.f - kr) / kg;
    const uint8_t* Y = frame.Plane(0);
    const uint8_t* U = frame.Plane(1);
    const uint8_t* V = frame.Plane(2);
    const size_t cw = frame.desc.PlaneWidth(1);
    const float scaleOut = outType == PixelType::U16 ? 65535.f : (outType == PixelType::U8 ? 255.f : 1.f);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float yy = (static_cast<float>(Y[static_cast<size_t>(y) * w + x]) - yOff) * yScale;
            const size_t ci = static_cast<size_t>(y / 2) * cw + x / 2;
            const float cb = (static_cast<float>(U[ci]) - 128.f) * cScale;
            const float cr = (static_cast<float>(V[ci]) - 128.f) * cScale;
            out.Set(x, y, 0, std::clamp(yy + crR * cr, 0.f, 1.f) * scaleOut);
            out.Set(x, y, 1, std::clamp(yy + cbG * cb + crG * cr, 0.f, 1.f) * scaleOut);
            out.Set(x, y, 2, std::clamp(yy + cbB * cb, 0.f, 1.f) * scaleOut);
        }
    return out;
}

void RefRgbToYuv420p(const PassImage& rgb, const ColorInfo& info, CpuFrame& out) {
    const uint32_t w = rgb.width, h = rgb.height;
    out.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    const bool full = info.range == ColorRange::Full;
    const float kr = info.matrix == ColorMatrix::Bt709 ? 0.2126f : 0.299f;
    const float kb = info.matrix == ColorMatrix::Bt709 ? 0.0722f : 0.114f;
    const float kg = 1.f - kr - kb;
    const float scale = rgb.type == PixelType::U8 ? 1.f / 255.f : rgb.type == PixelType::U16 ? 1.f / 65535.f : 1.f;
    const float yScale = full ? 255.f : 219.f, yOff = full ? 0.f : 16.f, cScale = full ? 255.f : 224.f;
    uint8_t* Y = out.Plane(0);
    uint8_t* U = out.Plane(1);
    uint8_t* V = out.Plane(2);
    const size_t cw = out.desc.PlaneWidth(1), ch = out.desc.PlaneHeight(1);
    std::vector<float> cb(static_cast<size_t>(w) * h), cr(static_cast<size_t>(w) * h);
    auto q = [](float v) { return static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L)); };
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float r = std::clamp(rgb.Get(x, y, 0) * scale, 0.f, 1.f);
            const float g = std::clamp(rgb.Get(x, y, 1) * scale, 0.f, 1.f);
            const float b = std::clamp(rgb.Get(x, y, 2) * scale, 0.f, 1.f);
            const float yy = kr * r + kg * g + kb * b;
            Y[static_cast<size_t>(y) * w + x] = q(yOff + yScale * yy);
            cb[static_cast<size_t>(y) * w + x] = (b - yy) / (2.f * (1.f - kb));
            cr[static_cast<size_t>(y) * w + x] = (r - yy) / (2.f * (1.f - kr));
        }
    for (size_t cy = 0; cy < ch; ++cy)
        for (size_t cx = 0; cx < cw; ++cx) {
            float sb = 0.f, sr = 0.f;
            int n = 0;
            for (uint32_t dy = 0; dy < 2; ++dy)
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    const uint32_t x = std::min<uint32_t>(static_cast<uint32_t>(cx * 2 + dx), w - 1);
                    const uint32_t y = std::min<uint32_t>(static_cast<uint32_t>(cy * 2 + dy), h - 1);
                    sb += cb[static_cast<size_t>(y) * w + x];
                    sr += cr[static_cast<size_t>(y) * w + x];
                    ++n;
                }
            U[cy * cw + cx] = q(128.f + cScale * sb / static_cast<float>(n));
            V[cy * cw + cx] = q(128.f + cScale * sr / static_cast<float>(n));
        }
}

CpuFrame Pattern8(uint32_t w, uint32_t h) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    for (size_t i = 0; i < f.desc.PlaneSize(0); ++i) f.Plane(0)[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
    for (size_t i = 0; i < f.desc.PlaneSize(1); ++i) f.Plane(1)[i] = static_cast<uint8_t>((i * 13 + 40) & 0xFF);
    for (size_t i = 0; i < f.desc.PlaneSize(2); ++i) f.Plane(2)[i] = static_cast<uint8_t>((i * 5 + 200) & 0xFF);
    return f;
}

CpuFrame Frame16(uint32_t w, uint32_t h, PixelFormat fmt, int depth, uint16_t y, uint16_t u, uint16_t v) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, fmt, static_cast<uint8_t>(depth)});
    std::fill(f.Plane16(0), f.Plane16(0) + f.desc.PlaneCols(0) * f.desc.PlaneHeight(0), y);
    std::fill(f.Plane16(1), f.Plane16(1) + f.desc.PlaneCols(1) * f.desc.PlaneHeight(1), u);
    std::fill(f.Plane16(2), f.Plane16(2) + f.desc.PlaneCols(2) * f.desc.PlaneHeight(2), v);
    return f;
}

}  // namespace

TEST_CASE("8-bit 4:2:0 conversions are bit-identical to the pre-10-bit implementation", "[convert][color][regression]") {
    const CpuFrame f = Pattern8(37, 23);  // odd sizes: the chroma edge handling too
    for (const ColorInfo ci : {ColorInfo{ColorMatrix::Bt709, ColorRange::Limited}, ColorInfo{ColorMatrix::Bt601, ColorRange::Full}}) {
        for (const PixelType t : {PixelType::F32, PixelType::F16, PixelType::U16, PixelType::U8}) {
            const PassImage a = YuvToRgb(f, ci, t), b = RefYuv420pToRgb(f, ci, t);
            REQUIRE(a.SameLayout(b));
            CHECK(a.data == b.data);
        }
        const PassImage rgba = YuvToRgba16f(f, ci), ref16 = RefYuv420pToRgb(f, ci, PixelType::F16);
        for (uint32_t y = 0; y < f.desc.height; ++y)
            for (uint32_t x = 0; x < f.desc.width; ++x)
                for (size_t c = 0; c < 3; ++c) REQUIRE(rgba.As<uint16_t>()[(static_cast<size_t>(y) * f.desc.width + x) * 4 + c] == ref16.As<uint16_t>()[(static_cast<size_t>(y) * f.desc.width + x) * 3 + c]);
        const PassImage rgb = YuvToRgb(f, ci, PixelType::F16);
        CpuFrame back, refBack;
        RgbToYuv420p(rgb, ci, back);
        RefRgbToYuv420p(rgb, ci, refBack);
        REQUIRE(back.desc == refBack.desc);
        CHECK(back.data == refBack.data);
    }
}

TEST_CASE("10-bit and 12-bit limited range scale with the depth", "[convert][color]") {
    const ColorInfo ci{ColorMatrix::Bt709, ColorRange::Limited};
    const PassImage white = YuvToRgb(Frame16(4, 2, PixelFormat::Yuv420p16, 10, 940, 512, 512), ci, PixelType::F32);
    for (size_t c = 0; c < 3; ++c) CHECK(std::fabs(white.Get(1, 1, c) - 1.f) < 1e-6f);
    const PassImage black = YuvToRgb(Frame16(4, 2, PixelFormat::Yuv420p16, 10, 64, 512, 512), ci, PixelType::F32);
    for (size_t c = 0; c < 3; ++c) CHECK(black.Get(0, 0, c) == 0.f);
    const PassImage grey = YuvToRgb(Frame16(4, 2, PixelFormat::Yuv422p16, 10, 502, 512, 512), ci, PixelType::F32);  // (502-64)/876 = 0.5
    for (size_t c = 0; c < 3; ++c) CHECK(std::fabs(grey.Get(3, 1, c) - 0.5f) < 1e-6f);
    const PassImage superWhite = YuvToRgb(Frame16(2, 2, PixelFormat::Yuv444p16, 10, 1023, 512, 512), ci, PixelType::F32);
    CHECK(superWhite.Get(0, 0, 0) == 1.f);
    // BT.709 pure blue at 10 bits is 4x the 8-bit code values 32 / 240 / 118
    const PassImage blue = YuvToRgb(Frame16(2, 2, PixelFormat::Yuv420p16, 10, 128, 960, 472), ci, PixelType::F32);
    CHECK(blue.Get(1, 1, 2) > 0.98f);
    CHECK(blue.Get(1, 1, 0) < 0.02f);
    CHECK(blue.Get(1, 1, 1) < 0.02f);
    const PassImage white12 = YuvToRgb(Frame16(2, 2, PixelFormat::Yuv444p16, 12, 3760, 2048, 2048), ci, PixelType::F32);
    CHECK(std::fabs(white12.Get(1, 1, 1) - 1.f) < 1e-6f);
    const ColorInfo full{ColorMatrix::Bt709, ColorRange::Full};
    const PassImage fullWhite = YuvToRgb(Frame16(2, 2, PixelFormat::Yuv420p16, 10, 1023, 512, 512), full, PixelType::F32);
    CHECK(std::fabs(fullWhite.Get(0, 0, 0) - 1.f) < 1e-6f);
    const PassImage fullGrey16 = YuvToRgb(Frame16(2, 2, PixelFormat::Yuv420p16, 16, 32768, 32768, 32768), full, PixelType::F32);
    CHECK(std::fabs(fullGrey16.Get(0, 0, 0) - 32768.f / 65535.f) < 1e-6f);
    // the RGBA16F path agrees with the generic one
    const CpuFrame g = Frame16(4, 2, PixelFormat::Yuv422p16, 10, 502, 600, 400);
    const PassImage a = YuvToRgba16f(g, ci), b = YuvToRgb(g, ci, PixelType::F16);
    for (size_t i = 0; i < 8; ++i)
        for (size_t c = 0; c < 3; ++c) CHECK(a.As<uint16_t>()[i * 4 + c] == b.As<uint16_t>()[i * 3 + c]);
    CHECK(a.As<uint16_t>()[3] == 0x3C00);  // alpha 1.0
    CpuFrame empty;
    empty.desc = FrameDesc{4, 2, PixelFormat::Yuv420p16, 10};
    CHECK_THROWS(YuvToRgb(empty, ci));
}

TEST_CASE("10-bit round trip keeps every luma code; 4:2:2 and 4:4:4 chroma are replicated and averaged per block", "[convert][color]") {
    // every 10-bit luma code in one row, full range: RGB -> back must give the code itself
    const ColorInfo full{ColorMatrix::Bt709, ColorRange::Full};
    CpuFrame ramp;
    ramp.Allocate(FrameDesc{1024, 1, PixelFormat::Yuv444p16, 10});
    for (uint32_t x = 0; x < 1024; ++x) {
        ramp.Plane16(0)[x] = static_cast<uint16_t>(x);
        ramp.Plane16(1)[x] = 512;
        ramp.Plane16(2)[x] = 512;
    }
    const PassImage rgb32 = YuvToRgb(ramp, full, PixelType::F32);
    CpuFrame back;
    RgbToYuv(rgb32, full, FrameDesc{0, 0, PixelFormat::Yuv444p16, 10}, back);
    REQUIRE(back.desc == ramp.desc);
    int exact = 0;
    for (uint32_t x = 0; x < 1024; ++x) exact += back.Plane16(0)[x] == x;
    CHECK(exact == 1024);
    CHECK(back.Plane16(1)[100] == 512);
    // through half floats the code can move by one at most
    const PassImage rgb16 = YuvToRgb(ramp, full, PixelType::F16);
    CpuFrame back16;
    RgbToYuv(rgb16, full, FrameDesc{0, 0, PixelFormat::Yuv444p16, 10}, back16);
    int maxDiff = 0;
    for (uint32_t x = 0; x < 1024; ++x) maxDiff = std::max(maxDiff, std::abs(static_cast<int>(back16.Plane16(0)[x]) - static_cast<int>(x)));
    CHECK(maxDiff <= 1);
    // limited range: the legal codes 64..940 come back exactly
    const ColorInfo lim{ColorMatrix::Bt709, ColorRange::Limited};
    CpuFrame backLim;
    RgbToYuv(YuvToRgb(ramp, lim, PixelType::F32), lim, FrameDesc{0, 0, PixelFormat::Yuv444p16, 10}, backLim);
    for (uint32_t x = 64; x <= 940; x += 7) CHECK(backLim.Plane16(0)[x] == x);
    CHECK(backLim.Plane16(0)[1000] == 940);  // above white clamps to white

    // 4:2:2: chroma shared by horizontal pairs, every row its own; 4:4:4: every pixel its own
    CpuFrame c422;
    c422.Allocate(FrameDesc{4, 2, PixelFormat::Yuv422p16, 10});
    std::fill(c422.Plane16(0), c422.Plane16(0) + 8, static_cast<uint16_t>(502));
    const uint16_t u[4] = {400, 600, 300, 700};  // (col 0, row 0), (col 1, row 0), (col 0, row 1), (col 1, row 1)
    for (int i = 0; i < 4; ++i) {
        c422.Plane16(1)[i] = u[i];
        c422.Plane16(2)[i] = 512;
    }
    const PassImage r422 = YuvToRgb(c422, lim, PixelType::F32);
    CHECK(r422.Get(0, 0, 2) == r422.Get(1, 0, 2));  // the pair shares its chroma
    CHECK(r422.Get(0, 0, 2) != r422.Get(2, 0, 2));
    CHECK(r422.Get(0, 0, 2) != r422.Get(0, 1, 2));  // rows differ: no vertical subsampling
    CpuFrame back422;
    RgbToYuv(r422, lim, FrameDesc{0, 0, PixelFormat::Yuv422p16, 10}, back422);
    for (int i = 0; i < 4; ++i) CHECK(std::abs(static_cast<int>(back422.Plane16(1)[i]) - u[i]) <= 1);
    CpuFrame c444;
    c444.Allocate(FrameDesc{2, 1, PixelFormat::Yuv444p, 8});
    c444.Plane(0)[0] = c444.Plane(0)[1] = 126;
    c444.Plane(1)[0] = 100;
    c444.Plane(1)[1] = 160;
    c444.Plane(2)[0] = c444.Plane(2)[1] = 128;
    const PassImage r444 = YuvToRgb(c444, lim, PixelType::F32);
    CHECK(r444.Get(0, 0, 2) != r444.Get(1, 0, 2));
    CpuFrame back444;
    RgbToYuv(r444, lim, FrameDesc{0, 0, PixelFormat::Yuv444p, 8}, back444);
    CHECK(back444.Plane(1)[0] == 100);
    CHECK(back444.Plane(1)[1] == 160);
    // RgbToYuv takes the size from the image and the layout from the target
    CpuFrame to10;
    RgbToYuv(r444, lim, FrameDesc{99, 99, PixelFormat::Yuv420p16, 10}, to10);
    CHECK(to10.desc == FrameDesc{2, 1, PixelFormat::Yuv420p16, 10});
    CHECK(std::abs(static_cast<int>(to10.Plane16(0)[0]) - 504) <= 1);  // 126 at 8 bits is 504 at 10
    CHECK(std::abs(static_cast<int>(to10.Plane16(1)[0]) - 520) <= 1);  // the mean of 100 and 160 is 130 -> 520
}
