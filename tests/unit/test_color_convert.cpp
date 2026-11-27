#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "convert/ColorConvert.h"
#include "convert/ColorPlan.h"

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
    CHECK(ToString(ColorMatrix::Bt709) == "bt709");  // stage_params.matrix of every SDR manifest
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

// TASK-0040: the viewport converts in bands of rows on several threads (8-12 ms single-threaded at 1080p); every pixel
// is computed alone, so any thread count - more threads than rows too - gives the same halves.
TEST_CASE("YuvToRgba16f in bands of rows is the same bit for bit whatever the thread count", "[convert][color][regression]") {
    struct Layout {
        uint32_t w, h;
        PixelFormat fmt;
        uint8_t depth;
    };
    for (const Layout lay : {Layout{37, 23, PixelFormat::Yuv420p, 8}, Layout{64, 40, PixelFormat::Yuv422p16, 10}, Layout{30, 5, PixelFormat::Yuv444p, 8},
                             Layout{48, 300, PixelFormat::Yuv420p16, 12}}) {
        CpuFrame f;
        f.Allocate(FrameDesc{lay.w, lay.h, lay.fmt, lay.depth});
        uint32_t seed = 12345;
        const uint32_t maxv = (1u << lay.depth) - 1u;
        if (f.desc.HighBitDepth()) {
            auto* p = reinterpret_cast<uint16_t*>(f.data.data());
            for (size_t i = 0; i < f.data.size() / 2; ++i) p[i] = static_cast<uint16_t>((seed = seed * 1664525u + 1013904223u) >> 16 & maxv);
        } else {
            for (auto& b : f.data) b = static_cast<uint8_t>((seed = seed * 1664525u + 1013904223u) >> 24);
        }
        const ColorInfo ci{ColorMatrix::Bt709, ColorRange::Limited};
        const PassImage one = YuvToRgba16f(f, ci, 1);
        for (const int threads : {2, 3, 7, 16, 0}) {
            INFO(lay.w << "x" << lay.h << " threads " << threads);
            CHECK(YuvToRgba16f(f, ci, threads).data == one.data);
        }
    }
}

// ---- TASK-0075 (HDR, PR TA): the BT.2020 matrix, one ResolveColor for every consumer, YuvToRgb in bands -----------------

namespace {

// ColorTagsFor as it was before ResolveColor (frozen here): the result tags of every SDR source must stay these.
ColorTags FrozenColorTagsFor(const ColorInfo& ci, const VideoStreamInfo& info) {
    ColorTags t;
    t.range = ci.range == ColorRange::Full ? 2 : 1;
    if (ci.matrix == ColorMatrix::Bt709) t.matrix = 1;
    else t.matrix = info.colorSpace == 5 ? 5 : 6;
    t.primaries = info.colorPrimaries;
    t.transfer = info.colorTrc;
    return t;
}

bool Same(const ColorInfo& a, const ColorInfo& b) { return a.matrix == b.matrix && a.range == b.range; }
bool Same(const ColorTags& a, const ColorTags& b) { return a.range == b.range && a.matrix == b.matrix && a.primaries == b.primaries && a.transfer == b.transfer; }

}  // namespace

// Pure BT.2020 red is (Kr, -Kr / (2 (1 - Kb)), 1/2): Y 294, Cb 387, Cr 960 at 10 bits limited. White (940/512/512) would not
// tell the matrices apart - Kr and Kb do not enter it.
TEST_CASE("BT.2020 matrix: pure red at 10 bits limited is Y 294, Cb 387, Cr 960 both ways", "[convert][color][hdr]") {
    const ColorInfo c2020{ColorMatrix::Bt2020, ColorRange::Limited};
    CHECK(ToString(ColorMatrix::Bt2020) == "bt2020");
    const PassImage red = YuvToRgb(Frame16(2, 2, PixelFormat::Yuv444p16, 10, 294, 387, 960), c2020, PixelType::F32);
    CHECK(std::fabs(red.Get(1, 1, 0) - 1.f) < 1e-3f);
    CHECK(std::fabs(red.Get(1, 1, 1)) < 1e-3f);
    CHECK(std::fabs(red.Get(1, 1, 2)) < 1e-3f);
    PassImage rgb;
    rgb.Allocate(2, 2, PixelType::F32, {"R", "G", "B"});
    for (uint32_t y = 0; y < 2; ++y)
        for (uint32_t x = 0; x < 2; ++x) rgb.Set(x, y, 0, 1.f);
    CpuFrame yuv;
    RgbToYuv(rgb, c2020, FrameDesc{0, 0, PixelFormat::Yuv444p16, 10}, yuv);
    CHECK(yuv.Plane16(0)[3] == 294);
    CHECK(yuv.Plane16(1)[3] == 387);
    CHECK(yuv.Plane16(2)[3] == 960);
    // the result of a BT.2020 conversion is tagged BT.2020 NCL, whatever the source said
    VideoStreamInfo i;
    i.colorSpace = 1;
    CHECK(ResultColorTags(c2020, i).matrix == 9);
}

// TA moved every consumer onto ResolveColor without changing what any of them reads; TB-2 (TASK-0079) stops every other source. For
// every tag the stream can carry, both ranges and both sides of the 720-line rule: a source on the legacy path has each role
// ColorInfoFromStream and the result tags the old ones; any other has its processing roles throw, its viewport / measure roles still
// the SDR reading, and the tags WrittenTags gives.
TEST_CASE("ResolveColor keeps legacy cells as before, stops every other cell and mixed signalling, refuses passthrough for IPT and RGB",
          "[convert][color][hdr][regression]") {
    VideoStreamInfo i;
    int checked = 0, wrong = 0, legacy = 0, stopped = 0;
    auto stops = [](auto&& role) {
        try {
            role();
        } catch (const ColourUnsupported&) {
            return true;
        }
        return false;
    };
    for (const uint32_t h : {576u, 1080u})
        for (int range = 0; range <= 2; ++range)
            for (int m = 0; m <= 18; ++m)
                for (int p = 0; p <= 18; ++p)
                    for (int t = 0; t <= 18; ++t) {
                        i.width = h * 16 / 9;
                        i.height = h;
                        i.codedHeight = h;
                        i.colorRange = range;
                        i.colorSpace = m;
                        i.colorPrimaries = p;
                        i.colorTrc = t;
                        ColourLayers in;
                        in.stream = in.container = RawTags{m, p, t, range, 0};
                        i.color = SourceColorOf(in, "420", "hevc");
                        const ColorInfo sdr = ColorInfoFromStream(i);
                        const ColorPlan plan = ResolveColor(i);
                        bool ok = plan.legacy == i.color.legacyPath && Same(plan.View(), sdr) && Same(plan.MeasureDecode(), sdr);
                        if (i.color.legacyPath) {
                            ++legacy;
                            ok = ok && plan.process == ColourStop::None && plan.all == ColourStop::None && Same(plan.Colour(), sdr) &&
                                 Same(plan.Guide(), sdr) && Same(plan.ResultEncode(), sdr) && Same(plan.ResultTags(), FrozenColorTagsFor(sdr, i)) &&
                                 plan.ResultMeta().Empty();
                        } else {
                            ++stopped;
                            const bool ipt = m == 15, rgb = m == 0 && (t == 16 || t == 18 || p == 9);
                            const ColourStop all = ipt ? ColourStop::DolbyVisionIpt : rgb ? ColourStop::RgbHdr : ColourStop::None;
                            const ColourStop process = all != ColourStop::None ? all : (m == 10 || m == 14) ? ColourStop::Matrix : ColourStop::NotYet;
                            ok = ok && plan.all == all && plan.process == process && !plan.why.empty() &&
                                 stops([&] { (void)plan.Colour(); }) && stops([&] { (void)plan.Guide(); }) && stops([&] { (void)plan.ResultEncode(); }) &&
                                 Same(plan.ResultTags(), WrittenTags(i.color));
                        }
                        if (!ok && ++wrong <= 5) {
                            INFO("height " << h << " range " << range << " matrix " << m << " primaries " << p << " trc " << t);
                            CHECK(ok);
                        }
                        ++checked;
                    }
    CHECK(wrong == 0);
    CHECK(checked == 2 * 3 * 19 * 19 * 19);
    CHECK(legacy > 0);
    CHECK(stopped > 0);
    // mixed signalling: SDR frames over PQ codec parameters (a ProRes colr) - stopped, not read as SDR
    ColourLayers mixed;
    mixed.stream = mixed.container = RawTags{9, 9, 16, 1, 0};
    mixed.containerRecords.mdcv = Mdcv{};  // a stale colr's records: not written with the frames' SDR tags
    mixed.containerRecords.mdcv->maxLum = {1000, 1};
    mixed.containerRecords.mdcv->hasLuminance = true;
    mixed.containerRecords.cll = Cll{1000, 400};
    FirstLookColor frame;
    frame.how = FirstLook::Decode;
    frame.tags = RawTags{1, 1, 1, 1, 0};
    mixed.first = frame;
    i.color = SourceColorOf(mixed, "422", "prores");
    REQUIRE(i.color.cls == SourceClass::Legacy);
    REQUIRE(i.color.hdr.cll);
    const ColorPlan m = ResolveColor(i);
    CHECK(m.process == ColourStop::NotYet);
    CHECK(m.why.find("layers disagree") != std::string::npos);
    CHECK(m.ResultTags().transfer == 1);  // the frames' tags as they are
    CHECK(m.ResultMeta().Empty());
    CHECK_THROWS_AS(RequireColour(m, ColourUse::Process, "process"), ColourUnsupported);
    CHECK_NOTHROW(RequireColour(m, ColourUse::Passthrough, "process"));
    // an IPT base: refused in every mode, with its kind
    ColourLayers ipt;
    DoviConf p5;
    p5.profile = 5;
    ipt.dovi = p5;
    i.color = SourceColorOf(ipt, "420", "hevc");
    try {
        RequireColour(ResolveColor(i), ColourUse::Passthrough, "process --passthrough: a.mkv");
        FAIL("passthrough of an IPT base was not refused");
    } catch (const ColourUnsupported& e) {
        CHECK(e.kind == ColourStop::DolbyVisionIpt);
        CHECK(std::string(e.what()).rfind("process --passthrough: a.mkv: a Dolby Vision IPT base layer", 0) == 0);
    }
}

// The guides (depth, flow), FG, export, compare and the thumbnails convert with YuvToRgb; in bands of rows it must give the very same
// values as the single-threaded walk it was, whatever the count - more threads than rows too.
TEST_CASE("YuvToRgb in bands of rows is the same bit for bit whatever the thread count", "[convert][color][hdr][regression]") {
    struct Layout {
        uint32_t w, h;
        PixelFormat fmt;
        uint8_t depth;
    };
    for (const Layout lay : {Layout{22, 37, PixelFormat::Yuv420p, 8}, Layout{22, 37, PixelFormat::Yuv422p, 8}, Layout{22, 37, PixelFormat::Yuv444p, 8},
                             Layout{22, 37, PixelFormat::Yuv420p16, 10}, Layout{22, 37, PixelFormat::Yuv422p16, 12}, Layout{22, 37, PixelFormat::Yuv444p16, 10},
                             Layout{48, 301, PixelFormat::Yuv420p16, 10}}) {
        CpuFrame f;
        f.Allocate(FrameDesc{lay.w, lay.h, lay.fmt, lay.depth});
        uint32_t seed = 4242;
        const uint32_t maxv = (1u << lay.depth) - 1u;
        if (f.desc.HighBitDepth()) {
            auto* p = reinterpret_cast<uint16_t*>(f.data.data());
            for (size_t k = 0; k < f.data.size() / 2; ++k) p[k] = static_cast<uint16_t>((seed = seed * 1664525u + 1013904223u) >> 16 & maxv);
        } else {
            for (auto& b : f.data) b = static_cast<uint8_t>((seed = seed * 1664525u + 1013904223u) >> 24);
        }
        for (const ColorInfo ci : {ColorInfo{ColorMatrix::Bt709, ColorRange::Limited}, ColorInfo{ColorMatrix::Bt2020, ColorRange::Full}})
            for (const PixelType t : {PixelType::F32, PixelType::F16, PixelType::U16, PixelType::U8}) {
                const PassImage one = YuvToRgb(f, ci, t, 1);
                for (const int threads : {2, 3, 8, 64, 0}) {
                    INFO(ToString(lay.fmt) << " " << int(lay.depth) << " bits " << lay.w << "x" << lay.h << " " << ToString(ci.matrix) << " threads " << threads);
                    CHECK(YuvToRgb(f, ci, t, threads).data == one.data);
                }
            }
    }
}

// A frame without its data is refused before the bands start: thrown inside a worker thread (or while workers run) it
// would end the process with std::terminate instead of reaching the caller's catch (thumbnails, stages).
TEST_CASE("YuvToRgb and YuvToRgba16f refuse a frame without data before converting in bands", "[convert][color][hdr][regression]") {
    const ColorInfo ci{ColorMatrix::Bt709, ColorRange::Limited};
    CpuFrame tall;
    tall.desc = FrameDesc{4, 300, PixelFormat::Yuv420p16, 10};  // >= 256 rows: the automatic count takes several threads too
    tall.data.resize(16);                                        // short, not empty
    for (const int threads : {4, 0}) {
        INFO("threads " << threads);
        CHECK_THROWS(YuvToRgb(tall, ci, PixelType::F16, threads));
        CHECK_THROWS(YuvToRgba16f(tall, ci, threads));
    }
}

// TASK-0080 (TC-1, guard of the SDR path): YuvToRgb of 10-bit 4:2:0 and 4:4:4 frames is frozen - the HDR->SDR branch of ForEachRgbOf
// (ColorInfo::convert) must leave the plain branch bit for bit. The 8-bit 4:2:0 path is held by the reference converters above.
TEST_CASE("The SDR branch of YuvToRgb is untouched: frozen 10-bit samples", "[convert][color][hdr][regression]") {
    auto fnv = [](const std::vector<uint8_t>& bytes) {
        uint64_t h = 1469598103934665603ull;
        for (const uint8_t b : bytes) h = (h ^ b) * 1099511628211ull;
        return h;
    };
    struct Case {
        PixelFormat fmt;
        ColorInfo ci;
        PixelType type;
        uint64_t hash;
    };
    const ColorInfo l709{ColorMatrix::Bt709, ColorRange::Limited}, f601{ColorMatrix::Bt601, ColorRange::Full}, l2020{ColorMatrix::Bt2020, ColorRange::Limited};
    const Case cases[] = {
        {PixelFormat::Yuv420p16, l709, PixelType::F32, 0xc76b94cfdca6a4b2ull},  {PixelFormat::Yuv420p16, l709, PixelType::F16, 0xc9a6e8f67b4d4969ull},
        {PixelFormat::Yuv420p16, l709, PixelType::U16, 0xa99976db72db8c0cull},  {PixelFormat::Yuv420p16, l709, PixelType::U8, 0x64e667bb7f2d9351ull},
        {PixelFormat::Yuv420p16, f601, PixelType::F32, 0x88f311b61c8e95f9ull},  {PixelFormat::Yuv420p16, l2020, PixelType::F32, 0x4d24493ad006ad8bull},
        {PixelFormat::Yuv444p16, l709, PixelType::F32, 0x6537dc22561faa0eull},  {PixelFormat::Yuv444p16, l2020, PixelType::F16, 0x85d500dc73567988ull},
    };
    for (const Case& c : cases) {
        CpuFrame f;
        f.Allocate(FrameDesc{22, 37, c.fmt, 10});
        uint32_t seed = 2026;
        auto* p = reinterpret_cast<uint16_t*>(f.data.data());
        for (size_t k = 0; k < f.data.size() / 2; ++k) p[k] = static_cast<uint16_t>((seed = seed * 1664525u + 1013904223u) >> 16 & 1023u);
        INFO(ToString(c.fmt) << " " << ToString(c.ci.matrix) << " " << ToString(c.type));
        CHECK(fnv(YuvToRgb(f, c.ci, c.type).data) == c.hash);
        if (c.type == PixelType::F16) {
            const PassImage a = YuvToRgba16f(f, c.ci);
            std::vector<uint8_t> rgb;
            for (size_t i = 0; i < a.data.size() / 8; ++i) rgb.insert(rgb.end(), a.data.begin() + i * 8, a.data.begin() + i * 8 + 6);
            CHECK(fnv(rgb) == c.hash);
        }
    }
}
