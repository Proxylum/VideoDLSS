#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

#include "convert/SceneCut.h"

using namespace dlssvid;
using Catch::Approx;

namespace {
// An 8-bit 4:2:0 frame (limited range) from a luma function of (x, y) in [0, 1]; neutral chroma.
CpuFrame Frame(uint32_t w, uint32_t h, const std::function<double(uint32_t, uint32_t)>& luma) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) f.Plane(0)[y * w + x] = static_cast<uint8_t>(std::lround(16.0 + 219.0 * std::clamp(luma(x, y), 0.0, 1.0)));
    std::fill(f.data.begin() + static_cast<std::ptrdiff_t>(f.desc.PlaneOffset(1)), f.data.end(), uint8_t{128});
    return f;
}
CpuFrame Flat(uint32_t w, uint32_t h, uint8_t y) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    std::fill(f.data.begin(), f.data.end(), y);
    return f;
}
}  // namespace

TEST_CASE("MakeLumaThumb: an integer box to <= 640 wide, even sides, luma normalised by the range and the bit depth", "[unit][scene_cut]") {
    const ColorInfo limited{}, full{ColorMatrix::Bt709, ColorRange::Full};
    for (const auto& [w, h, tw, th] : std::vector<std::array<uint32_t, 4>>{
             {1920, 1080, 640, 360}, {3840, 2160, 640, 360}, {1280, 720, 640, 360}, {2560, 1440, 640, 360}, {48, 32, 48, 32}, {1281, 721, 426, 240}}) {
        INFO(w << "x" << h);
        const LumaThumb t = MakeLumaThumb(Flat(w, h, 126), limited);
        CHECK(t.width == tw);
        CHECK(t.height == th);
        CHECK(t.frameWidth == w);
        CHECK(t.y.size() == static_cast<size_t>(tw) * th);
    }
    CHECK(MakeLumaThumb(Flat(64, 32, 16), limited).y[0] == 0.f);
    CHECK(MakeLumaThumb(Flat(64, 32, 235), limited).y[0] == 1.f);
    CHECK(MakeLumaThumb(Flat(64, 32, 126), limited).y[0] == Approx(110.0 / 219.0));
    CHECK(MakeLumaThumb(Flat(64, 32, 250), limited).y[0] == 1.f);  // super-white clamped
    CHECK(MakeLumaThumb(Flat(64, 32, 255), full).y[0] == 1.f);
    CHECK(MakeLumaThumb(Flat(64, 32, 0), full).y[0] == 0.f);
    // 10-bit limited: 64..940
    CpuFrame ten;
    ten.Allocate(FrameDesc{64, 32, PixelFormat::Yuv420p16, 10});
    for (size_t i = 0; i < 64 * 32; ++i) ten.Plane16(0)[i] = 940;
    CHECK(MakeLumaThumb(ten, limited).y[0] == 1.f);
    for (size_t i = 0; i < 64 * 32; ++i) ten.Plane16(0)[i] = 64;
    CHECK(MakeLumaThumb(ten, limited).y[0] == 0.f);
    // the box: a 2-pixel checker of a 1280-wide frame averages to grey
    const LumaThumb box = MakeLumaThumb(Frame(1280, 720, [](uint32_t x, uint32_t y) { return ((x + y) % 2) ? 1.0 : 0.0; }), limited);
    CHECK(box.y[0] == Approx(0.5).margin(1e-6));
    CHECK(MakeLumaThumb(CpuFrame{}, limited).Empty());
    CHECK(SceneCutScore(MakeLumaThumb(Flat(64, 32, 126), limited), MakeLumaThumb(Flat(64, 32, 126), limited)) == 0.0);
    CHECK(SceneCutScore(MakeLumaThumb(Flat(64, 32, 126), limited), MakeLumaThumb(Flat(66, 32, 126), limited)) == 1.0);  // another size
}

// PR-6 (spec §2d): a hard cut resets the history; the motion of one scene, a dissolve and a mild flash do not. A strong flash does
// (into it and out of it): the known false reset, one frame of history each (spec §2d «Риски»).
TEST_CASE("SceneCutDetector: a hard cut and a size change are cuts, motion, a dissolve and a mild flash are not", "[unit][scene_cut][regression]") {
    const uint32_t w = 256, h = 144;
    auto sceneA = [](int t) { return [t](uint32_t x, uint32_t y) { return 0.5 + 0.35 * std::sin((x + 2.0 * t) * 0.05) * std::cos(y * 0.07); }; };  // pans 2 px a frame
    auto sceneB = [](uint32_t x, uint32_t y) { return ((x / 32 + y / 32) % 2) ? 0.85 : 0.15; };
    auto sceneC = [](uint32_t x, uint32_t y) { return 0.3 + 0.4 * x / 256.0 + 0.1 * y / 144.0; };
    std::vector<CpuFrame> clip;
    for (int t = 0; t < 5; ++t) clip.push_back(Frame(w, h, sceneA(t)));                       // 0..4  A, panning
    for (int t = 0; t < 3; ++t) clip.push_back(Frame(w, h, sceneB));                          // 5..7  hard cut to B
    for (int k = 1; k <= 8; ++k)                                                               // 8..15 dissolve B -> C
        clip.push_back(Frame(w, h, [&, k](uint32_t x, uint32_t y) { return sceneB(x, y) + (sceneC(x, y) - sceneB(x, y)) * k / 8.0; }));
    clip.push_back(Frame(w, h, [&](uint32_t x, uint32_t y) { return sceneC(x, y) + 0.12; }));  // 16 a mild flash
    clip.push_back(Frame(w, h, sceneC));                                                       // 17
    clip.push_back(Frame(w, h, [&](uint32_t x, uint32_t y) { return sceneC(x, y) + 0.45; }));  // 18 a strong flash
    clip.push_back(Frame(w, h, sceneC));                                                       // 19
    clip.push_back(Frame(w / 2, h / 2, sceneC));                                               // 20 another size

    SceneCutDetector det(ColorInfo{});
    std::vector<int> cuts;
    double cutScore = 0.0;
    for (size_t i = 0; i < clip.size(); ++i) {
        double s = -1.0;
        if (det.Next(clip[i], 0.24, &s)) cuts.push_back(static_cast<int>(i));
        CHECK(s >= 0.0);
        if (i == 5) cutScore = s;
        if (i == 0) CHECK(s == 0.0);  // the first frame: nothing to compare
    }
    CHECK(cuts == std::vector<int>{5, 18, 19, 20});
    CHECK(cutScore > 0.24);
    // a higher threshold: the hard cut (≈ 0.35 here) only above it is missed
    SceneCutDetector strict(ColorInfo{});
    std::vector<int> few;
    for (size_t i = 0; i < 8; ++i)
        if (strict.Next(clip[i], cutScore + 0.01)) few.push_back(static_cast<int>(i));
    CHECK(few.empty());
    // Reset: the next frame starts again (the stage's --start), an empty frame keeps the last thumb
    det.Reset();
    CHECK_FALSE(det.Next(clip[5], 0.24));
    CHECK_FALSE(det.Next(CpuFrame{}, 0.24));
    CHECK_FALSE(det.Next(clip[6], 0.24));
    CHECK(det.Next(clip[0], 0.24));
    // PR-13: Score is the same number without a threshold (the stage scenes stores it); an empty frame does not move the last thumb
    SceneCutDetector a(ColorInfo{}), b(ColorInfo{});
    for (size_t i = 0; i < clip.size(); ++i) {
        double s = -1.0;
        a.Next(clip[i], 0.24, &s);
        CHECK(b.Score(clip[i]) == s);
    }
    CHECK(b.Score(CpuFrame{}) == 0.0);
    CHECK(b.Score(clip.back()) == 0.0);  // the same frame as the last one given
}
