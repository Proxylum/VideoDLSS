// Frame generation building blocks (stage 7): color_fg indexing, the synthetic camera, the blend baseline,
// backends and availability messages.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "gpu/D3D12Device.h"
#include "stages/fg/BlendFrameGenerator.h"
#include "stages/fg/IFrameGenerator.h"

using namespace dlssvid;
using Catch::Approx;

TEST_CASE("color_fg indexing: real frames at i * mult, generated frames between, (N - 1) * mult + 1 in total", "[fg][unit]") {
    CHECK(FgRealIndex(0, 2) == 0);
    CHECK(FgRealIndex(3, 2) == 6);
    CHECK(FgInterpIndex(3, 1, 2) == 7);
    CHECK(FgRealIndex(4, 2) == 8);
    CHECK(FgInterpIndex(2, 1, 3) == 7);
    CHECK(FgInterpIndex(2, 2, 3) == 8);
    CHECK(FgRealIndex(3, 3) == 9);
    CHECK(FgFrameCount(0, 2) == 0);
    CHECK(FgFrameCount(1, 2) == 1);
    CHECK(FgFrameCount(45, 2) == 89);
    CHECK(FgFrameCount(10, 3) == 28);
    CHECK(FgFrameCount(10, 4) == 37);
}

TEST_CASE("The synthetic DLSS-G camera: perspective with the frame aspect, its inverse and an identity previous clip", "[fg][unit]") {
    const FgCamera c = BuildFgCamera(1920, 1080);
    CHECK(c.aspect == Approx(1920.0 / 1080.0));
    CHECK(c.fovRadians == Approx(60.0 * 3.14159265358979 / 180.0));
    CHECK(c.nearPlane == Approx(0.1f));
    CHECK(c.farPlane == Approx(1000.f));
    float prod[16];
    Mul4x4(c.viewToClip, c.clipToView, prod);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) CHECK(prod[i * 4 + j] == Approx(i == j ? 1.f : 0.f).margin(1e-4));
    Mul4x4(c.clipToView, c.viewToClip, prod);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) CHECK(prod[i * 4 + j] == Approx(i == j ? 1.f : 0.f).margin(1e-3));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) CHECK(c.identity[i * 4 + j] == (i == j ? 1.f : 0.f));
    // a point on the far plane lands at clip z = w (depth 1), on the near plane at z = 0
    const float pFar[4] = {0.f, 0.f, 1000.f, 1.f}, pNear[4] = {0.f, 0.f, 0.1f, 1.f};
    auto transform = [&](const float* p, float* out) {
        for (int j = 0; j < 4; ++j) {
            out[j] = 0.f;
            for (int k = 0; k < 4; ++k) out[j] += p[k] * c.viewToClip[k * 4 + j];
        }
    };
    float o[4];
    transform(pFar, o);
    CHECK(o[2] / o[3] == Approx(1.f).margin(1e-4));
    transform(pNear, o);
    CHECK(o[2] / o[3] == Approx(0.f).margin(1e-4));
}

TEST_CASE("Blend baseline: lerp(prev, cur, k / mult) frames, ToRgbF32 scaling", "[fg][unit]") {
    PassImage u8;
    u8.Allocate(2, 1, PixelType::U8, {"R", "G", "B"});
    for (int i = 0; i < 6; ++i) u8.As<uint8_t>()[i] = static_cast<uint8_t>(i == 0 ? 255 : 51);
    const PassImage f = ToRgbF32(u8);
    CHECK(f.type == PixelType::F32);
    CHECK(f.Get(0, 0, 0) == Approx(1.f));
    CHECK(f.Get(1, 0, 2) == Approx(0.2f));

    D3D12Device dev({true, false});
    BlendFrameGenerator blend;
    FgConfig cfg;
    cfg.backend = "blend";
    cfg.width = 4;
    cfg.height = 2;
    cfg.multiplier = 4;
    blend.Init(dev, cfg);
    CHECK(blend.Diagnostics().ok);
    PassImage a = MakePassImage(PassKind::ColorFg, 4, 2, PixelType::F16), b = a;
    for (uint32_t y = 0; y < 2; ++y)
        for (uint32_t x = 0; x < 4; ++x)
            for (size_t c = 0; c < 3; ++c) {
                a.Set(x, y, c, 0.2f);
                b.Set(x, y, c, 0.8f);
            }
    FgInputs in;
    in.prev = FgFrame{nullptr, &a};
    in.cur = FgFrame{nullptr, &b};
    in.reset = false;
    in.frameIndex = 1;
    std::vector<PassImage> out;
    blend.Generate(in, out);
    REQUIRE(out.size() == 3);
    CHECK(out[0].Get(1, 1, 0) == Approx(0.35f).margin(2e-3));
    CHECK(out[1].Get(3, 0, 1) == Approx(0.5f).margin(2e-3));
    CHECK(out[2].Get(0, 1, 2) == Approx(0.65f).margin(2e-3));
    CHECK(out[0].width == 4);
    CHECK(out[0].type == PixelType::F16);
    CHECK(blend.Calls() == 1);
    CHECK(!blend.LastReset());
    CHECK(blend.Describe()["multiplier"] == 4);
    PassImage other;
    other.Allocate(3, 2, PixelType::F16, {"R", "G", "B"});
    in.cur = FgFrame{nullptr, &other};
    CHECK_THROWS(blend.Generate(in, out));
}

TEST_CASE("FG backends, availability and instructions", "[fg][unit]") {
    CHECK(FgBackends() == std::vector<std::string>{"dlssg", "rife", "blend"});
    CHECK(FgAvailable("blend").available);
    const FgAvailability bogus = FgAvailable("bogus");
    CHECK(!bogus.available);
    CHECK(bogus.reason.find("unknown") != std::string::npos);
    CHECK_THROWS(CreateFrameGenerator("bogus"));
    CHECK(CreateFrameGenerator("blend")->Name() == "blend");
    CHECK(CreateFrameGenerator("dlssg")->Name() == "dlssg");
    CHECK(CreateFrameGenerator("rife")->Name() == "rife");
    const FgAvailability dlssg = FgAvailable("dlssg");
    if (!dlssg.available) CHECK((dlssg.reason.find(kDlssgDllName) != std::string::npos || dlssg.reason.find("DLSS SDK") != std::string::npos));
    const FgAvailability rife = FgAvailable("rife");
    if (!rife.available) CHECK(rife.reason.find("TensorRT") != std::string::npos);
    CHECK(DlssgDllInstruction().find("docs/dll-setup.md") != std::string::npos);
    FgDiagnostics d;
    d.gpu = "x";
    d.multiFrameMax = 3;
    CHECK(d.ToJson()["multi_frame_max"] == 3);
    CHECK(d.ToJson()["ok"] == false);
}
