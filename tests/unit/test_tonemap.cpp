// Tonemapper (stage 6): passthrough identity, ACES / Reinhard / input transfers against the CPU reference, on WARP.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "GpuTextures.h"
#include "gpu/D3D12Device.h"
#include "stages/ParamSchema.h"
#include "stages/tonemap/Tonemapper.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Approx;

namespace {
std::vector<float> RunTonemap(D3D12Device& dev, Tonemapper& tm, uint32_t w, uint32_t h, const std::vector<float>& rgb, const TonemapOptions& o) {
    auto src = UploadRgba16f(dev, w, h, rgb);
    auto dst = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { tm.Run(cl, src.Get(), dst.Get(), w, h, o); });
    return ReadbackRgb(dev, dst.Get(), w, h);
}
}  // namespace

TEST_CASE("TonemapReference: known values and identity", "[tonemap][unit]") {
    TonemapOptions o;
    CHECK(o.IsIdentity());
    CHECK(TonemapReference(0.25f, o) == Approx(0.25f));
    CHECK(TonemapReference(1.7f, o) == Approx(1.f));   // clamped
    CHECK(TonemapReference(-0.2f, o) == Approx(0.f));
    o.inputTransfer = "linear";
    CHECK(!o.IsIdentity());
    CHECK(TonemapReference(0.5f, o) == Approx(0.7354f).margin(1e-3));  // sRGB encode of linear 0.5
    o.curve = "aces";
    CHECK(TonemapReference(0.5f, o) == Approx(0.8073f).margin(1e-3));  // ACES(0.5) = 0.6163 -> sRGB
    CHECK(TonemapReference(100.f, o) == Approx(1.f).margin(1e-3));     // highlights compress to white
    CHECK(TonemapReference(0.f, o) == Approx(0.f).margin(1e-3));
    o.curve = "reinhard";
    CHECK(TonemapReference(1.f, o) == Approx(0.7354f).margin(1e-3));  // 1 / (1 + 1) = 0.5 linear
    o.curve = "passthrough";
    o.exposureEv = 1.f;  // +1 stop doubles linear light: 0.25 -> 0.5
    CHECK(TonemapReference(0.25f, o) == Approx(0.7354f).margin(1e-3));
    o.exposureEv = 0.f;
    o.inputTransfer = "pq";
    CHECK(TonemapReference(0.58f, o) == Approx(1.f).margin(2e-2));  // ~203 nits = reference white -> 1.0
    CHECK(TonemapReference(0.f, o) == Approx(0.f).margin(1e-3));
    o.inputTransfer = "hlg";
    CHECK(TonemapReference(0.5f, o) > TonemapReference(0.25f, o));
    o.inputTransfer = "bogus";
    CHECK_THROWS(TonemapReference(0.5f, o));
    o.inputTransfer = "srgb";
    o.curve = "bogus";
    CHECK_THROWS(TonemapReference(0.5f, o));
    CHECK(!TonemapCurveId("bogus"));
    CHECK(TonemapCurveId("aces") == 1);
    CHECK(TonemapTransferId("hlg") == 3);
}

// TASK-0043: the schema said exposure 0.0 in -8..8 while the stage multiplied linear light by it - an explicit 0 (the
// default the form shows) or a negative value gave black frames. Exposure is in stops now; 0 is the identity.
TEST_CASE("Exposure is in EV stops: 0 is the identity, +1 doubles linear light, -1 halves", "[tonemap][unit][regression]") {
    TonemapOptions o;
    CHECK(o.exposureEv == 0.f);
    CHECK(o.ExposureGain() == 1.f);
    CHECK(o.IsIdentity());
    o.exposureEv = 1.f;
    CHECK(o.ExposureGain() == 2.f);
    CHECK_FALSE(o.IsIdentity());
    o.exposureEv = -1.f;
    CHECK(o.ExposureGain() == 0.5f);
    o.inputTransfer = "linear";
    CHECK(TonemapReference(0.5f, o) == Approx(TonemapReference(0.25f, TonemapOptions{"passthrough", "linear", 0.f})));
    o.exposureEv = -8.f;  // the bottom of the range: dark, not black
    CHECK(TonemapReference(1.f, o) > 0.01f);
    // the schema's default is the stage's default
    const StageSchema* nr = FindStageSchema("nr");
    REQUIRE(nr);
    CHECK(nr->Find("exposure")->def.get<double>() == TonemapOptions{}.exposureEv);
}

TEST_CASE("Tonemapper: passthrough on sRGB input copies (and clamps) exactly", "[tonemap][unit][gpu]") {
    D3D12Device dev({true, false});
    Tonemapper tm(dev);
    const uint32_t w = 9, h = 5;
    std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = -0.3f + 1.6f * static_cast<float>(i) / static_cast<float>(rgb.size() - 1);
    const std::vector<float> out = RunTonemap(dev, tm, w, h, rgb, TonemapOptions{});
    for (size_t i = 0; i < rgb.size(); ++i) {
        const float expect = std::clamp(HalfToFloat(FloatToHalf(rgb[i])), 0.f, 1.f);
        CHECK(out[i] == expect);  // bit-exact: no decode / encode round trip on the identity path
    }
}

TEST_CASE("Tonemapper: ACES, Reinhard, exposure and input transfers match the CPU reference", "[tonemap][unit][gpu]") {
    D3D12Device dev({true, false});
    Tonemapper tm(dev);
    const uint32_t w = 16, h = 4;
    std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = 4.f * static_cast<float>(i) / static_cast<float>(rgb.size() - 1);  // 0..4 (HDR for linear inputs)
    for (const char* transfer : {"srgb", "linear", "pq", "hlg"}) {
        for (const char* curve : {"passthrough", "aces", "reinhard"}) {
            for (float exposure : {0.f, -1.f}) {  // EV
                TonemapOptions o;
                o.curve = curve;
                o.inputTransfer = transfer;
                o.exposureEv = exposure;
                const std::vector<float> out = RunTonemap(dev, tm, w, h, rgb, o);
                for (size_t i = 0; i < rgb.size(); ++i) {
                    const float in = HalfToFloat(FloatToHalf(rgb[i]));
                    const float expect = TonemapReference(in, o);
                    INFO(transfer << " " << curve << " exposure " << exposure << " value " << in);
                    CHECK(out[i] == Approx(expect).margin(0.02));  // half precision + pow
                }
            }
        }
    }
}
