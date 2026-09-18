// NVIDIA Optical Flow Accelerator on a synthetic translation (skipped without an NVIDIA GPU).

#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "convert/MvConvert.h"
#include "convert/Warp.h"
#include "gpu/D3D12Device.h"

#ifdef DLSSVID_WITH_CUDA
#include "stages/flow/OfaFlowEstimator.h"
#endif

using namespace dlssvid;

#ifdef DLSSVID_WITH_CUDA

namespace {
PassImage Texture(uint32_t w, uint32_t h, int shiftX, int shiftY) {
    PassImage img = MakePassImage(PassKind::ColorSource, w, h, PixelType::F32);
    uint32_t seed = 99;
    std::vector<float> noise(static_cast<size_t>(w + 64) * (h + 64));
    for (auto& n : noise) {
        seed = seed * 1664525u + 1013904223u;
        n = ((seed >> 8) % 1000) / 1000.f;
    }
    // blocky noise (8 px blocks) so the flow is unambiguous
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const int sx = static_cast<int>(x) - shiftX + 32, sy = static_cast<int>(y) - shiftY + 32;
            const float v = noise[static_cast<size_t>(sy / 8) * (w + 64) + static_cast<size_t>(sx / 8)];
            img.Set(x, y, 0, v);
            img.Set(x, y, 1, 1.f - v);
            img.Set(x, y, 2, 0.5f * v + 0.25f);
        }
    return img;
}
}  // namespace

TEST_CASE("OFA recovers a global translation and passes the warp test", "[ofa][gpu]") {
    std::string reason;
    if (!OfaFlowEstimator::Available(&reason)) SKIP("OFA unavailable: " << reason);
    {
        D3D12Device dev;
        if (dev.IsWarp() || !dev.IsNvidia()) SKIP("no NVIDIA GPU");
    }
    const uint32_t w = 640, h = 384;
    const PassImage a = Texture(w, h, 0, 0), b = Texture(w, h, 6, -3);  // b = a moved by (+6, -3)
    OfaFlowEstimator ofa;
    FlowEstimatorConfig cfg;
    cfg.extra["input"] = "abgr";  // CPU frames in this test
    ofa.Init(cfg, w, h);
    FlowInput ia, ib;
    ia.rgb = &a;
    ib.rgb = &b;
    PassImage flow, conf;
    ofa.Estimate(ia, ib, flow, &conf);
    REQUIRE(flow.width == w);
    REQUIRE(flow.height == h);
    REQUIRE(conf.width == w);
    double su = 0, sv = 0;
    size_t n = 0;
    for (uint32_t y = 32; y < h - 32; ++y)
        for (uint32_t x = 32; x < w - 32; ++x) {
            su += flow.Get(x, y, 0);
            sv += flow.Get(x, y, 1);
            ++n;
        }
    INFO("mean flow " << su / n << ", " << sv / n << " (expected 6, -3), grid " << ofa.Describe()["grid"]);
    CHECK(std::fabs(su / n - 6.0) < 0.75);
    CHECK(std::fabs(sv / n + 3.0) < 0.75);
    MvConvertOptions opt;
    opt.dilateRadius = 0;
    const PassImage mv = ForwardFlowToBackwardMv(flow, nullptr, opt);
    const double psnr = WarpPsnr(a, b, mv);
    INFO("warp PSNR " << psnr << " dB");
    CHECK(psnr >= 28.0);
    ofa.Shutdown();
}

#else
TEST_CASE("OFA backend not built", "[ofa]") { SUCCEED("DLSSVID_WITH_CUDA is off"); }
#endif
