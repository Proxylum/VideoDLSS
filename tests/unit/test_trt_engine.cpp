#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>

#include "gpu/D3D12Device.h"

#ifdef DLSSVID_WITH_TENSORRT
#include "ml/TrtEngine.h"
#include "ml/TrtLoader.h"
#endif

using namespace dlssvid;

#ifdef DLSSVID_WITH_TENSORRT

TEST_CASE("TensorRT builds, caches and runs the tiny ONNX model", "[trt][gpu]") {
    std::string reason;
    if (!trt::Available(&reason)) SKIP("TensorRT unavailable: " << reason);
    {
        D3D12Device dev;
        if (dev.IsWarp() || !dev.IsNvidia()) SKIP("no NVIDIA GPU");
    }
    const std::filesystem::path onnx = std::filesystem::path(DLSSVID_SOURCE_DIR) / "tests" / "data" / "tiny.onnx";
    REQUIRE(std::filesystem::exists(onnx));
    TrtEngine::Options opt;
    opt.cacheDir = std::filesystem::path(DLSSVID_TEST_TMP) / "trt";
    opt.fp16 = false;
    const auto cache = TrtEngine::CachePathFor(onnx, opt);
    std::filesystem::remove(cache);

    auto engine = TrtEngine::FromOnnx(onnx, opt);
    REQUIRE(std::filesystem::exists(cache));
    REQUIRE(engine->Inputs().size() == 1);
    REQUIRE(engine->Outputs().size() == 2);
    CHECK(engine->Input("image").shape == std::vector<int64_t>{1, 3, 4, 6});
    CHECK(engine->Output("depth").elements == 24);

    std::vector<float> in(72);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i) * 0.1f - 2.f;
    engine->SetInput("image", in.data(), in.size());
    engine->Execute();
    std::vector<float> out(72), depth(24);
    engine->GetOutput("out", out.data(), out.size());
    engine->GetOutput("depth", depth.data(), depth.size());
    for (size_t i = 0; i < in.size(); ++i) CHECK(std::fabs(out[i] - std::max(0.f, in[i] * 2.f + 1.f)) < 1e-4f);
    for (size_t p = 0; p < 24; ++p) {
        const float expected = (out[p] + out[24 + p] + out[48 + p]) / 3.f;
        CHECK(std::fabs(depth[p] - expected) < 1e-4f);
    }
    CHECK_THROWS(engine->SetInput("image", in.data(), 10));
    CHECK_THROWS(engine->Input("nope"));

    // second construction is a cache hit
    auto again = TrtEngine::FromEngineFile(cache);
    CHECK(again->EnginePath() == cache);
    CHECK_THROWS(TrtEngine::FromOnnx("nope.onnx", opt));
    CHECK(!trt::LibraryVersion().empty());
}

#else
TEST_CASE("TensorRT backend not built", "[trt]") { SUCCEED("DLSSVID_WITH_TENSORRT is off"); }
#endif
