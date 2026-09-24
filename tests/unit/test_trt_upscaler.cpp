// Stage 5, TASK-0022: the TensorRT upscaler on the tiny nearest-x2 model (tests/data/tiny_sr.onnx) — the padded
// tiles must reassemble the frame exactly, whatever the frame size relative to the tile.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#include "gpu/D3D12Device.h"
#include "passes/PassImage.h"
#include "util/Half.h"
#ifdef DLSSVID_WITH_TENSORRT
#include "ml/TrtLoader.h"
#include "stages/upscale/TrtUpscaler.h"
#endif

using namespace dlssvid;

#ifdef DLSSVID_WITH_TENSORRT
namespace {

// A models folder with one model: `tiny-sr`, the nearest-x2 ONNX of tests/data (no download, no export).
std::filesystem::path TinyModelsDir(const char* name) {
    const auto dir = std::filesystem::path(DLSSVID_TEST_TMP) / "trt_upscaler" / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir / "cache");
    std::filesystem::copy_file(std::filesystem::path(DLSSVID_SOURCE_DIR) / "tests" / "data" / "tiny_sr.onnx", dir / "cache" / "tiny-sr.onnx");
    std::ofstream(dir / "registry.json") << R"({"$schema_version": 1, "models": [{"id": "tiny-sr", "stage": "upscale", "role": "test",
 "source": "tests/data/make_tiny_sr_onnx.py", "format": "onnx", "url": "", "sha256": "", "license": "test",
 "params": {"family": "onnx", "scale": 2, "tile": 16, "tile_pad": 2, "input_name": "image", "output_name": "upscaled"}}]})";
    return dir;
}

PassImage RandomRgba(uint32_t w, uint32_t h, uint32_t seed) {
    PassImage img;
    img.Allocate(w, h, PixelType::F16, {"R", "G", "B", "A"});
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    uint16_t* p = img.As<uint16_t>();
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        for (int c = 0; c < 3; ++c) p[i * 4 + c] = FloatToHalf(static_cast<float>(d(rng)) / 255.f);
        p[i * 4 + 3] = FloatToHalf(1.f);
    }
    return img;
}

// Pixels of `out` (RGBA16F, 2x) that differ from the nearest-x2 of `in`.
size_t Mismatches(const PassImage& in, const PassImage& out) {
    const uint16_t* p = in.As<uint16_t>();
    const uint16_t* o = out.As<uint16_t>();
    size_t bad = 0;
    for (uint32_t y = 0; y < out.height; ++y)
        for (uint32_t x = 0; x < out.width; ++x)
            for (int c = 0; c < 3; ++c) {
                const float expect = HalfToFloat(p[((y / 2) * in.width + x / 2) * 4 + c]);
                const float got = HalfToFloat(o[(y * out.width + x) * 4 + c]);
                if (std::abs(got - expect) > 1e-3f) ++bad;
            }
    return bad;
}

}  // namespace

TEST_CASE("TrtUpscaler runs the tiny nearest-x2 model through padded tiles and reassembles the frame exactly", "[trt][gpu][upscale]") {
    std::string reason;
    if (!trt::Available(&reason)) SKIP("TensorRT unavailable: " << reason);
    D3D12Device dev;
    if (dev.IsWarp() || !dev.IsNvidia()) SKIP("no NVIDIA GPU");
    const auto dir = TinyModelsDir("exact");
    TrtUpscaler up;
    UpscalerConfig cfg;
    cfg.backend = "trt";
    cfg.inputWidth = 37;
    cfg.inputHeight = 23;
    cfg.outputWidth = 74;
    cfg.outputHeight = 46;
    cfg.extra = {{"models_dir", dir.string()}, {"model", "tiny-sr"}, {"tile", 16}};
    up.Init(dev, cfg);
    CHECK(up.ProcessesOnCpu());
    CHECK(up.NativeScale() == 2);
    CHECK(up.Tile() == 16);
    CHECK(up.Pad() == 2);
    CHECK(up.Describe()["model"] == "tiny-sr");
    CHECK(up.Describe()["model_scale"] == 2);

    // a frame larger than the tile: windows overlap, the borders are owned by the edge windows
    const PassImage in = RandomRgba(37, 23, 7);
    PassImage out;
    up.EvaluateCpu(in, out);
    REQUIRE(out.width == 74);
    REQUIRE(out.height == 46);
    REQUIRE(out.ChannelCount() == 4);
    CHECK(Mismatches(in, out) == 0);
    const float alpha = HalfToFloat(out.As<uint16_t>()[3]);
    CHECK(alpha == 1.f);

    // a frame smaller than the tile: filled by reflection, still exact
    const PassImage smallIn = RandomRgba(10, 7, 11);  // not `small`: a Windows macro
    PassImage smallOut;
    up.EvaluateCpu(smallIn, smallOut);
    REQUIRE(smallOut.width == 20);
    REQUIRE(smallOut.height == 14);
    CHECK(Mismatches(smallIn, smallOut) == 0);

    // the tile from the registry when none is asked for; a model of another stage is refused
    TrtUpscaler byRegistry;
    cfg.extra = {{"models_dir", dir.string()}, {"model", "tiny-sr"}};
    byRegistry.Init(dev, cfg);
    CHECK(byRegistry.Tile() == 16);
    TrtUpscaler unknown;
    cfg.extra = {{"models_dir", dir.string()}, {"model", "nope"}};
    CHECK_THROWS(unknown.Init(dev, cfg));
    up.Shutdown();
}
#endif
