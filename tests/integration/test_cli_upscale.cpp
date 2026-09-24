// CLI: `dlssvid upscale` and `dlssvid compare` as processes.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "TestClips.h"
#include "stages/upscale/WorkerUpscaler.h"
#ifdef DLSSVID_WITH_TENSORRT
#include "gpu/D3D12Device.h"
#include "ml/TrtLoader.h"
#endif
#include "passes/PassSequence.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {
int Run(const std::string& args) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + " >nul 2>&1\"";
    return std::system(cmd.c_str());
}
std::string Q(const std::filesystem::path& p) { return "\"" + p.string() + "\""; }
}  // namespace

TEST_CASE("dlssvid upscale --backend nis writes color_sr and a preview; compare reports PSNR/SSIM", "[integration][cli][upscale]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 40;
    spec.height = 24;
    const auto dir = TempDir() / "cli_upscale";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto out = dir / "passes";
    REQUIRE(Run("upscale --backend nis --warp --scale 2 --frames 4 --video " + Q(dir / "sr.mkv") + " --codec ffv1 -i " + Q(clip) + " -o " + Q(out)) == 0);
    const PassReader sr = PassReader::Open(out / "color_sr");
    sr.Validate({80, 48, 4, PassKind::ColorSr});
    CHECK(sr.Man().stageParams["backend"] == "nis");
    CHECK(DecodeAll(dir / "sr.mkv").size() == 4);

    // identical inputs -> infinite PSNR, SSIM 1; the JSON summary is written
    REQUIRE(Run("compare --ref " + Q(clip) + " --test " + Q(clip) + " --frames 3 --json " + Q(dir / "same.json") + " -q") == 0);
    std::ifstream in(dir / "same.json");
    nlohmann::json j;
    in >> j;
    CHECK(j["summary"]["frames"] == 3);
    CHECK(j["summary"]["ssim_y_mean"].get<double>() > 0.9999);
    CHECK(j["frames"].size() == 3);
    // pass folder vs the preview video at the same size
    REQUIRE(Run("compare --ref " + Q(out / "color_sr") + " --test " + Q(dir / "sr.mkv") + " -q --json " + Q(dir / "sr.json")) == 0);
    std::ifstream in2(dir / "sr.json");
    nlohmann::json j2;
    in2 >> j2;
    CHECK(j2["summary"]["psnr_rgb_mean"].get<double>() > 38.0);
    CHECK(j2["summary"]["size"][0] == 80);

    // dlss on WARP (no NVIDIA adapter): the fallback to nis works, --no-fallback fails with the reason
    REQUIRE(Run("upscale --backend dlss --warp --frames 1 -i " + Q(clip) + " -o " + Q(dir / "vsr")) == 0);
    CHECK(Run("upscale --backend dlss --no-fallback --warp --frames 1 -i " + Q(clip) + " -o " + Q(dir / "vsr2")) != 0);
    // errors
    CHECK(Run("upscale --backend nope --warp -i " + Q(clip) + " -o " + Q(dir / "x")) != 0);
    CHECK(Run("upscale --backend nis --format npz --warp -i " + Q(clip) + " -o " + Q(dir / "x")) != 0);
    CHECK(Run("compare --ref " + Q(clip) + " --test " + Q(out / "color_sr")) != 0);  // sizes differ
}

#ifdef DLSSVID_WITH_TENSORRT
TEST_CASE("CLI upscale --backend trt runs a registry model from --models-dir", "[cli][upscale][trt][gpu]") {
    std::string reason;
    if (!trt::Available(&reason)) SKIP("TensorRT unavailable: " << reason);
    {
        D3D12Device dev;
        if (dev.IsWarp() || !dev.IsNvidia()) SKIP("no NVIDIA GPU");
    }
    const auto dir = TempDir() / "cli_upscale_trt";
    std::filesystem::remove_all(dir);
    const auto models = dir / "models";
    std::filesystem::create_directories(models / "cache");
    std::filesystem::copy_file(std::filesystem::path(DLSSVID_SOURCE_DIR) / "tests" / "data" / "tiny_sr.onnx", models / "cache" / "tiny-sr.onnx");
    std::ofstream(models / "registry.json") << R"({"$schema_version": 1, "models": [{"id": "tiny-sr", "stage": "upscale", "role": "test",
 "source": "tests/data/make_tiny_sr_onnx.py", "format": "onnx", "url": "", "sha256": "", "license": "test",
 "params": {"family": "onnx", "scale": 2, "tile": 16, "tile_pad": 2, "input_name": "image", "output_name": "upscaled"}}]})";
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 40;
    spec.height = 24;
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    REQUIRE(Run("upscale --backend trt --model tiny-sr --tile 16 --models-dir " + Q(models) + " --frames 1 -i " + Q(clip) + " -o " + Q(dir / "out")) == 0);
    CHECK(std::filesystem::exists(dir / "out" / "color_sr" / "manifest.json"));
    CHECK(Run("upscale --backend trt --model nope --models-dir " + Q(models) + " --no-fallback --frames 1 -i " + Q(clip) + " -o " + Q(dir / "bad")) != 0);
}
#endif

TEST_CASE("CLI upscale --backend worker runs the worker's stub with a window", "[cli][upscale][worker]") {
    const UpscalerAvailability a = WorkerUpscaler::Available();
    if (!a.available) SKIP("sr_worker unavailable: " << a.reason);
    const auto dir = TempDir() / "cli_upscale_worker";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 40;
    spec.height = 24;
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    REQUIRE(Run("upscale --backend worker --model stub --window 2 --overlap 1 --scale 2 -i " + Q(clip) + " -o " + Q(dir / "out")) == 0);
    CHECK(std::filesystem::exists(dir / "out" / "color_sr" / "manifest.json"));
}
