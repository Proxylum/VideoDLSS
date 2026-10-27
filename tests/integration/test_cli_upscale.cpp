// CLI: `dlssvid upscale` and `dlssvid compare` as processes.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <functional>
#include <string>

#include "TestClips.h"
#include "stages/upscale/WorkerUpscaler.h"
#ifdef DLSSVID_WITH_TENSORRT
#include "gpu/D3D12Device.h"
#include "ml/TrtLoader.h"
#endif
#include "passes/PassSequence.h"
#include "passes/ScenesPass.h"

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
    CHECK(j["summary"]["mae_rgb_mean"].get<double>() == 0.0);  // TASK-0050: the size of an edit, 0 for the same frames
    CHECK(j["frames"].size() == 3);
    // pass folder vs the preview video at the same size
    REQUIRE(Run("compare --ref " + Q(out / "color_sr") + " --test " + Q(dir / "sr.mkv") + " -q --json " + Q(dir / "sr.json")) == 0);
    std::ifstream in2(dir / "sr.json");
    nlohmann::json j2;
    in2 >> j2;
    CHECK(j2["summary"]["psnr_rgb_mean"].get<double>() > 38.0);
    CHECK(j2["summary"]["size"][0] == 80);
    CHECK(j2["summary"]["mae_rgb_mean"].get<double>() > 0.0);
    CHECK(j2["summary"]["mae_y_mean"].get<double>() > 0.0);
    CHECK(j2["frames"][0].contains("mae_rgb"));

    // dlss on WARP (no NVIDIA adapter): the fallback to nis works, --no-fallback fails with the reason
    REQUIRE(Run("upscale --backend dlss --warp --frames 1 -i " + Q(clip) + " -o " + Q(dir / "vsr")) == 0);
    CHECK(Run("upscale --backend dlss --no-fallback --warp --frames 1 -i " + Q(clip) + " -o " + Q(dir / "vsr2")) != 0);
    // errors
    CHECK(Run("upscale --backend nope --warp -i " + Q(clip) + " -o " + Q(dir / "x")) != 0);
    // every format of the schema is a pass format of the stage (TASK-0044: npz was rejected, «Запустить отдельно» failed)
    CHECK(Run("upscale --backend nis --format bmp --warp -i " + Q(clip) + " -o " + Q(dir / "x")) != 0);
    REQUIRE(Run("upscale --backend nis --format npz --warp --frames 1 -i " + Q(clip) + " -o " + Q(dir / "npz")) == 0);
    CHECK(std::filesystem::exists(dir / "npz" / "color_sr" / "manifest.json"));
    CHECK(Run("compare --ref " + Q(clip) + " --test " + Q(out / "color_sr")) != 0);  // sizes differ
}

namespace {
// A colour pass of frames [first, first + count): grey value(frame, x, y) in all three channels.
void GreyPass(const std::filesystem::path& dir, uint32_t w, uint32_t h, int64_t first, int64_t count, const std::function<float(int64_t, uint32_t, uint32_t)>& value) {
    PassWriter writer(dir, Manifest::ForPass(PassKind::ColorSr, w, h, FileFormat::Exr));
    for (int64_t f = first; f < first + count; ++f) {
        PassImage img = MakePassImage(PassKind::ColorSr, w, h, PixelType::F32);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < 3; ++c) img.Set(x, y, c, value(f, x, y));
        writer.WriteFrame(f, img);
    }
    writer.Finish();
}
nlohmann::json ReadJson(const std::filesystem::path& p) {
    std::ifstream in(p);
    nlohmann::json j;
    in >> j;
    return j;
}
}  // namespace

// PR-8 (E10): the low-band share of a difference and the PSNR of the low and the high part of an edit, measured on the CPU with the
// product's own filter (NrLowpassCpu). Ref and test start at frame 6, the edit's input I at 0: I is taken by the reference's frame
// number - by index its frames would be 0.12 darker and the high part far from the reference.
TEST_CASE("dlssvid compare --bands: a smooth difference is low band, a checkerboard high; --edit-from gives the low and high PSNR, takes the frames of I by the reference's frame numbers and refuses a frame I does not have", "[integration][cli]") {
    const uint32_t w = 64, h = 16;
    const auto dir = TempDir() / "cli_compare_bands";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    auto input = [](int64_t f) { return 0.3f + 0.02f * static_cast<float>(f); };
    auto smooth = [](uint32_t x) { return 0.1f * std::cos(static_cast<float>(x) * 6.2831853f / 64.f); };
    auto checker = [](uint32_t x, uint32_t y) { return ((x + y) % 2) ? 0.125f : -0.125f; };
    GreyPass(dir / "flat", w, h, 6, 4, [](int64_t, uint32_t, uint32_t) { return 0.5f; });
    GreyPass(dir / "smooth", w, h, 6, 4, [&](int64_t, uint32_t x, uint32_t) { return 0.5f + smooth(x); });
    GreyPass(dir / "checker", w, h, 6, 4, [&](int64_t, uint32_t x, uint32_t y) { return 0.5f + checker(x, y); });
    // the edit of I: a smooth part and a checkerboard; the references are I with only one of them
    GreyPass(dir / "I", w, h, 0, 12, [&](int64_t f, uint32_t, uint32_t) { return input(f); });
    GreyPass(dir / "edited", w, h, 6, 4, [&](int64_t f, uint32_t x, uint32_t y) { return input(f) + smooth(x) + checker(x, y); });
    GreyPass(dir / "low_only", w, h, 6, 4, [&](int64_t f, uint32_t x, uint32_t) { return input(f) + smooth(x); });
    GreyPass(dir / "high_only", w, h, 6, 4, [&](int64_t f, uint32_t x, uint32_t y) { return input(f) + checker(x, y); });

    REQUIRE(Run("compare --ref " + Q(dir / "flat") + " --test " + Q(dir / "smooth") + " --bands 2,4 -q --json " + Q(dir / "smooth.json")) == 0);
    const nlohmann::json s = ReadJson(dir / "smooth.json");
    REQUIRE(s["summary"]["frames"] == 4);
    REQUIRE(s["summary"]["bands"].size() == 2);
    CHECK(s["summary"]["bands"][0]["d"] == 2);
    CHECK(s["summary"]["bands"][1]["d"] == 4);
    CHECK(s["summary"]["bands"][0]["low_energy_y"].get<double>() > 0.8);
    CHECK(s["summary"]["bands"][0]["low_energy_rgb"].get<double>() > 0.8);
    CHECK(!s["summary"]["bands"][0].contains("psnr_y_low"));  // without --edit-from only the shares
    REQUIRE(s["frames"].size() == 4);
    CHECK(s["frames"][0]["bands"].size() == 2);

    REQUIRE(Run("compare --ref " + Q(dir / "flat") + " --test " + Q(dir / "checker") + " --bands 2,4 -q --json " + Q(dir / "checker.json")) == 0);
    const nlohmann::json c = ReadJson(dir / "checker.json");
    for (const auto& b : c["summary"]["bands"]) CHECK(b["low_energy_y"].get<double>() < 1e-9);

    // the low part of the edit alone reproduces I + smooth, the high part alone I + checkerboard
    REQUIRE(Run("compare --ref " + Q(dir / "low_only") + " --test " + Q(dir / "edited") + " --edit-from " + Q(dir / "I") + " --bands 2 -q --json " +
                Q(dir / "low.json")) == 0);
    const nlohmann::json lo = ReadJson(dir / "low.json");
    REQUIRE(lo["summary"]["frames"] == 4);
    CHECK(lo["summary"]["bands"][0]["psnr_y_low"].get<double>() > 35.0);
    CHECK(lo["summary"]["bands"][0]["psnr_y_high"].get<double>() < 25.0);
    REQUIRE(Run("compare --ref " + Q(dir / "high_only") + " --test " + Q(dir / "edited") + " --edit-from " + Q(dir / "I") + " --bands 2 -q --json " +
                Q(dir / "high.json")) == 0);
    const nlohmann::json hi = ReadJson(dir / "high.json");
    CHECK(hi["summary"]["bands"][0]["psnr_y_high"].get<double>() > 35.0);  // I by index would leave it 0.12 off: ~18 dB
    CHECK(hi["summary"]["bands"][0]["psnr_y_low"].get<double>() < 25.0);

    // I without the reference's frames 8, 9: an error, not a silent end; --edit-from needs --bands; bad cell sizes
    GreyPass(dir / "I_short", w, h, 0, 8, [&](int64_t f, uint32_t, uint32_t) { return input(f); });
    CHECK(Run("compare --ref " + Q(dir / "low_only") + " --test " + Q(dir / "edited") + " --edit-from " + Q(dir / "I_short") + " --bands 2 -q") != 0);
    CHECK(Run("compare --ref " + Q(dir / "low_only") + " --test " + Q(dir / "edited") + " --edit-from " + Q(dir / "I") + " -q") != 0);
    CHECK(Run("compare --ref " + Q(dir / "flat") + " --test " + Q(dir / "smooth") + " --bands 1 -q") != 0);
    CHECK(Run("compare --ref " + Q(dir / "flat") + " --test " + Q(dir / "smooth") + " --bands 2,x -q") != 0);
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

// PR-13: dlssvid upscale takes the clip's scene list for the backends that keep a history - dlss (on WARP it falls back to nis: the list
// stays, the label with it, as the fingerprint of the backend asked for counts it) - and not for nis, which keeps none
TEST_CASE("dlssvid upscale reads the scene list for DLSS SR, not for nis", "[integration][cli][upscale][scenes]") {
    const uint32_t w = 40, h = 24;
    ClipSpec spec;
    spec.frames = 4;
    spec.width = w;
    spec.height = h;
    const auto dir = TempDir() / "cli_upscale_scenes";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto clip = WriteClipWith(dir / "clip.mkv", spec, [&](int64_t i) {  // a hard cut at frame 2
        CpuFrame f = SyntheticFrame(i, w, h);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c) y[r * w + c] = static_cast<uint8_t>(i < 2 ? 40 + c + 4 * i : 220 - c);
        return f;
    });
    const auto out = dir / "passes";
    REQUIRE(Run("scenes -i " + Q(clip) + " -o " + Q(out)) == 0);
    REQUIRE(Run("upscale --backend nis --warp -i " + Q(clip) + " -o " + Q(out)) == 0);
    CHECK_FALSE(Manifest::Load(out / "color_sr").stageParams.contains("scene_cuts_id"));
    REQUIRE(Run("upscale --backend dlss --warp -i " + Q(clip) + " -o " + Q(out)) == 0);  // WARP: no DLSS, nis in its place
    const Manifest m = Manifest::Load(out / "color_sr");
    CHECK(m.stageParams["fallback"] == true);
    CHECK(m.stageParams["scene_cuts_id"] == SceneCutsId({2}));
    REQUIRE(Run("upscale --backend dlss --warp --no-scenes -i " + Q(clip) + " -o " + Q(out)) == 0);
    CHECK_FALSE(Manifest::Load(out / "color_sr").stageParams.contains("scene_cuts_id"));
}
