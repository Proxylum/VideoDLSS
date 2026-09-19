// FgStage end to end (stage 7): the blend baseline on WARP (color_fg with x2 / x3 frames, fps, manifest,
// preview video, colour from a pass folder), dlssg without a GPU (disabled or the instruction), DLSS Frame
// Generation on an NVIDIA GPU (skipped without the DLL) and RIFE through TensorRT (skipped without the model).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/ImageMetrics.h"
#include "passes/PassSequence.h"
#include "stages/fg/FgStage.h"
#if defined(DLSSVID_WITH_TENSORRT)
#include "ml/ModelRegistry.h"
#endif

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Approx;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "fg" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

std::vector<PassImage> DecodeRgb(const std::filesystem::path& clip) {
    VideoDecoder dec(clip);
    const ColorInfo ci = ColorInfoFromStream(dec.Info());
    std::vector<PassImage> out;
    CpuFrame f;
    while (dec.NextFrame(f)) out.push_back(Yuv420pToRgb(f, ci, PixelType::F32));
    return out;
}

}  // namespace

TEST_CASE("FgStage with the blend baseline on WARP: x2 frames, doubled fps, manifest, preview video", "[integration][fg]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("blend");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    FgStageOptions o;
    o.backend = "blend";
    o.multiplier = 2;
    o.outputDir = dir / "passes";
    o.videoOut = dir / "fg.mkv";
    o.videoCodec = "ffv1";
    o.sourceFile = "clip.mkv";
    const FgRunResult r = RunFg(dec, dev, o);
    CHECK(r.stats.frames == 5);
    CHECK(r.stats.generated == 4);
    CHECK(r.stats.backend == "blend");
    CHECK(r.stats.colorSource == "video");
    CHECK(r.stats.multiplier == 2);
    CHECK(!r.stats.disabled);
    CHECK(r.backend["backend"] == "blend");

    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().pass == "color_fg");
    CHECK(rd.Man().frameCount == FgFrameCount(5, 2));
    CHECK(rd.Man().fps.num == spec.fpsNum * 2);
    CHECK(rd.Man().fps.den == spec.fpsDen);
    CHECK(rd.Man().stageParams["multiplier"] == 2);
    CHECK(rd.Man().stageParams["generated"] == 4);
    CHECK(rd.Man().stageParams["color_source"] == "video");
    const std::vector<PassImage> src = DecodeRgb(clip);
    REQUIRE(src.size() == 5);
    for (int i = 0; i < 5; ++i) {
        const ImageMetrics m = CompareImages(src[static_cast<size_t>(i)], rd.ReadFrame(FgRealIndex(i, 2)));
        CHECK((m.psnrY > 50.0 || !std::isfinite(m.psnrY)));  // real frames are copies (half precision)
    }
    for (int i = 0; i < 4; ++i) {
        const PassImage g = rd.ReadFrame(FgInterpIndex(i, 1, 2));
        const PassImage& a = src[static_cast<size_t>(i)];
        const PassImage& b = src[static_cast<size_t>(i) + 1];
        for (uint32_t y = 0; y < g.height; y += 5)
            for (uint32_t x = 0; x < g.width; x += 7) CHECK(g.Get(x, y, 1) == Approx(0.5f * (a.Get(x, y, 1) + b.Get(x, y, 1))).margin(3e-3));
    }
    CHECK(DecodeAll(dir / "fg.mkv").size() == 9);
    VideoDecoder vd(dir / "fg.mkv");
    CHECK(vd.Info().frameRate.num == spec.fpsNum * 2 * vd.Info().frameRate.den / spec.fpsDen);
}

TEST_CASE("FgStage: x3 from a colour pass folder, frame layout and weights", "[integration][fg]") {
    const uint32_t w = 32, h = 16;
    const int frames = 4;
    ClipSpec spec;
    spec.frames = frames;
    spec.width = w;
    spec.height = h;
    const auto dir = Dir("x3");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    {
        PassWriter writer(dir / "passes" / "color_nr", Manifest::ForPass(PassKind::ColorNr, w, h, FileFormat::Exr));
        for (int i = 0; i < frames; ++i) {
            PassImage img = MakePassImage(PassKind::ColorNr, w, h, PixelType::F16);
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x)
                    for (size_t c = 0; c < 3; ++c) img.Set(x, y, c, 0.1f * static_cast<float>(i) + 0.02f * static_cast<float>(c));
            writer.WriteFrame(i, img);
        }
        writer.Finish();
    }
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    FgStageOptions o;
    o.backend = "blend";
    o.multiplier = 3;
    o.colorDir = dir / "passes" / "color_nr";
    o.outputDir = dir / "passes";
    const FgRunResult r = RunFg(dec, dev, o);
    CHECK(r.stats.frames == frames);
    CHECK(r.stats.generated == (frames - 1) * 2);
    CHECK(r.stats.colorSource == "pass");
    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().frameCount == FgFrameCount(frames, 3));
    CHECK(rd.Man().fps.num == spec.fpsNum * 3);
    // between real 1 (0.1) and real 2 (0.2): 0.1333 and 0.1667
    CHECK(rd.ReadFrame(FgRealIndex(1, 3)).Get(3, 3, 0) == Approx(0.1f).margin(2e-3));
    CHECK(rd.ReadFrame(FgInterpIndex(1, 1, 3)).Get(3, 3, 0) == Approx(0.1f + 0.1f / 3.f).margin(2e-3));
    CHECK(rd.ReadFrame(FgInterpIndex(1, 2, 3)).Get(3, 3, 0) == Approx(0.1f + 0.2f / 3.f).margin(2e-3));
    CHECK(rd.ReadFrame(FgRealIndex(2, 3)).Get(3, 3, 0) == Approx(0.2f).margin(2e-3));
    CHECK(rd.ReadFrame(FgRealIndex(3, 3)).Get(3, 3, 2) == Approx(0.34f).margin(2e-3));
    CHECK(!rd.HasFrame(FgRealIndex(3, 3) + 1));
    FgStageOptions bad = o;
    bad.multiplier = 5;
    VideoDecoder dec2(clip);
    CHECK_THROWS(RunFg(dec2, dev, bad));
}

TEST_CASE("FgStage: dlssg without an NVIDIA GPU / DLL is disabled in a pipeline or fails with the instruction", "[integration][fg]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 32;
    spec.height = 16;
    const auto dir = Dir("unavailable");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    D3D12Device dev({true, false});  // WARP
    FgStageOptions o;
    o.backend = "dlssg";
    o.outputDir = dir / "passes";
    o.disableWhenUnavailable = true;
    VideoDecoder dec(clip);
    const FgRunResult r = RunFg(dec, dev, o);
    CHECK(r.stats.disabled);
    CHECK(!r.stats.disabledReason.empty());
    CHECK(r.stats.frames == 0);
    CHECK(!std::filesystem::exists(r.outDir / Manifest::kFileName));
    o.disableWhenUnavailable = false;
    VideoDecoder dec2(clip);
    CHECK_THROWS_WITH(RunFg(dec2, dev, o), Catch::Matchers::ContainsSubstring("fg:"));
}

TEST_CASE("DLSS Frame Generation produces color_fg on an NVIDIA GPU", "[integration][fg][gpu][dlssg]") {
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("DLSS Frame Generation needs an NVIDIA GPU");
    const FgAvailability avail = FgAvailable("dlssg");
    if (!avail.available) SKIP(avail.reason);
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 256;
    spec.height = 144;
    const auto dir = Dir("dlssg");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    FgStageOptions o;
    o.backend = "dlssg";
    o.multiplier = 2;
    o.outputDir = dir / "passes";
    o.sourceFile = "clip.mkv";
    VideoDecoder dec(clip);
    FgRunResult r;
    try {
        r = RunFg(dec, dev, o);
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (what.find("not available on this GPU") != std::string::npos) SKIP(what);
        throw;
    }
    CHECK(r.stats.frames == 6);
    CHECK(r.stats.generated == 5);
    CHECK(r.diagnostics["ok"] == true);
    CHECK(r.diagnostics["available"] == true);
    CHECK(std::string(r.diagnostics["create_result"]).find("Success") != std::string::npos);
    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().frameCount == FgFrameCount(6, 2));
    const std::vector<PassImage> src = DecodeRgb(clip);
    // generated frame i+0.5 must resemble both neighbours (and not be garbage)
    for (int i = 1; i < 5; ++i) {
        const PassImage g = rd.ReadFrame(FgInterpIndex(i, 1, 2));
        const ImageMetrics ma = CompareImages(src[static_cast<size_t>(i)], g), mb = CompareImages(src[static_cast<size_t>(i) + 1], g);
        CHECK(ma.psnrY > 12.0);
        CHECK(mb.psnrY > 12.0);
        for (uint32_t y = 0; y < g.height; y += 13)
            for (uint32_t x = 0; x < g.width; x += 11) {
                CHECK(g.Get(x, y, 0) >= 0.f);
                CHECK(g.Get(x, y, 0) <= 1.f);
            }
    }
    // x3 needs Multi Frame Generation: either it works or it is refused with the instruction
    FgStageOptions o3 = o;
    o3.multiplier = 3;
    o3.outputDir = dir / "passes3";
    VideoDecoder dec3(clip);
    if (r.diagnostics["multi_frame_max"].get<int>() >= 2) {
        const FgRunResult r3 = RunFg(dec3, dev, o3);
        CHECK(r3.stats.generated == 10);
        CHECK(PassReader::Open(r3.outDir).Man().frameCount == FgFrameCount(6, 3));
    } else {
        CHECK_THROWS_WITH(RunFg(dec3, dev, o3), Catch::Matchers::ContainsSubstring("multiplier"));
    }
}

TEST_CASE("RIFE through TensorRT produces color_fg", "[integration][fg][gpu][rife]") {
#if defined(DLSSVID_WITH_TENSORRT)
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("RIFE through TensorRT needs an NVIDIA GPU");
    const FgAvailability avail = FgAvailable("rife");
    if (!avail.available) SKIP(avail.reason);
    const ModelRegistry registry = ModelRegistry::Load(ModelRegistry::DefaultRegistryPath());
    const ModelEntry* e = registry.Find("rife49");
    if (!e || !registry.IsCached(*e)) SKIP("rife49 ONNX is not in models/cache (run `dlssvid fg --backend rife` once to download it)");
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 160;
    spec.height = 96;
    const auto dir = Dir("rife");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    FgStageOptions o;
    o.backend = "rife";
    o.multiplier = 2;
    o.outputDir = dir / "passes";
    VideoDecoder dec(clip);
    const FgRunResult r = RunFg(dec, dev, o);
    CHECK(r.stats.frames == 4);
    CHECK(r.stats.generated == 3);
    CHECK(r.diagnostics["ok"] == true);
    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().frameCount == 7);
    const std::vector<PassImage> src = DecodeRgb(clip);
    const PassImage g = rd.ReadFrame(FgInterpIndex(1, 1, 2));
    CHECK(CompareImages(src[1], g).psnrY > 12.0);
    CHECK(CompareImages(src[2], g).psnrY > 12.0);
#else
    SKIP("built without TensorRT");
#endif
}
