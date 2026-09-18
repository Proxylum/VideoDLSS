// NrStage end to end (stage 6): the stub model on WARP (color_nr pass, preview video, color_sr input,
// guides through the cache slot, masks, two passes, model resolution, temporal filter, the A/B without
// guides), the ngx backend without a GPU / DLL (disabled or the instruction, never a crash) and the real
// NGX Feature 18 on an NVIDIA GPU with nvngx_dlssnr.dll (skipped otherwise).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "io/VideoDecoder.h"
#include "passes/ImageMetrics.h"
#include "passes/PassSequence.h"
#include "pipeline/Pipeline.h"
#include "stages/nr/NrStage.h"
#include "stages/nr/StubNrBackend.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Approx;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "nr" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// checker + gradient colour frames (local contrast for the stub to enhance)
PassImage ColorFrame(int64_t i, uint32_t w, uint32_t h) {
    PassImage img = MakePassImage(PassKind::ColorSr, w, h, PixelType::F16);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float checker = ((x / 3 + y / 3) % 2) ? 0.65f : 0.35f;
            for (size_t c = 0; c < 3; ++c) img.Set(x, y, c, std::clamp(checker * (1.f - 0.05f * static_cast<float>(i)) + 0.1f * static_cast<float>(c) * x / w, 0.f, 1.f));
        }
    return img;
}

void WritePass(const std::filesystem::path& dir, PassKind kind, uint32_t w, uint32_t h, int frames, FileFormat fmt, const std::function<void(int64_t, PassImage&)>& fill) {
    PassWriter writer(dir, Manifest::ForPass(kind, w, h, fmt));
    for (int i = 0; i < frames; ++i) {
        PassImage img = MakePassImage(kind, w, h);
        fill(i, img);
        writer.WriteFrame(i, img);
    }
    writer.Finish();
}

}  // namespace

TEST_CASE("NrStage writes color_nr with the stub on WARP, encodes a preview and reports stats", "[integration][nr]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("stub");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    NrStageOptions o;
    o.backend = "stub";
    o.outputDir = dir / "passes";
    o.videoOut = dir / "nr.mkv";
    o.videoCodec = "ffv1";
    o.sourceFile = "clip.mkv";
    o.intensity = 1.5f;
    const NrRunResult r = RunNr(dec, dev, o);
    CHECK(r.stats.frames == 4);
    CHECK(r.stats.backend == "stub");
    CHECK(r.stats.colorSource == "video");
    CHECK(r.stats.width == 48);
    CHECK(r.stats.height == 32);
    CHECK(r.stats.workWidth == 48);
    CHECK(r.stats.passes == 1);
    CHECK(!r.stats.disabled);
    CHECK(r.stats.MeanMs() > 0.0);
    CHECK(r.backend["backend"] == "stub");
    CHECK(r.diagnostics["ok"] == true);

    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().pass == "color_nr");
    CHECK(rd.Man().width == 48);
    CHECK(rd.Man().frameCount == 4);
    CHECK(rd.Man().colorspace == "srgb");
    CHECK(rd.Man().stageParams["backend"] == "stub");
    CHECK(rd.Man().stageParams["passes"] == 1);
    CHECK(rd.Man().stageParams["color_source"] == "video");
    CHECK(rd.Man().stageParams.contains("ms_per_frame"));
    CHECK(rd.Man().stageParams["tonemap"]["curve"] == "passthrough");

    // the edit is visible but sane
    VideoDecoder dec2(clip);
    const ColorInfo ci = ColorInfoFromStream(dec2.Info());
    CpuFrame f;
    REQUIRE(dec2.NextFrame(f));
    const ImageMetrics m = CompareImages(Yuv420pToRgb(f, ci, PixelType::F32), rd.ReadFrame(0));
    CHECK(m.psnrY < 60.0);
    CHECK(m.psnrY > 15.0);
    const PassImage out0 = rd.ReadFrame(0);
    for (uint32_t y = 0; y < out0.height; y += 7)
        for (uint32_t x = 0; x < out0.width; x += 5) {
            CHECK(out0.Get(x, y, 0) >= 0.f);
            CHECK(out0.Get(x, y, 0) <= 1.f);
        }
    CHECK(DecodeAll(dir / "nr.mkv").size() == 4);
}

TEST_CASE("NrStage: color_sr input, guides through the slot, masks, two passes, model resolution and the temporal filter", "[integration][nr]") {
    const uint32_t w = 96, h = 64, gw = 48, gh = 32;
    const int frames = 3;
    ClipSpec spec;
    spec.frames = frames;
    spec.width = gw;
    spec.height = gh;
    const auto dir = Dir("full");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto passes = dir / "passes";
    WritePass(passes / "color_sr", PassKind::ColorSr, w, h, frames, FileFormat::Exr, [&](int64_t i, PassImage& img) { img = ColorFrame(i, w, h); });
    WritePass(passes / "depth_dlss", PassKind::DepthDlss, gw, gh, frames, FileFormat::Exr, [&](int64_t, PassImage& img) {
        for (uint32_t y = 0; y < gh; ++y)
            for (uint32_t x = 0; x < gw; ++x) img.Set(x, y, 0, 0.5f);
    });
    WritePass(passes / "mv_dlss", PassKind::MvDlss, gw, gh, frames, FileFormat::Exr, [&](int64_t, PassImage&) {});
    WritePass(passes / "mask_ui", PassKind::Mask, w, h, frames, FileFormat::Png, [&](int64_t, PassImage& img) {
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) img.Set(x, y, 0, x < w / 2 ? 255.f : 0.f);  // left half protected
    });
    WritePass(passes / "mask_skin", PassKind::Mask, w, h, frames, FileFormat::Png, [&](int64_t, PassImage& img) {
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) img.Set(x, y, 0, x >= w / 2 && y >= h / 2 ? 255.f : 0.f);  // bottom-right quarter: skin
    });

    D3D12Device dev({true, false});
    NrStageOptions o;
    o.backend = "stub";
    o.outputDir = passes;
    o.colorDir = passes / "color_sr";
    o.depthDir = passes / "depth_dlss";
    o.mvDir = passes / "mv_dlss";
    o.masksDir = passes;
    o.passes = 2;
    o.modelScale = 0.5;
    o.temporal = 0.5f;
    o.skinBlend = 0.f;  // skin regions keep the input as well
    o.intensity = 2.f;
    o.sourceFile = "clip.mkv";

    Pipeline pipeline(dev, 4);
    auto stage = std::make_unique<NrStage>(o);
    NrStage* raw = stage.get();
    pipeline.AddStage(std::move(stage));
    pipeline.Init();
    VideoDecoder dec(clip);
    CpuFrame f;
    int64_t n = 0;
    while (dec.NextFrame(f)) {
        GpuFrameCache::Slot& slot = pipeline.Cache().Acquire(f.index, f.desc);
        pipeline.ProcessFrame(f);
        REQUIRE(slot.passes.count("color_nr") == 1);
        CHECK(slot.passes.at("color_nr").width == w);
        CHECK(slot.passes.count("depth_dlss") == 1);
        CHECK(slot.passes.count("mv_dlss") == 1);
        ++n;
    }
    pipeline.Finish();
    CHECK(n == frames);
    const NrStage::Stats& st = raw->GetStats();
    CHECK(st.frames == frames);
    CHECK(st.colorSource == "color_sr");
    CHECK(st.width == w);
    CHECK(st.height == h);
    CHECK(st.workWidth == gw);
    CHECK(st.workHeight == gh);
    CHECK(st.passes == 2);
    CHECK(st.depthUsed);
    CHECK(st.mvUsed);
    CHECK(st.masks == std::vector<std::string>{"ui", "skin"});
    const auto* stub0 = dynamic_cast<const StubNrBackend*>(raw->Backend(0));
    const auto* stub1 = dynamic_cast<const StubNrBackend*>(raw->Backend(1));
    REQUIRE(stub0);
    REQUIRE(stub1);
    CHECK(stub0->Calls() == frames);
    CHECK(stub1->Calls() == frames);
    CHECK(stub0->LastHadDepth());
    CHECK(stub0->LastHadMv());
    CHECK(!stub0->LastReset());  // history kept after the first frame
    pipeline.Shutdown();

    const PassReader rd = PassReader::Open(passes / "color_nr");
    CHECK(rd.Man().stageParams["passes"] == 2);
    CHECK(rd.Man().stageParams["model_scale"] == 0.5);
    CHECK(rd.Man().stageParams["masks"] == nlohmann::json::array({"ui", "skin"}));
    CHECK(rd.Man().stageParams["depth_used"] == true);
    CHECK(rd.Man().stageParams["mv_used"] == true);
    const PassImage in0 = ColorFrame(0, w, h);
    const PassImage out0 = rd.ReadFrame(0);
    double protectedDiff = 0.0, skinDiff = 0.0, editedDiff = 0.0;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const double d = std::fabs(out0.Get(x, y, 1) - in0.Get(x, y, 1));
            if (x < w / 2) protectedDiff += d;
            else if (y >= h / 2) skinDiff += d;
            else editedDiff += d;
        }
    CHECK(protectedDiff / (w / 2 * h) < 2e-3);        // ui mask: the input passes through
    CHECK(skinDiff / (w / 2 * h / 2) < 2e-3);         // skin mask with blend 0: the input as well
    CHECK(editedDiff / (w / 2 * h / 2) > 0.01);       // the rest is edited

    // A/B: the same run without guides
    o.useGuides = false;
    o.outputDir = dir / "passes_noguides";
    VideoDecoder dec2(clip);
    const NrRunResult r2 = RunNr(dec2, dev, o);
    CHECK(r2.stats.frames == frames);
    CHECK(!r2.stats.depthUsed);
    CHECK(!r2.stats.mvUsed);
    CHECK(PassReader::Open(r2.outDir).Man().stageParams["guides"] == false);
}

TEST_CASE("NrStage: the ngx backend without an NVIDIA GPU / DLL is disabled in a pipeline or fails with the instruction, never crashes", "[integration][nr]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 32;
    spec.height = 16;
    const auto dir = Dir("unavailable");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    D3D12Device dev({true, false});  // WARP: never an NVIDIA adapter
    NrStageOptions o;
    o.backend = "ngx";
    o.outputDir = dir / "passes";
    o.disableWhenUnavailable = true;
    VideoDecoder dec(clip);
    const NrRunResult r = RunNr(dec, dev, o);
    CHECK(r.stats.disabled);
    CHECK(!r.stats.disabledReason.empty());
    CHECK(r.stats.frames == 0);
    CHECK(!std::filesystem::exists(r.outDir / Manifest::kFileName));
    o.disableWhenUnavailable = false;
    VideoDecoder dec2(clip);
    CHECK_THROWS_WITH(RunNr(dec2, dev, o), Catch::Matchers::ContainsSubstring("nr:"));
}

TEST_CASE("Neural Rendering (NGX Feature 18) produces color_nr on an NVIDIA GPU with nvngx_dlssnr.dll", "[integration][nr][gpu][ngx]") {
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("Neural Rendering needs an NVIDIA GPU");
    const NrAvailability avail = NrAvailable("ngx");
    if (!avail.available) SKIP(avail.reason);
    const NvidiaDriverVersion drv = NvidiaDriverFromUmd(dev.UmdDriverVersion());
    if (drv.valid && !drv.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor)) SKIP("driver " + drv.ToString() + " is older than 616.56 (TASK-0011)");
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 256;
    spec.height = 144;
    const auto dir = Dir("ngx");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    NrStageOptions o;
    o.backend = "ngx";
    o.outputDir = dir / "passes";
    o.sourceFile = "clip.mkv";
    VideoDecoder dec(clip);
    NrRunResult r;
    try {
        r = RunNr(dec, dev, o);
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (what.find("FAIL_FeatureNotSupported") != std::string::npos) SKIP("the DLL refuses this GPU — patch it with `dlssvid nr-patch` (" + what + ")");
        throw;
    }
    CHECK(r.stats.frames == 4);
    CHECK(r.diagnostics["ok"] == true);
    CHECK(std::string(r.diagnostics["dll_sha256"]).size() == 64);
    CHECK(std::string(r.diagnostics["create_result"]).find("Success") != std::string::npos);
    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().frameCount == 4);
    VideoDecoder dec2(clip);
    const ColorInfo ci = ColorInfoFromStream(dec2.Info());
    CpuFrame f;
    REQUIRE(dec2.NextFrame(f));
    const ImageMetrics m = CompareImages(Yuv420pToRgb(f, ci, PixelType::F32), rd.ReadFrame(0));
    CHECK(m.psnrY > 12.0);  // the model edits the frame but does not destroy it
    const PassImage out0 = rd.ReadFrame(0);
    for (uint32_t y = 0; y < out0.height; y += 9)
        for (uint32_t x = 0; x < out0.width; x += 7) {
            CHECK(out0.Get(x, y, 1) >= 0.f);
            CHECK(out0.Get(x, y, 1) <= 1.f);
        }
}
