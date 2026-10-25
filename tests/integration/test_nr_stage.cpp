// NrStage end to end (stage 6): the stub model on WARP (color_nr pass, preview video, color_sr input,
// guides through the cache slot, masks, two passes, model resolution, temporal filter, the A/B without
// guides), the ngx backend without a GPU / DLL (disabled or the instruction, never a crash) and the real
// NGX Feature 18 on an NVIDIA GPU with nvngx_dlssnr.dll (skipped otherwise).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "convert/MvConvert.h"
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
    CHECK(rd.Man().stageParams["ms_per_frame_gpu"].is_number());
    CHECK_FALSE(rd.Man().stageParams.contains("ms_per_frame"));  // the wall time is process's to write (D8)
    CHECK(rd.Man().stageParams["tonemap"]["curve"] == "passthrough");

    // the edit is visible but sane
    VideoDecoder dec2(clip);
    const ColorInfo ci = ColorInfoFromStream(dec2.Info());
    CpuFrame f;
    REQUIRE(dec2.NextFrame(f));
    const ImageMetrics m = CompareImages(YuvToRgb(f, ci, PixelType::F32), rd.ReadFrame(0));
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

// TASK-0047: frames before writeFrom go through the model for its history (dlssvid nr --warmup) and are not output - no pass
// frame, no video frame, no slot; a run of warm-up frames alone leaves no color_nr folder.
TEST_CASE("NrStage writeFrom: earlier frames build the history and are not output", "[integration][nr][preview]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("write_from");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    D3D12Device dev({true, false});
    NrStageOptions o;
    o.backend = "stub";
    o.outputDir = dir / "passes";
    o.videoOut = dir / "nr.mkv";
    o.videoCodec = "ffv1";
    o.temporal = 0.5f;
    o.writeFrom = 2;
    Pipeline pipeline(dev, 4);
    auto stage = std::make_unique<NrStage>(o);
    NrStage* raw = stage.get();
    pipeline.AddStage(std::move(stage));
    pipeline.Init();
    VideoDecoder dec(clip);
    CpuFrame f;
    while (dec.NextFrame(f)) {
        GpuFrameCache::Slot& slot = pipeline.Cache().Acquire(f.index, f.desc);
        pipeline.ProcessFrame(f);
        CHECK(slot.passes.count("color_nr") == (f.index >= 2 ? 1u : 0u));
        const auto* s = dynamic_cast<const StubNrBackend*>(raw->Backend(0));
        REQUIRE(s);
        CHECK(s->LastReset() == (f.index == 0));  // the model resets on the first warm-up frame only: the written frames keep its history
    }
    pipeline.Finish();
    const NrStage::Stats& st = raw->GetStats();
    CHECK(st.frames == 5);
    CHECK(st.warmup == 2);
    CHECK(st.written == 3);
    CHECK(st.firstWritten == 2);
    CHECK(st.lastWritten == 4);
    const auto* stub = dynamic_cast<const StubNrBackend*>(raw->Backend(0));
    REQUIRE(stub);
    CHECK(stub->Calls() == 5);  // the model saw the warm-up frames
    pipeline.Shutdown();
    const PassReader rd = PassReader::Open(dir / "passes" / "color_nr");
    CHECK(rd.Man().firstFrame == 2);
    CHECK(rd.Man().lastFrame == 4);
    CHECK(rd.Man().frameCount == 3);
    CHECK_FALSE(rd.HasFrame(0));
    CHECK(rd.Man().stageParams["range"] == nlohmann::json{{"first", 2}, {"last", 4}, {"warmup", 2}});
    const auto video = DecodeAll(dir / "nr.mkv");
    REQUIRE(video.size() == 3);
    CHECK(video[0].pts == 0);

    // nothing to write: no folder, no empty manifest, no video file
    NrStageOptions late = o;
    late.outputDir = dir / "late";
    late.videoOut = dir / "late.mkv";
    late.writeFrom = 100;
    VideoDecoder dec2(clip);
    const NrRunResult r = RunNr(dec2, dev, late);
    CHECK(r.stats.frames == 5);
    CHECK(r.stats.written == 0);
    CHECK_FALSE(std::filesystem::exists(dir / "late" / "color_nr"));
    CHECK_FALSE(std::filesystem::exists(late.videoOut));  // the encoder opens at the first output frame
}

// TASK-0048 (spec §2e: E3-E5): the A/B switches reach the backend - the mv scale, the zero guides at CreateFeature sized from the
// pass folders, the depth guide dropped - and the pass names them (stageParams.ab); a default run's manifest has no "ab".
TEST_CASE("NrStage A/B switches reach the backend: mv scale auto, zero guides at creation, no depth guide", "[integration][nr][ab]") {
    const uint32_t w = 96, h = 64, gw = 48, gh = 32;
    const int frames = 2;
    ClipSpec spec;
    spec.frames = frames;
    spec.width = gw;
    spec.height = gh;
    const auto dir = Dir("ab");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto passes = dir / "passes";
    WritePass(passes / "color_sr", PassKind::ColorSr, w, h, frames, FileFormat::Exr, [&](int64_t i, PassImage& img) { img = ColorFrame(i, w, h); });
    WritePass(passes / "depth_dlss", PassKind::DepthDlss, gw, gh, frames, FileFormat::Exr, [&](int64_t, PassImage& img) {
        for (uint32_t y = 0; y < gh; ++y)
            for (uint32_t x = 0; x < gw; ++x) img.Set(x, y, 0, 0.5f);
    });
    WritePass(passes / "mv_dlss", PassKind::MvDlss, gw, gh, frames, FileFormat::Exr, [&](int64_t, PassImage&) {});
    D3D12Device dev({true, false});
    auto options = [&](const char* name) {
        NrStageOptions o;
        o.backend = "stub";
        o.outputDir = dir / name;
        o.colorDir = passes / "color_sr";
        o.depthDir = passes / "depth_dlss";
        o.mvDir = passes / "mv_dlss";
        return o;
    };
    struct Out {
        NrConfig config;
        NrStage::Stats stats;
        bool hadDepth = false, hadMv = false;
        nlohmann::json stageParams;
    };
    auto run = [&](const NrStageOptions& o) {
        Pipeline pipeline(dev, 4);
        auto stage = std::make_unique<NrStage>(o);
        NrStage* raw = stage.get();
        pipeline.AddStage(std::move(stage));
        pipeline.Init();
        VideoDecoder dec(clip);
        CpuFrame f;
        while (dec.NextFrame(f)) {
            pipeline.Cache().Acquire(f.index, f.desc);
            pipeline.ProcessFrame(f);
        }
        pipeline.Finish();
        Out out;
        const auto* stub = dynamic_cast<const StubNrBackend*>(raw->Backend(0));
        REQUIRE(stub);
        out.config = stub->Config();
        out.stats = raw->GetStats();
        out.hadDepth = stub->LastHadDepth();
        out.hadMv = stub->LastHadMv();
        pipeline.Shutdown();
        out.stageParams = Manifest::Load(o.outputDir / "color_nr").stageParams;
        return out;
    };

    NrStageOptions a = options("auto");
    a.mvScale = "auto";
    a.createWithMv = true;
    const Out ra = run(a);
    CHECK(ra.config.mvScale == "auto");
    CHECK(ra.config.createWithMv);
    CHECK(ra.config.createMvWidth == gw);  // from the mv folder, before any frame
    CHECK(ra.config.createMvHeight == gh);
    CHECK(ra.config.createDepthWidth == gw);
    CHECK(ra.stats.mvScaleX == 2.f);  // mv 48x32 -> model 96x64
    CHECK(ra.stats.mvScaleY == 2.f);
    CHECK(ra.hadDepth);
    CHECK(ra.hadMv);
    CHECK(ra.stageParams["ab"] == nlohmann::json{{"mv_scale", "auto"}, {"create_with_mv", true}});

    NrStageOptions b = options("no_depth");
    b.depthGuide = false;
    const Out rb = run(b);
    CHECK_FALSE(rb.hadDepth);
    CHECK(rb.hadMv);
    CHECK_FALSE(rb.stats.depthUsed);
    CHECK(rb.stats.mvUsed);
    CHECK(rb.stageParams["depth_used"] == false);
    CHECK(rb.stageParams["ab"] == nlohmann::json{{"depth_guide", false}});
    CHECK(rb.stats.mvScaleX == 1.f);

    const Out rc = run(options("default"));
    CHECK_FALSE(rc.stageParams.contains("ab"));  // a default run's manifest is as it was
    CHECK(rc.config.mvScale == "unit");

    NrStageOptions bad = options("bad");
    bad.mvScale = "pixels";
    VideoDecoder dec1(clip);
    CHECK_THROWS_WITH(RunNr(dec1, dev, bad), Catch::Matchers::ContainsSubstring("must be unit | auto"));
    NrStageOptions still = options("still");
    still.useGuides = false;
    still.createWithMv = true;
    VideoDecoder dec2(clip);
    CHECK_THROWS_WITH(RunNr(dec2, dev, still), Catch::Matchers::ContainsSubstring("drop --no-guides"));
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

// PR-7 (spec §2b): the edit strength is NrCompose's, once after the last model pass - two stub passes are not linear, so a gain inside
// the backend or per pass would not give twice the edit of one at s = 2.
// PR-5 (spec §2c): passes 1..4 - one backend (one feature, one history) each, in a chain: the next reads the last one's output. The
// VRAM of each pass's feature goes to the stats and the manifest: the stub has no feature, 0 each (none when DXGI reports no memory).
TEST_CASE("NrStage runs four passes, one backend each", "[integration][nr]") {
    const uint32_t w = 64, h = 32;
    const int frames = 3;
    ClipSpec spec;
    spec.frames = frames;
    spec.width = w;
    spec.height = h;
    const auto dir = Dir("four_passes");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    WritePass(dir / "src" / "color_sr", PassKind::ColorSr, w, h, frames, FileFormat::Exr, [&](int64_t i, PassImage& img) { img = ColorFrame(i, w, h); });
    D3D12Device dev({true, false});
    NrStageOptions o;
    o.backend = "stub";
    o.colorDir = dir / "src" / "color_sr";
    o.outputDir = dir / "p4";
    o.passes = 4;
    Pipeline pipeline(dev, 4);
    auto stage = std::make_unique<NrStage>(o);
    NrStage* raw = stage.get();
    pipeline.AddStage(std::move(stage));
    pipeline.Init();
    VideoDecoder dec(clip);
    CpuFrame f;
    while (dec.NextFrame(f)) {
        pipeline.Cache().Acquire(f.index, f.desc);
        pipeline.ProcessFrame(f);
    }
    pipeline.Finish();
    CHECK(raw->GetStats().passes == 4);
    const std::vector<double> vram = raw->GetStats().vramMibPerPass;
    CHECK((vram.empty() || vram == std::vector<double>(4, 0.0)));
    for (size_t p = 0; p < 4; ++p) {
        const auto* stub = dynamic_cast<const StubNrBackend*>(raw->Backend(p));
        REQUIRE(stub);
        CHECK(stub->Calls() == frames);
        CHECK(!stub->LastReset());  // its own history after the first frame
    }
    CHECK(raw->Backend(4) == nullptr);
    pipeline.Shutdown();
    const PassReader rd = PassReader::Open(dir / "p4" / "color_nr");
    CHECK(rd.Man().stageParams["passes"] == 4);
    if (vram.empty()) CHECK_FALSE(rd.Man().stageParams.contains("vram_mib_per_pass"));
    else CHECK(rd.Man().stageParams["vram_mib_per_pass"] == nlohmann::json(vram));
    // each pass edits the last one's output: the edit grows with the passes
    auto edit = [&](int passes) {
        NrStageOptions q = o;
        q.outputDir = dir / ("e" + std::to_string(passes));
        q.passes = passes;
        VideoDecoder d2(clip);
        const NrRunResult r = RunNr(d2, dev, q);
        const PassImage in = ColorFrame(1, w, h), out = PassReader::Open(r.outDir).ReadFrame(1);
        double sum = 0.0;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < 3; ++c) sum += std::fabs(static_cast<double>(out.Get(x, y, c)) - in.Get(x, y, c));
        return sum / (static_cast<double>(w) * h * 3);
    };
    const double e1 = edit(1), e2 = edit(2), e4 = edit(4);
    CHECK(e2 > e1);
    CHECK(e4 > e2);
}

// PR-6 (spec §2d): a hard cut in the source resets every pass's history and the flicker filter's - the frame at the cut is the one
// a run without the filter makes; at scene_cut 0 the filter blends the last scene into the new one. The detector reads the decoded
// source, not the colour pass.
TEST_CASE("NrStage resets the history at a scene cut of the source", "[integration][nr][regression]") {
    const uint32_t w = 64, h = 32;
    const int frames = 6, cutAt = 3;
    ClipSpec spec;
    spec.frames = frames;
    spec.width = w;
    spec.height = h;
    const auto dir = Dir("scene_cut");
    // scene A brightens 4 levels a frame (no cut: 0.02), a hard cut at frame 3 to scene B, the gradient inverted (≈ 0.5)
    const auto clip = WriteClipWith(dir / "clip.mkv", spec, [&](int64_t i) {
        CpuFrame f = SyntheticFrame(i, w, h);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c) y[r * w + c] = static_cast<uint8_t>(i < cutAt ? 40 + c + 4 * i : 220 - c);
        return f;
    });
    D3D12Device dev({true, false});
    auto options = [&](float sceneCut, float temporal, const char* name) {
        NrStageOptions o;
        o.backend = "stub";
        o.outputDir = dir / name;
        o.passes = 2;
        o.temporal = temporal;
        o.temporalThreshold = 1.f;  // the colour gate open: the blend is the filter's weight alone
        o.sceneCut = sceneCut;
        return o;
    };
    {  // frame by frame: Reset reaches both passes at the cut, and only there after the first frame
        NrStageOptions o = options(0.24f, 0.5f, "on");
        VideoDecoder dec(clip);
        o.color = ColorInfoFromStream(dec.Info());
        Pipeline pipeline(dev, 4);
        auto stage = std::make_unique<NrStage>(o);
        NrStage* raw = stage.get();
        pipeline.AddStage(std::move(stage));
        pipeline.Init();
        CpuFrame f;
        while (dec.NextFrame(f)) {
            pipeline.Cache().Acquire(f.index, f.desc);
            pipeline.ProcessFrame(f);
            for (size_t p = 0; p < 2; ++p) {
                const auto* stub = dynamic_cast<const StubNrBackend*>(raw->Backend(p));
                REQUIRE(stub);
                INFO("frame " << f.index << " pass " << p);
                CHECK(stub->LastReset() == (f.index == 0 || f.index == cutAt));
            }
        }
        pipeline.Finish();
        CHECK(raw->GetStats().sceneCutActive);
        CHECK(raw->GetStats().sceneCuts == std::vector<int64_t>{cutAt});
        CHECK(raw->GetStats().sceneCutCount == 1);
        pipeline.Shutdown();
    }
    auto run = [&](float sceneCut, float temporal, const char* name) {
        VideoDecoder dec(clip);
        return RunNr(dec, dev, options(sceneCut, temporal, name));
    };
    const NrRunResult off = run(0.f, 0.5f, "off"), plain = run(0.24f, 0.f, "plain");
    CHECK_FALSE(off.stats.sceneCutActive);
    CHECK(off.stats.sceneCuts.empty());
    CHECK(plain.stats.sceneCuts == std::vector<int64_t>{cutAt});  // guides on: the model's history is reset without the filter too
    auto worst = [&](const char* a, const char* b, int64_t i) {
        const PassImage x = PassReader::Open(dir / a / "color_nr").ReadFrame(i), y = PassReader::Open(dir / b / "color_nr").ReadFrame(i);
        double d = 0.0;
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c)
                for (size_t ch = 0; ch < 3; ++ch) d = std::max(d, std::fabs(static_cast<double>(x.Get(c, r, ch)) - y.Get(c, r, ch)));
        return d;
    };
    CHECK(worst("on", "plain", cutAt) < 1e-3);      // at the cut the filter does not blend
    CHECK(worst("off", "plain", cutAt) > 0.05);     // without the reset it blends scene A into B
    CHECK(worst("on", "plain", cutAt - 1) > 1e-3);  // the filter does blend within a scene
    CHECK(worst("on", "plain", cutAt + 1) > 1e-3);  // and again after the cut
    const PassReader on = PassReader::Open(dir / "on" / "color_nr");
    CHECK(on.Man().stageParams["scene_cuts"] == nlohmann::json::array({cutAt}));
    CHECK(on.Man().stageParams["scene_cut_count"] == 1);
    CHECK_THAT(on.Man().stageParams["scene_cut_threshold"].get<double>(), Catch::Matchers::WithinAbs(0.24, 1e-6));
    const PassReader rdOff = PassReader::Open(dir / "off" / "color_nr");
    CHECK(rdOff.Man().stageParams["scene_cut_threshold"] == 0.0);
    CHECK_FALSE(rdOff.Man().stageParams.contains("scene_cuts"));  // not checked
    // still mode without the filter: no history to reset, the detector does not run
    auto stillRun = [&](float sceneCut, float temporal, const char* name) {
        NrStageOptions so = options(sceneCut, temporal, name);
        so.useGuides = false;
        VideoDecoder d2(clip);
        return RunNr(d2, dev, so);
    };
    const NrRunResult s = stillRun(0.24f, 0.f, "still");
    CHECK_FALSE(s.stats.sceneCutActive);
    CHECK_FALSE(PassReader::Open(dir / "still" / "color_nr").Man().stageParams.contains("scene_cuts"));
    // still mode with the filter (spec D12): the detector runs for the filter alone - no blend of scene A into B
    const NrRunResult st = stillRun(0.24f, 0.5f, "still_t"), stOff = stillRun(0.f, 0.5f, "still_t_off");
    CHECK(st.stats.sceneCutActive);
    CHECK(st.stats.sceneCuts == std::vector<int64_t>{cutAt});
    CHECK(worst("still_t", "still", cutAt) < 1e-3);
    CHECK(worst("still_t_off", "still", cutAt) > 0.05);
    CHECK(worst("still_t", "still", cutAt + 1) > 1e-3);
}

TEST_CASE("NrStage applies the edit strength once, after the last model pass", "[integration][nr][regression]") {
    const uint32_t w = 64, h = 32;
    const int frames = 2;
    ClipSpec spec;
    spec.frames = frames;
    spec.width = w;
    spec.height = h;
    const auto dir = Dir("strength");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    // a soft checker: twice the stub's two-pass edit stays inside [0, 1], so saturate clips (almost) nothing
    auto soft = [&](int64_t i) {
        PassImage img = MakePassImage(PassKind::ColorSr, w, h, PixelType::F16);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < 3; ++c)
                    img.Set(x, y, c, (((x / 4 + y / 4) % 2) ? 0.55f : 0.45f) + 0.05f * static_cast<float>(c) * x / w - 0.01f * static_cast<float>(i));
        return img;
    };
    WritePass(dir / "src" / "color_sr", PassKind::ColorSr, w, h, frames, FileFormat::Exr, [&](int64_t i, PassImage& img) { img = soft(i); });
    D3D12Device dev({true, false});
    auto run = [&](float s) {
        NrStageOptions o;
        o.backend = "stub";
        o.colorDir = dir / "src" / "color_sr";
        o.outputDir = dir / ("s" + std::to_string(static_cast<int>(s * 10)));
        o.passes = 2;
        o.intensity = s;
        VideoDecoder dec(clip);
        const NrRunResult r = RunNr(dec, dev, o);
        CHECK(r.stats.frames == frames);
        return PassReader::Open(r.outDir).ReadFrame(1);
    };
    const PassImage in = soft(1), one = run(1.f), two = run(2.f);
    double edit = 0.0;  // the stub edits: else e2 = 2 e1 = 0 would pass for any strength
    int64_t checked = 0;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < 3; ++c) {
                const double i0 = in.Get(x, y, c), e1 = one.Get(x, y, c) - i0, e2 = two.Get(x, y, c) - i0;
                if (i0 + 2.0 * e1 < 0.005 || i0 + 2.0 * e1 > 0.995) continue;  // saturate clips the extrapolation there
                INFO(x << "," << y << " c" << c);
                CHECK(std::fabs(e2 - 2.0 * e1) < 2e-3);
                edit = std::max(edit, std::fabs(e1));
                ++checked;
            }
    INFO("checked " << checked);
    REQUIRE(checked > static_cast<int64_t>(w) * h * 3 * 9 / 10);
    CHECK(edit > 0.02);
    const PassReader rd = PassReader::Open(dir / "s20" / "color_nr");
    CHECK(rd.Man().stageParams["intensity"] == 2.0);
    CHECK(rd.Man().stageParams["model_intensity"] == 1.0);  // what the model got, also the mark of a PR-7 pass
}

// PR-7 (E2): intensity 0 is the frame itself - no backend is created (ngx on WARP has neither an NVIDIA GPU nor the DLL), its guides
// and masks are not read, the output is the tone-mapped input.
TEST_CASE("NrStage intensity 0: no model, no DLL, no guides - the tone-mapped input is written", "[integration][nr][regression]") {
    const uint32_t w = 48, h = 32;
    const int frames = 2;
    ClipSpec spec;
    spec.frames = frames;
    spec.width = w;
    spec.height = h;
    const auto dir = Dir("bypass");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto passes = dir / "passes";
    WritePass(passes / "color_sr", PassKind::ColorSr, w, h, frames, FileFormat::Exr, [&](int64_t i, PassImage& img) { img = ColorFrame(i, w, h); });
    WritePass(passes / "depth_dlss", PassKind::DepthDlss, w, h, frames, FileFormat::Exr, [&](int64_t, PassImage&) {});
    WritePass(passes / "mv_dlss", PassKind::MvDlss, w, h, frames, FileFormat::Exr, [&](int64_t, PassImage&) {});
    WritePass(passes / "mask_ui", PassKind::Mask, w, h, frames, FileFormat::Png, [&](int64_t, PassImage& img) {
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) img.Set(x, y, 0, 255.f);
    });
    D3D12Device dev({true, false});  // WARP: an ngx backend would fail
    NrStageOptions o;
    o.backend = "ngx";
    o.disableWhenUnavailable = false;
    o.colorDir = passes / "color_sr";
    o.depthDir = passes / "depth_dlss";
    o.mvDir = passes / "mv_dlss";
    o.masksDir = passes;
    o.outputDir = dir / "out";
    o.temporal = 0.5f;
    o.intensity = 0.f;
    VideoDecoder dec(clip);
    const NrRunResult r = RunNr(dec, dev, o);
    CHECK_FALSE(r.stats.disabled);
    CHECK(r.stats.bypass);
    CHECK(r.stats.passes == 0);
    CHECK(r.stats.frames == frames);
    CHECK_FALSE(r.stats.depthUsed);
    CHECK_FALSE(r.stats.mvUsed);
    CHECK(r.stats.masks.empty());
    CHECK(r.backend["bypass"] == true);
    CHECK(r.diagnostics.empty());
    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().stageParams["bypass"] == true);
    CHECK(rd.Man().stageParams["passes"] == 0);
    CHECK(rd.Man().stageParams["intensity"] == 0.0);
    CHECK_FALSE(rd.Man().stageParams.contains("model_intensity"));
    CHECK(rd.Man().stageParams["ms_per_frame_gpu"].is_number());  // the estimates take this run's wall time beside it
    for (int i = 0; i < frames; ++i) {
        const PassImage in = ColorFrame(i, w, h), out = rd.ReadFrame(i);
        double worst = 0.0;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < 3; ++c) worst = std::max(worst, std::fabs(static_cast<double>(out.Get(x, y, c)) - in.Get(x, y, c)));
        CHECK(worst < 1e-3);  // passthrough tonemap: the input, the ui mask and the filter changed nothing
    }
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
    // what creating the feature cost (TASK-0048, spec E6 / E8): measured on the real feature, the VRAM by DXGI for this process
    CHECK(r.diagnostics["init_ms"].get<double>() > 0.0);
    CHECK(r.diagnostics["create_ms"].get<double>() > 0.0);
    CHECK(r.diagnostics["vram_feature_bytes"].is_number());
    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().frameCount == 4);
    VideoDecoder dec2(clip);
    const ColorInfo ci = ColorInfoFromStream(dec2.Info());
    CpuFrame f;
    REQUIRE(dec2.NextFrame(f));
    const ImageMetrics m = CompareImages(YuvToRgb(f, ci, PixelType::F32), rd.ReadFrame(0));
    CHECK(m.psnrY > 12.0);  // the model edits the frame but does not destroy it
    const PassImage out0 = rd.ReadFrame(0);
    for (uint32_t y = 0; y < out0.height; y += 9)
        for (uint32_t x = 0; x < out0.width; x += 7) {
            CHECK(out0.Get(x, y, 1) >= 0.f);
            CHECK(out0.Get(x, y, 1) <= 1.f);
        }
}

// PR-5 (spec §2c): four passes on the real model - four features in one process (E6f: 933 MiB each at 4K, safe), each its VRAM in
// the stats and the manifest.
TEST_CASE("Neural Rendering runs four passes on NGX: four features, the VRAM of each", "[integration][nr][gpu][ngx]") {
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("Neural Rendering needs an NVIDIA GPU");
    const NrAvailability avail = NrAvailable("ngx");
    if (!avail.available) SKIP(avail.reason);
    const NvidiaDriverVersion drv = NvidiaDriverFromUmd(dev.UmdDriverVersion());
    if (drv.valid && !drv.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor)) SKIP("driver " + drv.ToString() + " is older than 616.56 (TASK-0011)");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 256;
    spec.height = 144;
    const auto dir = Dir("ngx_four");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    NrStageOptions o;
    o.backend = "ngx";
    o.outputDir = dir / "passes";
    o.passes = 4;
    VideoDecoder dec(clip);
    NrRunResult r;
    try {
        r = RunNr(dec, dev, o);
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (what.find("FAIL_FeatureNotSupported") != std::string::npos) SKIP("the DLL refuses this GPU — patch it with `dlssvid nr-patch` (" + what + ")");
        throw;
    }
    CHECK(r.stats.frames == 3);
    CHECK(r.stats.passes == 4);
    REQUIRE(r.stats.vramMibPerPass.size() == 4);
    // a real feature holds memory (E6: 504 MiB at 1440p, 933 at 4K): 0 would mean the window missed its CreateFeature; the passes
    // are the same feature, so about the same size
    const double first = r.stats.vramMibPerPass[0];
    CHECK(first > 0.0);
    for (const double v : r.stats.vramMibPerPass) {
        CHECK(v > 0.5 * first);
        CHECK(v < 2.0 * first);
    }
    const PassReader rd = PassReader::Open(r.outDir);
    CHECK(rd.Man().stageParams["passes"] == 4);
    CHECK(rd.Man().stageParams["vram_mib_per_pass"] == nlohmann::json(r.stats.vramMibPerPass));
    const PassImage out = rd.ReadFrame(2);
    for (uint32_t y = 0; y < out.height; y += 9)
        for (uint32_t x = 0; x < out.width; x += 7)
            for (size_t c = 0; c < 3; ++c) {
                CHECK(std::isfinite(out.Get(x, y, c)));
                CHECK(out.Get(x, y, c) >= 0.f);
                CHECK(out.Get(x, y, c) <= 1.f);
            }
}

// Audit pass-mv-yup (TASK-0038): an mv_dlss pass stored with y up reaches the cache slot - and so the NR, upscale and FG
// backends, which all upload their guides through PassReader::ReadFrame - in the DLSS convention (y down).
TEST_CASE("NrStage puts a y_up mv_dlss pass into the cache slot in the DLSS convention", "[integration][nr][mv][y_up]") {
    const uint32_t w = 64, h = 48, gw = 32, gh = 24;
    const int frames = 2;
    ClipSpec spec;
    spec.frames = frames;
    spec.width = gw;
    spec.height = gh;
    const auto dir = Dir("yup");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto passes = dir / "passes";
    auto mvDown = [&](int64_t i) {  // what the backends must get
        PassImage mv = MakePassImage(PassKind::MvDlss, gw, gh);
        for (uint32_t y = 0; y < gh; ++y)
            for (uint32_t x = 0; x < gw; ++x) {
                mv.Set(x, y, 0, 0.5f + static_cast<float>(i));
                mv.Set(x, y, 1, 0.25f + 0.01f * static_cast<float>(y));
            }
        return mv;
    };
    WritePass(passes / "color_sr", PassKind::ColorSr, w, h, frames, FileFormat::Exr, [&](int64_t i, PassImage& img) { img = ColorFrame(i, w, h); });
    WritePass(passes / "depth_dlss", PassKind::DepthDlss, gw, gh, frames, FileFormat::Exr, [&](int64_t, PassImage& img) {
        for (uint32_t y = 0; y < gh; ++y)
            for (uint32_t x = 0; x < gw; ++x) img.Set(x, y, 0, 0.5f);
    });
    WritePass(passes / "mv_dlss", PassKind::MvDlss, gw, gh, frames, FileFormat::Exr, [&](int64_t i, PassImage& img) { img = FlipMvY(mvDown(i)); });
    Manifest m = Manifest::Load(passes / "mv_dlss");
    m.mv.yUp = true;  // stored y up, as another tool would
    m.Save(passes / "mv_dlss");

    D3D12Device dev({true, false});
    NrStageOptions o;
    o.backend = "stub";
    o.outputDir = passes;
    o.colorDir = passes / "color_sr";
    o.depthDir = passes / "depth_dlss";
    o.mvDir = passes / "mv_dlss";
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
        REQUIRE(slot.passes.count("mv_dlss") == 1);
        const PassImage got = pipeline.Cache().DownloadPass(slot, "mv_dlss").ConvertTo(PixelType::F32);
        const PassImage want = mvDown(f.index);
        REQUIRE(got.SameLayout(want));
        CHECK(PassImage::MaxAbsDiff(got, want) < 2e-3);  // y down; the stored (y-up) vectors would be off by ~0.5
        CHECK(got.Get(4, 10, 1) > 0.f);
        ++n;
    }
    pipeline.Finish();
    CHECK(n == frames);
    const auto* stub0 = dynamic_cast<const StubNrBackend*>(raw->Backend(0));
    REQUIRE(stub0);
    CHECK(stub0->LastHadMv());
    pipeline.Shutdown();
}
