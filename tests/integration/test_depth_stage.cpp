// DepthStage with the stub estimator: passes, manifests, windows/overlap, GPU textures, TAE.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "convert/DepthConvert.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "io/VideoDecoder.h"
#include "passes/PassSequence.h"
#include "pipeline/Pipeline.h"
#include "stages/depth/DepthStage.h"
#include "stages/depth/StubDepthEstimator.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {
std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "depth" / name;
    std::filesystem::remove_all(d);
    return d;
}
}  // namespace

TEST_CASE("DepthStage writes depth_raw and depth_dlss for every frame", "[integration][depth]") {
    ClipSpec spec;
    spec.frames = 7;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(TempDir() / "depth_stub.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    DepthStageOptions o;
    o.backend = "stub";
    o.outputDir = Dir("stub");
    o.dlss.zNear = 0.5f;
    o.dlss.zFar = 20.f;
    o.sourceFile = "depth_stub.mkv";
    int frames = 0;
    o.onFrame = [&](int64_t, double) { ++frames; };
    const DepthRunResult r = RunDepth(dec, dev, o);
    CHECK(r.stats.frames == 7);
    CHECK(frames == 7);
    CHECK(r.stats.windows == 7);
    CHECK(r.estimator["backend"] == "stub");

    const PassReader raw = PassReader::Open(r.rawDir);
    CHECK(raw.Man().pass == "depth_raw");
    CHECK(raw.Man().frameCount == 7);
    CHECK(raw.Man().depth.units == "meters");
    CHECK(raw.Man().depth.zNear == 0.5f);
    CHECK(raw.Man().fps.num == 24);
    CHECK(raw.Man().stageParams.contains("tae_mean"));
    CHECK(raw.Man().stageParams["stabilize"] == "scale");  // metric stub -> scale over the window by default (TASK-0038)
    CHECK(raw.Man().stageParams["stabilize_requested"] == "auto");
    CHECK(raw.Man().stageParams["stabilize_window"] == 8);
    CHECK(r.stabilize == "scale");
    raw.Validate({48, 32, 7, PassKind::DepthRaw});

    // depth equals the stub's formula on the decoded frame
    VideoDecoder dec2(clip);
    CpuFrame f;
    REQUIRE(dec2.NextFrame(f));
    const PassImage rgb = YuvToRgb(f, ColorInfoFromStream(dec2.Info()), PixelType::F32);
    const PassImage expected = StubDepthEstimator::Expected(rgb);
    CHECK(PassImage::MaxAbsDiff(raw.ReadFrame(0), expected) < 1e-5);

    const PassReader dlss = PassReader::Open(r.dlssDir);
    dlss.Validate({48, 32, 7, PassKind::DepthDlss});
    const PassImage d0 = dlss.ReadFrame(0);
    const float z = expected.Get(10, 10, 0);
    CHECK(std::fabs(d0.Get(10, 10, 0) - LinearToReverseZ(z, 0.5f, 20.f)) < 1e-6f);
}

TEST_CASE("DepthStage windows with overlap emit every frame exactly once", "[integration][depth]") {
    ClipSpec spec;
    spec.frames = 11;
    spec.width = 32;
    spec.height = 16;
    const auto clip = WriteClip(TempDir() / "depth_window.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    DepthStageOptions o;
    o.outputDir = Dir("window");
    o.writeDlss = false;
    o.uploadToGpu = false;
    o.color = ColorInfoFromStream(dec.Info());
    o.fps = dec.Info().frameRate;
    Pipeline pipeline(dev, 2);
    auto est = std::make_unique<StubDepthEstimator>(4, 1);  // window 4, overlap 1
    StubDepthEstimator* raw = est.get();
    auto stage = std::make_unique<DepthStage>(o, std::move(est));
    DepthStage* stagePtr = stage.get();
    pipeline.AddStage(std::move(stage));
    pipeline.Init();
    CpuFrame f;
    while (dec.NextFrame(f)) pipeline.ProcessFrame(f);
    pipeline.Finish();
    CHECK(stagePtr->GetStats().frames == 11);
    CHECK(raw->Calls() >= 3);
    pipeline.Shutdown();
    const PassReader r = PassReader::Open(o.outputDir / "depth_raw");
    CHECK(r.Man().frameCount == 11);
    CHECK(r.Man().firstFrame == 0);
    CHECK(r.Man().lastFrame == 10);
    CHECK(r.MissingFrames().empty());
    CHECK_FALSE(std::filesystem::exists(o.outputDir / "depth_dlss"));

    // the estimator's windows and the alignment delay interleave: every frame still comes out exactly once, in order
    VideoDecoder dec3(clip);
    o.outputDir = Dir("window_delay");
    o.stabilizeWindow = 2;
    Pipeline p2(dev, 2);
    auto stage2 = std::make_unique<DepthStage>(o, std::make_unique<StubDepthEstimator>(4, 1));
    DepthStage* s2 = stage2.get();
    p2.AddStage(std::move(stage2));
    p2.Init();
    while (dec3.NextFrame(f)) p2.ProcessFrame(f);
    p2.Finish();
    CHECK(s2->GetStats().frames == 11);
    p2.Shutdown();
    const PassReader r2 = PassReader::Open(o.outputDir / "depth_raw");
    CHECK(r2.Man().frameCount == 11);
    CHECK(r2.MissingFrames().empty());
    CHECK(r2.Man().stageParams["stabilize_window"] == 2);
}

TEST_CASE("DepthStage uploads depth into the GPU frame cache slot", "[integration][depth][gpu]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 24;
    spec.height = 16;
    const auto clip = WriteClip(TempDir() / "depth_gpu.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    DepthStageOptions o;
    o.backend = "stub";
    o.outputDir.clear();  // no files, GPU only
    o.color = ColorInfoFromStream(dec.Info());
    Pipeline pipeline(dev, 4);
    pipeline.AddStage(std::make_unique<DepthStage>(o));
    pipeline.Init();
    CpuFrame f;
    REQUIRE(dec.NextFrame(f));
    GpuFrameCache::Slot& slot = pipeline.Cache().Acquire(f.index, f.desc);
    pipeline.ProcessFrame(f);
    pipeline.Finish();
    REQUIRE(slot.passes.count("depth_raw") == 1);
    const PassImage back = pipeline.Cache().DownloadPass(slot, "depth_raw");
    const PassImage rgb = YuvToRgb(f, o.color, PixelType::F32);
    CHECK(PassImage::MaxAbsDiff(back, StubDepthEstimator::Expected(rgb)) < 1e-5);
    pipeline.Shutdown();
}

TEST_CASE("GpuFrameCache pass textures round-trip 1, 2 and 3 channels", "[integration][gpu]") {
    D3D12Device dev({true, false});
    GpuFrameCache cache(dev, 1);
    GpuFrameCache::Slot& slot = cache.Acquire(0, FrameDesc{8, 8, PixelFormat::Yuv420p});
    PassImage d = MakePassImage(PassKind::DepthRaw, 13, 7);
    PassImage mv = MakePassImage(PassKind::MvRaw, 13, 7);
    PassImage rgb = MakePassImage(PassKind::ColorSource, 13, 7);  // F16
    for (uint32_t y = 0; y < 7; ++y)
        for (uint32_t x = 0; x < 13; ++x) {
            d.Set(x, y, 0, x * 0.5f + y);
            mv.Set(x, y, 0, x - 6.f);
            mv.Set(x, y, 1, y * 0.25f);
            rgb.Set(x, y, 0, x / 13.f);
            rgb.Set(x, y, 1, y / 7.f);
            rgb.Set(x, y, 2, 0.5f);
        }
    cache.UploadPass(slot, "depth_raw", d);
    cache.UploadPass(slot, "mv_raw", mv);
    cache.UploadPass(slot, "color", rgb);
    CHECK(slot.passes.size() == 3);
    CHECK(cache.DownloadPass(slot, "depth_raw").data == d.data);
    CHECK(cache.DownloadPass(slot, "mv_raw").data == mv.data);
    const PassImage c = cache.DownloadPass(slot, "color");
    CHECK(c.channels == rgb.channels);
    CHECK(c.data == rgb.data);
    CHECK_THROWS(cache.DownloadPass(slot, "nope"));
}

// Audit depth-7 (TASK-0038): metric depth is aligned by default - before, auto meant "none" for metric models, so the
// TAE of the default run was the TAE of the raw model.
namespace {
class FlickerDepthEstimator final : public IDepthEstimator {  // the stub's depth, x1.15 / x0.87 on alternate frames
public:
    std::string Name() const override { return "flicker"; }
    bool IsMetric() const override { return true; }
    void Init(const DepthEstimatorConfig&) override { n_ = 0; }
    void Estimate(const std::vector<const PassImage*>& rgb, std::vector<PassImage>& out) override {
        out.clear();
        for (const PassImage* f : rgb) {
            PassImage d = StubDepthEstimator::Expected(*f);
            const float k = n_++ % 2 == 0 ? 1.15f : 0.87f;
            for (uint32_t y = 0; y < d.height; ++y)
                for (uint32_t x = 0; x < d.width; ++x) d.Set(x, y, 0, d.Get(x, y, 0) * k);
            out.push_back(std::move(d));
        }
    }

private:
    int n_ = 0;
};

DepthStage::Stats RunFlicker(const std::filesystem::path& clip, const std::filesystem::path& out, DepthStabilize mode) {
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    DepthStageOptions o;
    o.outputDir = out;
    o.stabilize = mode;
    o.color = ColorInfoFromStream(dec.Info());
    o.fps = dec.Info().frameRate;
    Pipeline pipeline(dev, 4);
    auto stage = std::make_unique<DepthStage>(o, std::make_unique<FlickerDepthEstimator>());
    DepthStage* raw = stage.get();
    pipeline.AddStage(std::move(stage));
    pipeline.Init();
    CpuFrame f;
    while (dec.NextFrame(f)) {
        pipeline.Cache().Acquire(f.index, f.desc);
        pipeline.ProcessFrame(f);
    }
    pipeline.Finish();
    const DepthStage::Stats st = raw->GetStats();
    pipeline.Shutdown();
    return st;
}
}  // namespace

TEST_CASE("DepthStage aligns metric depth by default: TAE below the unaligned run (regression TASK-0038 depth-7)", "[integration][depth][regression]") {
    ClipSpec spec;
    spec.frames = 48;  // most frames get the full ±8 window; the shrunk windows at the ends keep part of the flicker
    spec.width = 32;
    spec.height = 16;
    const auto clip = WriteClip(TempDir() / "depth_flicker.mkv", spec);
    const DepthStage::Stats aligned = RunFlicker(clip, Dir("flicker_auto"), DepthStabilize::Auto);
    const DepthStage::Stats raw = RunFlicker(clip, Dir("flicker_none"), DepthStabilize::None);
    CHECK(aligned.frames == 48);
    CHECK(raw.frames == 48);
    CHECK(aligned.MeanTae() < 0.6 * raw.MeanTae());  // before the fix both were the raw model's flicker (ratio 1)
    const Manifest m = Manifest::Load(TempDir() / "depth" / "flicker_auto" / "depth_dlss");
    CHECK(m.stageParams["stabilize"] == "scale");
    CHECK(m.stageParams["stabilize_requested"] == "auto");
    CHECK(m.stageParams["stabilize_window"] == 8);
    CHECK(m.stageParams["stabilize_scale"] != nlohmann::json::array({1.0, 1.0}));
    CHECK(Manifest::Load(TempDir() / "depth" / "flicker_none" / "depth_raw").stageParams["stabilize"] == "none");
}

TEST_CASE("DepthStabilize: auto is scale for metric, scale+shift for relative; explicit modes are kept", "[integration][depth]") {
    using M = TemporalStabilizer::Mode;
    CHECK(ResolveStabilizeMode(DepthStabilize::Auto, true) == M::ScaleOnly);
    CHECK(ResolveStabilizeMode(DepthStabilize::Auto, false) == M::ScaleShift);
    CHECK(ResolveStabilizeMode(DepthStabilize::ScaleShift, true) == M::ScaleShift);  // before: silently scale for metric
    CHECK(ResolveStabilizeMode(DepthStabilize::Scale, false) == M::ScaleOnly);
    CHECK(ResolveStabilizeMode(DepthStabilize::None, true) == M::None);
    CHECK(ResolveStabilizeMode(DepthStabilize::None, false) == M::None);
    CHECK(ParseDepthStabilize("auto") == DepthStabilize::Auto);
    CHECK(ParseDepthStabilize("scale_shift") == DepthStabilize::ScaleShift);
    CHECK_THROWS(ParseDepthStabilize("bogus"));
    for (const auto s : {DepthStabilize::Auto, DepthStabilize::None, DepthStabilize::Scale, DepthStabilize::ScaleShift})
        CHECK(ParseDepthStabilize(DepthStabilizeName(s)) == s);

    ClipSpec spec;
    spec.frames = 5;
    spec.width = 24;
    spec.height = 16;
    const auto clip = WriteClip(TempDir() / "depth_scale_shift.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    DepthStageOptions o;
    o.backend = "stub";
    o.outputDir = Dir("scale_shift");
    o.stabilize = DepthStabilize::ScaleShift;
    const DepthRunResult r = RunDepth(dec, dev, o);
    CHECK(r.stabilize == "scale_shift");
    const Manifest m = Manifest::Load(o.outputDir / "depth_raw");
    CHECK(m.stageParams["stabilize"] == "scale_shift");
    CHECK(m.stageParams.contains("stabilize_shift"));
}

TEST_CASE("GpuFrameCache drops pass textures when a slot is reused for another frame", "[integration][gpu]") {
    D3D12Device dev({true, false});
    GpuFrameCache cache(dev, 1);
    FrameDesc desc;
    desc.width = 16;
    desc.height = 8;
    GpuFrameCache::Slot& slot = cache.Acquire(0, desc);
    cache.UploadPass(slot, "depth_raw", MakePassImage(PassKind::DepthRaw, 16, 8));
    REQUIRE(slot.passes.count("depth_raw") == 1);
    cache.Acquire(0, desc);  // the same frame again: kept
    CHECK(slot.passes.count("depth_raw") == 1);
    GpuFrameCache::Slot& reused = cache.Acquire(1, desc);  // another frame in the only slot
    CHECK(reused.passes.empty());  // a delayed depth must not leave frame 0's texture for frame 1's readers
    CHECK(cache.Find(0) == nullptr);
}
