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
    CHECK(raw.Man().stageParams["stabilize"] == "none");  // metric stub -> no alignment by default
    raw.Validate({48, 32, 7, PassKind::DepthRaw});

    // depth equals the stub's formula on the decoded frame
    VideoDecoder dec2(clip);
    CpuFrame f;
    REQUIRE(dec2.NextFrame(f));
    const PassImage rgb = Yuv420pToRgb(f, ColorInfoFromStream(dec2.Info()), PixelType::F32);
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
    const PassImage rgb = Yuv420pToRgb(f, o.color, PixelType::F32);
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
