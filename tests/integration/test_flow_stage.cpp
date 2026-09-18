// FlowStage with the stub estimator: one-frame latency, zero tail, mv_dlss with depth, GPU textures.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "io/VideoDecoder.h"
#include "passes/PassSequence.h"
#include "pipeline/Pipeline.h"
#include "stages/depth/DepthStage.h"
#include "stages/flow/FlowStage.h"
#include "stages/flow/StubFlowEstimator.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {
std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "flow" / name;
    std::filesystem::remove_all(d);
    return d;
}
}  // namespace

TEST_CASE("FlowStage writes mv_raw for every frame and mv_dlss shifted by one", "[integration][flow]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(TempDir() / "flow_stub.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    FlowStageOptions o;
    o.backend = "stub";
    o.estimator.extra["dx"] = 2.f;
    o.estimator.extra["dy"] = -1.f;
    o.outputDir = Dir("stub");
    o.convert.dilateRadius = 0;
    o.sourceFile = "flow_stub.mkv";
    int frames = 0;
    o.onFrame = [&](int64_t, double) { ++frames; };
    const FlowRunResult r = RunFlow(dec, dev, o);
    CHECK(r.stats.frames == 6);
    CHECK(r.stats.pairs == 5);
    CHECK(frames == 5);
    CHECK(std::fabs(r.stats.meanMagnitude - std::sqrt(5.0)) < 1e-4);
    CHECK(r.estimator["backend"] == "stub");

    const PassReader raw = PassReader::Open(r.rawDir);
    raw.Validate({48, 32, 6, PassKind::MvRaw});
    CHECK(raw.Man().mv.direction == "forward");
    CHECK(raw.ReadFrame(0).Get(5, 5, 0) == 2.f);
    CHECK(raw.ReadFrame(4).Get(5, 5, 1) == -1.f);
    CHECK(raw.ReadFrame(5).Get(5, 5, 0) == 0.f);  // last frame: no next frame -> zero
    CHECK(raw.Man().stageParams.contains("mean_magnitude_px"));

    const PassReader dlss = PassReader::Open(r.dlssDir);
    dlss.Validate({48, 32, 6, PassKind::MvDlss});
    CHECK(dlss.Man().mv.direction == "backward");
    CHECK(dlss.ReadFrame(0).Get(5, 5, 0) == 0.f);   // first frame: zero
    CHECK(dlss.ReadFrame(1).Get(20, 20, 0) == -2.f);  // inverted forward flow of frame 0
    CHECK(dlss.ReadFrame(1).Get(20, 20, 1) == 1.f);
    CHECK(dlss.ReadFrame(5).Get(20, 20, 0) == -2.f);
    CHECK(dlss.Man().stageParams.contains("warp_psnr_mean_db"));
}

TEST_CASE("FlowStage scales mv_dlss to the target and uses depth from a pass folder", "[integration][flow]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 32;
    spec.height = 16;
    const auto clip = WriteClip(TempDir() / "flow_target.mkv", spec);
    // depth from the stub depth stage
    const auto depthRoot = Dir("depth_for_flow");
    {
        VideoDecoder dec(clip);
        D3D12Device dev({true, false});
        DepthStageOptions d;
        d.backend = "stub";
        d.outputDir = depthRoot;
        d.writeDlss = false;
        RunDepth(dec, dev, d);
    }
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    FlowStageOptions o;
    o.backend = "stub";
    o.estimator.extra["dx"] = 1.f;
    o.outputDir = Dir("target");
    o.targetWidth = 64;
    o.targetHeight = 32;
    o.depthDir = depthRoot / "depth_raw";
    const FlowRunResult r = RunFlow(dec, dev, o);
    const PassReader dlss = PassReader::Open(r.dlssDir);
    CHECK(dlss.Man().width == 64);
    CHECK(dlss.Man().height == 32);
    CHECK(dlss.Man().mv.refWidth == 64);
    CHECK(dlss.Man().stageParams["depth_occlusion"] == true);
    CHECK(dlss.ReadFrame(2).Get(30, 15, 0) == -2.f);  // 1 px at source = 2 px at 2x target
    CHECK(r.stats.MeanWarpPsnr() > 0.0);
}

TEST_CASE("FlowStage uploads mv_raw and mv_dlss into the GPU cache", "[integration][flow][gpu]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 24;
    spec.height = 16;
    const auto clip = WriteClip(TempDir() / "flow_gpu.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    FlowStageOptions o;
    o.backend = "stub";
    o.estimator.extra["dx"] = 3.f;
    o.outputDir.clear();
    o.color = ColorInfoFromStream(dec.Info());
    Pipeline pipeline(dev, 8);
    pipeline.AddStage(std::make_unique<FlowStage>(o));
    pipeline.Init();
    CpuFrame f;
    while (dec.NextFrame(f)) {
        pipeline.Cache().Acquire(f.index, f.desc);
        pipeline.ProcessFrame(f);
    }
    pipeline.Finish();
    GpuFrameCache::Slot* s0 = pipeline.Cache().Find(0);
    GpuFrameCache::Slot* s1 = pipeline.Cache().Find(1);
    REQUIRE(s0);
    REQUIRE(s1);
    CHECK(s0->passes.count("mv_raw") == 1);
    CHECK(s1->passes.count("mv_dlss") == 1);
    CHECK(pipeline.Cache().DownloadPass(*s0, "mv_raw").Get(3, 3, 0) == 3.f);
    CHECK(pipeline.Cache().DownloadPass(*s1, "mv_dlss").Get(10, 10, 0) == -3.f);
    pipeline.Shutdown();
}

TEST_CASE("Stub flow estimator honours its configuration", "[flow]") {
    StubFlowEstimator est;
    FlowEstimatorConfig cfg;
    cfg.extra["dx"] = 0.5f;
    est.Init(cfg, 8, 4);
    const PassImage rgb = MakePassImage(PassKind::ColorSource, 8, 4, PixelType::F32);
    FlowInput a, b;
    a.rgb = &rgb;
    b.rgb = &rgb;
    PassImage flow, conf;
    est.Estimate(a, b, flow, &conf);
    CHECK(flow.width == 8);
    CHECK(flow.Get(1, 1, 0) == 0.5f);
    CHECK(conf.Get(1, 1, 0) == 1.f);
    CHECK(est.Calls() == 1);
    FlowInput none;
    CHECK_THROWS(est.Estimate(none, none, flow, nullptr));
    CHECK_THROWS(CreateFlowEstimator("nope"));
}
