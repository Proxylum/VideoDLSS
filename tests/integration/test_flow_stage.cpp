// FlowStage with the stub estimator: one-frame latency, zero tail, mv_dlss with depth, GPU textures.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "io/VideoDecoder.h"
#include "passes/PassSequence.h"
#include "passes/ScenesPass.h"
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

// PR-13 (regression): no motion across a cut of the clip's list - mv_raw of the frame before the cut (prev -> cut) and mv_dlss of the
// cut frame (cut -> prev) are zero, every other pair keeps its vectors; the cut pair is no part of the magnitude and the warp PSNR; both
// passes carry the label. A cut at frame 1 (the first pair) opens the writers at the right size; one beyond --frames changes nothing.
TEST_CASE("FlowStage writes zero vectors across a cut of the scene list", "[integration][flow][scenes][regression]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(TempDir() / "flow_cut.mkv", spec);
    D3D12Device dev({true, false});
    auto list = [](std::vector<int64_t> cuts) {
        SceneList l;
        l.cuts = std::move(cuts);
        l.id = SceneCutsId(l.cuts);
        l.threshold = 0.24;
        l.frames = 6;
        l.reachedEnd = true;
        return l;
    };
    auto run = [&](const char* name, const SceneList& cuts, int64_t maxFrames = -1) {
        VideoDecoder dec(clip);
        FlowStageOptions o;
        o.backend = "stub";
        o.estimator.extra["dx"] = 2.f;
        o.estimator.extra["dy"] = -1.f;
        o.outputDir = Dir(name);
        o.convert.dilateRadius = 0;
        o.sceneCuts = cuts;
        return RunFlow(dec, dev, o, maxFrames);
    };
    const FlowRunResult r = run("cut3", list({3}));
    CHECK(r.stats.pairs == 5);
    CHECK(r.stats.cutPairs == 1);
    CHECK(r.stats.sceneCuts == std::vector<int64_t>{3});
    CHECK(std::fabs(r.stats.meanMagnitude - std::sqrt(5.0)) < 1e-4);  // the four pairs within a shot
    CHECK(r.stats.psnrCount == 4);
    const PassReader raw = PassReader::Open(r.rawDir), dlss = PassReader::Open(r.dlssDir);
    for (int64_t f : {0, 1, 3, 4}) {
        INFO("mv_raw " << f);
        CHECK(raw.ReadFrame(f).Get(5, 5, 0) == 2.f);
    }
    CHECK(raw.ReadFrame(2).Get(5, 5, 0) == 0.f);  // 2 -> 3 crosses the cut
    CHECK(raw.ReadFrame(2).Get(5, 5, 1) == 0.f);
    for (int64_t f : {1, 2, 4, 5}) {
        INFO("mv_dlss " << f);
        CHECK(dlss.ReadFrame(f).Get(20, 20, 0) == -2.f);
    }
    CHECK(dlss.ReadFrame(3).Get(20, 20, 0) == 0.f);  // 3 -> 2 crosses it too
    CHECK(dlss.ReadFrame(3).Get(20, 20, 1) == 0.f);
    for (const PassReader* rd : {&raw, &dlss}) {
        CHECK(rd->Man().stageParams["scene_cuts_id"] == SceneCutsId({3}));
        CHECK(rd->Man().stageParams["scene_cut_count"] == 1);
        CHECK(rd->Man().stageParams["scene_cuts"] == nlohmann::json::array({3}));
        CHECK(SceneLabelOf(rd->Man()) == SceneCutsId({3}));
    }

    const FlowRunResult first = run("cut1", list({1}));  // the first pair: the writers still get the estimator's size
    PassReader::Open(first.rawDir).Validate({48, 32, 6, PassKind::MvRaw});
    PassReader::Open(first.dlssDir).Validate({48, 32, 6, PassKind::MvDlss});
    CHECK(PassReader::Open(first.rawDir).ReadFrame(0).Get(5, 5, 0) == 0.f);
    CHECK(PassReader::Open(first.dlssDir).ReadFrame(1).Get(20, 20, 0) == 0.f);
    CHECK(PassReader::Open(first.dlssDir).ReadFrame(2).Get(20, 20, 0) == -2.f);

    const FlowRunResult part = run("beyond", list({5}), 4);  // frames 0..3: the cut at 5 is not in the run
    CHECK(part.stats.cutPairs == 0);
    CHECK(PassReader::Open(part.rawDir).Man().stageParams["scene_cut_count"] == 0);
    CHECK(PassReader::Open(part.rawDir).ReadFrame(2).Get(5, 5, 0) == 2.f);

    const FlowRunResult none = run("none", SceneList{});  // no list: no label
    CHECK(none.stats.cutPairs == 0);
    CHECK_FALSE(PassReader::Open(none.rawDir).Man().stageParams.contains("scene_cuts_id"));
    CHECK_FALSE(PassReader::Open(none.dlssDir).Man().stageParams.contains("scene_cuts_id"));
}
