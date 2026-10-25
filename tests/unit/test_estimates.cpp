// Estimates (stage 9, MR D): baselines scale with the source's megapixels, a recorded ms/frame wins, the outcome
// combines the plan, the stages and the source into size, rate, time and disk.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "passes/Manifest.h"
#include "pipeline/Estimates.h"
#include "stages/ParamSchema.h"
#include "stages/nr/NrStage.h"

using namespace dlssvid;
using Catch::Approx;

TEST_CASE("Baselines scale with the source size, the recorded ms/frame wins, bytes per frame follow the layouts", "[unit][estimates]") {
    const double ref = 1920.0 * 800 / 1e6;
    CHECK(BaselineMsPerFrame("depth", "da3", ref) == Approx(350.0));
    CHECK(BaselineMsPerFrame("depth", "da3", 2 * ref) == Approx(700.0));
    CHECK(BaselineMsPerFrame("depth", "vda", ref) == Approx(300.0));
    CHECK(BaselineMsPerFrame("flow", "ofa", ref) == Approx(500.0));
    CHECK(BaselineMsPerFrame("upscale", "dlss", ref) == Approx(420.0));
    CHECK(BaselineMsPerFrame("nr", "ngx", ref) == Approx(480.0));
    CHECK(BaselineMsPerFrame("fg", "dlssg", ref) == Approx(780.0));
    CHECK(BaselineMsPerFrame("encode", "", 3840.0 * 1600 / 1e6) == Approx(150.0));
    CHECK(BaselineMsPerFrame("nr", "stub", ref) < 50.0);
    CHECK(BaselineMsPerFrame("bogus", "", ref) == 0.0);
    CHECK(BaselineMsPerFrame("depth", "da3", 0.0) == Approx(350.0));  // unknown size: the reference

    const auto root = std::filesystem::path(DLSSVID_TEST_TMP) / "estimates";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    CHECK(!RecordedMsPerFrame(root, "nr"));
    Manifest m = Manifest::ForPass(PassKind::ColorNr, 8, 4, FileFormat::Exr);
    m.stageParams["ms_per_frame"] = 123.5;
    m.stageParams["ms_per_frame_gpu"] = 20.0;  // as process records it: the wall time beside the stage's GPU time (D8)
    m.Save(root / "color_nr");
    CHECK(RecordedMsPerFrame(root, "nr") == Approx(123.5));
    Manifest d = Manifest::ForPass(PassKind::DepthRaw, 8, 4, FileFormat::Exr);
    d.Save(root / "depth_raw");
    CHECK(!RecordedMsPerFrame(root, "depth"));  // no record in the manifest
    // TASK-0039: by model - the time of Small is not the time of Large
    CHECK(BaselineMsPerFrame("depth", "vda", ref, "metric-vda-large") > BaselineMsPerFrame("depth", "vda", ref, "metric-vda-small"));
    CHECK(BaselineMsPerFrame("depth", "vda", ref, "metric-vda-small") == Approx(300.0));
    CHECK(BaselineMsPerFrame("depth", "worker:vda", ref) > BaselineMsPerFrame("depth", "vda", ref));
    d.stageParams["ms_per_frame"] = 250.0;
    d.fingerprint = "sha256:made-by-process";  // the version block (tool) is written for stamped passes only
    d.tool = {{"backend", "vda"}, {"model", "metric-vda-small"}};
    d.Save(root / "depth_raw");
    CHECK(!RecordedMsPerFrame(root, "depth", {{"model", "metric-vda-large"}}));
    CHECK(!RecordedMsPerFrame(root, "depth", {{"backend", "da3"}, {"model", "da3metric-large"}}));
    CHECK(RecordedMsPerFrame(root, "depth", {{"backend", "vda"}, {"model", "metric-vda-small"}}) == Approx(250.0));
    CHECK(RecordedMsPerFrame(root, "depth", {{"model", ""}}) == Approx(250.0));  // only fields given and non-empty count
    CHECK(RecordedMsPerFrame(root, "depth") == Approx(250.0));

    CHECK(BytesPerFrame("depth", 1920, 800, 1.0, 1) == 1920ull * 800 * 4 * 2);
    CHECK(BytesPerFrame("flow", 1920, 800, 1.0, 1) == 1920ull * 800 * 8 * 2);
    CHECK(BytesPerFrame("upscale", 1920, 800, 2.0, 1) == 3840ull * 1600 * 8);
    CHECK(BytesPerFrame("nr", 1920, 800, 1.5, 1) == 2880ull * 1200 * 8);
    CHECK(BytesPerFrame("fg", 1920, 800, 2.0, 2) == 3840ull * 1600 * 8 * 2);
    CHECK(BytesPerFrame("encode", 1920, 800, 2.0, 2) == 0);
}

TEST_CASE("EstimateRun combines the plan, the stages and the source", "[unit][estimates]") {
    ProcessPlan plan;
    plan.passesRoot = "Z:/nowhere";
    for (const char* name : {"depth", "flow", "upscale", "nr", "fg"}) {
        StageDecision d;
        d.name = name;
        d.action = (std::string(name) == "nr" || std::string(name) == "fg") ? "run" : "reuse";
        d.params = EffectiveStageParams(name, nlohmann::json::object());
        plan.stages.push_back(d);
    }
    std::vector<ProcessStage> stages = DefaultProcessStages();  // upscale x2, fg x2
    SourceInfo src;
    src.width = 1920;
    src.height = 800;
    src.fps = 24.0;
    src.frames = 240;
    src.audio = true;
    RunOutcome o = EstimateRun(plan, stages, src);
    CHECK(o.width == 3840);
    CHECK(o.height == 1600);
    CHECK(o.fps == Approx(48.0));
    CHECK(o.audio);
    CHECK(o.stagesTotal == 6);
    CHECK(o.stagesToRun == 3);  // nr, fg and the encode
    CHECK(o.reused == std::vector<std::string>{"depth", "flow", "upscale"});
    CHECK(o.seconds == Approx(240 * (0.48 + 0.78) + 480 * 0.15).margin(1.0));
    CHECK(o.fullSeconds == Approx(240 * (0.35 + 0.5 + 0.42 + 0.48 + 0.78) + 480 * 0.15).margin(1.0));
    CHECK(o.newBytes == BytesPerFrame("nr", 1920, 800, 2.0, 2) * 240 + BytesPerFrame("fg", 1920, 800, 2.0, 2) * 240);
    REQUIRE(o.stages.size() == 6);
    CHECK(o.stages[3].stage == "nr");
    CHECK(o.stages[3].seconds == Approx(240 * 0.48).margin(0.5));
    CHECK(!o.stages[3].recorded);
    CHECK(o.stages[0].seconds == 0.0);  // reused
    CHECK(o.stages[0].bytes == 0);
    CHECK(o.stages[5].stage == "encode");
    CHECK(o.stages[5].frames == 480);
    CHECK(o.stages[4].frames == 479);  // 2N-1 generated + real frames

    // frame generation off: the rate stays, the encode covers the source frames, no upscale: the size stays
    for (auto& s : stages)
        if (s.name == "fg" || s.name == "upscale") s.enabled = false;
    plan.stages.clear();
    o = EstimateRun(plan, stages, src);
    CHECK(o.fps == Approx(24.0));
    CHECK(o.width == 1920);
    CHECK(o.stagesTotal == 1);
    CHECK(o.stages.back().frames == 240);
}

// D8 (TASK-0049): a standalone `dlssvid nr` (the GUI's «Запустить отдельно») recorded its 16 ms GPU time as ms_per_frame and
// `process` adopted the pass with it - the NR ETA came out ~25x short; passes and model_scale did not move the estimate at all.
TEST_CASE("NR estimate follows passes and model scale; a GPU-only ms/frame is not wall time", "[unit][estimates][regression]") {
    const auto root = std::filesystem::path(DLSSVID_TEST_TMP) / "estimates_nr";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    auto save = [&](int passes, const nlohmann::json& times) {
        Manifest m = Manifest::ForPass(PassKind::ColorNr, 2560, 1440, FileFormat::Exr);
        m.frameCount = 300;
        m.stageParams = {{"backend", "ngx"}, {"passes", passes}, {"model_scale", 1.0}, {"work_size", {2560, 1440}}};
        m.stageParams.update(times);
        m.Save(root / "color_nr");
    };
    auto params = [](int passes, double scale) { return EffectiveStageParams("nr", {{"passes", passes}, {"model_scale", scale}}); };
    const nlohmann::json ngx = {{"backend", "ngx"}};

    // GPU-only times are not wall times
    for (const nlohmann::json& times : {nlohmann::json{{"ms_per_frame_gpu", 15.9}}, nlohmann::json{{"ms_per_frame", 15.9}},
                                        nlohmann::json{{"ms_per_frame", 12.0}, {"ms_per_frame_gpu", 15.9}}}) {
        INFO(times.dump());
        save(1, times);
        CHECK_FALSE(RecordedMsPerFrame(root, "nr", ngx, params(1, 1.0), 300));
    }
    // a process record: as it is for the same run
    save(1, {{"ms_per_frame", 410.0}, {"ms_per_frame_gpu", 15.9}});
    CHECK(*RecordedMsPerFrame(root, "nr", ngx, params(1, 1.0), 300) == Approx(410.0));
    CHECK(*RecordedMsPerFrame(root, "nr", ngx) == Approx(410.0));

    CHECK(NrModelMsPerPass("ngx", 2560.0 * 1440) == Approx(17.6));
    CHECK(NrModelMsPerPass("ngx", 3840.0 * 2160) == Approx(17.6 * 2.25));
    CHECK(NrModelMsPerPass("stub", 2560.0 * 1440) == 0.0);

    // two passes: the model's step plus the second feature created once, spread over the frames
    const auto two = RecordedMsPerFrame(root, "nr", ngx, params(2, 1.0), 300);
    REQUIRE(two);
    CHECK(*two > 410.0 + 17.6);
    CHECK(*two < 410.0 + 17.6 + 10.0);
    CHECK(*RecordedMsPerFrame(root, "nr", ngx, params(2, 1.0), 30) > *two);  // a short run carries more of the creation
    CHECK(*RecordedMsPerFrame(root, "nr", ngx, params(2, 1.0), 1000000) == Approx(410.0 + 17.6).margin(0.01));
    // the model's resolution
    CHECK(*RecordedMsPerFrame(root, "nr", ngx, params(1, 0.5), 300) == Approx(410.0 - 17.6 + 17.6 / 4));
    CHECK(*RecordedMsPerFrame(root, "nr", ngx, params(1, 2.0), 300) == Approx(410.0 - 17.6 + 17.6 * 4));
    CHECK(*RecordedMsPerFrame(root, "nr", ngx, params(2, 0.5), 300) < *two);
    // a two-pass record moved to one pass: the creation is not taken back
    save(2, {{"ms_per_frame", 464.0}, {"ms_per_frame_gpu", 33.5}});
    CHECK(*RecordedMsPerFrame(root, "nr", ngx, params(1, 1.0), 300) == Approx(464.0 - 17.6));
    CHECK(*RecordedMsPerFrame(root, "nr", ngx, params(2, 1.0), 300) == Approx(464.0));

    // EstimateRun: NR works on the upscaled frame (1280x720 x2 = 2560x1440)
    ProcessPlan plan;
    plan.passesRoot = root;
    StageDecision d;
    d.name = "nr";
    d.action = "run";
    d.params = EffectiveStageParams("nr", {{"passes", 2}});
    d.tool = ngx;
    plan.stages.push_back(d);
    const std::vector<ProcessStage> stages = DefaultProcessStages();
    SourceInfo src;
    src.width = 1280;
    src.height = 720;
    src.fps = 24.0;
    src.frames = 300;
    save(1, {{"ms_per_frame", 410.0}, {"ms_per_frame_gpu", 15.9}});
    RunOutcome o = EstimateRun(plan, stages, src);
    REQUIRE(!o.stages.empty());
    CHECK(o.stages[0].recorded);
    CHECK(o.stages[0].msPerFrame == Approx(*two));
    save(1, {{"ms_per_frame_gpu", 15.9}});  // a standalone run's pass: the baseline, moved to two passes - not 16 ms
    const double base = BaselineMsPerFrame("nr", "ngx", 1280.0 * 720 / 1e6);
    o = EstimateRun(plan, stages, src);
    CHECK_FALSE(o.stages[0].recorded);
    CHECK(o.stages[0].msPerFrame > base + 17.6);
    CHECK(o.stages[0].msPerFrame < base + 17.6 + 10.0);
    plan.stages[0].params = EffectiveStageParams("nr", nlohmann::json::object());
    o = EstimateRun(plan, stages, src);
    CHECK(o.stages[0].msPerFrame == Approx(base));

    CHECK(NrWorkSize(2560, 1440, 0.5) == std::pair<uint32_t, uint32_t>{1280, 720});
    CHECK(NrWorkSize(1281, 721, 1.0) == std::pair<uint32_t, uint32_t>{1280, 720});
}
