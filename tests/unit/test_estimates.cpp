// Estimates (stage 9, MR D): baselines scale with the source's megapixels, a recorded ms/frame wins, the outcome
// combines the plan, the stages and the source into size, rate, time and disk.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "passes/Manifest.h"
#include "pipeline/Estimates.h"
#include "stages/ParamSchema.h"

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
    m.Save(root / "color_nr");
    CHECK(RecordedMsPerFrame(root, "nr") == Approx(123.5));
    Manifest d = Manifest::ForPass(PassKind::DepthRaw, 8, 4, FileFormat::Exr);
    d.Save(root / "depth_raw");
    CHECK(!RecordedMsPerFrame(root, "depth"));  // no record in the manifest

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
