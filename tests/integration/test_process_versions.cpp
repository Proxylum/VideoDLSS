// Stage 9, MR A: pass fingerprints and versions in RunProcess / PlanProcess with the deterministic backends on WARP —
// a pass is reused only when its fingerprint matches, a changed parameter recomputes the stage and everything below
// it (regression for the stage-8 gap), replaced versions go to <pass>.v/ and come back without recomputing, gc keeps
// two, passes made before fingerprints are adopted, a changed source recomputes everything.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "TestClips.h"
#include "passes/Manifest.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ProcessRunner.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "versions" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

std::vector<ProcessStage> StubStages() {
    std::vector<ProcessStage> s = DefaultProcessStages();
    for (const char* spec : {"depth.backend=stub", "flow.backend=stub", "upscale.backend=nis", "upscale.scale=2", "nr.backend=stub", "fg.backend=blend", "fg.multiplier=2"})
        ApplyParamSpec(s, spec);
    return s;
}

ProcessOptions Options(const std::filesystem::path& dir, const std::filesystem::path& clip) {
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages();
    o.codec = "ffv1";
    o.warp = true;
    return o;
}

const ProcessStageReport& Report(const ProcessResult& r, const std::string& name) {
    for (const auto& s : r.stages)
        if (s.name == name) return s;
    throw std::runtime_error("no report for " + name);
}

const StageDecision& Decision(const ProcessPlan& p, const std::string& name) {
    for (const auto& s : p.stages)
        if (s.name == name) return s;
    throw std::runtime_error("no decision for " + name);
}

std::vector<std::string> Actions(const ProcessPlan& p) {
    std::vector<std::string> out;
    for (const auto& s : p.stages) out.push_back(s.action);
    return out;
}

std::vector<std::string> Statuses(const ProcessResult& r) {
    std::vector<std::string> out;
    for (const auto& s : r.stages)
        if (s.name != "encode") out.push_back(s.status);
    return out;
}

const std::vector<std::string> kStages{"depth", "flow", "upscale", "nr", "fg"};

}  // namespace

TEST_CASE("RunProcess stamps every pass with its fingerprint; PlanProcess predicts the run and then a full reuse", "[integration][process][versions]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 64;
    spec.height = 36;
    const auto dir = Dir("stamp");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o = Options(dir, clip);

    const ProcessPlan plan0 = PlanProcess(o);
    REQUIRE(plan0.stages.size() == 5);
    CHECK(Actions(plan0) == std::vector<std::string>(5, "run"));
    for (const auto& s : plan0.stages) {
        CHECK(s.kind == "missing");
        CHECK(!s.current);
        CHECK(s.fingerprint.rfind("sha256:", 0) == 0);
        CHECK(!s.retire);
    }
    CHECK(plan0.runCount == 5);
    CHECK(plan0.reuseCount == 0);
    CHECK(plan0.finalPass == "color_fg");
    CHECK(plan0.frames == -1);  // MKV carries no frame count: the run covers the whole source
    CHECK(Decision(plan0, "nr").inputs.size() == 3);  // color_sr, depth_dlss, mv_dlss — as the run will produce them
    CHECK(Decision(plan0, "nr").inputs.at("color_sr") == Decision(plan0, "upscale").fingerprint);
    CHECK(Decision(plan0, "fg").inputs.at("color_nr") == Decision(plan0, "nr").fingerprint);
    CHECK(Decision(plan0, "nr").params["intensity"] == 1);
    CHECK(Decision(plan0, "nr").tool["backend"] == "stub");
    CHECK(!std::filesystem::exists(o.passesRoot));  // planning writes nothing
    CHECK(plan0.ToJson()["stages"].size() == 5);

    const ProcessResult r = RunProcess(o);
    CHECK(Statuses(r) == std::vector<std::string>(5, "ran"));
    for (const auto& name : kStages) {
        const ProcessStageReport& s = Report(r, name);
        CHECK(s.fingerprint == Decision(plan0, name).fingerprint);  // the plan and the run agree
        CHECK(s.decision["kind"] == "missing");
        CHECK(s.retired.empty());
        CHECK_THAT(s.reason, ContainsSubstring("missing"));
    }
    const Manifest nr = Manifest::Load(o.passesRoot / "color_nr");
    CHECK(nr.fingerprint == Report(r, "nr").fingerprint);
    CHECK(nr.inputs.at("color_sr") == Report(r, "upscale").fingerprint);
    CHECK(nr.inputs.at("depth_dlss") == Report(r, "depth").fingerprint);
    CHECK(nr.inputs.at("mv_dlss") == Report(r, "flow").fingerprint);
    CHECK(nr.tool["backend"] == "stub");
    CHECK(!nr.tool["app"].get<std::string>().empty());
    CHECK(nr.paramsCanonical["intensity"] == 1);
    CHECK(nr.paramsCanonical["backend"] == "stub");
    CHECK(nr.created.size() == 20);
    CHECK(Manifest::Load(o.passesRoot / "depth_raw").fingerprint == Manifest::Load(o.passesRoot / "depth_dlss").fingerprint);
    CHECK(Manifest::Load(o.passesRoot / "mv_raw").fingerprint == Manifest::Load(o.passesRoot / "mv_dlss").fingerprint);
    CHECK(Manifest::Load(o.passesRoot / "color_fg").inputs.at("color_nr") == nr.fingerprint);
    CHECK(Report(r, "encode").fingerprint == Report(r, "fg").fingerprint);
    CHECK(r.ToJson()["stages"][3]["fingerprint"] == nr.fingerprint);
    CHECK(r.ToJson()["stages"][3]["decision"]["kind"] == "missing");

    const ProcessPlan plan1 = PlanProcess(o);
    CHECK(Actions(plan1) == std::vector<std::string>(5, "reuse"));
    for (const auto& s : plan1.stages) {
        CHECK(s.kind == "exact");
        CHECK(s.current);
        CHECK(s.currentFingerprint == s.fingerprint);
    }
    CHECK(plan1.runCount == 0);
    CHECK(plan1.reuseCount == 5);
    const ProcessResult again = RunProcess(o);
    CHECK(Statuses(again) == std::vector<std::string>(5, "reused"));
    for (const auto& name : kStages) {
        CHECK(Report(again, name).reason.empty());
        CHECK(Report(again, name).fingerprint == Report(r, name).fingerprint);
    }
    CHECK(ListPassVersions(o.passesRoot).size() == 7);  // seven current passes, no history
}

TEST_CASE("Regression (stage-8 gap): a changed NR parameter recomputes nr and fg, keeps the stages above and the old versions", "[integration][process][versions]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("params");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o = Options(dir, clip);
    const ProcessResult base = RunProcess(o);
    const std::string oldNr = Report(base, "nr").fingerprint, oldFg = Report(base, "fg").fingerprint;

    ApplyParamSpec(o.stages, "nr.intensity=1.4");
    const ProcessPlan plan = PlanProcess(o);
    CHECK(Actions(plan) == std::vector<std::string>{"reuse", "reuse", "reuse", "run", "run"});
    const StageDecision& nr = Decision(plan, "nr");
    CHECK(nr.kind == "params_changed");
    CHECK(nr.diff["params"]["intensity"] == nlohmann::json({1, 1.4}));
    CHECK_THAT(nr.detail, ContainsSubstring("intensity: 1 -> 1.4"));
    CHECK(nr.retire);
    CHECK(nr.currentFingerprint == oldNr);
    CHECK(nr.fingerprint != oldNr);
    const StageDecision& fg = Decision(plan, "fg");
    CHECK(fg.kind == "input_changed");
    CHECK(fg.diff["inputs"] == nlohmann::json({"color_nr"}));
    CHECK(fg.inputs.at("color_nr") == nr.fingerprint);
    CHECK(plan.runCount == 2);

    const ProcessResult r = RunProcess(o);
    CHECK(Statuses(r) == std::vector<std::string>{"reused", "reused", "reused", "ran", "ran"});
    CHECK(Report(r, "nr").status == "ran");  // before MR A this was "reused": the parameter change went unnoticed
    CHECK_THAT(Report(r, "nr").reason, ContainsSubstring("params_changed"));
    CHECK(Report(r, "nr").fingerprint == nr.fingerprint);
    CHECK(!Report(r, "nr").retired.empty());
    CHECK(!Report(r, "fg").retired.empty());
    CHECK(Report(r, "depth").retired.empty());
    CHECK(Manifest::Load(o.passesRoot / "color_nr").paramsCanonical["intensity"] == 1.4);
    const auto nrVersions = ListPassVersions(o.passesRoot, "color_nr");
    REQUIRE(nrVersions.size() == 2);
    CHECK(nrVersions[0].current);
    CHECK(nrVersions[1].id == Report(r, "nr").retired);
    CHECK(nrVersions[1].fingerprint == oldNr);
    CHECK(nrVersions[1].params["intensity"] == 1);
    CHECK(Manifest::Exists(o.passesRoot / "color_nr.v" / Report(r, "nr").retired));
    const auto fgVersions = ListPassVersions(o.passesRoot, "color_fg");
    REQUIRE(fgVersions.size() == 2);
    CHECK(fgVersions[1].fingerprint == oldFg);
    CHECK(fgVersions[1].inputs.at("color_nr") == oldNr);
    CHECK(ListPassVersions(o.passesRoot, "color_sr").size() == 1);
    CHECK(DecodeAll(o.output).size() == 7);
}

TEST_CASE("Switching a parameter back restores the previous version instead of recomputing; gc keeps two versions", "[integration][process][versions]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("restore");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o = Options(dir, clip);
    const ProcessResult base = RunProcess(o);
    const std::string nr10 = Report(base, "nr").fingerprint, fg10 = Report(base, "fg").fingerprint;
    ApplyParamSpec(o.stages, "nr.intensity=1.4");
    const ProcessResult run14 = RunProcess(o);
    const std::string nr14 = Report(run14, "nr").fingerprint;
    REQUIRE(nr14 != nr10);

    ApplyParamSpec(o.stages, "nr.intensity=1.0");
    const ProcessPlan plan = PlanProcess(o);
    CHECK(Actions(plan) == std::vector<std::string>{"reuse", "reuse", "reuse", "restore", "restore"});
    CHECK(Decision(plan, "nr").kind == "restored");
    CHECK(Decision(plan, "nr").version == Report(run14, "nr").retired);
    CHECK(Decision(plan, "nr").fingerprint == nr10);
    CHECK(Decision(plan, "fg").fingerprint == fg10);
    CHECK(plan.runCount == 0);
    CHECK(plan.reuseCount == 5);
    const ProcessResult back = RunProcess(o);
    CHECK(Statuses(back) == std::vector<std::string>(5, "reused"));
    CHECK(Report(back, "nr").restored == Report(run14, "nr").retired);
    CHECK(Report(back, "fg").restored == Report(run14, "fg").retired);
    CHECK_THAT(Report(back, "nr").reason, ContainsSubstring("restored"));
    CHECK(Manifest::Load(o.passesRoot / "color_nr").fingerprint == nr10);
    CHECK(Manifest::Load(o.passesRoot / "color_nr").paramsCanonical["intensity"] == 1);
    auto versions = ListPassVersions(o.passesRoot, "color_nr");
    REQUIRE(versions.size() == 2);
    CHECK(versions[1].fingerprint == nr14);  // the 1.4 version waits in the history
    CHECK(PlanProcess(o).runCount == 0);

    // a third value: the run keeps two versions per pass (the current one included)
    ApplyParamSpec(o.stages, "nr.intensity=1.8");
    CHECK(o.keepVersions == 2);
    const ProcessResult run18 = RunProcess(o);
    CHECK(Report(run18, "nr").status == "ran");
    CHECK(ListPassVersions(o.passesRoot, "color_fg").size() == 2);  // nobody refers to color_fg versions: current + 1
    versions = ListPassVersions(o.passesRoot, "color_nr");
    REQUIRE((versions.size() == 2 || versions.size() == 3));  // + the version the kept color_fg one lists as its input
    CHECK(versions[0].current);
    if (versions.size() == 3) CHECK(ReferencedFingerprints(o.passesRoot).count(versions[2].fingerprint) == 1);
    CHECK(ListPassVersions(o.passesRoot, "depth_dlss").size() == 1);

    // keepVersions = 0 keeps everything
    o.keepVersions = 0;
    const size_t before = ListPassVersions(o.passesRoot, "color_nr").size();
    ApplyParamSpec(o.stages, "nr.intensity=1.6");  // within the schema range (0..2): the runner validates parameters now
    RunProcess(o);
    CHECK(ListPassVersions(o.passesRoot, "color_nr").size() == before + 1);
}

TEST_CASE("Passes made before fingerprints are adopted on the first run; another source recomputes everything", "[integration][process][versions]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("legacy");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o = Options(dir, clip);
    const ProcessResult base = RunProcess(o);
    // strip the version block the way a pass written by stage 8 (or a standalone `dlssvid nr`) looks
    for (const auto& name : PassesUnderRoot(o.passesRoot)) {
        const auto file = o.passesRoot / name / Manifest::kFileName;
        nlohmann::json j;
        std::ifstream(file) >> j;
        for (const char* key : {"fingerprint", "inputs", "tool", "params_canonical", "created"}) j.erase(key);
        std::ofstream(file) << j.dump(2);
        CHECK(Manifest::Load(o.passesRoot / name).fingerprint.empty());
    }
    const ProcessPlan plan = PlanProcess(o);
    CHECK(Actions(plan) == std::vector<std::string>(5, "reuse"));
    for (const auto& s : plan.stages) {
        CHECK(s.kind == "legacy");
        CHECK(s.current);
        CHECK(s.currentFingerprint.empty());
    }
    CHECK(Decision(plan, "nr").inputs.at("color_sr") == Decision(plan, "upscale").fingerprint);  // adopted passes count with their new fingerprint
    const ProcessResult adopted = RunProcess(o);
    CHECK(Statuses(adopted) == std::vector<std::string>(5, "reused"));
    for (const auto& name : kStages) {
        CHECK_THAT(Report(adopted, name).reason, ContainsSubstring("adopted"));
        CHECK(Report(adopted, name).fingerprint == Report(base, name).fingerprint);  // the same parameters -> the same fingerprint as the original run
    }
    const Manifest nr = Manifest::Load(o.passesRoot / "color_nr");
    CHECK(nr.fingerprint == Report(base, "nr").fingerprint);
    CHECK(nr.created.size() == 20);
    CHECK(nr.paramsCanonical["intensity"] == 1);
    CHECK(PlanProcess(o).runCount == 0);
    CHECK(ListPassVersions(o.passesRoot).size() == 7);  // adoption creates no history

    // another source: every stage runs again, the old versions are kept aside
    spec.frames = 4;
    o.input = WriteClip(dir / "clip2.mkv", spec);
    const ProcessPlan plan2 = PlanProcess(o);
    CHECK(Actions(plan2) == std::vector<std::string>(5, "run"));
    for (const auto& s : plan2.stages) {
        CHECK(s.kind == "source_changed");
        CHECK(s.retire);
    }
    const ProcessResult other = RunProcess(o);
    CHECK(Statuses(other) == std::vector<std::string>(5, "ran"));
    CHECK(!Report(other, "depth").retired.empty());
    CHECK(ListPassVersions(o.passesRoot, "depth_dlss").size() == 2);
    CHECK(other.framesIn == 4);
}

TEST_CASE("Forced runs, incomplete passes completed in place, a disabled stage leaves no fingerprint", "[integration][process][versions]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("forced");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o = Options(dir, clip);
    o.frames = 3;
    const ProcessResult base = RunProcess(o);
    CHECK(base.framesIn == 3);

    SECTION("--no-skip-existing recomputes in place when the fingerprint is the same") {
        o.skipExisting = false;
        const ProcessPlan plan = PlanProcess(o);
        CHECK(Actions(plan) == std::vector<std::string>(5, "run"));
        CHECK(Decision(plan, "depth").kind == "forced");
        CHECK(!Decision(plan, "depth").retire);
        const ProcessResult r = RunProcess(o);
        CHECK(Statuses(r) == std::vector<std::string>(5, "ran"));
        CHECK(Report(r, "nr").retired.empty());
        CHECK(ListPassVersions(o.passesRoot).size() == 7);
        CHECK(!std::filesystem::exists(o.passesRoot / "color_nr.v"));
    }
    SECTION("more frames of the same source: the passes are incomplete and completed in place") {
        o.frames = 5;
        const ProcessPlan plan = PlanProcess(o);
        CHECK(Actions(plan) == std::vector<std::string>(5, "run"));
        CHECK(Decision(plan, "depth").kind == "incomplete");
        CHECK_THAT(Decision(plan, "depth").detail, ContainsSubstring("3 of 5 frames"));
        CHECK(!Decision(plan, "depth").retire);
        CHECK(Decision(plan, "depth").fingerprint == Report(base, "depth").fingerprint);  // the frame count is not part of the fingerprint
        const ProcessResult r = RunProcess(o);
        CHECK(Statuses(r) == std::vector<std::string>(5, "ran"));
        CHECK(r.framesIn == 5);
        CHECK(ListPassVersions(o.passesRoot).size() == 7);
        CHECK(PlanProcess(o).runCount == 0);
    }
    SECTION("nr ngx on WARP is disabled: the retired version stays in the history, fg follows color_sr") {
        ApplyParamSpec(o.stages, "nr.backend=ngx");
        o.disableUnavailable = true;
        const ProcessPlan plan = PlanProcess(o);
        CHECK(Decision(plan, "nr").kind == "params_changed");
        CHECK_THAT(Decision(plan, "nr").detail, ContainsSubstring("backend: stub -> ngx"));
        const ProcessResult r = RunProcess(o);
        CHECK(Report(r, "nr").status == "disabled");
        CHECK(Report(r, "nr").fingerprint.empty());
        CHECK(!Report(r, "nr").retired.empty());
        CHECK(!Manifest::Exists(o.passesRoot / "color_nr"));
        CHECK(ListPassVersions(o.passesRoot, "color_nr").size() == 1);  // the stub version, in the history
        CHECK(Report(r, "fg").status == "ran");
        CHECK(Report(r, "fg").decision["kind"] == "input_changed");
        CHECK(Manifest::Load(o.passesRoot / "color_fg").inputs.count("color_nr") == 0);
        CHECK(Manifest::Load(o.passesRoot / "color_fg").inputs.at("color_sr") == Report(r, "upscale").fingerprint);
        CHECK(r.finalPass == o.passesRoot / "color_fg");
    }
}

TEST_CASE("Schema defaults are the same run; --force recomputes one stage in place; bad parameters are refused", "[integration][process][versions]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("schema");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o = Options(dir, clip);
    const ProcessResult base = RunProcess(o);
    CHECK(Statuses(base) == std::vector<std::string>(5, "ran"));
    CHECK(Manifest::Load(o.passesRoot / "color_nr").stageParams["ms_per_frame"].get<double>() > 0);  // recorded for the estimates
    CHECK(Manifest::Load(o.passesRoot / "depth_dlss").stageParams["ms_per_frame"].get<double>() > 0);
    CHECK(Manifest::Load(o.passesRoot / "color_nr").paramsCanonical["style"] == "natural");  // the defaults are part of the fingerprint

    // an explicit default is not a change (before MR D: params_changed, the keys were absent when the pass was made)
    ApplyParamSpec(o.stages, "nr.intensity=1.0");
    ApplyParamSpec(o.stages, "nr.passes=1");
    ApplyParamSpec(o.stages, "fg.multiplier=2");
    CHECK(PlanProcess(o).runCount == 0);

    // --force: the stage alone, in place (same fingerprint), the stages below stay reused
    o.forceStages = {"nr"};
    const ProcessPlan plan = PlanProcess(o);
    CHECK(Decision(plan, "nr").action == "run");
    CHECK(Decision(plan, "nr").kind == "forced");
    CHECK(!Decision(plan, "nr").retire);
    CHECK(Decision(plan, "fg").action == "reuse");
    const ProcessResult forced = RunProcess(o);
    CHECK(Statuses(forced) == std::vector<std::string>{"reused", "reused", "reused", "ran", "reused"});
    CHECK(Report(forced, "nr").retired.empty());
    CHECK(ListPassVersions(o.passesRoot, "color_nr").size() == 1);
    o.forceStages = {"bogus"};
    CHECK_THROWS_WITH(PlanProcess(o), ContainsSubstring("--force"));
    o.forceStages.clear();

    // validation against the schema, before anything runs
    ApplyParamSpec(o.stages, "nr.intensity=5");
    CHECK_THROWS_WITH(PlanProcess(o), ContainsSubstring("nr.intensity must be within"));
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("nr.intensity must be within"));
    ApplyParamSpec(o.stages, "nr.intensity=1.0");
    ApplyParamSpec(o.stages, "fg.multiplier=7");
    CHECK_THROWS_WITH(PlanProcess(o), ContainsSubstring("fg.multiplier must be one of"));
    ApplyParamSpec(o.stages, "fg.multiplier=2");
    CHECK(PlanProcess(o).runCount == 0);

    // a hash the caller knows is used as given (the project's cache): another value means another source
    o.sourceHash = "sha256:not-the-same-file";
    const ProcessPlan other = PlanProcess(o);
    CHECK(other.sourceHash == "sha256:not-the-same-file");
    CHECK(Decision(other, "depth").kind == "source_changed");
    o.sourceHash = "garbage";  // not a hash: ignored, the file is hashed
    CHECK(PlanProcess(o).sourceHash == plan.sourceHash);
}
