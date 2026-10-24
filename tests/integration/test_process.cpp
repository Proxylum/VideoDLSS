// The whole pipeline as one run (stage 8): RunProcess over the pass cache with the deterministic backends on WARP,
// pass reuse on a rerun, --frames, audio copy, nr disabled without a DLL, passthrough, parameter specs.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>

#include "TestClips.h"
#include "passes/PassSequence.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ProcessRunner.h"
#include "stages/fg/IFrameGenerator.h"
#include "viewport/Project.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Approx;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "process" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

std::vector<ProcessStage> StubStages(int multiplier = 2) {
    std::vector<ProcessStage> s = DefaultProcessStages();
    for (const char* spec : {"depth.backend=stub", "flow.backend=stub", "upscale.backend=nis", "upscale.scale=2", "nr.backend=stub", "fg.backend=blend"}) ApplyParamSpec(s, spec);
    ApplyParamSpec(s, "fg.multiplier=" + std::to_string(multiplier));
    return s;
}

const ProcessStageReport& Report(const ProcessResult& r, const std::string& name) {
    for (const auto& s : r.stages)
        if (s.name == name) return s;
    throw std::runtime_error("no report for " + name);
}

}  // namespace

TEST_CASE("ApplyParamSpec parses stage.key=value into typed JSON", "[integration][process]") {
    std::vector<ProcessStage> s = DefaultProcessStages();
    REQUIRE(s.size() == 5);
    ApplyParamSpec(s, "nr.intensity=1.5");
    ApplyParamSpec(s, "fg.multiplier=3");
    ApplyParamSpec(s, "upscale.backend=nis");
    ApplyParamSpec(s, "nr.no_guides=true");
    ApplyParamSpec(s, "depth.model=da3mono-large");
    for (const auto& st : s) {
        if (st.name == "nr") {
            CHECK(st.params["intensity"] == 1.5);
            CHECK(st.params["no_guides"] == true);
        }
        if (st.name == "fg") CHECK(st.params["multiplier"] == 3);
        if (st.name == "upscale") CHECK(st.params["backend"] == "nis");
        if (st.name == "depth") CHECK(st.params["model"] == "da3mono-large");
    }
    CHECK_THROWS(ApplyParamSpec(s, "nr=1"));
    CHECK_THROWS(ApplyParamSpec(s, "bogus.key=1"));
    CHECK_THROWS(ApplyParamSpec(s, "nr.=1"));
    CHECK(ProcessStageOrder() == std::vector<std::string>{"depth", "flow", "upscale", "nr", "fg"});
    CHECK(!PassComplete(TempDir() / "process" / "nowhere", 1));
}

TEST_CASE("RunProcess: stub stages on WARP produce a result with doubled frames and audio; a rerun reuses the passes", "[integration][process]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 64;
    spec.height = 36;
    spec.audio = true;
    const auto dir = Dir("stub");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    o.codec = "ffv1";
    o.warp = true;
    std::vector<std::string> seen;
    o.progress = [&](const std::string& stage, int64_t, int64_t) {
        if (seen.empty() || seen.back() != stage) seen.push_back(stage);
    };
    const ProcessResult r = RunProcess(o);
    REQUIRE(r.stages.size() == 6);
    for (const char* name : {"depth", "flow", "upscale", "nr", "fg"}) CHECK(Report(r, name).status == "ran");
    CHECK(Report(r, "encode").status == "ran");
    CHECK(Report(r, "fg").frames == 5);
    CHECK(r.finalPass == dir / "passes" / "color_fg");
    CHECK(r.framesIn == 5);
    CHECK(r.framesOut == FgFrameCount(5, 2));
    CHECK(r.outputFps.ToDouble() == Approx(48.0));
    CHECK(r.audio);
    CHECK(seen == std::vector<std::string>{"depth", "flow", "upscale", "nr", "fg", "encode"});
    REQUIRE(std::filesystem::exists(o.output));
    CHECK(DecodeAll(o.output).size() == 9);
    const StreamSummary sum = Summarize(o.output);
    CHECK(sum.videoStreams == 1);
    CHECK(sum.audioStreams == 1);
    CHECK(sum.audioPackets > 0);
    const PassReader sr = PassReader::Open(dir / "passes" / "color_sr");
    CHECK(sr.Man().width == 128);
    CHECK(sr.Man().frameCount == 5);
    const nlohmann::json j = r.ToJson();
    CHECK(j["stages"].size() == 6);
    CHECK(j["frames_out"] == 9);

    // rerun: every stage's pass is complete -> reused, only the encode runs
    const ProcessResult again = RunProcess(o);
    for (const char* name : {"depth", "flow", "upscale", "nr", "fg"}) CHECK(Report(again, name).status == "reused");
    CHECK(Report(again, "encode").status == "ran");
    CHECK(again.framesOut == 9);
    // forced recompute
    ProcessOptions forced = o;
    forced.skipExisting = false;
    CHECK(Report(RunProcess(forced), "depth").status == "ran");
}

// TASK-0042 (review): a PNG16 colour pass written before the fix is black. The "+png.2" revision changes the fingerprint
// of PNG runs, but a pass without a fingerprint (a standalone `dlssvid nr --format png16`) would be adopted and stamped
// with it - and a pass adopted under exr parameters is PNG all the same. Judged by the file: not reused, not adopted.
TEST_CASE("RunProcess: a PNG16 colour pass from before TASK-0042 is recomputed, with or without a fingerprint", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("old_png");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    ApplyParamSpec(o.stages, "nr.format=png");
    o.codec = "ffv1";
    o.warp = true;
    RunProcess(o);
    const auto nrDir = o.passesRoot / "color_nr";
    REQUIRE(Manifest::Load(nrDir).format == FileFormat::Png);
    CHECK(Report(RunProcess(o), "nr").status == "reused");  // written by this build: +png.2
    auto stampOld = [&](bool keepFingerprint) {
        Manifest m = Manifest::Load(nrDir);
        m.tool["app"] = std::string(kPassToolVersion);  // what the builds before stamped
        if (!keepFingerprint) {
            m.fingerprint.clear();
            m.tool = nlohmann::json::object();
        }
        m.Save(nrDir);
    };
    stampOld(true);
    const size_t versions = ListPassVersions(o.passesRoot, "color_nr").size();
    ProcessResult r = RunProcess(o);
    CHECK(Report(r, "nr").status == "ran");
    CHECK(Report(r, "nr").reason.find("TASK-0042") != std::string::npos);
    // review of TASK-0043: the recomputed pass keeps its fingerprint, so fg built on the black one would be reused
    CHECK(Report(r, "fg").status == "ran");
    CHECK(ListPassVersions(o.passesRoot, "color_nr").size() == versions);  // discarded, not kept to be restored
    stampOld(false);
    CHECK(Report(RunProcess(o), "nr").status == "ran");  // before: adopted as legacy and stamped +png.2 for good
    CHECK(Report(RunProcess(o), "nr").status == "reused");
}

// TASK-0043: the fingerprint hashed the schema's view of the parameters while the stages read the raw ones - an alias ran
// at its value under a fingerprint saying the default. The parameters are normalised for both now (the handoff of the
// effective parameters to the stages is covered by the mirror test in test_param_schema: with the defaults in agreement
// it cannot be told apart from the outside).
TEST_CASE("RunProcess: the stage runs with the parameters its fingerprint hashes", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("run_params");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    for (auto& st : o.stages)
        if (st.name == "nr") st.params["model-scale"] = 0.5;  // the alias the standalone CLI and old projects use
    o.codec = "ffv1";
    o.warp = true;
    RunProcess(o);
    const Manifest m = Manifest::Load(o.passesRoot / "color_nr");
    CHECK(m.paramsCanonical["model_scale"] == 0.5);          // the fingerprint: before, 1.0
    CHECK(m.stageParams["model_scale"] == 0.5);              // what ran
    CHECK_FALSE(m.paramsCanonical.contains("model-scale"));
    CHECK(m.stageParams["tonemap"]["exposure_ev"] == 0.0);  // the schema's default reached the stage
}

// TASK-0043: nr.exposure was a linear multiplier while the schema said 0 - an explicit 0 gave black frames under the
// fingerprint of the identity run. In stops 0 is the identity, so such a pass must not be reused, restored or adopted;
// a pass made with the old default (multiplier 1) stays valid.
TEST_CASE("RunProcess: an nr pass made with the old linear exposure 0 is recomputed, one made with 1 is reused", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("old_exposure");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    o.codec = "ffv1";
    o.warp = true;
    RunProcess(o);
    const auto nrDir = o.passesRoot / "color_nr";
    REQUIRE(Manifest::Load(nrDir).stageParams["tonemap"].contains("exposure_ev"));  // what this build writes
    auto stampOld = [&](double multiplier, bool keepFingerprint) {
        Manifest m = Manifest::Load(nrDir);
        nlohmann::json& t = m.stageParams["tonemap"];
        t.erase("exposure_ev");
        t["exposure"] = multiplier;  // what the builds before wrote
        if (!keepFingerprint) m.fingerprint.clear();  // a standalone `dlssvid nr` pass, adopted as legacy
        m.Save(nrDir);
    };
    stampOld(1.0, true);
    CHECK(Report(RunProcess(o), "nr").status == "reused");
    stampOld(0.0, true);
    const ProcessResult r = RunProcess(o);
    CHECK(Report(r, "nr").status == "ran");
    CHECK_THAT(Report(r, "nr").reason, Catch::Matchers::ContainsSubstring("old linear exposure"));
    // review: color_nr keeps its fingerprint (exposure 0 = the identity hashes as before), so the color_fg built on the
    // black pass matched too and stayed - the result black. It reruns now, and a rerun after it reuses again.
    CHECK(Report(r, "fg").status == "ran");
    CHECK_THAT(Report(r, "fg").reason, Catch::Matchers::ContainsSubstring("stale"));
    CHECK(Report(RunProcess(o), "fg").status == "reused");
    CHECK(Manifest::Load(nrDir).stageParams["tonemap"].contains("exposure_ev"));
    // the black pass went into the history: it is not restored for the same fingerprint either
    stampOld(0.0, true);
    ProcessOptions again = o;
    again.stages = StubStages(2);
    ApplyParamSpec(again.stages, "nr.intensity=1.5");
    RunProcess(again);  // retires the old-exposure pass
    CHECK(Report(RunProcess(o), "nr").status == "ran");  // back to intensity 1: the retired black version is not restored
    stampOld(0.0, false);
    CHECK(Report(RunProcess(o), "nr").status == "ran");  // nor adopted without a fingerprint
}

// TASK-0045: with the guides off nr reads no depth / mv, yet both were inputs of its fingerprint - a new depth recomputed it.
TEST_CASE("RunProcess: an nr pass with the guides off is not recomputed when the depth changes", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("guides_off");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    ApplyParamSpec(o.stages, "nr.guides=false");
    o.codec = "ffv1";
    o.warp = true;
    RunProcess(o);
    const auto inputs = Manifest::Load(o.passesRoot / "color_nr").inputs;
    CHECK_FALSE(inputs.count("depth_dlss"));
    CHECK_FALSE(inputs.count("mv_dlss"));
    // another depth_dlss; upscale off - it reads the depth, a new color_sr would recompute nr through its colour input
    ApplyParamSpec(o.stages, "depth.z_far=500");
    for (auto& s : o.stages) s.enabled = s.name == "depth" || s.name == "flow" || s.name == "nr";
    const ProcessResult r = RunProcess(o);
    CHECK(Report(r, "depth").status == "ran");
    CHECK(Report(r, "nr").status == "reused");  // before: input_changed
}

TEST_CASE("RunProcess: --frames, passthrough, an unavailable nr stage and a run without colour stages", "[integration][process]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 48;
    spec.height = 32;
    spec.audio = true;
    const auto dir = Dir("variants");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.passesRoot = dir / "passes";
    o.stages = StubStages(3);
    o.codec = "ffv1";
    o.warp = true;

    SECTION("frames limit and x3") {
        o.output = dir / "three.mkv";
        o.frames = 3;
        const ProcessResult r = RunProcess(o);
        CHECK(r.framesIn == 3);
        CHECK(r.framesOut == FgFrameCount(3, 3));
        CHECK(r.outputFps.ToDouble() == Approx(72.0));
        CHECK(DecodeAll(o.output).size() == 7);
    }
    SECTION("nr ngx on WARP is disabled when asked, fg still runs from color_sr") {
        o.output = dir / "nongx.mkv";
        ApplyParamSpec(o.stages, "nr.backend=ngx");
        o.disableUnavailable = true;
        const ProcessResult r = RunProcess(o);
        CHECK(Report(r, "nr").status == "disabled");
        CHECK(!Report(r, "nr").reason.empty());
        CHECK(Report(r, "fg").status == "ran");
        CHECK(r.finalPass == dir / "passes" / "color_fg");
        CHECK(r.framesOut == FgFrameCount(6, 3));
        o.disableUnavailable = false;
        o.output = dir / "nongx_fail.mkv";
        o.passesRoot = dir / "passes_fail";
        CHECK_THROWS_WITH(RunProcess(o), Catch::Matchers::ContainsSubstring("stage 'nr' failed"));
    }
    SECTION("no colour stage: the source is encoded unchanged") {
        o.output = dir / "guides_only.mkv";
        o.passesRoot = dir / "passes_guides";
        for (auto& s : o.stages) s.enabled = s.name == "depth" || s.name == "flow";
        const ProcessResult r = RunProcess(o);
        CHECK(r.finalPass.empty());
        CHECK(Report(r, "encode").status == "passthrough");
        CHECK(r.framesOut == 6);
        CHECK(Summarize(o.output).audioStreams == 1);
    }
    SECTION("passthrough option") {
        o.output = dir / "pt.mkv";
        o.passthrough = true;
        const ProcessResult r = RunProcess(o);
        REQUIRE(r.stages.size() == 1);
        CHECK(r.stages[0].status == "passthrough");
        CHECK(r.framesOut == 6);
        CHECK(r.audio);
    }
    SECTION("stages from a project file and unknown stages") {
        Project p = Project::Create(clip, dir / "passes_project");
        for (auto& st : p.stages) {
            if (st.name == "depth" || st.name == "flow" || st.name == "nr") st.params["backend"] = "stub";
            if (st.name == "upscale") st.params["backend"] = "nis";
            if (st.name == "fg") st.params["backend"] = "blend";
        }
        o.stages = StagesFromProject(p);
        o.passesRoot = p.passesRoot;
        o.output = dir / "project.mkv";
        o.frames = 2;
        const ProcessResult r = RunProcess(o);
        CHECK(r.framesOut == FgFrameCount(2, 2));
        o.stages.push_back(ProcessStage{"bogus", true, {}});
        CHECK_THROWS(RunProcess(o));
    }
}

// Audit fg-2 (TASK-0038): an FG stage that fails mid-run - here it reads a color_nr shorter than the source - is
// reported failed with --disable-unavailable, in this process and in its own; the result is encoded from color_nr.
TEST_CASE("RunProcess: an fg stage that throws mid-run is failed with disableUnavailable; the result comes from color_nr",
          "[integration][process][fg][isolation]") {
    for (const bool isolated : {false, true}) {
        INFO((isolated ? "in its own process" : "in this process"));
        ClipSpec spec;
        spec.frames = 5;
        spec.width = 48;
        spec.height = 32;
        const auto dir = Dir(isolated ? "fg_throws_child" : "fg_throws");
        const auto clip = WriteClip(dir / "clip.mkv", spec);
        ProcessOptions o;
        o.input = clip;
        o.output = dir / "r1.mkv";
        o.passesRoot = dir / "passes";
        o.stages = StubStages(2);
        o.codec = "ffv1";
        o.warp = true;
        o.frames = 3;
        RunProcess(o);  // color_nr of 3 frames
        for (auto& s : o.stages) s.enabled = s.name == "fg";
        o.frames = -1;
        o.forceStages = {"fg"};  // the MKV has no frame count: a 3-frame color_fg counts as complete, so force the rerun
        o.output = dir / "r2.mkv";
        o.disableUnavailable = true;
        if (isolated) o.fgExecutable = DLSSVID_CLI_PATH;
        const ProcessResult r = RunProcess(o);
        const ProcessStageReport& fg = Report(r, "fg");
        CHECK(fg.status == "failed");
        CHECK_THAT(fg.reason, Catch::Matchers::ContainsSubstring("has no frame 3"));
        CHECK(fg.fingerprint.empty());
        CHECK(fg.details["isolated"] == isolated);
        CHECK(r.finalPass == o.passesRoot / "color_nr");
        CHECK(r.framesOut == 3);
        CHECK(r.warnings.size() == 1);
        CHECK(Report(r, "encode").status == "ran");
        CHECK(!std::filesystem::exists(o.passesRoot / "color_fg"));    // the half-written pass of this run is removed,
        CHECK(!std::filesystem::exists(o.passesRoot / "color_fg.v"));  // not kept as a version
        o.disableUnavailable = false;
        o.output = dir / "r3.mkv";
        CHECK_THROWS_WITH(RunProcess(o), Catch::Matchers::ContainsSubstring("stage 'fg' failed"));
        CHECK(!std::filesystem::exists(o.output));
    }
}

// Review of TASK-0038 (fg-iso-4): fg disabled while a color_fg with its own fingerprint stays in place (a forced rerun is
// not moved to the history) - the result is the stage's input, not the pass left on disk.
TEST_CASE("RunProcess: fg disabled over a color_fg kept in place encodes color_nr, not that pass", "[integration][process][fg][regression]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("fg_disabled_in_place");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "r1.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    o.codec = "ffv1";
    o.warp = true;
    RunProcess(o);  // color_fg of 9 frames made by blend
    ApplyParamSpec(o.stages, "fg.backend=dlssg");  // unavailable on WARP
    o.disableUnavailable = true;
    // the blend pass stamped with the fingerprint the dlssg run needs: it stays in place under --force
    const std::string want = [&] {
        for (const auto& s : PlanProcess(o).stages)
            if (s.name == "fg") return s.fingerprint;
        return std::string();
    }();
    REQUIRE(!want.empty());
    Manifest m = Manifest::Load(o.passesRoot / "color_fg");
    m.fingerprint = want;
    m.Save(o.passesRoot / "color_fg");
    o.forceStages = {"fg"};
    o.output = dir / "r2.mkv";
    const ProcessResult r = RunProcess(o);
    CHECK(Report(r, "fg").status == "disabled");
    CHECK(std::filesystem::exists(o.passesRoot / "color_fg" / Manifest::kFileName));  // left alone
    CHECK(r.finalPass == o.passesRoot / "color_nr");  // before: the 9-frame color_fg of another backend was encoded
    CHECK(r.framesOut == 5);
}

TEST_CASE("RunProcess: fg disabled over an older color_fg encodes color_nr, not the stale pass", "[integration][process][fg]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("fg_disabled_stale");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "r1.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    o.codec = "ffv1";
    o.warp = true;
    o.frames = 3;
    RunProcess(o);  // color_fg of 3 real frames
    o.frames = -1;
    ApplyParamSpec(o.stages, "fg.backend=dlssg");  // unavailable on WARP
    o.disableUnavailable = true;
    o.output = dir / "r2.mkv";
    const ProcessResult r = RunProcess(o);
    CHECK(Report(r, "fg").status == "disabled");
    CHECK(r.finalPass == o.passesRoot / "color_nr");  // the input of the disabled stage, never a color_fg left from another run
    CHECK(r.framesOut == 3);                           // color_nr of the first run (the MKV has no frame count: reused)
}
