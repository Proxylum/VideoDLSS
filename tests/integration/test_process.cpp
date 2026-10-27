// The whole pipeline as one run (stage 8): RunProcess over the pass cache with the deterministic backends on WARP,
// pass reuse on a rerun, --frames, audio copy, nr disabled without a DLL, passthrough, parameter specs.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <functional>
#include <map>

#include "TestClips.h"
#include "passes/PassSequence.h"
#include "passes/formats/PngIO.h"
#include "pipeline/ControlMask.h"
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
    REQUIRE(s.size() == 6);  // PR-13: scenes first
    CHECK(s.front().name == "scenes");
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
    CHECK(ProcessStageOrder() == std::vector<std::string>{"scenes", "depth", "flow", "upscale", "nr", "fg"});
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
    REQUIRE(r.stages.size() == 7);
    for (const char* name : {"scenes", "depth", "flow", "upscale", "nr", "fg"}) CHECK(Report(r, name).status == "ran");
    CHECK(Report(r, "encode").status == "ran");
    CHECK(Report(r, "fg").frames == 5);
    CHECK(r.finalPass == dir / "passes" / "color_fg");
    CHECK(r.framesIn == 5);
    CHECK(r.framesOut == FgFrameCount(5, 2));
    CHECK(r.outputFps.ToDouble() == Approx(48.0));
    CHECK(r.audio);
    CHECK(seen == std::vector<std::string>{"scenes", "depth", "flow", "upscale", "nr", "fg", "encode"});
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
    CHECK(j["stages"].size() == 7);
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

// PR-7 (review): nr.intensity went to NGX before PR-7 - a pass with intensity != 1 is another picture now (0 was a black frame).
// Its fingerprint differs (+nr.gain / +nr.i0), so DecideStage reached the branch that keeps the current pass as a version: the
// panel's «Вернуть» could bring it back. Discarded now, with or without a fingerprint, and the color_fg made from it too.
TEST_CASE("RunProcess: an nr pass from before PR-7 with intensity != 1 is discarded, not kept as a version", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("old_intensity");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    ApplyParamSpec(o.stages, "nr.intensity=0.5");
    o.codec = "ffv1";
    o.warp = true;
    RunProcess(o);
    const auto nrDir = o.passesRoot / "color_nr";
    REQUIRE(Manifest::Load(nrDir).stageParams["model_intensity"] == 1.0);  // what this build writes
    REQUIRE(Manifest::Load(nrDir).tool.value("app", std::string()) == std::string(kPassToolVersion) + "+nr.gain");
    CHECK(Report(RunProcess(o), "nr").status == "reused");
    auto stampOld = [&](bool keepFingerprint) {
        Manifest m = Manifest::Load(nrDir);
        m.stageParams.erase("model_intensity");
        m.tool["app"] = std::string(kPassToolVersion);  // what the builds before stamped: no +nr.gain, so another fingerprint
        m.fingerprint = keepFingerprint ? "sha256:" + std::string(64, '0') : std::string();
        if (!keepFingerprint) m.tool = nlohmann::json::object();  // a standalone `dlssvid nr` pass, adopted as legacy
        m.Save(nrDir);
    };
    stampOld(true);
    const size_t nrVersions = ListPassVersions(o.passesRoot, "color_nr").size(), fgVersions = ListPassVersions(o.passesRoot, "color_fg").size();
    const ProcessResult r = RunProcess(o);
    CHECK(Report(r, "nr").status == "ran");
    CHECK_THAT(Report(r, "nr").reason, Catch::Matchers::ContainsSubstring("before PR-7"));
    CHECK(Report(r, "fg").status == "ran");
    CHECK(ListPassVersions(o.passesRoot, "color_nr").size() == nrVersions);  // discarded: before, it went to color_nr.v
    CHECK(ListPassVersions(o.passesRoot, "color_fg").size() == fgVersions);
    CHECK(Manifest::Load(nrDir).stageParams.contains("model_intensity"));
    CHECK(Report(RunProcess(o), "nr").status == "reused");
    stampOld(false);
    const size_t nrLegacy = ListPassVersions(o.passesRoot, "color_nr").size(), fgLegacy = ListPassVersions(o.passesRoot, "color_fg").size();
    const ProcessResult legacy = RunProcess(o);
    CHECK(Report(legacy, "nr").status == "ran");  // nor adopted without a fingerprint
    CHECK_THAT(Report(legacy, "nr").reason, Catch::Matchers::ContainsSubstring("before PR-7"));
    CHECK(ListPassVersions(o.passesRoot, "color_nr").size() == nrLegacy);  // discarded, not kept as a version
    CHECK(ListPassVersions(o.passesRoot, "color_fg").size() == fgLegacy);
    // the stage disabled after the discard (no DLL - the GUI runs --disable-unavailable): fg reads color_sr then, and the color_fg
    // made from the discarded pass went to the history; discarded too now
    stampOld(true);
    const size_t fgBefore = ListPassVersions(o.passesRoot, "color_fg").size();
    ProcessOptions ngx = o;
    ngx.stages = StubStages(2);
    ApplyParamSpec(ngx.stages, "nr.intensity=0.5");
    ApplyParamSpec(ngx.stages, "nr.backend=ngx");
    ngx.disableUnavailable = true;
    const ProcessResult off = RunProcess(ngx);
    CHECK(Report(off, "nr").status == "disabled");
    CHECK_FALSE(Manifest::Exists(nrDir));
    CHECK(Report(off, "fg").status == "ran");
    CHECK_THAT(Report(off, "fg").reason, Catch::Matchers::ContainsSubstring("discarded"));
    CHECK(ListPassVersions(o.passesRoot, "color_fg").size() == fgBefore);
    // a pass of before PR-7 at intensity 1 is the same picture: kept
    ProcessOptions one = o;
    one.stages = StubStages(2);
    RunProcess(one);
    Manifest m = Manifest::Load(nrDir);
    m.stageParams.erase("model_intensity");
    m.Save(nrDir);
    CHECK(Report(RunProcess(one), "nr").status == "reused");
}

// PR-5 (spec §2c): above 2 passes the model reworks its own output again - the plan notes it (the card's line); 1..2 and the bypass
// (intensity 0: no model) say nothing.
TEST_CASE("PlanProcess notes nr passes 3 and 4", "[integration][process]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("plan_passes");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.codec = "ffv1";
    o.warp = true;
    auto info = [&](std::initializer_list<const char*> specs) {
        ProcessOptions q = o;
        q.stages = StubStages(2);
        for (const char* s : specs) ApplyParamSpec(q.stages, s);
        for (const auto& d : PlanProcess(q).stages)
            if (d.name == "nr") return d.info;
        return std::string("no nr");
    };
    CHECK(info({"nr.passes=1"}).empty());
    CHECK(info({"nr.passes=2"}).empty());
    CHECK_THAT(info({"nr.passes=3"}), Catch::Matchers::ContainsSubstring("3 прохода"));
    CHECK_THAT(info({"nr.passes=4"}), Catch::Matchers::ContainsSubstring("синтетичнее"));
    CHECK(info({"nr.passes=4", "nr.intensity=0"}).empty());
}

// PR-13 (regression, replaces the PR-6 check): a pass without a fingerprint (a standalone command, a build before PR-13) is adopted
// only when it was made with the list the stage reads now - its label. Without one, with another one, or reset by NR's own detector of
// PR-6, it is another picture: recomputed, the old one kept in the history. With no cut in the list a pass of before PR-13 is the same run.
namespace {
std::filesystem::path MaskFile(const std::filesystem::path& file, uint32_t w, uint32_t h, uint16_t v) {
    PassImage img;
    img.Allocate(w, h, PixelType::U16, {"A"});
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) img.Set(x, y, 0, static_cast<float>(v));
    WritePng(file, img);
    return file;
}
// a copy: the plan is a temporary of the caller
StageDecision Decision(const ProcessPlan& p, const std::string& name) {
    for (const auto& d : p.stages)
        if (d.name == name) return d;
    throw std::runtime_error("no decision for " + name);
}
}  // namespace

// PR-8: the plan reads the control mask NR will read - other proportions are a warning (the mask stretches), an unreadable one stops NR
// before it runs, and a mask file edited in place stops it at reuse too (its stamp in the manifest is the old one).
TEST_CASE("the plan warns about a control mask of other proportions and stops NR on an unreadable one; a mask file edited in place stops NR at run and at reuse", "[integration][process][mask]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("plan_mask");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.codec = "ffv1";
    o.warp = true;
    o.stages = StubStages(2);
    for (auto& s : o.stages) s.enabled = s.name == "nr";
    ImportControlMask(MaskFile(dir / "square.png", 32, 32, 30000), o.passesRoot);  // 1:1 for a 3:2 frame
    const StageDecision d = Decision(PlanProcess(o), "nr");
    REQUIRE(d.warnings.size() == 1);
    CHECK_THAT(d.warnings[0], Catch::Matchers::ContainsSubstring("32×32 растянется на кадр 48×32"));
    CHECK_THAT(d.warnings[0], Catch::Matchers::ContainsSubstring("33 %"));
    ImportControlMask(MaskFile(dir / "near.png", 96, 65, 30000), o.passesRoot);  // 1.477 vs 1.5: 1.5 % off, still a warning
    CHECK(Decision(PlanProcess(o), "nr").warnings.size() == 1);
    ImportControlMask(MaskFile(dir / "same.png", 96, 64, 30000), o.passesRoot);  // the frame's proportions
    CHECK(Decision(PlanProcess(o), "nr").warnings.empty());
    CHECK(Decision(PlanProcess(o), "nr").error.empty());
    const ProcessResult first = RunProcess(o);
    CHECK(Report(first, "nr").status == "ran");

    // unstamped (unreadable as a control mask): an error while NR would run, a warning while it is reused
    const auto cur = o.passesRoot / kControlMaskPass;
    Manifest m = Manifest::Load(cur);
    const Manifest good = m;
    m.stageParams["static"] = false;
    m.Save(cur);
    const StageDecision un = Decision(PlanProcess(o), "nr");
    CHECK(un.action == "reuse");  // its fingerprint is the mask's manifest stamp: the same
    REQUIRE(un.warnings.size() == 1);
    CHECK_THAT(un.warnings[0], Catch::Matchers::ContainsSubstring("не читается (not static one-frame) и не применяется"));
    ProcessOptions forced = o;
    forced.forceStages = {"nr"};
    CHECK_THAT(Decision(PlanProcess(forced), "nr").error, Catch::Matchers::ContainsSubstring("маска управления не читается (not static one-frame)"));
    CHECK_THROWS_WITH(RunProcess(forced), Catch::Matchers::ContainsSubstring("маска управления не читается"));
    good.Save(cur);

    // the frame file edited in place: an error at reuse as well - the plan would reuse NR made with the old pixels
    MaskFile(cur / good.FrameFileName(0), 96, 64, 60000);
    const StageDecision edited = Decision(PlanProcess(o), "nr");
    CHECK(edited.action == "reuse");
    CHECK_THAT(edited.error, Catch::Matchers::ContainsSubstring("изменён вне программы"));
    CHECK_THROWS_WITH(RunProcess(o), Catch::Matchers::ContainsSubstring("изменён вне программы"));
    CHECK_THROWS_WITH(RunProcess(forced), Catch::Matchers::ContainsSubstring("изменён вне программы"));
}

// Review of PR-8: C0 changed the output of NR with protect and skin masks of different sizes (the UI was unprotected before) - such passes
// carry +nr.msize and the old ones are recomputed; masks of one size keep the fingerprint of before.
TEST_CASE("RunProcess: an nr pass with protect and skin masks of different sizes carries the revision of their fix; one size keeps its fingerprint", "[integration][process][mask][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("mask_sizes_revision");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    auto mask = [&](const char* name, uint32_t w, uint32_t h) {
        PassWriter wr(dir / "passes" / name, Manifest::ForPass(PassKind::Mask, w, h, FileFormat::Png));
        for (int64_t f = 0; f < 2; ++f) wr.WriteFrame(f, MakePassImage(PassKind::Mask, w, h, PixelType::U8));
        wr.Finish();
    };
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.codec = "ffv1";
    o.warp = true;
    o.stages = StubStages(2);
    for (auto& s : o.stages) s.enabled = s.name == "nr";
    mask("mask_ui", 48, 32);
    mask("mask_skin", 48, 32);
    REQUIRE(Report(RunProcess(o), "nr").status == "ran");
    CHECK(Manifest::Load(o.passesRoot / "color_nr").tool.value("app", std::string()).find("nr.msize") == std::string::npos);  // one size: as before
    std::filesystem::remove_all(o.passesRoot / "mask_skin");
    mask("mask_skin", 24, 16);
    const ProcessResult r = RunProcess(o);
    CHECK(Report(r, "nr").status == "ran");
    CHECK(Manifest::Load(o.passesRoot / "color_nr").tool.value("app", std::string()).find("+nr.msize") != std::string::npos);
    CHECK(Report(RunProcess(o), "nr").status == "reused");  // the revision is stable
}

// PR-8 (regression of the race the design closes): a mask replaced from the CLI while process runs. Between the plan and NR's Init the
// pass would say it was made with the new mask under the old fingerprint: the stage fails after the run, before the stamp. Replaced after
// the first NR frame (Init and Setup done) it changes nothing in this run: the mask is on the GPU, the pass gets the label of the one applied.
TEST_CASE("a control mask replaced after the plan fails the stage instead of stamping a wrong fingerprint; replaced after the first nr frame it changes nothing in this run", "[integration][process][mask][regression]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("mask_race");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.codec = "ffv1";
    o.warp = true;
    o.stages = StubStages(2);
    for (auto& s : o.stages) s.enabled = s.name == "nr";
    const ControlMaskImport a = ImportControlMask(MaskFile(dir / "a.png", 48, 32, 10000), o.passesRoot);
    MaskFile(dir / "b.png", 48, 32, 50000);
    MaskFile(dir / "c.png", 48, 32, 30000);
    ProcessOptions race = o;
    race.beforeStage = [&](const std::string& stage) {
        if (stage == "nr") ImportControlMask(dir / "b.png", o.passesRoot);
    };
    CHECK_THROWS_WITH(RunProcess(race), Catch::Matchers::ContainsSubstring("the control mask changed while nr ran"));
    REQUIRE(Manifest::Exists(o.passesRoot / "color_nr"));
    CHECK(Manifest::Load(o.passesRoot / "color_nr").fingerprint.empty());  // not stamped
    // the next run: the unstamped pass was made with the mask now there (b), complete - adopted under b, not under a
    const ProcessResult next = RunProcess(o);
    CHECK(Report(next, "nr").status == "reused");
    CHECK(Report(next, "nr").reason.find("adopted") != std::string::npos);
    const std::string bFp = ReadControlMaskInfo(o.passesRoot / kControlMaskPass)->fingerprint;
    CHECK(Manifest::Load(o.passesRoot / "color_nr").inputs.at("mask_control") == bFp);
    CHECK(bFp != a.mask.fingerprint);

    // after the first frame of nr: the run keeps the mask it uploaded and names it
    ProcessOptions late = o;
    late.forceStages = {"nr"};
    bool replaced = false;
    late.progress = [&](const std::string& stage, int64_t done, int64_t) {
        if (stage == "nr" && done >= 1 && !replaced) {
            replaced = true;
            ImportControlMask(dir / "c.png", o.passesRoot);
        }
    };
    const ProcessResult lr = RunProcess(late);
    CHECK(replaced);
    CHECK(Report(lr, "nr").status == "ran");
    const Manifest nr = Manifest::Load(o.passesRoot / "color_nr");
    CHECK(nr.inputs.at("mask_control") == bFp);  // planned and applied: b
    CHECK(ControlMaskLabelOf(nr) == bFp);
}

TEST_CASE("RunProcess adopts a pass without a fingerprint only when it was made with the current scene list", "[integration][process][regression]") {
    const uint32_t w = 48, h = 32;
    const int cutAt = 2;
    ClipSpec spec;
    spec.frames = 4;
    spec.width = w;
    spec.height = h;
    const auto dir = Dir("scene_label");
    const auto clip = WriteClipWith(dir / "clip.mkv", spec, [&](int64_t i) {  // a hard cut at frame 2
        CpuFrame f = SyntheticFrame(i, w, h);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c) y[r * w + c] = static_cast<uint8_t>(i < cutAt ? 40 + c + 4 * i : 220 - c);
        return f;
    });
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    o.codec = "ffv1";
    o.warp = true;
    RunProcess(o);
    const auto nrDir = o.passesRoot / "color_nr";
    const std::string id = Manifest::Load(nrDir).stageParams.value("scene_cuts_id", std::string());
    REQUIRE(id.rfind("cuts:", 0) == 0);  // what this build writes: the list's label
    CHECK(Manifest::Load(nrDir).inputs.at("scenes") == id);
    CHECK(Manifest::Load(nrDir).stageParams["scene_cuts"] == nlohmann::json::array({cutAt}));
    const Manifest made = Manifest::Load(nrDir);
    auto standalone = [&](const std::function<void(nlohmann::json&)>& label) {  // the pass as a standalone `dlssvid nr` leaves it
        Manifest m = made;
        m.fingerprint.clear();
        m.tool = nlohmann::json::object();
        m.inputs.clear();
        label(m.stageParams);
        m.Save(nrDir);
    };
    auto nrOf = [](const ProcessResult& r) -> const ProcessStageReport& {
        for (const auto& s : r.stages)
            if (s.name == "nr") return s;
        throw std::runtime_error("no nr");
    };
    standalone([](nlohmann::json&) {});
    CHECK(nrOf(RunProcess(o)).status == "reused");  // the same list: adopted
    standalone([](nlohmann::json& p) {
        for (const char* k : {"scene_cuts_id", "scene_cut_threshold", "scene_cuts", "scene_cut_count"}) p.erase(k);
    });
    const size_t before = ListPassVersions(o.passesRoot, "color_nr").size();
    ProcessResult r = RunProcess(o);
    CHECK(nrOf(r).status == "ran");
    CHECK_THAT(nrOf(r).reason, Catch::Matchers::ContainsSubstring("made without the scene list"));
    CHECK(!nrOf(r).retired.empty());  // a real picture of the run without resets: kept
    CHECK(ListPassVersions(o.passesRoot, "color_nr").size() == before + 1);
    standalone([](nlohmann::json& p) { p.erase("scene_cuts_id"); });  // reset by its own detector (PR-6): scene_cut_count 1, no label
    r = RunProcess(o);
    CHECK(nrOf(r).status == "ran");
    CHECK_THAT(nrOf(r).reason, Catch::Matchers::ContainsSubstring("PR-6"));
    // a rederive of scenes in the same plan with the same list (0.24 -> 0.3): a standalone pass with the label is still adopted - a
    // rewritten list is not a changed input
    standalone([](nlohmann::json&) {});
    ApplyParamSpec(o.stages, "scenes.threshold=0.3");
    const ProcessPlan plan = PlanProcess(o);
    std::string planned;
    for (const auto& d : plan.stages) {
        if (d.name == "scenes") CHECK(d.sceneCuts.rederive);
        if (d.name == "nr") {
            CHECK(d.action == "reuse");
            planned = d.fingerprint;
        }
    }
    r = RunProcess(o);
    CHECK(nrOf(r).status == "reused");
    CHECK(Manifest::Load(nrDir).fingerprint == planned);
    // no cut in the list (0.6): a pass without a label is the same run - adopted
    ApplyParamSpec(o.stages, "scenes.threshold=0.6");
    RunProcess(o);
    standalone([](nlohmann::json& p) {
        for (const char* k : {"scene_cuts_id", "scene_cut_threshold", "scene_cuts", "scene_cut_count"}) p.erase(k);
    });
    CHECK(nrOf(RunProcess(o)).status == "reused");
    CHECK_FALSE(Manifest::Load(nrDir).inputs.count("scenes"));
}

// PR-13 (regression): nr.scene_cut moved to scenes.threshold - a run with it is refused before anything runs, with where it went
TEST_CASE("RunProcess refuses nr.scene_cut and names scenes.threshold", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("moved_key");
    ProcessOptions o;
    o.input = WriteClip(dir / "clip.mkv", spec);
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    o.codec = "ffv1";
    o.warp = true;
    ApplyParamSpec(o.stages, "nr.scene_cut=0.3");
    CHECK_THROWS_WITH(PlanProcess(o), Catch::Matchers::ContainsSubstring("nr.scene_cut moved to scenes.threshold"));
    CHECK_THROWS_WITH(RunProcess(o), Catch::Matchers::ContainsSubstring("scenes.threshold"));
    CHECK_FALSE(std::filesystem::exists(o.passesRoot / "color_nr"));
}

// TASK-0047: with the source's frame count unknown (mkv) PassComplete looked at the last frame only, so a color_nr of a part
// of the clip (dlssvid nr --start, no fingerprint) was adopted as the clip's pass. A pass of the clip starts at frame 0.
TEST_CASE("RunProcess: a color_nr of a part of the clip (nr --start) is recomputed, not adopted", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("part_of_clip");
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
    {  // what a standalone ranged run leaves: frames 2..3, no fingerprint
        Manifest m = Manifest::Load(nrDir);
        m.fingerprint.clear();
        m.tool = nlohmann::json::object();
        m.firstFrame = 2;
        m.frameCount = 2;
        m.stageParams["range"] = {{"first", 2}, {"last", 3}, {"warmup", 0}};
        for (int64_t i : {0, 1}) std::filesystem::remove(nrDir / m.FrameFileName(i));
        m.Save(nrDir);
    }
    CHECK_FALSE(PassComplete(nrDir, -1, Manifest::Load(nrDir).sourceHash));  // before: true, the last frame was there
    const ProcessResult r = RunProcess(o);
    CHECK(Report(r, "nr").status == "ran");
    CHECK(Report(r, "nr").reason.find("part of the clip") != std::string::npos);
    CHECK(Manifest::Load(nrDir).firstFrame == 0);
    CHECK(Report(r, "fg").status == "reused");  // the pass fg was made from is back, under the same fingerprint
}

// Review of TASK-0047: the frame-0 rule was in PassComplete only - with nr off in the project, fg took a color_nr of a part of
// the clip as its colour («has no frame 0») and the encode put frame F at t=0. Such a pass is no input and not the result.
TEST_CASE("RunProcess: a part of the clip under the passes root is neither fg's input nor the result", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("part_not_input");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    for (auto& s : o.stages)
        if (s.name == "nr") s.enabled = false;
    o.codec = "ffv1";
    o.warp = true;
    const ProcessResult first = RunProcess(o);
    {  // what `dlssvid nr --start 2` leaves under the root: frames 2..3
        const Manifest sr = Manifest::Load(o.passesRoot / "color_sr");
        PassWriter w(o.passesRoot / "color_nr", Manifest::ForPass(PassKind::ColorNr, sr.width, sr.height, FileFormat::Exr));
        for (int64_t i : {2, 3}) w.WriteFrame(i, MakePassImage(PassKind::ColorNr, sr.width, sr.height));
        w.Finish();
    }
    ProcessResult r = RunProcess(o);
    CHECK(Report(r, "fg").status == "reused");  // its colour is color_sr, as before
    CHECK(r.framesOut == first.framesOut);
    for (auto& s : o.stages)
        if (s.name == "fg") s.enabled = false;
    std::filesystem::remove_all(o.passesRoot / "color_fg");  // a disabled stage's pass would still be the result
    r = RunProcess(o);
    CHECK(r.finalPass.filename() == "color_sr");  // the result is not the part of the clip (before: color_nr, 2 frames)
    CHECK(r.framesOut == 4);
}

// TASK-0048 (spec §2e): the A/B switches are not parameters - an nr pass a standalone run made with them (stage_params.ab) is
// never reused, restored or adopted as the default run, with or without a fingerprint; fg built on it reruns.
TEST_CASE("RunProcess: an nr pass made with the A/B switches of a standalone run is not adopted, reused or restored", "[integration][process][nr]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("ab_pass");
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
    auto markAb = [&](bool keepFingerprint) {
        Manifest m = Manifest::Load(nrDir);
        m.stageParams["ab"] = {{"mv_scale", "auto"}};
        if (!keepFingerprint) {
            m.fingerprint.clear();
            m.tool = nlohmann::json::object();
        }
        m.Save(nrDir);
    };
    markAb(true);
    ProcessResult r = RunProcess(o);
    CHECK(Report(r, "nr").status == "ran");
    CHECK(Report(r, "nr").reason.find("A/B") != std::string::npos);
    CHECK(Report(r, "fg").status == "reused");  // a real result of an experiment, not a broken pass: nothing below is stale
    CHECK(Report(RunProcess(o), "nr").status == "reused");  // the rerun is the default run
    markAb(false);  // what a standalone run leaves: no fingerprint
    const size_t versions = ListPassVersions(o.passesRoot, "color_nr").size();
    r = RunProcess(o);
    CHECK(Report(r, "nr").status == "ran");  // not adopted
    CHECK(Report(r, "fg").status == "reused");
    // kept in the history to compare (before: deleted, and fg deleted and recomputed with it)
    const auto history = ListPassVersions(o.passesRoot, "color_nr");
    CHECK(history.size() == versions + 1);
    bool abKept = false;
    for (const auto& v : history) abKept = abKept || (!v.current && Manifest::Load(v.dir).stageParams.contains("ab"));
    CHECK(abKept);
}

// Review of TASK-0048: an A/B color_nr under the passes root was fg's colour and the result when nr is off in the project
TEST_CASE("RunProcess: an A/B pass under the passes root is neither fg's input nor the result", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("ab_not_input");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);
    for (auto& s : o.stages)
        if (s.name == "nr") s.enabled = false;
    o.codec = "ffv1";
    o.warp = true;
    const ProcessResult first = RunProcess(o);
    {  // what `dlssvid nr -o <passes> --mv-scale auto` leaves: the whole clip, stage_params.ab
        const Manifest sr = Manifest::Load(o.passesRoot / "color_sr");
        Manifest m = Manifest::ForPass(PassKind::ColorNr, sr.width, sr.height, FileFormat::Exr);
        m.stageParams["ab"] = {{"mv_scale", "auto"}};
        PassWriter w(o.passesRoot / "color_nr", m);
        for (int64_t i = 0; i < 4; ++i) w.WriteFrame(i, MakePassImage(PassKind::ColorNr, sr.width, sr.height));
        w.Finish();
    }
    ProcessResult r = RunProcess(o);
    CHECK(Report(r, "fg").status == "reused");  // its colour is color_sr, as before
    CHECK(r.framesOut == first.framesOut);
    for (auto& s : o.stages)
        if (s.name == "fg") s.enabled = false;
    std::filesystem::remove_all(o.passesRoot / "color_fg");
    r = RunProcess(o);
    CHECK(r.finalPass.filename() == "color_sr");  // the result is not the A/B pass
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
    // a black pass the builds before PR-7 put into the history (under another fingerprint): not restored for its own either
    stampOld(0.0, true);
    REQUIRE(RetirePassVersion(o.passesRoot, "color_nr"));
    CHECK(Report(RunProcess(o), "nr").status == "ran");  // no current pass: restore() would bring the black version back, «reused»
    // under another fingerprint the black pass is discarded now, not kept as a version (review of PR-7)
    stampOld(0.0, true);
    const size_t versions = ListPassVersions(o.passesRoot, "color_nr").size();
    ProcessOptions again = o;
    again.stages = StubStages(2);
    ApplyParamSpec(again.stages, "nr.intensity=1.5");
    CHECK(Report(RunProcess(again), "nr").status == "ran");
    CHECK(ListPassVersions(o.passesRoot, "color_nr").size() == versions);
    CHECK(Report(RunProcess(o), "nr").status == "ran");  // back to intensity 1: the black version in the history is skipped
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

// PR-13 (regression): upscale reads depth_dlss / mv_dlss with DLSS alone. With nis a new depth and new vectors keep color_sr; nr and
// fg, which read them, are recomputed. Before, every upscale backend listed them as inputs and was recomputed for nothing.
TEST_CASE("RunProcess: an upscale without guides is not recomputed by a new depth and new vectors", "[integration][process][regression]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("upscale_inputs");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages(2);  // upscale nis
    o.codec = "ffv1";
    o.warp = true;
    RunProcess(o);
    CHECK(Manifest::Load(o.passesRoot / "color_sr").inputs.empty());
    CHECK(Manifest::Load(o.passesRoot / "color_nr").inputs.count("mv_dlss") == 1);
    const std::string sr = Manifest::Load(o.passesRoot / "color_sr").fingerprint;

    ApplyParamSpec(o.stages, "depth.z_far=500");  // a new depth, so new vectors (flow reads depth_raw)
    const ProcessPlan plan = PlanProcess(o);
    std::map<std::string, std::string> actions;
    for (const auto& d : plan.stages) actions[d.name] = d.action;
    CHECK(actions["depth"] == "run");
    CHECK(actions["flow"] == "run");
    CHECK(actions["upscale"] == "reuse");
    CHECK(actions["nr"] == "run");
    CHECK(actions["fg"] == "run");
    const ProcessResult r = RunProcess(o);
    CHECK(Report(r, "upscale").status == "reused");
    CHECK(Report(r, "nr").status == "ran");
    CHECK(Report(r, "fg").status == "ran");
    CHECK(Manifest::Load(o.passesRoot / "color_sr").fingerprint == sr);

    // DLSS reads them: in the plan (no DLL on WARP: the plan alone)
    ApplyParamSpec(o.stages, "upscale.backend=dlss");
    for (const auto& d : PlanProcess(o).stages)
        if (d.name == "upscale") {
            CHECK(d.inputs.count("depth_dlss") == 1);
            CHECK(d.inputs.count("mv_dlss") == 1);
        }
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
        REQUIRE(r.warnings.size() == 2);  // the failure, and the list of the first run covers frames 0..2 only (PR-13)
        CHECK(std::any_of(r.warnings.begin(), r.warnings.end(), [](const std::string& w) { return w.find("known for frames 0..2 only") != std::string::npos; }));
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
