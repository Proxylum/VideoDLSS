// PR-13 (spec §2d «Общий список склеек»): the stage scenes - one analysis of the source's luma, the list and every frame's score
// in a data pass; a new threshold over the stored scores without decoding; threshold 0 is off. In RunProcess it runs first; the
// stages below do not read the list yet, so their passes do not change with it.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "convert/SceneCut.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/ScenesPass.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ProcessRunner.h"
#include "pipeline/ScenesPlan.h"
#include "stages/nr/NrStage.h"
#include "stages/scenes/ScenesStage.h"
#include "util/Sha256.h"
#include "util/Subprocess.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "scenes" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

constexpr uint32_t kW = 64, kH = 32;
constexpr int64_t kCutAt = 3;

// scene A brightens 4 levels a frame (score ≈ 0.02), a hard cut at frame 3 to scene B, the gradient inverted (≈ 0.5): as in test_nr_stage
std::filesystem::path CutClip(const std::filesystem::path& dir, int frames = 6) {
    ClipSpec spec;
    spec.frames = frames;
    spec.width = kW;
    spec.height = kH;
    return WriteClipWith(dir / "clip.mkv", spec, [](int64_t i) {
        CpuFrame f = SyntheticFrame(i, kW, kH);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < kH; ++r)
            for (uint32_t c = 0; c < kW; ++c) y[r * kW + c] = static_cast<uint8_t>(i < kCutAt ? 40 + c + 4 * i : 220 - c);
        return f;
    });
}

// The clips of the equivalence with PR-6: its cut at 3 in 8 and in 10 bit (the low bits a static ramp), and a flash - a bright
// frame 3 over a slow scene, two scores above the threshold
std::filesystem::path EquivalenceClip(const std::filesystem::path& dir, const std::string& kind) {
    ClipSpec spec;
    spec.frames = 7;
    spec.width = kW;
    spec.height = kH;
    if (kind == "10bit") spec.pixFmt = "yuv420p10le";
    const FrameDesc desc = FrameDescFor(spec);
    return WriteClipWith(dir / (kind + ".mkv"), spec, [&](int64_t i) {
        CpuFrame f = SyntheticFrame(i, desc);
        for (uint32_t r = 0; r < kH; ++r)
            for (uint32_t c = 0; c < kW; ++c) {
                const int y = kind == "flash" ? (i == 3 ? 235 : 40 + static_cast<int>(c) + 2 * static_cast<int>(i))
                                              : (i < kCutAt ? 40 + static_cast<int>(c) + 4 * static_cast<int>(i) : 220 - static_cast<int>(c));
                if (desc.HighBitDepth()) f.Plane16(0)[r * kW + c] = static_cast<uint16_t>((y << 2) | ((r + c) & 3));
                else f.Plane(0)[r * kW + c] = static_cast<uint8_t>(y);
            }
        return f;
    });
}

ScenesStageOptions Options(const std::filesystem::path& root, const std::filesystem::path& clip, double threshold = 0.24) {
    VideoDecoder probe(clip);
    ScenesStageOptions o;
    o.threshold = threshold;
    o.outputDir = root;
    o.sourceFile = clip.filename().string();
    o.sourceHash = "sha256:" + Sha256File(clip);
    o.sourceFormat = probe.Info().decoded.Name();
    return o;
}

ProcessOptions StubProcess(const std::filesystem::path& dir, const std::filesystem::path& clip) {
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = DefaultProcessStages();
    for (const char* spec : {"depth.backend=stub", "flow.backend=stub", "upscale.backend=nis", "upscale.scale=2", "nr.backend=stub", "fg.backend=blend", "fg.multiplier=2"})
        ApplyParamSpec(o.stages, spec);
    o.codec = "ffv1";
    o.warp = true;
    return o;
}

const StageDecision& Decision(const ProcessPlan& p, const std::string& name) {
    for (const auto& s : p.stages)
        if (s.name == name) return s;
    throw std::runtime_error("no decision for " + name);
}

const ProcessStageReport& Report(const ProcessResult& r, const std::string& name) {
    for (const auto& s : r.stages)
        if (s.name == name) return s;
    throw std::runtime_error("no report for " + name);
}

struct CliResult {
    uint32_t code = 0;
    std::string out;
};

CliResult RunCli(std::vector<std::string> args) {
    args.insert(args.begin(), DLSSVID_CLI_PATH);
    CliResult r;
    r.code = Subprocess::Run(args, &r.out);
    return r;
}

nlohmann::json ReadJson(const std::filesystem::path& p) {
    nlohmann::json j;
    std::ifstream(p) >> j;
    return j;
}

}  // namespace

TEST_CASE("AnalyseScenes finds the hard cut and stores every frame's score beside the list", "[integration][scenes]") {
    const auto dir = Dir("analyse");
    const auto clip = CutClip(dir);
    const auto root = dir / "passes";
    const ScenesStageOptions o = Options(root, clip);
    VideoDecoder decoder(clip);
    std::vector<int64_t> calls;
    const ScenesRunResult r = AnalyseScenes(decoder, o, -1, [&](int64_t n) { calls.push_back(n); });
    CHECK(calls == std::vector<int64_t>{1, 2, 3, 4, 5, 6});
    CHECK(r.list.cuts == std::vector<int64_t>{kCutAt});
    CHECK(r.list.id == SceneCutsId({kCutAt}));
    CHECK(r.list.frames == 6);
    CHECK(r.list.reachedEnd);
    CHECK(!r.list.off);
    CHECK(!r.rederived);
    CHECK(r.msPerFrame > 0.0);
    CHECK(r.outDir == root / kScenesPass);

    const Manifest m = Manifest::Load(r.outDir);
    CHECK(m.IsDataPass());
    CHECK(m.frameCount == 6);
    CHECK(m.firstFrame == 0);
    CHECK(m.width == kW);
    CHECK(m.height == kH);
    CHECK(m.sourceHash == o.sourceHash);
    CHECK(m.stageParams["threshold"] == 0.24);
    CHECK(m.stageParams["off"] == false);
    CHECK(m.stageParams["scene_cuts"] == nlohmann::json::array({kCutAt}));
    CHECK(m.stageParams["scene_cut_count"] == 1);
    CHECK(m.stageParams["cuts_id"] == r.list.id);
    CHECK(!m.stageParams.contains("rederived"));
    const nlohmann::json& a = m.stageParams["analysis"];
    CHECK(a["algorithm"] == kScenesAlgorithm);
    CHECK(a["max_width"] == 640);
    CHECK(a["reached_end"] == true);
    CHECK(a["frames"] == 6);
    CHECK(a["source_format"] == o.sourceFormat);
    CHECK(a["scores_file"] == kScenesScoresFile);
    CHECK(a["scores_count"] == 6);

    const auto scores = ReadSceneScores(r.outDir, m);
    REQUIRE(scores);
    REQUIRE(scores->scores.size() == 6);
    CHECK(scores->scores[0] == 0.f);  // frame 0 has no frame before it
    CHECK(scores->scores[1] < 0.05f);
    CHECK(scores->scores[2] < 0.05f);
    CHECK(scores->scores[3] > 0.24f);
    CHECK(scores->scores[4] < 0.05f);
    const auto l = ReadSceneList(r.outDir, o.sourceHash);
    REQUIRE(l);
    CHECK(l->cuts == r.list.cuts);
    CHECK(l->reachedEnd);

    CHECK(PassComplete(r.outDir, 6, o.sourceHash));
    CHECK(!PassComplete(r.outDir, 7, o.sourceHash));
    CHECK(PassComplete(r.outDir, 0, o.sourceHash));  // the length unknown: the analysis reached the end of the source
    CHECK(!PassComplete(r.outDir, 6, "sha256:other"));
    CHECK(ScoresUsable(m, o.sourceHash, o.sourceFormat, 6));
    CHECK(ScoresUsable(m, o.sourceHash, o.sourceFormat, 0));
    CHECK(!ScoresUsable(m, o.sourceHash, o.sourceFormat, 7));
    CHECK(!ScoresUsable(m, o.sourceHash, "yuv444p", 6));  // another decoded layout: other scores
    CHECK(!ScoresUsable(m, "sha256:other", o.sourceFormat, 6));
    Manifest otherAlgorithm = m;
    otherAlgorithm.stageParams["analysis"]["algorithm"] = "luma-mad.320";
    CHECK(!ScoresUsable(otherAlgorithm, o.sourceHash, o.sourceFormat, 6));
    const auto found = FindSceneScores(root, o.sourceHash, o.sourceFormat, 6);
    REQUIRE(found);
    CHECK(found->version.empty());
    CHECK(found->scores.scores == scores->scores);

    // the synthetic clip of the other tests has no cut at the default threshold: their passes and the golden references stay as they were
    ClipSpec plain;
    plain.frames = 6;
    plain.width = 96;
    plain.height = 54;
    const auto plainClip = WriteClip(dir / "plain.mkv", plain);
    VideoDecoder plainDecoder(plainClip);
    const ScenesRunResult p = AnalyseScenes(plainDecoder, Options(dir / "plain", plainClip), -1);
    CHECK(p.list.cuts.empty());
    CHECK(p.list.id.empty());
}

TEST_CASE("AnalyseScenes over the first frames only: the list is incomplete for the whole clip", "[integration][scenes]") {
    const auto dir = Dir("frames");
    const auto clip = CutClip(dir);
    const auto root = dir / "passes";
    const ScenesStageOptions o = Options(root, clip);
    VideoDecoder decoder(clip);
    const ScenesRunResult r = AnalyseScenes(decoder, o, 2);
    CHECK(r.list.frames == 2);
    CHECK(!r.list.reachedEnd);
    CHECK(r.list.cuts.empty());  // the cut lies beyond the frames analysed
    const Manifest m = Manifest::Load(r.outDir);
    CHECK(m.frameCount == 2);
    CHECK(m.lastFrame == 1);
    CHECK(m.stageParams["analysis"]["reached_end"] == false);
    CHECK(PassComplete(r.outDir, 2, o.sourceHash));
    CHECK(!PassComplete(r.outDir, 6, o.sourceHash));
    CHECK(!PassComplete(r.outDir, 0, o.sourceHash));  // the length unknown and the end not reached
    CHECK(FindSceneScores(root, o.sourceHash, o.sourceFormat, 2));
    CHECK(!FindSceneScores(root, o.sourceHash, o.sourceFormat, 6));
    CHECK(!FindSceneScores(root, o.sourceHash, o.sourceFormat, 0));
}

TEST_CASE("RederiveScenes applies another threshold to the stored scores without the source; threshold 0 is off", "[integration][scenes]") {
    const auto dir = Dir("rederive");
    const auto clip = CutClip(dir);
    const auto root = dir / "passes";
    const ScenesStageOptions o = Options(root, clip);
    {
        VideoDecoder decoder(clip);
        AnalyseScenes(decoder, o, -1);
    }
    auto src = FindSceneScores(root, o.sourceHash, o.sourceFormat, 6);
    REQUIRE(src);
    std::filesystem::remove(clip);  // no decoder can open it now

    ScenesStageOptions o6 = o;
    o6.threshold = 0.6;
    const ScenesRunResult r6 = RederiveScenes(src->manifest, src->scores, o6);
    CHECK(r6.rederived);
    CHECK(r6.list.cuts.empty());
    CHECK(r6.list.id.empty());
    CHECK(r6.list.frames == 6);
    CHECK(r6.list.reachedEnd);
    CHECK(r6.msPerFrame == src->scores.analysis["ms_per_frame"].get<double>());  // the decoding the scores cost, for the estimates
    REQUIRE(!r6.list.nearMisses.empty());
    CHECK(r6.list.nearMisses.front().first == kCutAt);  // the cut is just below the new threshold: shown to check for a missed one
    const Manifest m6 = Manifest::Load(r6.outDir);
    CHECK(m6.stageParams["rederived"] == true);
    CHECK(m6.stageParams["threshold"] == 0.6);
    CHECK(m6.stageParams["scene_cut_count"] == 0);
    CHECK(m6.stageParams["analysis"]["ms_per_frame"] == src->scores.analysis["ms_per_frame"]);
    CHECK(m6.width == kW);
    const auto kept = ReadSceneScores(r6.outDir, m6);
    REQUIRE(kept);
    CHECK(kept->scores == src->scores.scores);  // the scores travel with every list made of them

    // the way back to 0.24 over the same scores: the list of the analysis, bit for bit
    src = FindSceneScores(root, o.sourceHash, o.sourceFormat, 6);
    REQUIRE(src);
    const ScenesRunResult back = RederiveScenes(src->manifest, src->scores, o);
    CHECK(back.list.cuts == std::vector<int64_t>{kCutAt});
    CHECK(back.list.id == SceneCutsId({kCutAt}));

    // threshold 0 without scores: an «off» pass, no analysis
    ScenesStageOptions off = o;
    off.threshold = 0.0;
    off.outputDir = dir / "off";
    const ScenesRunResult r0 = WriteOffScenes(off, kW, kH, Rational{24, 1}, 6);
    CHECK(r0.list.off);
    CHECK(r0.list.cuts.empty());
    CHECK(!std::filesystem::exists(r0.outDir / kScenesScoresFile));
    const auto l0 = ReadSceneList(r0.outDir, o.sourceHash);
    REQUIRE(l0);
    CHECK(l0->off);
    CHECK(l0->frames == 6);
    CHECK(l0->id.empty());
    CHECK(PassComplete(r0.outDir, 100, o.sourceHash));  // off stands for any length
    CHECK(!ReadSceneScores(r0.outDir, Manifest::Load(r0.outDir)));
    CHECK(!FindSceneScores(dir / "off", o.sourceHash, o.sourceFormat, 6));
}

TEST_CASE("RunProcess: scenes runs first; a new threshold goes over the stored scores, 0 is off, the way back restores", "[integration][process][scenes]") {
    const auto dir = Dir("process");
    const auto clip = CutClip(dir);
    ProcessOptions o = StubProcess(dir, clip);
    o.keepVersions = 0;  // every list stays in the history: the way back restores one

    const ProcessPlan plan0 = PlanProcess(o);
    REQUIRE(plan0.stages.front().name == "scenes");
    const StageDecision& s0 = plan0.stages.front();
    CHECK(s0.action == "run");
    CHECK(s0.kind == "missing");
    CHECK(!s0.sceneCuts.known);
    CHECK(s0.sceneCuts.id == kScenesPendingId);
    CHECK(s0.info == "список склеек посчитается при обработке");
    CHECK(plan0.ToJson()["stages"][0]["scene_cuts"]["known"] == false);
    CHECK(plan0.runCount == 6);

    std::vector<std::string> seen;
    o.progress = [&](const std::string& stage, int64_t, int64_t) {
        if (seen.empty() || seen.back() != stage) seen.push_back(stage);
    };
    const ProcessResult r0 = RunProcess(o);
    o.progress = {};
    CHECK(!seen.empty());
    CHECK(seen.front() == "scenes");
    CHECK(Report(r0, "scenes").status == "ran");
    CHECK(Report(r0, "scenes").fingerprint == s0.fingerprint);
    const auto list0 = ReadSceneList(o.passesRoot / kScenesPass, "");
    REQUIRE(list0);
    CHECK(list0->cuts == std::vector<int64_t>{kCutAt});
    CHECK(Manifest::Load(o.passesRoot / kScenesPass).fingerprint == s0.fingerprint);
    CHECK(Manifest::Load(o.passesRoot / "color_nr").inputs.at("scenes") == list0->id);  // nr reads the list (PR-13)
    CHECK(SceneLabelOf(Manifest::Load(o.passesRoot / "color_nr")) == list0->id);
    for (const auto& e : std::filesystem::recursive_directory_iterator(o.passesRoot)) {  // a planned id never reaches a manifest
        if (e.path().filename() != Manifest::kFileName) continue;
        std::ifstream f(e.path());
        std::stringstream text;
        text << f.rdbuf();
        CHECK(text.str().find("pending:") == std::string::npos);
    }

    // a rerun: the list is known in the plan
    const ProcessPlan plan1 = PlanProcess(o);
    const StageDecision& s1 = Decision(plan1, "scenes");
    CHECK(s1.action == "reuse");
    CHECK(s1.sceneCuts.known);
    CHECK(s1.sceneCuts.count == 1);
    CHECK(s1.sceneCuts.cuts == std::vector<int64_t>{kCutAt});
    CHECK(s1.sceneCuts.id == list0->id);
    CHECK(s1.info == "склеек: 1");
    CHECK(plan1.runCount == 0);

    // 0.24 -> 0.3: the same list, over the stored scores
    ApplyParamSpec(o.stages, "scenes.threshold=0.3");
    const ProcessPlan plan2 = PlanProcess(o);
    const StageDecision& s2 = Decision(plan2, "scenes");
    CHECK(s2.action == "run");
    CHECK(s2.kind == "params_changed");
    CHECK(s2.sceneCuts.rederive);
    CHECK(s2.sceneCuts.known);
    CHECK(s2.sceneCuts.count == 1);
    CHECK(s2.sceneCuts.before == 1);
    CHECK(s2.sceneCuts.id == list0->id);
    CHECK_THAT(s2.detail, ContainsSubstring("no decoding"));
    CHECK(plan2.runCount == 1);
    CHECK(plan2.ToJson()["stages"][0]["scene_cuts"]["rederive"] == true);
    const ProcessResult r2 = RunProcess(o);
    CHECK(Report(r2, "scenes").status == "ran");
    for (const char* name : {"depth", "flow", "upscale", "nr", "fg"}) CHECK(Report(r2, name).status == "reused");
    const Manifest m2 = Manifest::Load(o.passesRoot / kScenesPass);
    CHECK(m2.stageParams["rederived"] == true);
    CHECK(m2.stageParams["threshold"] == 0.3);
    CHECK(m2.fingerprint == s2.fingerprint);
    CHECK(ListPassVersions(o.passesRoot, kScenesPass).size() == 2);  // the 0.24 list waits in the history

    // 0.6: no cut left; the cut is among the near misses
    ApplyParamSpec(o.stages, "scenes.threshold=0.6");
    const ProcessPlan plan3 = PlanProcess(o);
    const StageDecision& s3 = Decision(plan3, "scenes");
    CHECK(s3.sceneCuts.rederive);
    CHECK(s3.sceneCuts.count == 0);
    CHECK(s3.sceneCuts.before == 1);
    CHECK(s3.sceneCuts.id.empty());
    CHECK(s3.info == "склеек не найдено");
    RunProcess(o);
    const auto list3 = ReadSceneList(o.passesRoot / kScenesPass, "");
    REQUIRE(list3);
    CHECK(list3->cuts.empty());
    REQUIRE(!list3->nearMisses.empty());
    CHECK(list3->nearMisses.front().first == kCutAt);

    // 0: off; the scores stay for the way back
    ApplyParamSpec(o.stages, "scenes.threshold=0");
    const ProcessPlan plan4 = PlanProcess(o);
    const StageDecision& s4 = Decision(plan4, "scenes");
    CHECK(s4.sceneCuts.off);
    CHECK(s4.sceneCuts.rederive);
    CHECK(s4.info == "порог 0: без сбросов на склейках");
    RunProcess(o);
    const auto list4 = ReadSceneList(o.passesRoot / kScenesPass, "");
    REQUIRE(list4);
    CHECK(list4->off);
    CHECK(list4->cuts.empty());
    CHECK(std::filesystem::exists(o.passesRoot / kScenesPass / kScenesScoresFile));

    // back to 0.24: the version of the first run comes back, nothing is analysed
    ApplyParamSpec(o.stages, "scenes.threshold=0.24");
    const ProcessPlan plan5 = PlanProcess(o);
    const StageDecision& s5 = Decision(plan5, "scenes");
    CHECK(s5.action == "restore");
    CHECK(s5.fingerprint == s0.fingerprint);
    CHECK(s5.sceneCuts.known);
    CHECK(s5.sceneCuts.count == 1);
    CHECK(s5.sceneCuts.id == list0->id);
    const ProcessResult r5 = RunProcess(o);
    CHECK(!Report(r5, "scenes").restored.empty());
    CHECK(ReadSceneList(o.passesRoot / kScenesPass, "")->cuts == std::vector<int64_t>{kCutAt});

    // --force scenes: analysed again from the source, not over the stored scores
    o.forceStages = {"scenes"};
    const ProcessPlan plan6 = PlanProcess(o);
    CHECK(Decision(plan6, "scenes").kind == "forced");
    CHECK(!Decision(plan6, "scenes").sceneCuts.rederive);
    CHECK(!Decision(plan6, "scenes").sceneCuts.known);
    RunProcess(o);
    CHECK(!Manifest::Load(o.passesRoot / kScenesPass).stageParams.contains("rederived"));
    CHECK(ReadSceneList(o.passesRoot / kScenesPass, "")->cuts == std::vector<int64_t>{kCutAt});
}

TEST_CASE("dlssvid scenes: the list on the console and in JSON; a second threshold over the stored scores, the first list kept", "[integration][cli][scenes]") {
    const auto dir = Dir("cli");
    const auto clip = CutClip(dir);
    const auto root = dir / "passes";
    const CliResult a = RunCli({"scenes", "-i", clip.string(), "-o", root.string(), "--json", (dir / "a.json").string()});
    INFO(a.out);
    REQUIRE(a.code == 0);
    CHECK(a.out.find("cut       3") != std::string::npos);
    CHECK(a.out.find("1 cut(s)") != std::string::npos);
    const nlohmann::json ja = ReadJson(dir / "a.json");
    CHECK(ja["cuts"] == nlohmann::json::array({kCutAt}));
    CHECK(ja["count"] == 1);
    CHECK(ja["rederived"] == false);
    CHECK(ja["reached_end"] == true);
    CHECK(ja["id"] == SceneCutsId({kCutAt}));

    const CliResult b = RunCli({"scenes", "-i", clip.string(), "-o", root.string(), "--threshold", "0.6", "--json", (dir / "b.json").string()});
    INFO(b.out);
    REQUIRE(b.code == 0);
    CHECK(b.out.find("stored scores, no decoding") != std::string::npos);
    CHECK(b.out.find("near      3") != std::string::npos);
    const nlohmann::json jb = ReadJson(dir / "b.json");
    CHECK(jb["rederived"] == true);
    CHECK(jb["count"] == 0);
    CHECK(jb["id"] == "");
    CHECK(ListPassVersions(root, kScenesPass).size() == 2);  // the 0.24 list went to the history

    const CliResult c = RunCli({"scenes", "-i", clip.string(), "-o", root.string(), "--reanalyze", "--json", (dir / "c.json").string()});
    REQUIRE(c.code == 0);
    CHECK(ReadJson(dir / "c.json")["rederived"] == false);
    CHECK(ReadJson(dir / "c.json")["count"] == 1);

    const CliResult bad = RunCli({"scenes", "-i", clip.string(), "-o", root.string(), "--threshold", "2"});
    CHECK(bad.code != 0);
    CHECK(bad.out.find("scenes.threshold") != std::string::npos);
}

// PR-13 (regression): at 0.24 the list of the stage scenes is the one NR's own detector of PR-6 reset its history at - the module
// it called, as it called it (every decoded source frame, Next(frame, 0.24f)). No score lies within 1e-4 of the threshold, so the
// stored float32 scores and the detector's double agree on every frame.
TEST_CASE("Regression: the scene list at 0.24 is the one NR's detector of PR-6 found (8 bit, 10 bit, a flash)", "[integration][scenes][regression]") {
    const auto dir = Dir("equivalence");
    D3D12Device dev({true, false});
    const std::vector<std::pair<std::string, std::vector<int64_t>>> clips{{"cut", {kCutAt}}, {"10bit", {kCutAt}}, {"flash", {3, 4}}};
    for (const auto& [kind, expected] : clips) {
        INFO(kind);
        const auto clip = EquivalenceClip(dir, kind);
        const ScenesStageOptions o = Options(dir / kind, clip);
        VideoDecoder decoder(clip);
        const ScenesRunResult r = AnalyseScenes(decoder, o, -1);
        const auto scores = ReadSceneScores(r.outDir, Manifest::Load(r.outDir));
        REQUIRE(scores);
        REQUIRE(scores->scores.size() == 7);
        for (const float v : scores->scores) CHECK(std::fabs(static_cast<double>(v) - 0.24) > 1e-4);
        CHECK(r.list.cuts == expected);

        std::vector<int64_t> pr6;
        {
            VideoDecoder d(clip);
            SceneCutDetector det(ColorInfoFromStream(d.Info()));
            CpuFrame f;
            while (d.NextFrame(f))
                if (det.Next(f, 0.24f)) pr6.push_back(f.index);
        }
        CHECK(r.list.cuts == pr6);

        NrStageOptions no;  // the stage resets at exactly these frames now (its own detector gave the same, ab44a61)
        no.backend = "stub";
        no.outputDir = dir / kind / "nr";
        no.temporal = 0.5f;
        no.sceneCuts = r.list;
        VideoDecoder dn(clip);
        CHECK(RunNr(dn, dev, no).stats.sceneCuts == r.list.cuts);
    }
}

// PR-13 (regression), the rule of the spec: an empty list is no input - on a clip without cuts every pass below is the one a run without
// the stage makes, fingerprint for fingerprint. A broken list on disk with the stage off: the stages that read it are warned and run
// without it.
TEST_CASE("Regression: on a clip without cuts the stages below have the fingerprints of a run without scenes", "[integration][process][scenes][regression]") {
    const auto dir = Dir("no_cuts");
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    std::filesystem::create_directories(dir / "with");
    std::filesystem::create_directories(dir / "without");
    const ProcessOptions with = StubProcess(dir / "with", clip);
    ProcessOptions without = StubProcess(dir / "without", clip);
    for (auto& st : without.stages)
        if (st.name == "scenes") st.enabled = false;
    const ProcessResult a = RunProcess(with), b = RunProcess(without);
    CHECK(Report(a, "scenes").status == "ran");
    REQUIRE(ReadSceneList(with.passesRoot / kScenesPass, ""));
    CHECK(ReadSceneList(with.passesRoot / kScenesPass, "")->cuts.empty());
    for (const char* name : {"depth", "flow", "upscale", "nr", "fg"}) {
        INFO(name);
        CHECK(Report(a, name).fingerprint == Report(b, name).fingerprint);
    }
    for (const char* pass : {"mv_dlss", "color_sr", "color_nr", "color_fg"}) {
        INFO(pass);
        CHECK(Manifest::Load(with.passesRoot / pass).inputs.count("scenes") == 0);
        CHECK(SceneLabelOf(Manifest::Load(with.passesRoot / pass)).empty());
    }
    CHECK(PlanProcess(with).runCount == 0);

    std::filesystem::create_directories(without.passesRoot / kScenesPass);
    std::ofstream(without.passesRoot / kScenesPass / Manifest::kFileName) << "{ broken";
    const ProcessPlan plan = PlanProcess(without);
    const StageDecision& nr = Decision(plan, "nr");
    CHECK(nr.action == "reuse");
    REQUIRE(!nr.warnings.empty());
    CHECK_THAT(nr.warnings.front(), ContainsSubstring("the scene list is not used"));
    CHECK(Decision(plan, "upscale").warnings.empty());  // nis reads no list
}
