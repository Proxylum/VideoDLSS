// The NR preview's core (PR-10, core/pipeline/NrPreview): the gate over the plan, the key, the command line, the check of a run,
// the cache, its clean-up, and the .preview folder being no pass, no version and no viewport source.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <process.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "passes/ControlMaskPass.h"
#include "passes/Manifest.h"
#include "passes/PassImage.h"
#include "passes/PassSequence.h"
#include "passes/ScenesPass.h"
#include "passes/formats/ExrIO.h"
#include "passes/formats/PngIO.h"
#include "pipeline/NrPreview.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ProcessRunner.h"
#include "viewport/FrameStore.h"

using namespace dlssvid;
namespace fs = std::filesystem;

namespace {

fs::path Dir(const char* name) {
    const fs::path d = fs::path(DLSSVID_TEST_TMP) / "nr_preview" / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

void WriteColor(const fs::path& dir, PassKind kind, int64_t first, int64_t last, uint32_t w = 8, uint32_t h = 4, float v = 0.25f) {
    Manifest m = Manifest::ForPass(kind, w, h, FileFormat::Exr);
    PassWriter writer(dir, m);
    for (int64_t f = first; f <= last; ++f) {
        PassImage img = MakePassImage(kind, w, h);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < img.channels.size(); ++c) img.Set(x, y, c, v + 0.01f * static_cast<float>(f) + 0.02f * static_cast<float>(x) + 0.03f * static_cast<float>(c));
        writer.WriteFrame(f, img);
    }
    writer.Finish();
}

StageDecision Decision(const std::string& name, const std::string& action, std::vector<std::string> outputs) {
    StageDecision d;
    d.name = name;
    d.action = action;
    d.outputs = std::move(outputs);
    return d;
}

// a plan of upscale -> nr over `root` with color_sr on disk (frames 0..59), and optionally depth / flow / scenes above
ProcessPlan Plan(const fs::path& root) {
    if (!Manifest::Exists(root / "color_sr")) WriteColor(root / "color_sr", PassKind::ColorSr, 0, 59);
    ProcessPlan plan;
    plan.passesRoot = root;
    plan.sourceHash = "sha256:" + std::string(64, 'c');
    plan.frames = 60;
    plan.stages.push_back(Decision("upscale", "reuse", {"color_sr"}));
    StageDecision nr = Decision("nr", "run", {"color_nr"});
    nr.params = {{"backend", "stub"}, {"intensity", 1.0}, {"format", "exr"}, {"dll_dir", "C:/somewhere"}};
    nr.tool = {{"app", "0.1.0"}, {"backend", "stub"}, {"model", ""}, {"model_hash", ""}, {"dll_file", ""}, {"dll", ""}};
    nr.inputs = {{"color_sr", "sha256:sr"}};
    nr.sourceFormat = "yuv420p";
    plan.stages.push_back(nr);
    return plan;
}

StageDecision& Nr(ProcessPlan& plan) {
    for (auto& d : plan.stages)
        if (d.name == "nr") return d;
    throw std::runtime_error("no nr");
}

NrPreviewEnv Env(const fs::path& dir) {
    NrPreviewEnv env;
    env.source = dir / "clip.mkv";
    env.cli = dir / "fake_cli.exe";
    if (!fs::exists(env.cli)) std::ofstream(env.cli, std::ios::binary) << "a fake cli";
    env.fps = 24.0;
    env.msPerFrame = 300.0;
    env.startSeconds = 2.0;
    return env;
}

bool Has(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }
std::string After(const std::vector<std::string>& v, const std::string& flag) {
    const auto it = std::find(v.begin(), v.end(), flag);
    return it != v.end() && it + 1 != v.end() ? *(it + 1) : std::string();
}

// what a successful `dlssvid nr` leaves in a job's work folder
void FakeRun(const NrPreviewJob& job, int64_t written = -1, const std::string& controlMask = {}, const std::string& dll = {}) {
    WriteNrPreviewParams(job);
    const int64_t n = written >= 0 ? written : job.frames;
    if (n > 0) WriteColor(job.workDir / "color_nr", PassKind::ColorNr, job.first, job.first + n - 1);
    nlohmann::json stats = {{"frames", job.warmup + n}, {"frames_written", n}, {"warmup_frames", job.warmup}, {"ms_per_frame_wall", 12.5}};
    stats["control_mask"] = controlMask.empty() ? nlohmann::json(nullptr) : nlohmann::json{{"fingerprint", controlMask}};
    nlohmann::json run = {{"stats", stats}, {"diagnostics", nlohmann::json::object()}};
    if (!dll.empty()) run["diagnostics"]["dll_sha256"] = dll;
    std::ofstream(job.workDir / "nr.json") << run.dump(2);
}

void Age(const fs::path& p, std::chrono::minutes by) { fs::last_write_time(p, fs::last_write_time(p) - by); }

}  // namespace

TEST_CASE("NR preview key: stable, changes with every part of the run and not with the passes root path", "[unit][preview]") {
    const auto dir = Dir("key");
    const auto root = dir / "passes";
    ProcessPlan plan = Plan(root);
    const NrPreviewEnv env = Env(dir);
    const NrPreviewJob base = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    REQUIRE(base.gate.ok());
    CHECK(base.key.size() == 12);
    CHECK(base.key.find_first_not_of("0123456789abcdef") == std::string::npos);
    CHECK(PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env).key == base.key);
    CHECK(base.entryDir == NrPreviewRoot(root) / base.key);
    CHECK(base.workDir.parent_path() == NrPreviewRoot(root));
    CHECK(base.workDir.filename().string().rfind(base.key + ".part-" + std::to_string(_getpid()) + "-", 0) == 0);
    const std::string escaped = nlohmann::json(root.string()).dump();  // as the key JSON writes a path: backslashes doubled
    CHECK(base.keyJson.find(escaped.substr(1, escaped.size() - 2)) == std::string::npos);  // no absolute path in the key
    CHECK(base.keyJson.find("\"masks\":{}") != std::string::npos);  // no mask on disk

    auto keyWith = [&](const std::function<void(ProcessPlan&, NrPreviewEnv&)>& change, NrPreviewMode mode = NrPreviewMode::Frame, int64_t first = 5) {
        ProcessPlan p = plan;
        NrPreviewEnv e = env;
        change(p, e);
        const NrPreviewJob j = PrepareNrPreview(p, mode, first, 1.0, e);
        REQUIRE(j.gate.ok());
        return j.key;
    };
    auto none = [](ProcessPlan&, NrPreviewEnv&) {};
    CHECK(keyWith(none, NrPreviewMode::Frame, 6) != base.key);
    CHECK(keyWith(none, NrPreviewMode::History) != base.key);
    CHECK(keyWith(none, NrPreviewMode::Segment) != base.key);
    CHECK(keyWith([](ProcessPlan& p, NrPreviewEnv&) { Nr(p).params["intensity"] = 1.2; }) != base.key);
    CHECK(keyWith([](ProcessPlan& p, NrPreviewEnv&) { p.sourceHash = "sha256:" + std::string(64, 'd'); }) != base.key);
    CHECK(keyWith([](ProcessPlan& p, NrPreviewEnv&) { Nr(p).sourceFormat = "yuv444p10"; }) != base.key);
    CHECK(keyWith([](ProcessPlan& p, NrPreviewEnv&) { Nr(p).tool["dll"] = "sha256:other"; }) != base.key);
    CHECK(keyWith([](ProcessPlan& p, NrPreviewEnv&) { Nr(p).tool["app"] = "0.1.0+nrcm"; }) != base.key);
    CHECK(keyWith([](ProcessPlan& p, NrPreviewEnv&) { Nr(p).inputs["color_sr"] = "sha256:sr2"; }) != base.key);
    CHECK(keyWith([](ProcessPlan&, NrPreviewEnv& e) { e.warp = true; }) != base.key);
    // the bytes of an input's manifest: a pass rewritten in place under the plan's id (a legacy input)
    {
        Manifest m = Manifest::Load(root / "color_sr");
        const Manifest keep = m;
        m.stageParams["edited"] = true;  // the manifest's bytes (created is written only with a fingerprint)
        m.Save(root / "color_sr");
        CHECK(keyWith(none) != base.key);
        keep.Save(root / "color_sr");
        CHECK(keyWith(none) == base.key);
    }
    // the executable that runs it
    std::ofstream(env.cli, std::ios::binary) << "another build of the cli";
    CHECK(keyWith(none) != base.key);
    std::ofstream(env.cli, std::ios::binary) << "a fake cli";
    fs::last_write_time(env.cli, fs::last_write_time(env.cli) + std::chrono::seconds(5));
    CHECK(keyWith(none) != base.key);
    // the root moved elsewhere with its passes: the same key
    const NrPreviewJob here = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    fs::copy(root, dir / "moved", fs::copy_options::recursive);
    ProcessPlan moved = plan;
    moved.passesRoot = dir / "moved";
    CHECK(PrepareNrPreview(moved, NrPreviewMode::Frame, 5, 0, env).key == here.key);
}

TEST_CASE("NR preview gate: reused upstream passes; run, restore and pending refuse with the stages; frame mode drops depth, mv and scenes", "[unit][preview]") {
    const auto root = Dir("gate");
    ProcessPlan plan = Plan(root);
    plan.stages.insert(plan.stages.begin(), Decision("flow", "reuse", {"mv_raw", "mv_dlss"}));
    plan.stages.insert(plan.stages.begin(), Decision("depth", "reuse", {"depth_raw", "depth_dlss"}));
    plan.stages.insert(plan.stages.begin(), Decision("scenes", "reuse", {kScenesPass}));
    Nr(plan).inputs = {{"color_sr", "a"}, {"depth_dlss", "b"}, {"mv_dlss", "c"}, {"mask_ui", "d"}, {kScenesPass, "e"}};
    auto stage = [&](ProcessPlan& p, const char* name) -> StageDecision& {
        for (auto& d : p.stages)
            if (d.name == name) return d;
        throw std::runtime_error(name);
    };

    // every producer reused: all passed (the mask has no producer: as it is); the frame mode never passes the scene list
    NrPreviewGate g = GateNrPreview(plan, NrPreviewMode::History);
    REQUIRE(g.ok());
    CHECK(g.passed.size() == 5);
    g = GateNrPreview(plan, NrPreviewMode::Frame);
    REQUIRE(g.ok());
    CHECK_FALSE(Has(g.passed, kScenesPass));
    CHECK(g.dropped == std::vector<std::string>{kScenesPass});

    // flow would run: a frame drops the vectors, history and segments refuse with the stage
    ProcessPlan run = plan;
    stage(run, "flow").action = "run";
    g = GateNrPreview(run, NrPreviewMode::Frame);
    REQUIRE(g.ok());
    CHECK(Has(g.dropped, "mv_dlss"));
    CHECK(Has(g.passed, "color_sr"));
    g = GateNrPreview(run, NrPreviewMode::Segment);
    CHECK(g.why == NrPreviewBlock::Upstream);
    CHECK(g.stages == std::vector<std::string>{"flow"});
    CHECK(g.passed.empty());

    // the colour is never optional: upscale to run or to restore refuses every mode; the stages in plan order
    ProcessPlan up = run;
    stage(up, "upscale").action = "run";
    g = GateNrPreview(up, NrPreviewMode::History);
    CHECK(g.why == NrPreviewBlock::Upstream);
    CHECK(g.stages == std::vector<std::string>{"flow", "upscale"});
    ProcessPlan restore = plan;
    stage(restore, "upscale").action = "restore";
    stage(restore, "upscale").version = "20261027-101010";
    g = GateNrPreview(restore, NrPreviewMode::Frame);
    CHECK(g.why == NrPreviewBlock::Restore);
    CHECK(g.detail == "20261027-101010");
    CHECK(g.stages == std::vector<std::string>{"upscale"});

    // the scene list not known yet: pending
    ProcessPlan pending = plan;
    stage(pending, "scenes").action = "run";
    stage(pending, "scenes").pending = true;
    stage(pending, "depth").pending = true;
    g = GateNrPreview(pending, NrPreviewMode::History);
    CHECK(g.why == NrPreviewBlock::Pending);
    CHECK(g.stages == std::vector<std::string>{"scenes", "depth"});
    CHECK(GateNrPreview(pending, NrPreviewMode::Frame).ok());  // a frame drops the list and the depth
}

TEST_CASE("NR preview gate: no nr decision, a decision error and a missing DLL are refused with their reason", "[unit][preview]") {
    const auto root = Dir("gate_refused");
    ProcessPlan plan = Plan(root);
    ProcessPlan off = plan;
    off.stages.pop_back();
    CHECK(GateNrPreview(off, NrPreviewMode::Frame).why == NrPreviewBlock::NrOff);
    ProcessPlan err = plan;
    Nr(err).error = "not enough video memory";
    const NrPreviewGate e = GateNrPreview(err, NrPreviewMode::Frame);
    CHECK(e.why == NrPreviewBlock::NrError);
    CHECK(e.detail == "not enough video memory");
    ProcessPlan dll = plan;
    Nr(dll).tool["dll_file"] = "nvngx_dlssnr.dll";
    CHECK(GateNrPreview(dll, NrPreviewMode::Frame).why == NrPreviewBlock::NoDll);
    Nr(dll).tool["dll"] = "sha256:found";
    CHECK(GateNrPreview(dll, NrPreviewMode::Frame).ok());
    // the range: past the clip's end; the warm-up shortened at its start; a segment cut at its end
    const NrPreviewEnv env = Env(root);
    CHECK(PrepareNrPreview(plan, NrPreviewMode::Frame, 60, 0, env).gate.why == NrPreviewBlock::PastEnd);
    const NrPreviewJob start = PrepareNrPreview(plan, NrPreviewMode::History, 2, 0, env);
    CHECK(start.warmup == 2);
    CHECK(start.frames == 1);
    const NrPreviewJob seg = PrepareNrPreview(plan, NrPreviewMode::Segment, 50, 3.0, env);
    CHECK(seg.frames == 10);  // 72 asked, the clip ends at 59
    CHECK(seg.warmup == kNrPreviewHistoryFrames);
    CHECK(std::abs(seg.estimatedSeconds - (2.0 + 16 * 0.3)) < 1e-9);
    // the disk: a segment beyond the free space less the reserve
    NrPreviewEnv tight = env;
    tight.freeBytes = kNrPreviewDiskReserve + 1024;
    CHECK(PrepareNrPreview(plan, NrPreviewMode::Segment, 0, 1.0, tight).gate.why == NrPreviewBlock::Disk);
    tight.freeBytes = 0;  // not checked
    CHECK(PrepareNrPreview(plan, NrPreviewMode::Segment, 0, 1.0, tight).gate.ok());
    // a control mask that does not read stops the preview before it starts (Init would)
    Nr(plan).inputs[kControlMaskPass] = "sha256:m";
    fs::create_directories(root / kControlMaskPass);
    Manifest broken = Manifest::ForPass(PassKind::Mask, 4, 4, FileFormat::Png);
    broken.pass = kControlMaskPass;
    broken.Save(root / kControlMaskPass);
    const NrPreviewJob mask = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    CHECK(mask.gate.why == NrPreviewBlock::ControlMask);
    CHECK_FALSE(mask.gate.detail.empty());
}

TEST_CASE("NR preview command line: schema keys only in params.json, the masks root, scenes or --no-scenes, no --dll-dir, --warp on WARP", "[unit][preview]") {
    const auto dir = Dir("args");
    const auto root = dir / "passes";
    ProcessPlan plan = Plan(root);
    NrPreviewEnv env = Env(dir);
    NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::History, 10, 0, env);
    REQUIRE(job.gate.ok());
    CHECK_FALSE(job.params.contains("dll_dir"));  // not an option of dlssvid nr: --params-file would refuse it
    CHECK(job.params["intensity"] == 1.0);
    const std::vector<std::string> a = job.Args();
    CHECK(a.front() == "nr");
    CHECK(After(a, "-o") == job.workDir.string());
    CHECK(Has(a, "--no-auto"));
    CHECK(After(a, "--source-hash") == plan.sourceHash);
    CHECK(After(a, "--params-file") == (job.workDir / "params.json").string());
    CHECK(After(a, "--start") == "10");
    CHECK(After(a, "--frames") == "1");
    CHECK(After(a, "--warmup") == "6");
    CHECK(After(a, "--masks-dir") == root.string());
    CHECK(After(a, "--color-dir") == (root / "color_sr").string());
    CHECK(Has(a, "--no-scenes"));
    CHECK_FALSE(Has(a, "--depth-dir"));
    CHECK_FALSE(Has(a, "--dll-dir"));
    CHECK_FALSE(Has(a, "--warp"));
    env.warp = true;
    plan.stages.insert(plan.stages.begin(), Decision("scenes", "reuse", {kScenesPass}));
    Nr(plan).inputs[kScenesPass] = "cuts:x";
    job = PrepareNrPreview(plan, NrPreviewMode::History, 10, 0, env);
    const std::vector<std::string> b = job.Args();
    CHECK(Has(b, "--warp"));
    CHECK(After(b, "--scenes-dir") == (root / kScenesPass).string());
    CHECK_FALSE(Has(b, "--no-scenes"));
    WriteNrPreviewParams(job);
    std::ifstream in(job.workDir / "params.json");
    CHECK(nlohmann::json::parse(in) == job.params);
}

TEST_CASE("NR preview cache: an entry is found only with its sidecar, the same key JSON and the full range; a run is checked before it is published", "[unit][preview]") {
    const auto dir = Dir("cache");
    const auto root = dir / "passes";
    ProcessPlan plan = Plan(root);
    const NrPreviewEnv env = Env(dir);
    const NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::Segment, 20, 0.2, env);
    REQUIRE(job.gate.ok());
    REQUIRE(job.frames == 5);
    CHECK_FALSE(FindNrPreview(job));
    FakeRun(job);
    CHECK_FALSE(FindNrPreview(job));  // a work folder is no entry
    const NrPreviewVerdict v = VerifyNrPreviewRun(job);
    CHECK(v.ok());
    CHECK(v.cacheable);
    const NrPreviewEntry e = PublishNrPreview(job, 3.5, true);
    CHECK(e.dir == job.entryDir);
    CHECK_FALSE(fs::exists(job.workDir));
    CHECK(fs::exists(job.entryDir / "preview.json"));
    CHECK(e.first == 20);
    CHECK(e.last == 24);
    CHECK(e.bytes > 0);
    const auto hit = FindNrPreview(job);
    REQUIRE(hit);
    CHECK(hit->dir == job.entryDir);
    CHECK(hit->seconds == 3.5);
    CHECK(hit->stats["ms_per_frame_wall"] == 12.5);
    // a hit marks the entry as shown now
    Age(job.entryDir / "preview.json", std::chrono::minutes(30));
    const auto before = fs::last_write_time(job.entryDir / "preview.json");
    REQUIRE(FindNrPreview(job));
    CHECK(fs::last_write_time(job.entryDir / "preview.json") > before);
    // another key JSON under the same name (a collision of 12 hex), a frame gone: no hit
    NrPreviewJob other = job;
    other.keyJson += " ";
    CHECK_FALSE(FindNrPreview(other));
    fs::remove(job.entryDir / "color_nr" / Manifest::Load(job.entryDir / "color_nr").FrameFileName(22));
    CHECK_FALSE(FindNrPreview(job));
    // a second instance published the same key first: its entry wins, this run's folder goes
    const NrPreviewJob again = PrepareNrPreview(plan, NrPreviewMode::Segment, 20, 0.2, env);
    REQUIRE(again.key == job.key);
    FakeRun(again);
    const NrPreviewEntry replaced = PublishNrPreview(again, 1.0, true);  // the broken entry is replaced
    CHECK(replaced.dir == job.entryDir);
    CHECK(FindNrPreview(again));
    const NrPreviewJob third = PrepareNrPreview(plan, NrPreviewMode::Segment, 20, 0.2, env);
    FakeRun(third);
    const NrPreviewEntry kept = PublishNrPreview(third, 9.0, true);
    CHECK(kept.seconds == 1.0);  // the entry already there
    CHECK_FALSE(fs::exists(third.workDir));
    // not cacheable (another DLL): shown from the work folder, nothing published
    const NrPreviewJob dll = PrepareNrPreview(plan, NrPreviewMode::Frame, 30, 0, env);
    FakeRun(dll);
    const NrPreviewEntry uncached = PublishNrPreview(dll, 1.0, false);
    CHECK(uncached.dir == dll.workDir);
    CHECK_FALSE(uncached.cacheable);
    CHECK_FALSE(fs::exists(dll.entryDir));
}

TEST_CASE("NR preview check: fewer frames, another control mask, an input changed meanwhile and another DLL", "[unit][preview][regression]") {
    const auto dir = Dir("verify");
    const auto root = dir / "passes";
    ProcessPlan plan = Plan(root);
    const NrPreviewEnv env = Env(dir);
    NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::Segment, 10, 0.2, env);
    REQUIRE(job.frames == 5);
    FakeRun(job, 3);
    NrPreviewVerdict v = VerifyNrPreviewRun(job);
    CHECK(v.why == NrPreviewReject::Frames);
    CHECK(v.written == 3);
    fs::remove_all(job.workDir);
    CHECK(VerifyNrPreviewRun(job).why == NrPreviewReject::NoSummary);
    // the control mask the run applied is not the one the job planned
    job.controlMask = "sha256:planned";
    FakeRun(job, -1, "sha256:other");
    CHECK(VerifyNrPreviewRun(job).why == NrPreviewReject::ControlMask);
    fs::remove_all(job.workDir);
    FakeRun(job, -1, "sha256:planned");
    CHECK(VerifyNrPreviewRun(job).ok());
    // a process from a terminal rewrote color_sr during the run
    Manifest m = Manifest::Load(root / "color_sr");
    m.stageParams["edited"] = true;
    m.Save(root / "color_sr");
    v = VerifyNrPreviewRun(job);
    CHECK(v.why == NrPreviewReject::InputChanged);
    CHECK(v.detail == "color_sr");
    // the run loaded another DLL than the key names: shown, not cached
    ProcessPlan ngx = Plan(dir / "ngx");
    Nr(ngx).tool["dll_file"] = "nvngx_dlssnr.dll";
    Nr(ngx).tool["dll"] = "sha256:" + std::string(64, 'a');
    const NrPreviewJob d = PrepareNrPreview(ngx, NrPreviewMode::Frame, 3, 0, env);
    CHECK(d.dllSha256 == std::string(64, 'a'));
    FakeRun(d, -1, {}, std::string(64, 'b'));
    v = VerifyNrPreviewRun(d);
    CHECK(v.ok());
    CHECK_FALSE(v.cacheable);
    fs::remove_all(d.workDir);
    FakeRun(d, -1, {}, std::string(64, 'a'));
    CHECK(VerifyNrPreviewRun(d).cacheable);
}

TEST_CASE("NR preview GC: at most 16 entries and the byte cap, pinned entries kept, stale part and save folders removed, unknown names kept, a locked file is reported", "[unit][preview]") {
    const auto root = Dir("gc");
    const fs::path base = NrPreviewRoot(root);
    auto entry = [&](int i, uint64_t bytes, int ageMinutes) {
        char name[16];
        std::snprintf(name, sizeof name, "%012x", 0xa00 + i);
        const fs::path d = base / name;
        fs::create_directories(d / "color_nr");
        std::ofstream(d / "color_nr" / "f.exr") << "x";
        std::ofstream(d / "preview.json") << nlohmann::json{{"format", "dlssvid-nr-preview"}, {"key", name}, {"bytes", bytes}}.dump();
        Age(d / "preview.json", std::chrono::minutes(ageMinutes));
        return d;
    };
    std::vector<fs::path> e;
    for (int i = 0; i < 18; ++i) e.push_back(entry(i, 1000, 100 - i));  // e[0] the oldest
    NrPreviewGcReport r = GcNrPreviews(root, {e[1]});
    CHECK(r.removed.size() == 2);
    CHECK_FALSE(fs::exists(e[0]));
    CHECK(fs::exists(e[1]));  // pinned: kept although old
    CHECK_FALSE(fs::exists(e[2]));
    CHECK(fs::exists(e[3]));
    CHECK(NrPreviewUsageOf(root).entries == 16);
    CHECK(NrPreviewUsageOf(root).bytes == 16000);
    // the byte cap: the oldest go until the rest fits
    r = GcNrPreviews(root, {}, 16, 10500);
    CHECK(r.removed.size() == 6);
    CHECK(NrPreviewUsageOf(root).bytes == 10000);
    // work and save folders: ours and not running go, another process's fresh one stays, its old one goes; other names stay
    const fs::path ours = base / ("aaaaaaaaaaaa.part-" + std::to_string(_getpid()) + "-7");
    const fs::path running = base / ("aaaaaaaaaaab.part-" + std::to_string(_getpid()) + "-8");
    const fs::path theirsFresh = base / "aaaaaaaaaaac.part-999999-1";
    const fs::path theirsOld = base / "save-999999-2";
    const fs::path unknown = base / "notes";
    for (const auto& d : {ours, running, theirsFresh, theirsOld, unknown}) fs::create_directories(d);
    Age(theirsOld, std::chrono::minutes(11));
    GcNrPreviews(root, {running});
    CHECK_FALSE(fs::exists(ours));
    CHECK(fs::exists(running));
    CHECK(fs::exists(theirsFresh));
    CHECK_FALSE(fs::exists(theirsOld));
    CHECK(fs::exists(unknown));
    // a file held by another program: reported, not thrown; removed at a later clean-up
    const fs::path held = e[17] / "color_nr" / "f.exr";
    const HANDLE h = CreateFileW(held.wstring().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(h != INVALID_HANDLE_VALUE);
    r = RemoveNrPreviews(root, {});
    CloseHandle(h);
    CHECK(r.failed == std::vector<fs::path>{e[17]});
    CHECK(fs::exists(held));
    RemoveNrPreviews(root, {});
    CHECK_FALSE(fs::exists(e[17]));
    CHECK(NrPreviewUsageOf(root).entries == 0);
    CHECK(fs::exists(unknown));
}

TEST_CASE("The .preview folder is no pass, no version and no viewport source", "[unit][preview]") {
    const auto root = Dir("invisible");
    WriteColor(root / "color_sr", PassKind::ColorSr, 0, 3);
    const fs::path entry = NrPreviewRoot(root) / "abcdef012345";
    WriteColor(entry / "color_nr", PassKind::ColorNr, 1, 2);
    std::ofstream(entry / "preview.json") << "{}";
    CHECK(PassesUnderRoot(root) == std::vector<std::string>{"color_sr"});
    for (const auto& v : ListPassVersions(root)) CHECK(v.pass == "color_sr");
    CHECK(GcPassVersions(root, 1).removed.empty());
    CHECK(fs::exists(entry / "color_nr" / Manifest::kFileName));
    for (const auto& s : FrameStore::DiscoverPasses(root)) CHECK(s.name == "color_sr");
    CHECK(FrameStore::DiscoverPassVersions(root).empty());
    CHECK_FALSE(Manifest::Exists(root / kPreviewDir));
    CHECK_FALSE(Manifest::Exists(NrPreviewRoot(root)));
}

TEST_CASE("NR preview: a frame saved as PNG 16-bit or as the EXR frame file", "[unit][preview]") {
    const auto dir = Dir("save");
    const auto root = dir / "passes";
    ProcessPlan plan = Plan(root);
    const NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::Segment, 4, 0.1, Env(dir));
    FakeRun(job);
    const NrPreviewEntry e = PublishNrPreview(job, 1.0, true);
    const PassImage colour = PassReader::Open(e.ColorDir()).ReadFrame(5).ConvertNormalizedTo(PixelType::F32);
    SaveNrPreviewFrame(e, 5, dir / "frame.png");
    const PassImage png = ReadPng(dir / "frame.png").ConvertNormalizedTo(PixelType::F32);
    CHECK(png.width == colour.width);
    CHECK(PassImage::MaxAbsDiff(png, colour) <= 1.0 / 65535.0 + 1e-6);
    CHECK(PassImage::MaxAbsDiff(png, colour) < 1e-3);
    SaveNrPreviewFrame(e, 5, dir / "frame.exr");
    CHECK(fs::file_size(dir / "frame.exr") == fs::file_size(e.ColorDir() / Manifest::Load(e.ColorDir()).FrameFileName(5)));
    CHECK_THROWS(SaveNrPreviewFrame(e, 40, dir / "none.png"));
    for (const auto& de : fs::directory_iterator(NrPreviewRoot(root))) CHECK(de.path().filename().string().rfind("save-", 0) != 0);  // no leftover

    // a pass in PNG (nr.format png) saved under .exr: an EXR, not PNG bytes with that name (review of PR-10)
    NrPreviewEntry p = e;
    p.dir = NrPreviewRoot(root) / "0123456789ab";
    {
        PassWriter w(p.ColorDir(), Manifest::ForPass(PassKind::ColorNr, 8, 4, FileFormat::Png));
        w.WriteFrame(5, PassReader::Open(e.ColorDir()).ReadFrame(5));
        w.Finish();
    }
    REQUIRE(Manifest::Load(p.ColorDir()).format == FileFormat::Png);
    const PassImage pngPass = PassReader::Open(p.ColorDir()).ReadFrame(5).ConvertNormalizedTo(PixelType::F32);
    SaveNrPreviewFrame(p, 5, dir / "from_png.exr");
    const PassImage exr = ReadExr(dir / "from_png.exr").ConvertNormalizedTo(PixelType::F32);
    CHECK(exr.width == pngPass.width);
    CHECK(PassImage::MaxAbsDiff(exr, pngPass) < 1e-3);  // half floats of 16-bit values
}

TEST_CASE("NR preview: a mask NR opens by name though the plan leaves it out is in the key and in the check of a run", "[unit][preview][regression]") {
    const auto dir = Dir("masks");
    const auto root = dir / "passes";
    ProcessPlan plan = Plan(root);
    const NrPreviewEnv env = Env(dir);
    const std::string none = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env).key;
    {  // frames 100..101 only: no mask of the whole clip, the plan leaves it out - NrStage::Init opens it by name all the same
        PassWriter w(root / "mask_ui", Manifest::ForPass(PassKind::Mask, 8, 4, FileFormat::Exr));
        w.WriteFrame(100, MakePassImage(PassKind::Mask, 8, 4));
        w.WriteFrame(101, MakePassImage(PassKind::Mask, 8, 4));
        w.Finish();
    }
    const NrPreviewJob with = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    REQUIRE(with.gate.ok());
    CHECK(with.key != none);
    CHECK(with.masks.count("mask_ui") == 1);
    Manifest m = Manifest::Load(root / "mask_ui");
    m.stageParams = {{"note", "other"}};  // restamped
    m.Save(root / "mask_ui");
    const NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    CHECK(job.key != with.key);
    FakeRun(job);
    m.stageParams = {{"note", "third"}};  // rewritten during the run
    m.Save(root / "mask_ui");
    const NrPreviewVerdict v = VerifyNrPreviewRun(job);
    CHECK(v.why == NrPreviewReject::InputChanged);
    CHECK(v.detail == "mask_ui");
    const NrPreviewJob gone = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    FakeRun(gone);
    fs::remove_all(root / "mask_ui");  // removed during the run
    CHECK(VerifyNrPreviewRun(gone).why == NrPreviewReject::InputChanged);
    const NrPreviewJob added = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    FakeRun(added);
    {
        PassWriter w(root / "mask_face", Manifest::ForPass(PassKind::Mask, 8, 4, FileFormat::Exr));
        w.WriteFrame(0, MakePassImage(PassKind::Mask, 8, 4));
        w.Finish();
    }
    CHECK(VerifyNrPreviewRun(added).why == NrPreviewReject::InputChanged);  // written during the run

    // a control mask the plan did not have, applied by the run (written meanwhile): refused, as process refuses it
    const NrPreviewJob plain = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    REQUIRE(plain.controlMask.empty());
    FakeRun(plain, -1, std::string(64, 'e'));
    CHECK(VerifyNrPreviewRun(plain).why == NrPreviewReject::ControlMask);

    // the bypass (intensity 0) opens no mask: none in the key, a mask written during its run does not refuse it
    ProcessPlan bypass = plan;
    Nr(bypass).params["intensity"] = 0.0;
    const NrPreviewJob b = PrepareNrPreview(bypass, NrPreviewMode::Frame, 5, 0, env);
    REQUIRE(b.gate.ok());
    CHECK(b.masks.empty());
    FakeRun(b);
    Manifest face = Manifest::Load(root / "mask_face");
    face.stageParams = {{"note", "other"}};
    face.Save(root / "mask_face");
    CHECK(VerifyNrPreviewRun(b).ok());
}

TEST_CASE("NR preview: over the disk budget the key is still set - a preview in the cache needs no room", "[unit][preview][regression]") {
    const auto dir = Dir("disk_cached");
    const auto root = dir / "passes";
    ProcessPlan plan = Plan(root);
    NrPreviewEnv env = Env(dir);
    const NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    REQUIRE(job.gate.ok());
    FakeRun(job);
    PublishNrPreview(job, 1.0, true);
    env.freeBytes = 1;  // no room
    const NrPreviewJob low = PrepareNrPreview(plan, NrPreviewMode::Frame, 5, 0, env);
    CHECK(low.gate.why == NrPreviewBlock::Disk);
    CHECK(low.key == job.key);
    CHECK(FindNrPreview(low).has_value());
}

TEST_CASE("NR preview: a seek through the video counts the CLI's demux before it in the estimate; the colour pass is read by index", "[unit][preview][regression]") {
    const auto dir = Dir("scan");
    const auto root = dir / "passes";
    ProcessPlan plan = Plan(root);
    plan.frames = 600;
    NrPreviewEnv env = Env(dir);
    std::ofstream(env.source, std::ios::binary) << std::string(4u << 20, 'x');  // a 4 MiB "clip"
    // with color_sr: by index, nothing to scan
    const NrPreviewJob pass = PrepareNrPreview(plan, NrPreviewMode::Frame, 300, 0, env);
    REQUIRE(pass.gate.ok());
    CHECK(pass.scanBytes == 0);
    // the colour from the video (upscale off)
    plan.stages.erase(plan.stages.begin());
    Nr(plan).inputs.clear();
    const NrPreviewJob nearJob = PrepareNrPreview(plan, NrPreviewMode::History, 54, 0, env);  // from 48: the frames are counted
    REQUIRE(nearJob.gate.ok());
    CHECK(nearJob.scanBytes == 0);
    const NrPreviewJob farJob = PrepareNrPreview(plan, NrPreviewMode::Frame, 300, 0, env);
    REQUIRE(farJob.gate.ok());
    CHECK(std::abs(static_cast<double>(farJob.scanBytes) - (4u << 20) * (300.0 + 1 + 64) / 600.0) <= 1.0);  // up to the frames read and 64
    CHECK(farJob.estimatedSeconds == Catch::Approx(env.startSeconds + env.msPerFrame / 1000.0 + static_cast<double>(farJob.scanBytes) / kNrPreviewScanRate));
    plan.frames = 0;  // the length unknown: the whole file
    CHECK(PrepareNrPreview(plan, NrPreviewMode::Frame, 300, 0, env).scanBytes == (4u << 20));
}
