// The NR preview end to end without Qt (PR-10): a real plan, the job's `dlssvid nr` as a process on the stub (WARP), the check, the
// publication and the cache - and nothing of the project changes: the plan, the pass versions and the passes stay as they were.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "passes/Manifest.h"
#include "passes/PassSequence.h"
#include "pipeline/NrPreview.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ProcessRunner.h"
#include "stages/nr/INrBackend.h"
#include "util/Subprocess.h"

using namespace dlssvid;
using namespace dlssvid::test;
namespace fs = std::filesystem;

namespace {

fs::path Dir(const char* name) {
    const auto d = TempDir() / "nr_preview_flow" / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

ProcessOptions Options(const fs::path& dir, const fs::path& clip) {
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.codec = "ffv1";
    o.warp = true;
    o.stages = DefaultProcessStages();
    for (const char* spec : {"depth.backend=stub", "flow.backend=stub", "upscale.backend=nis", "upscale.scale=2", "nr.backend=stub"}) ApplyParamSpec(o.stages, spec);
    for (auto& s : o.stages) s.enabled = s.name != "fg";
    return o;
}

NrPreviewEnv Env(const fs::path& clip) {
    NrPreviewEnv env;
    env.source = clip;
    env.cli = DLSSVID_CLI_PATH;
    env.warp = true;
    env.fps = 24.0;
    env.msPerFrame = 10.0;
    env.startSeconds = 0.5;
    return env;
}

uint32_t Run(const NrPreviewJob& job, std::string* out = nullptr) {
    WriteNrPreviewParams(job);
    std::vector<std::string> args = job.Args();
    args.insert(args.begin(), DLSSVID_CLI_PATH);
    return Subprocess::Run(args, out);
}

nlohmann::json Versions(const fs::path& root) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& v : ListPassVersions(root)) j.push_back({v.pass, v.id, v.current, v.fingerprint, v.frames});
    return j;
}

}  // namespace

TEST_CASE("NR preview flow without Qt: prepare, run the CLI, verify, publish, find; plan and versions unchanged", "[integration][preview]") {
    ClipSpec spec;
    spec.frames = 12;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("flow");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const ProcessOptions o = Options(dir, clip);
    REQUIRE(RunProcess(o).stages.size() > 0);
    const auto root = o.passesRoot;
    const ProcessPlan plan = PlanProcess(o);
    const nlohmann::json planBefore = plan.ToJson();
    const nlohmann::json versionsBefore = Versions(root);

    // a frame with history at 5: the warm-up shortened to the clip's start - the process run's frame 5, bit for bit
    const NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::History, 5, 0, Env(clip));
    REQUIRE(job.gate.ok());
    CHECK(job.warmup == 5);
    std::string out;
    REQUIRE(Run(job, &out) == 0);
    INFO(out);
    const NrPreviewVerdict v = VerifyNrPreviewRun(job);
    REQUIRE(v.ok());
    const NrPreviewEntry e = PublishNrPreview(job, 1.0, v.cacheable);
    CHECK(e.dir == job.entryDir);
    const PassImage preview = PassReader::Open(e.ColorDir()).ReadFrame(5);
    const PassImage processed = PassReader::Open(root / "color_nr").ReadFrame(5);
    CHECK(PassImage::MaxAbsDiff(preview.ConvertNormalizedTo(PixelType::F32), processed.ConvertNormalizedTo(PixelType::F32)) == 0.0);
    REQUIRE(FindNrPreview(PrepareNrPreview(PlanProcess(o), NrPreviewMode::History, 5, 0, Env(clip))));  // the same key again: cached

    // a frame at 0 without history
    const NrPreviewJob zero = PrepareNrPreview(plan, NrPreviewMode::Frame, 0, 0, Env(clip));
    REQUIRE(zero.gate.ok());
    REQUIRE(Run(zero) == 0);
    REQUIRE(VerifyNrPreviewRun(zero).ok());
    PublishNrPreview(zero, 1.0, true);

    // nothing of the project changed: the plan, the versions, the passes' manifests; .preview is under the root only
    CHECK(PlanProcess(o).ToJson() == planBefore);
    CHECK(Versions(root) == versionsBefore);
    for (const auto& p : PassesUnderRoot(root)) CHECK(p.front() != '.');
    CHECK(NrPreviewUsageOf(root).entries == 2);
}

TEST_CASE("NR preview verification rejects a run whose input manifest changed meanwhile", "[integration][preview][regression]") {
    ClipSpec spec;
    spec.frames = 8;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("changed");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const ProcessOptions o = Options(dir, clip);
    RunProcess(o);
    const NrPreviewJob job = PrepareNrPreview(PlanProcess(o), NrPreviewMode::Frame, 3, 0, Env(clip));
    REQUIRE(job.gate.ok());
    REQUIRE(Run(job) == 0);
    // a `process` from a terminal rewrote the colour pass during the run (here: its manifest after the run, before the check)
    const fs::path manifest = o.passesRoot / "color_sr" / Manifest::kFileName;
    std::string bytes;
    {
        std::ifstream in(manifest, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    std::ofstream(manifest, std::ios::binary | std::ios::app) << "\n";
    const NrPreviewVerdict v = VerifyNrPreviewRun(job);
    CHECK(v.why == NrPreviewReject::InputChanged);
    CHECK(v.detail == "color_sr");
    std::ofstream(manifest, std::ios::binary | std::ios::trunc) << bytes;
    CHECK(VerifyNrPreviewRun(job).ok());
}

TEST_CASE("NR preview key changes with the DLL sha", "[integration][preview]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("dll");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions o = Options(dir, clip);
    RunProcess(o);
    ApplyParamSpec(o.stages, "nr.backend=ngx");
    const fs::path dlls = dir / "nvidia";
    fs::create_directories(dlls);
    std::ofstream(dlls / "nvngx_dlssnr.dll", std::ios::binary) << "a fake snippet";
    REQUIRE(_putenv_s("DLSSVID_NVIDIA_DLL_DIR", dlls.string().c_str()) == 0);
    auto key = [&] {
        ProcessPlan plan = PlanProcess(o);
        for (auto& d : plan.stages) d.error.clear();  // no GPU here: the key is what is checked
        const NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::Frame, 1, 0, Env(clip));
        REQUIRE(job.gate.ok());
        CHECK_FALSE(job.dllSha256.empty());
        return job.key;
    };
    const std::string a = key();
    CHECK(key() == a);
    std::ofstream(dlls / "nvngx_dlssnr.dll", std::ios::binary) << "another fake snippet, of another size";
    const std::string b = key();
    _putenv_s("DLSSVID_NVIDIA_DLL_DIR", "");
    CHECK(b != a);
}

// PR-10 (spec §9 of the design): the preview's command on the real model - a frame, a frame with history, and one kill during the
// model's start-up, after which the next preview works. One feature per process: no stress probe. Only with an NVIDIA GPU, the DLL and
// the driver; runs at CI idle (GPU tests).
TEST_CASE("NR preview command on the real DLL: a frame, a frame with history, and a kill during start-up", "[integration][nr][gpu][ngx][preview]") {
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("Neural Rendering needs an NVIDIA GPU");
    const NrAvailability avail = NrAvailable("ngx");
    if (!avail.available) SKIP(avail.reason);
    const NvidiaDriverVersion drv = NvidiaDriverFromUmd(dev.UmdDriverVersion());
    if (drv.valid && !drv.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor)) SKIP("driver " + drv.ToString() + " is older than 616.56 (TASK-0011)");
    ClipSpec spec;
    spec.frames = 12;
    spec.width = 256;
    spec.height = 144;
    const auto dir = Dir("ngx");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ProcessOptions upstream = Options(dir, clip);  // the passes the preview reads, NR not run
    upstream.warp = false;
    for (auto& s : upstream.stages) s.enabled = s.name != "fg" && s.name != "nr";
    RunProcess(upstream);
    ProcessOptions o = Options(dir, clip);  // the project's plan: NR on NGX
    o.warp = false;
    ApplyParamSpec(o.stages, "nr.backend=ngx");
    const ProcessPlan plan = PlanProcess(o);
    NrPreviewEnv env = Env(clip);
    env.warp = false;
    auto run = [&](NrPreviewMode mode, int64_t first) {
        const NrPreviewJob job = PrepareNrPreview(plan, mode, first, 0, env);
        REQUIRE(job.gate.ok());
        CHECK_FALSE(job.dllSha256.empty());
        std::string out;
        REQUIRE(Run(job, &out) == 0);
        const NrPreviewVerdict v = VerifyNrPreviewRun(job);
        REQUIRE(v.ok());
        CHECK(v.cacheable);  // the CLI loaded the DLL the plan hashed
        return PublishNrPreview(job, 1.0, v.cacheable);
    };
    const NrPreviewEntry frame = run(NrPreviewMode::Frame, 5);
    CHECK(frame.dir.filename().string().size() == 12);
    CHECK(Manifest::Load(frame.ColorDir()).firstFrame == 5);
    const NrPreviewEntry history = run(NrPreviewMode::History, 8);
    CHECK(history.warmup == kNrPreviewHistoryFrames);
    CHECK(history.stats["frames"] == 7);
    // one kill while the model starts (the GUI's «Отменить», processing preempting it): the next preview runs as before
    {
        const NrPreviewJob job = PrepareNrPreview(plan, NrPreviewMode::Frame, 3, 0, env);
        WriteNrPreviewParams(job);
        std::vector<std::string> args = job.Args();
        args.insert(args.begin(), DLSSVID_CLI_PATH);
        Subprocess p;
        p.Start(args, {}, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // NGX init and CreateFeature take ≈ 1.6-2 s (E8)
        p.Terminate();
        p.Wait(10000);
        std::error_code ec;
        std::filesystem::remove_all(job.workDir, ec);
    }
    const NrPreviewEntry after = run(NrPreviewMode::Frame, 3);
    CHECK(Manifest::Load(after.ColorDir()).frameCount == 1);
}

