// `dlssvid process` (project-driven and with parameters) and `dlssvid bench` as processes (stage 8).

#include <catch2/catch_test_macros.hpp>

#include <windows.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <vector>

#include "TestClips.h"
#include "passes/Manifest.h"
#include "passes/PassSequence.h"
#include "util/Subprocess.h"
#include "viewport/Project.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

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

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "cli_process" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

nlohmann::json ReadJson(const std::filesystem::path& p) {
    std::ifstream in(p);
    nlohmann::json j;
    in >> j;
    return j;
}

}  // namespace

TEST_CASE("dlssvid process --project runs the saved stages and encodes the result with audio", "[integration][cli][process]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    spec.audio = true;
    const auto dir = Dir("project");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    Project p = Project::Create(clip, dir / "passes");
    for (auto& st : p.stages) {
        if (st.name == "depth" || st.name == "flow" || st.name == "nr") st.params["backend"] = "stub";
        if (st.name == "upscale") st.params["backend"] = "nis";
        if (st.name == "fg") st.params["backend"] = "blend";
    }
    p.resultVideo = dir / "result.mkv";
    p.Save(dir / "clip.dlssvid.json");
    const auto json = dir / "report.json";
    const CliResult r = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "--warp", "--codec", "ffv1", "--json", json.string()});
    CHECK(r.code == 0);
    CHECK(r.out.find("result:") != std::string::npos);
    REQUIRE(std::filesystem::exists(dir / "result.mkv"));
    CHECK(DecodeAll(dir / "result.mkv").size() == 7);
    CHECK(Summarize(dir / "result.mkv").audioStreams == 1);
    const nlohmann::json j = ReadJson(json);
    CHECK(j["frames_out"] == 7);
    CHECK(j["audio"] == true);
    CHECK(j["stages"].size() == 6);
    CHECK(std::filesystem::exists(dir / "passes" / "color_fg" / Manifest::kFileName));
    // the same project again: passes reused
    const CliResult again = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "--warp", "--codec", "ffv1", "--json", json.string()});
    CHECK(again.code == 0);
    const nlohmann::json j2 = ReadJson(json);
    CHECK(j2["stages"][0]["status"] == "reused");
    CHECK(j2["stages"][4]["status"] == "reused");
    // explicit parameters override the project (x3 needs a fresh color_fg: --no-skip-existing)
    const CliResult x3 = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "-o", (dir / "x3.mkv").string(), "--multiplier", "3", "--warp", "--codec", "ffv1",
                                 "--no-skip-existing", "--frames", "3"});
    CHECK(x3.code == 0);
    CHECK(DecodeAll(dir / "x3.mkv").size() == 7);
    // bad spec / unknown stage
    CHECK(RunCli({"process", "-i", clip.string(), "-o", (dir / "bad.mkv").string(), "--warp", "--stages", "bogus"}).code != 0);
    CHECK(RunCli({"process", "-i", clip.string(), "-o", (dir / "bad2.mkv").string(), "--warp", "--param", "nr=1"}).code != 0);
}

// A project as a preset for another video (TASK-0043, found while studying Visual Enhancer): the other file got the
// project's cached source hash, so the project's passes were reused for it and its result was written over the project's.
TEST_CASE("dlssvid process --project with another -i hashes that video and keeps the project's passes and result", "[integration][cli][process][regression]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("project_other_input");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    ClipSpec otherSpec = spec;
    otherSpec.width = 64;  // other content
    const auto other = WriteClip(dir / "other" / "other.mkv", otherSpec);
    Project p = Project::Create(clip, dir / "passes");
    for (auto& st : p.stages) {
        if (st.name == "depth" || st.name == "flow" || st.name == "nr") st.params["backend"] = "stub";
        if (st.name == "upscale") st.params["backend"] = "nis";
        if (st.name == "fg") st.params["backend"] = "blend";
    }
    p.resultVideo = dir / "result.mkv";
    REQUIRE(!p.SourceHash().empty());  // the GUI saves the hash of the video it opened
    const auto projectFile = dir / "clip.dlssvid.json";
    p.Save(projectFile);
    REQUIRE(RunCli({"process", "--project", projectFile.string(), "--warp", "--codec", "ffv1"}).code == 0);
    const auto resultTime = std::filesystem::last_write_time(dir / "result.mkv");
    const std::string projectHash = Manifest::Load(dir / "passes" / "color_nr").sourceHash;
    REQUIRE(!projectHash.empty());

    const auto json = dir / "other.json";
    const CliResult r = RunCli({"process", "--project", projectFile.string(), "-i", other.string(), "--warp", "--codec", "ffv1", "--json", json.string()});
    CHECK(r.code == 0);
    const nlohmann::json j = ReadJson(json);
    for (const auto& s : j["stages"]) CHECK(s["status"] != "reused");  // before: every stage reused the project's passes
    // its own result and passes next to it; the project's are untouched
    const auto otherResult = dir / "other" / "other_result.mkv";
    REQUIRE(std::filesystem::exists(otherResult));
    CHECK(DecodeAll(otherResult).size() == 5);
    const Manifest otherNr = Manifest::Load(dir / "other" / "other_result_passes" / "color_nr");
    CHECK(otherNr.sourceHash != projectHash);
    CHECK(otherNr.width == 128);  // 64 x2: the other video's frames
    CHECK(std::filesystem::last_write_time(dir / "result.mkv") == resultTime);
    CHECK(Manifest::Load(dir / "passes" / "color_nr").sourceHash == projectHash);

    // the project's own video moved elsewhere (the old path is gone) is still the project's: its passes are reused
    std::filesystem::rename(clip, dir / "moved.mkv");
    const CliResult moved = RunCli({"process", "--project", projectFile.string(), "-i", (dir / "moved.mkv").string(), "--warp", "--codec", "ffv1",
                                    "-o", (dir / "moved_result.mkv").string(), "--json", json.string()});
    CHECK(moved.code == 0);
    CHECK(ReadJson(json)["stages"][3]["status"] == "reused");  // nr
}

// TASK-0045: an imported mask has no fingerprint, it stands for its source and length - a mask re-imported with other
// content and as many frames reused the NR pass made from the old one. `import --stamp` fingerprints the content.
TEST_CASE("dlssvid import --stamp: a mask re-imported with other content recomputes NR", "[integration][cli][process][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("mask_stamp");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto passes = dir / "passes";
    auto writeMask = [&](uint8_t value) {
        PassWriter w(passes / "mask_skin", Manifest::ForPass(PassKind::Mask, 16, 8, FileFormat::Png));
        for (int64_t f = 0; f < 2; ++f) {
            PassImage m = MakePassImage(PassKind::Mask, 16, 8, PixelType::U8);
            std::fill(m.data.begin(), m.data.end(), value);
            w.WriteFrame(f, m);
        }
        w.Finish();
    };
    const std::vector<std::string> process{"process", "-i", clip.string(), "-o", (dir / "result.mkv").string(), "--passes", passes.string(), "--warp", "--codec", "ffv1",
                                           "--stages", "nr", "--nr-backend", "stub", "--json", (dir / "r.json").string()};
    writeMask(255);
    REQUIRE(RunCli({"import", "-i", (passes / "mask_skin").string(), "--stamp"}).code == 0);
    const std::string stamped = Manifest::Load(passes / "mask_skin").fingerprint;
    CHECK(stamped.rfind("sha256:", 0) == 0);
    REQUIRE(RunCli(process).code == 0);
    REQUIRE(RunCli(process).code == 0);
    CHECK(ReadJson(dir / "r.json")["stages"][0]["status"] == "reused");
    writeMask(64);  // other content, the same frames
    REQUIRE(RunCli({"import", "-i", (passes / "mask_skin").string(), "--stamp"}).code == 0);
    CHECK(Manifest::Load(passes / "mask_skin").fingerprint != stamped);
    REQUIRE(RunCli(process).code == 0);
    const nlohmann::json j = ReadJson(dir / "r.json");
    CHECK(j["stages"][0]["status"] == "ran");
    CHECK(j["stages"][0]["reason"].get<std::string>().find("input") != std::string::npos);
}

TEST_CASE("dlssvid process --passthrough and dlssvid bench", "[integration][cli][process]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("bench");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const CliResult pt = RunCli({"process", "-i", clip.string(), "-o", (dir / "pt.mkv").string(), "--passthrough", "--warp", "--codec", "ffv1"});
    CHECK(pt.code == 0);
    CHECK(DecodeAll(dir / "pt.mkv").size() == 5);
    const auto json = dir / "bench.json";
    const CliResult b = RunCli({"bench", "-i", clip.string(), "--frames", "3", "--warp", "--codec", "ffv1", "--param", "depth.backend=stub", "--param", "flow.backend=stub", "--param",
                                "upscale.backend=nis", "--param", "nr.backend=stub", "--param", "fg.backend=blend", "--json", json.string()});
    CHECK(b.code == 0);
    CHECK(b.out.find("GPU:") != std::string::npos);
    CHECK(b.out.find("ms/frame") != std::string::npos);
    const nlohmann::json j = ReadJson(json);
    CHECK(j["frames_in"] == 3);
    CHECK(j["stages"].size() == 6);
    CHECK(j["stages"][5]["name"] == "encode");
    CHECK(j.contains("gpu"));
    CHECK(j["stages"][2]["ms_per_frame"].get<double>() >= 0.0);
}

// Audit fg-2 (TASK-0038): FG runs in its own process; a real death of that process (TerminateProcess 0xC0000005, as a
// driver fault would end it) fails only the stage - the result is encoded from color_nr.
namespace {
const nlohmann::json& StageJson(const nlohmann::json& report, const std::string& name) {
    for (const auto& s : report["stages"])
        if (s["name"] == name) return s;
    throw std::runtime_error("no stage " + name);
}

std::vector<std::string> StubProcessArgs(const std::filesystem::path& clip, const std::filesystem::path& out, const std::filesystem::path& passes) {
    return {"process", "-i", clip.string(), "-o", out.string(), "--passes", passes.string(), "--warp", "--codec", "ffv1", "--param", "depth.backend=stub",
            "--param", "flow.backend=stub", "--param", "upscale.backend=nis", "--param", "nr.backend=stub", "--param", "fg.backend=blend"};
}

bool EndsWith(const std::string& s, const std::string& tail) { return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0; }
}  // namespace

TEST_CASE("dlssvid process: a crashing fg process fails only its stage: the result is encoded from color_nr and the report says fg failed",
          "[integration][cli][process][fg][isolation]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 48;
    spec.height = 32;
    spec.audio = true;
    const auto dir = Dir("fg_crash");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto passes = dir / "p";

    std::vector<std::string> args = StubProcessArgs(clip, dir / "crash.mkv", passes);
    args.insert(args.end(), {"--disable-unavailable", "--fg-crash-after", "2", "--json", (dir / "crash.json").string()});
    const CliResult r = RunCli(args);
    CHECK(r.code == 0);  // before the fix the whole run died with 0xC0000005
    CHECK(r.out.find("fg failed") != std::string::npos);
    CHECK(r.out.find("warning:") != std::string::npos);
    const nlohmann::json j = ReadJson(dir / "crash.json");
    const nlohmann::json& fg = StageJson(j, "fg");
    CHECK(fg["status"] == "failed");
    CHECK(fg["details"]["isolated"] == true);
    CHECK(fg["details"]["crashed"] == true);
    CHECK(fg["details"]["exit_code"] == 3221225477u);
    CHECK(fg["details"]["partial"] == "removed");
    CHECK(fg["reason"].get<std::string>().find("0xC0000005") != std::string::npos);
    CHECK(fg["fingerprint"] == "");
    const nlohmann::json& enc = StageJson(j, "encode");
    CHECK(enc["status"] == "ran");
    CHECK(EndsWith(enc["out_dir"].get<std::string>(), "color_nr"));
    CHECK(EndsWith(j["final_pass"].get<std::string>(), "color_nr"));
    CHECK(j["frames_out"] == 6);
    CHECK(j["audio"] == true);
    CHECK(j["warnings"].size() == 1);
    CHECK(DecodeAll(dir / "crash.mkv").size() == 6);
    CHECK(!std::filesystem::exists(passes / "color_fg"));    // the debris of the dead process is gone
    CHECK(!std::filesystem::exists(passes / "color_fg.v"));  // and did not go to the history

    // strict (no --disable-unavailable): the run fails as before, no result
    std::vector<std::string> strict = StubProcessArgs(clip, dir / "strict.mkv", passes);
    strict.insert(strict.end(), {"--fg-crash-after", "2"});
    const CliResult s = RunCli(strict);
    CHECK(s.code == 1);
    CHECK(s.out.find("stage 'fg' failed") != std::string::npos);
    CHECK(!std::filesystem::exists(dir / "strict.mkv"));
    CHECK(!std::filesystem::exists(passes / "color_fg"));

    // without the hook: the stages above are reused, fg runs in its own process, timing without its start-up
    std::vector<std::string> ok = StubProcessArgs(clip, dir / "ok.mkv", passes);
    ok.insert(ok.end(), {"--disable-unavailable", "--json", (dir / "ok.json").string()});
    const CliResult o = RunCli(ok);
    CHECK(o.code == 0);
    const nlohmann::json jo = ReadJson(dir / "ok.json");
    for (const char* name : {"depth", "flow", "upscale", "nr"}) CHECK(StageJson(jo, name)["status"] == "reused");
    const nlohmann::json& fgOk = StageJson(jo, "fg");
    CHECK(fgOk["status"] == "ran");
    CHECK(fgOk["details"]["isolated"] == true);
    CHECK(fgOk["details"]["startup_seconds"].get<double>() >= 0.0);
    CHECK(fgOk["ms_per_frame"].get<double>() <= 1000.0 * fgOk["seconds"].get<double>() / fgOk["frames"].get<double>() + 1e-6);
    CHECK(jo["frames_out"] == 11);
    CHECK(jo["warnings"].empty());
    CHECK(!Manifest::Load(passes / "color_fg").fingerprint.empty());
}

TEST_CASE("dlssvid process: an fg process that crashes after writing its result keeps color_fg", "[integration][cli][process][fg][isolation]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("fg_crash_on_exit");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    std::vector<std::string> args = StubProcessArgs(clip, dir / "r.mkv", dir / "p");
    args.insert(args.end(), {"--disable-unavailable", "--fg-crash-on-exit", "--json", (dir / "r.json").string()});
    const CliResult r = RunCli(args);
    CHECK(r.code == 0);
    const nlohmann::json j = ReadJson(dir / "r.json");
    CHECK(StageJson(j, "fg")["status"] == "ran");  // a complete pass is not thrown away for a crash on exit
    REQUIRE(j["warnings"].size() == 1);
    CHECK(j["warnings"][0].get<std::string>().find("crashed on exit") != std::string::npos);
    CHECK(EndsWith(j["final_pass"].get<std::string>(), "color_fg"));
    CHECK(j["frames_out"] == 11);
    CHECK(!Manifest::Load(dir / "p" / "color_fg").fingerprint.empty());
}

TEST_CASE("dlssvid process: killing the run ends its fg process too", "[integration][cli][process][fg][isolation]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("fg_kill");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    Subprocess run;
    run.Start({DLSSVID_CLI_PATH, "process", "-i", clip.string(), "-o", (dir / "o.mkv").string(), "--passes", (dir / "p").string(), "--warp", "--codec", "ffv1",
               "--stages", "fg", "--param", "fg.backend=blend", "--fg-stall"});
    const std::regex started(R"(fg runs in its own process \(pid (\d+)\))");
    DWORD pid = 0;
    const ULONGLONG deadline = GetTickCount64() + 60000;
    while (!pid && GetTickCount64() < deadline) {
        const auto line = run.ReadLine(1000);
        if (!line) {
            if (!run.Running()) break;
            continue;
        }
        std::smatch m;
        if (std::regex_search(*line, m, started)) pid = static_cast<DWORD>(std::stoul(m[1].str()));
    }
    REQUIRE(pid != 0);
    HANDLE child = OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, pid);
    REQUIRE(child != nullptr);
    run.Terminate();  // as TaskQueue::cancel does
    const DWORD waited = WaitForSingleObject(child, 10000);
    if (waited != WAIT_OBJECT_0) TerminateProcess(child, 1);  // clean up before failing
    CloseHandle(child);
    CHECK(waited == WAIT_OBJECT_0);  // without the job object the stalled child would live on
}

// TASK-0039: the depth model is checked against the registry before anything is decoded or hashed - before, an unknown
// id listed the whole registry after the source hash, a DA3 id for vda reached the Python export (KeyError 'hf_file').
TEST_CASE("dlssvid depth refuses an unknown or foreign VDA model and a bad window before decoding", "[integration][cli][depth][models]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("vda_refuse");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const std::string models = (std::filesystem::path(DLSSVID_SOURCE_DIR) / "models").string();
    const auto out = dir / "passes";
    auto depth = [&](std::vector<std::string> extra) {
        std::vector<std::string> args{"depth", "--warp", "--models-dir", models, "-i", clip.string(), "-o", out.string()};
        args.insert(args.end(), extra.begin(), extra.end());
        return RunCli(args);
    };
    const CliResult unknown = depth({"--backend", "vda", "--model", "metric-vda-huge"});
    CHECK(unknown.code != 0);
    CHECK(unknown.out.find("metric-vda-huge") != std::string::npos);
    CHECK(unknown.out.find("metric-vda-small") != std::string::npos);
    CHECK(unknown.out.find("metric-vda-large") != std::string::npos);
    CHECK(unknown.out.find("sea-raft") == std::string::npos);  // the VDA models, not the whole registry
    const CliResult foreign = depth({"--backend", "vda", "--model", "da3metric-large"});
    CHECK(foreign.code != 0);
    CHECK(foreign.out.find("not a VDA model") != std::string::npos);
    CHECK(depth({"--backend", "vda", "--window", "40"}).code != 0);  // out of range
    const CliResult bad = depth({"--backend", "vda", "--window", "10"});
    CHECK(bad.code != 0);
    CHECK(bad.out.find("multiple of 4") != std::string::npos);
    CHECK(!std::filesystem::exists(out / "depth_raw"));  // refused before the decoder and the stage
}

// TASK-0039: `--depth-backend da3` over a project with VDA Large leaves no VDA model (it would be "not a DA3 model"),
// and the plan prints the advice of the stages: the licence of a non-commercial model.
TEST_CASE("dlssvid process --plan: a backend switch drops the VDA model, a non-commercial model is a warning", "[integration][cli][process][models]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("vda_plan");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const std::string models = (std::filesystem::path(DLSSVID_SOURCE_DIR) / "models").string();
    Project p = Project::Create(clip, dir / "passes");
    for (auto& st : p.stages) {
        st.enabled = st.name == "depth";
        if (st.name == "depth") st.params = {{"backend", "vda"}, {"model", "metric-vda-large"}, {"models_dir", models}};
    }
    p.Save(dir / "clip.dlssvid.json");
    const auto json = dir / "plan.json";
    const CliResult nc = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "--plan", "--warp", "--json", json.string()});
    CHECK(nc.code == 0);
    CHECK(nc.out.find("warning: depth: metric-vda-large") != std::string::npos);
    CHECK(nc.out.find("CC-BY-NC-4.0") != std::string::npos);
    const CliResult da3 = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "--plan", "--warp", "--depth-backend", "da3", "--json", json.string()});
    CHECK(da3.code == 0);  // before: "process: depth: 'metric-vda-large' is not a DA3 model"
    const nlohmann::json plan = ReadJson(json);
    REQUIRE(plan["stages"].size() == 1);
    CHECK(plan["stages"][0]["params"]["backend"] == "da3");
    CHECK(plan["stages"][0]["params"]["model"] == "");  // the schema default, not the VDA model
    CHECK(plan["stages"][0]["warnings"].empty());
}
