// `dlssvid process` (project-driven and with parameters) and `dlssvid bench` as processes (stage 8).

#include <catch2/catch_test_macros.hpp>

#include <windows.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <vector>

#include "TestClips.h"
#include "io/VideoDecoder.h"
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

// a stage of a plan or run report by name: the indices move when a stage is added (PR-13: scenes first)
const nlohmann::json& StageJson(const nlohmann::json& report, const std::string& name) {
    for (const auto& s : report["stages"])
        if (s["name"] == name) return s;
    throw std::runtime_error("no stage " + name);
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
    CHECK(j["stages"].size() == 7);  // scenes .. fg + encode
    CHECK(std::filesystem::exists(dir / "passes" / "color_fg" / Manifest::kFileName));
    // PR-11: the result says what made it - the comment in every container, the JSON in Matroska; dlssvid info prints both
    {
        const auto tags = ReadContainerTags(dir / "result.mkv");
        REQUIRE(tags.count("comment"));
        const std::string note = tags.at("comment");
        CHECK(note.rfind("VideoDLSS ", 0) == 0);
        CHECK(note.find("| upscale: nis x2") != std::string::npos);
        CHECK(note.find("| nr: stub") != std::string::npos);
        CHECK(note.find("| fg: blend x2") != std::string::npos);
        CHECK(note.find(dir.string()) == std::string::npos);
        REQUIRE(tags.count("dlssvid_settings"));
        CHECK(nlohmann::json::parse(tags.at("dlssvid_settings"))["stages"]["nr"]["backend"] == "stub");
        const CliResult info = RunCli({"info", (dir / "result.mkv").string(), "--warp"});
        CHECK(info.code == 0);
        CHECK(info.out.find("comment:     VideoDLSS ") != std::string::npos);
        CHECK(info.out.find("settings:    {") != std::string::npos);
    }
    // the same project again: passes reused
    const CliResult again = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "--warp", "--codec", "ffv1", "--json", json.string()});
    CHECK(again.code == 0);
    const nlohmann::json j2 = ReadJson(json);
    CHECK(StageJson(j2, "depth")["status"] == "reused");
    CHECK(StageJson(j2, "fg")["status"] == "reused");
    // explicit parameters override the project (x3 needs a fresh color_fg: --no-skip-existing)
    const CliResult x3 = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "-o", (dir / "x3.mkv").string(), "--multiplier", "3", "--warp", "--codec", "ffv1",
                                 "--no-skip-existing", "--frames", "3"});
    CHECK(x3.code == 0);
    CHECK(DecodeAll(dir / "x3.mkv").size() == 7);
    CHECK(ReadContainerTags(dir / "x3.mkv").at("comment").find("| fg: blend x3") != std::string::npos);  // what ran, not the project
    // --no-settings-note, and encode.settings_note false in the project: none
    CHECK(RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "-o", (dir / "plain.mkv").string(), "--warp", "--codec", "ffv1",
                  "--no-settings-note"}).code == 0);
    CHECK_FALSE(ReadContainerTags(dir / "plain.mkv").count("comment"));
    CHECK_FALSE(ReadContainerTags(dir / "plain.mkv").count("dlssvid_settings"));
    p.settingsNote = false;
    p.Save(dir / "quiet.dlssvid.json");
    CHECK(RunCli({"process", "--project", (dir / "quiet.dlssvid.json").string(), "-o", (dir / "quiet.mkv").string(), "--warp", "--codec", "ffv1"}).code == 0);
    CHECK_FALSE(ReadContainerTags(dir / "quiet.mkv").count("comment"));
    // MOV (ProRes): the comment the mov muxer keeps (spec §2g: «по факту теста»); no Matroska JSON there
    CHECK(RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "-o", (dir / "result.mov").string(), "--warp", "--codec", "prores_ks"}).code == 0);
    const auto mov = ReadContainerTags(dir / "result.mov");
    CHECK(mov.count("comment"));
    CHECK_FALSE(mov.count("dlssvid_settings"));
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
    CHECK(StageJson(ReadJson(json), "nr")["status"] == "reused");
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

// TASK-0045, review: `import --stamp` saved the manifest as the run saw it (--y-up / --near / --far / --relative are for
// the run only), hashed the frames but not how they are read (the same files with mv.y_up are other vectors) and, with
// --to-dlss, stamped the source instead of the converted pass the stages read.
TEST_CASE("dlssvid import --stamp: the fingerprint alone, how the files are read, the converted pass", "[integration][cli][passes][regression]") {
    const auto dir = Dir("stamp_layout");
    auto writePass = [&](PassKind kind, const std::filesystem::path& to) {
        PassWriter w(to, Manifest::ForPass(kind, 16, 8, FileFormat::Exr));
        for (int64_t f = 0; f < 2; ++f) w.WriteFrame(f, MakePassImage(kind, 16, 8));
        w.Finish();
    };
    const auto mv = dir / "mv_dlss";
    writePass(PassKind::MvDlss, mv);
    const std::string before = Manifest::Load(mv).ToJson().dump();
    REQUIRE(RunCli({"import", "-i", mv.string(), "--stamp", "--y-up"}).code == 0);
    Manifest m = Manifest::Load(mv);
    const std::string stamped = m.fingerprint;
    CHECK(stamped.rfind("sha256:", 0) == 0);
    CHECK_FALSE(m.mv.yUp);  // before: --y-up written into manifest.json
    m.fingerprint.clear();
    CHECK(m.ToJson().dump() == before);  // the fingerprint is the only change
    m.mv.yUp = true;  // the same files read y up
    m.Save(mv);
    REQUIRE(RunCli({"import", "-i", mv.string(), "--stamp"}).code == 0);
    CHECK(Manifest::Load(mv).fingerprint != stamped);  // before: the same fingerprint, the stages below reused

    const auto raw = dir / "mv_raw";
    const auto converted = dir / "converted";
    writePass(PassKind::MvRaw, raw);
    REQUIRE(RunCli({"import", "-i", raw.string(), "--to-dlss", converted.string(), "--stamp"}).code == 0);
    CHECK(Manifest::Load(raw).fingerprint.empty());  // the source is left as it was
    CHECK(Manifest::Load(converted).fingerprint.rfind("sha256:", 0) == 0);
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
    CHECK(j["stages"].size() == 7);
    CHECK(j["stages"].back()["name"] == "encode");
    CHECK(j.contains("gpu"));
    CHECK(StageJson(j, "upscale")["ms_per_frame"].get<double>() >= 0.0);
}

// Audit fg-2 (TASK-0038): FG runs in its own process; a real death of that process (TerminateProcess 0xC0000005, as a
// driver fault would end it) fails only the stage - the result is encoded from color_nr.
namespace {
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

// PR-9: --param-file - the stages in layers. Here the card files: a card preset replaces the form's keys (the project's engineer keys
// stay), a bare object merges over it, in command-line order; the backend options and --param win; a file switches its stage on, also
// one --stages leaves out
namespace {
Project StubProject(const std::filesystem::path& clip, const std::filesystem::path& dir) {
    Project p = Project::Create(clip, dir / "passes");
    for (auto& st : p.stages) {
        if (st.name == "depth" || st.name == "flow" || st.name == "nr") st.params["backend"] = "stub";
        if (st.name == "upscale") st.params["backend"] = "nis";
        if (st.name == "fg") st.params["backend"] = "blend";
    }
    return p;
}

std::string WriteJsonFile(const std::filesystem::path& p, const nlohmann::json& j) {
    std::ofstream(p, std::ios::binary) << j.dump(2);
    return p.string();
}

nlohmann::json PresetJson(const char* scope, const char* name, const nlohmann::json& stages) {
    return {{"format", "dlssvid-preset"}, {"schema", 1}, {"scope", scope}, {"name", name}, {"stages", stages}};
}

nlohmann::json NrCard(const nlohmann::json& params) {
    nlohmann::json s = nlohmann::json::object();
    s["nr"] = {{"params", params}};
    return s;
}

// a pipeline preset over the stub methods: scenes at 0.3, NIS with a sharpness, nr at 0.6, fg off, ProRes 10-bit 4:2:2
nlohmann::json AnimePreset() {
    nlohmann::json stages = nlohmann::json::object();
    stages["scenes"] = {{"enabled", true}, {"params", {{"threshold", 0.3}}}};
    stages["depth"] = {{"enabled", true}, {"params", {{"backend", "stub"}}}};
    stages["flow"] = {{"enabled", true}, {"params", {{"backend", "stub"}}}};
    stages["upscale"] = {{"enabled", true}, {"params", {{"backend", "nis"}, {"scale", 2}, {"sharpness", 0.8}}}};
    stages["nr"] = {{"enabled", true}, {"params", {{"intensity", 0.6}, {"backend", "stub"}}}};
    stages["fg"] = {{"enabled", false}, {"params", {{"backend", "blend"}, {"multiplier", 2}}}};
    nlohmann::json j = PresetJson("pipeline", "Anime", stages);
    j["encode"] = {{"codec", "prores_ks"}, {"bit_depth", "10"}, {"chroma", "422"}, {"options", nlohmann::json::object()}};
    return j;
}
}  // namespace

TEST_CASE("dlssvid process --param-file: project < stages < files < backend options < param; a preset keeps the project's engineer keys; the stage is switched on",
          "[integration][cli][process][presets]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("param_file");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    Project p = StubProject(clip, dir);
    for (auto& st : p.stages)
        if (st.name == "nr") {
            st.enabled = false;
            st.params = {{"backend", "ngx"}, {"format", "png"}, {"intensity", 1.4}, {"exposure", 1.0}};
        }
    p.Save(dir / "p.dlssvid.json");
    const std::string card = WriteJsonFile(dir / "card.json", PresetJson("stage", "Soft", NrCard({{"intensity", 0.7}, {"local_tone", 0.5}})));
    const std::string bare = WriteJsonFile(dir / "bare.json", {{"temporal", 0.3}, {"intensity", 0.9}});
    const auto json = dir / "plan.json";
    const CliResult r = RunCli({"process", "--project", (dir / "p.dlssvid.json").string(), "--plan", "--warp", "--json", json.string(), "--stages", "depth,flow,upscale",
                                "--param-file", "nr=" + card, "--param-file", "nr=" + bare, "--param", "nr.passes=2", "--nr-backend", "stub"});
    INFO(r.out);
    REQUIRE(r.code == 0);
    const nlohmann::json plan = ReadJson(json);
    nlohmann::json nr = StageJson(plan, "nr")["params"];  // on: the files switched it on, past --stages and the project
    CHECK(nr["intensity"] == 0.9);   // the bare file, after the card preset
    CHECK(nr["local_tone"] == 0.5);  // the card preset
    CHECK(nr["temporal"] == 0.3);
    CHECK(nr["exposure"] == 0.0);    // the card preset holds the card: the project's 1 EV is back at its default
    CHECK(nr["passes"] == 2);        // --param
    CHECK(nr["format"] == "png");    // an engineer key of the project stays
    CHECK(nr["backend"] == "stub");  // --nr-backend over the project's ngx
    CHECK_THROWS(StageJson(plan, "fg"));  // --stages left it out
    // without stage=: a card preset goes to its stage
    const CliResult c = RunCli({"process", "--project", (dir / "p.dlssvid.json").string(), "--plan", "--warp", "--json", json.string(), "--param-file", card,
                                "--nr-backend", "stub"});
    INFO(c.out);
    REQUIRE(c.code == 0);
    CHECK(c.out.find("a card preset of nr") != std::string::npos);
    CHECK(StageJson(ReadJson(json), "nr")["params"]["intensity"] == 0.7);
}

TEST_CASE("dlssvid process --param-file with a pipeline preset: enabled, settings and the encoder from it; --stages, stage files, backend options, --codec, --bit-depth and --param win",
          "[integration][cli][process][presets]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("pipeline_preset");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    Project p = StubProject(clip, dir);
    p.resultVideo = dir / "clip_result.mp4";
    p.codecOptions = {{"preset", "p7"}, {"b", "20M"}};
    for (auto& st : p.stages)
        if (st.name == "depth") st.params["max_res"] = 512;  // a key of the machine: no preset touches it
    p.Save(dir / "p.dlssvid.json");
    std::string projectText;
    {
        std::ifstream in(dir / "p.dlssvid.json", std::ios::binary);
        projectText.assign(std::istreambuf_iterator<char>(in), {});
    }
    const std::string anime = WriteJsonFile(dir / "anime.json", AnimePreset());
    const std::string soft = WriteJsonFile(dir / "soft.json", PresetJson("stage", "Soft", NrCard({{"intensity", 0.8}})));
    const auto json = dir / "plan.json";
    auto plan = [&](std::vector<std::string> extra) {
        std::vector<std::string> args{"process", "--project", (dir / "p.dlssvid.json").string(), "--plan", "--warp", "--json", json.string()};
        args.insert(args.end(), extra.begin(), extra.end());
        const CliResult r = RunCli(args);
        INFO(r.out);
        REQUIRE(r.code == 0);
        return std::pair{ReadJson(json), r.out};
    };
    {  // the stage file written before the pipeline preset still wins: the preset is layer 2 wherever it stands
        const auto [j, out] = plan({"--param-file", "nr=" + soft, "--param-file", anime});
        CHECK_THROWS(StageJson(j, "fg"));  // switched off by the preset
        CHECK(StageJson(j, "nr")["params"]["intensity"] == 0.8);
        CHECK(StageJson(j, "upscale")["params"]["sharpness"] == 0.8);
        CHECK(StageJson(j, "scenes")["params"]["threshold"] == 0.3);
        CHECK(StageJson(j, "depth")["params"]["max_res"] == 512);
        CHECK(j["encode"]["codec"] == "prores_ks");
        CHECK(j["encode"]["bit_depth"] == "10");
        CHECK(j["encode"]["chroma"] == "422");
        CHECK(j["encode"]["options"] == nlohmann::json{{"preset", "p7"}});  // the preset's «Авто» bitrate; the project's other option stays
        CHECK(j["encode"]["settings_note"] == true);
        CHECK(std::filesystem::path(j["encode"]["output"].get<std::string>()).filename() == "clip_result.mov");
        CHECK(out.find("encode: prores_ks, bit depth 10, chroma 422") != std::string::npos);
    }
    {  // the flags over the preset; the container follows the codec that encodes
        const auto [j, out] = plan({"--param-file", anime, "--codec", "ffv1", "--bit-depth", "12", "--param", "nr.intensity=0.5"});
        CHECK(j["encode"]["codec"] == "ffv1");
        CHECK(j["encode"]["bit_depth"] == "12");
        CHECK(std::filesystem::path(j["encode"]["output"].get<std::string>()).filename() == "clip_result.mkv");
        CHECK(StageJson(j, "nr")["params"]["intensity"] == 0.5);
    }
    {
        const auto [j, out] = plan({"--param-file", anime, "--codec", "h264_nvenc", "--bit-depth", "auto", "--chroma", "auto", "--stages",
                                    "scenes,depth,flow,upscale,nr,fg", "--fg-backend", "blend"});
        CHECK(std::filesystem::path(j["encode"]["output"].get<std::string>()).filename() == "clip_result.mp4");
        CHECK(StageJson(j, "fg")["params"]["backend"] == "blend");  // --stages switched it back on
    }
    {  // --codec-opt replaces the options, as with a project
        const auto [j, out] = plan({"--param-file", anime, "--codec-opt", "b=30M"});
        CHECK(j["encode"]["options"] == nlohmann::json{{"b", "30M"}});
    }
    std::ifstream in(dir / "p.dlssvid.json", std::ios::binary);
    CHECK(std::string(std::istreambuf_iterator<char>(in), {}) == projectText);  // the project file is not changed
}

TEST_CASE("dlssvid process and bench --param-file refuse: an unknown stage, an unknown key of a bare file, a bare object without stage=, two pipeline presets, a preset without the stage, a value out of range, an impossible encode",
          "[integration][cli][process][presets][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("param_file_refused");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const std::string bare = WriteJsonFile(dir / "bare.json", {{"intensty", 1.0}});
    const std::string anime = WriteJsonFile(dir / "anime.json", AnimePreset());
    const std::string card = WriteJsonFile(dir / "card.json", PresetJson("stage", "Soft", NrCard({{"intensity", 0.8}})));
    const std::string five = WriteJsonFile(dir / "five.json", PresetJson("stage", "Five", NrCard({{"intensity", 5}})));
    nlohmann::json h264 = AnimePreset();
    h264["encode"] = {{"codec", "h264_nvenc"}, {"bit_depth", 10}};
    const std::string h264file = WriteJsonFile(dir / "h264.json", h264);
    auto refused = [&](const char* cmd, std::vector<std::string> extra, const char* says) {
        std::vector<std::string> args{cmd, "-i", clip.string(), "--warp"};
        if (std::string(cmd) == "process") args.push_back("--plan");
        args.insert(args.end(), extra.begin(), extra.end());
        const CliResult r = RunCli(args);
        INFO(r.out);
        CHECK(r.code != 0);
        CHECK(r.out.find(says) != std::string::npos);
    };
    refused("process", {"--param-file", "export=" + card}, "export=");  // not a stage: a path, and there is no such file
    refused("process", {"--param-file", "nr=" + bare}, "'intensty' not nr settings");
    refused("process", {"--param-file", bare}, "needs stage=<file.json>");
    refused("bench", {"--param-file", bare}, "needs stage=<file.json>");
    refused("process", {"--param-file", anime, "--param-file", anime}, "one pipeline preset at a time");
    refused("process", {"--param-file", "depth=" + card}, "has no settings for depth");
    refused("process", {"--param-file", "nr=" + five}, "nr.intensity must be within");
    refused("bench", {"--param-file", five}, "nr.intensity must be within");
    refused("process", {"--param-file", h264file}, "encode.bit_depth 10: H.264 NVENC does not encode 10-bit");
    refused("process", {"--param-file", "nr="}, "no file");
}

TEST_CASE("dlssvid bench --param-file with a pipeline preset runs its stages with its encoder; --codec wins", "[integration][cli][bench][presets]") {
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("bench_preset");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    nlohmann::json preset = AnimePreset();
    preset["encode"] = {{"codec", "ffv1"}, {"bit_depth", "auto"}, {"chroma", "auto"}, {"options", nlohmann::json::object()}};
    const std::string file = WriteJsonFile(dir / "ffv1.json", preset);
    const auto json = dir / "bench.json";
    const CliResult b = RunCli({"bench", "-i", clip.string(), "--frames", "3", "--warp", "--param-file", file, "--json", json.string()});
    INFO(b.out);
    REQUIRE(b.code == 0);
    const nlohmann::json j = ReadJson(json);
    CHECK(j["output_format"]["codec"] == "ffv1");
    CHECK(StageJson(j, "nr")["status"] == "ran");  // the preset's stub
    CHECK_THROWS(StageJson(j, "fg"));  // off in the preset: not run, not in the report
    const CliResult pr = RunCli({"bench", "-i", clip.string(), "--frames", "3", "--warp", "--param-file", file, "--codec", "prores_ks", "--json", json.string()});
    INFO(pr.out);
    REQUIRE(pr.code == 0);
    CHECK(ReadJson(json)["output_format"]["codec"] == "prores_ks");
}
