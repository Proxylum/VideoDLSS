// `dlssvid nr`, `dlssvid nr --check` and `dlssvid nr-patch` as processes (stage 6).

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "TestClips.h"
#include "passes/Manifest.h"
#include "passes/PassSequence.h"
#include "util/Sha256.h"
#include "util/Subprocess.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::WithinAbs;

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
    const auto d = TempDir() / "cli_nr" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

bool HavePython() {
    static const bool ok = std::system("python -c \"pass\" >nul 2>&1") == 0;
    return ok;
}

nlohmann::json ReadJson(const std::filesystem::path& p) {
    std::ifstream in(p);
    nlohmann::json j;
    in >> j;
    return j;
}

}  // namespace

TEST_CASE("dlssvid nr --backend stub --warp writes color_nr and a JSON summary", "[integration][cli][nr]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("stub");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto passes = dir / "passes";
    const auto json = dir / "nr.json";
    const CliResult r = RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", passes.string(), "--frames", "3", "--format", "png16", "--json", json.string(),
                                "--intensity", "1.2", "--style", "cinematic", "--model-scale", "0.5", "--temporal", "0.3"});
    CHECK(r.code == 0);
    CHECK(r.out.find("color_nr:") != std::string::npos);
    CHECK(r.out.find("backend:     stub") != std::string::npos);
    CHECK(std::filesystem::exists(passes / "color_nr" / Manifest::kFileName));
    {
        // png16 colour is the full 16-bit range (TASK-0042: 0 / 1 of 65535 - black frames)
        const PassImage f = PassReader::Open(passes / "color_nr").ReadFrame(0).ConvertNormalizedTo(PixelType::F32);
        double mean = 0.0;
        for (uint32_t y = 0; y < f.height; ++y)
            for (uint32_t x = 0; x < f.width; ++x) mean += f.Get(x, y, 1);
        CHECK(mean / (static_cast<double>(f.width) * f.height) > 0.05);
    }
    const nlohmann::json j = ReadJson(json);
    CHECK(j["stats"]["frames"] == 3);
    CHECK(j["stats"]["color_source"] == "video");
    CHECK(j["stats"]["work_size"] == nlohmann::json::array({24, 16}));
    CHECK(j["backend"]["backend"] == "stub");
    CHECK(j["diagnostics"]["ok"] == true);
    // D8: the standalone run's manifest keeps the stage's GPU time only; the JSON summary has both times for nr_ab.py
    const Manifest nm = Manifest::Load(passes / "color_nr");
    CHECK(nm.stageParams["ms_per_frame_gpu"].is_number());
    CHECK_FALSE(nm.stageParams.contains("ms_per_frame"));
    CHECK(j["stats"]["ms_per_frame_wall"].get<double>() >= j["stats"]["ms_per_frame_gpu"].get<double>());
    // underscore spellings (the GUI passes project JSON keys as options) are accepted too
    const CliResult r2 = RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / "passes2").string(), "--frames", "1", "--model_scale", "1", "--no_guides"});
    CHECK(r2.code == 0);
    // missing input
    CHECK(RunCli({"nr", "--backend", "stub", "--warp", "-o", (dir / "passes3").string()}).code != 0);

    // TASK-0044: the standalone run checks every value against the schema (before: `--intensity 5` went to NGX); styles 3..6
    // are refused too since PR-2 E7 (TASK-0052)
    for (const std::vector<std::string>& bad : std::vector<std::vector<std::string>>{
             {"--intensity", "5"}, {"--passes", "5"}, {"--passes", "0"}, {"--exposure", "20"}, {"--style", "7"}, {"--style", "4"}, {"--local-tone", "3"}}) {
        std::vector<std::string> args{"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / "bad").string(), "--frames", "1"};
        args.insert(args.end(), bad.begin(), bad.end());
        const CliResult b = RunCli(args);
        INFO(bad[0] << " " << bad[1]);
        CHECK(b.code != 0);
        CHECK(b.out.find("must be") != std::string::npos);
    }
    CHECK(RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / "cinematic").string(), "--frames", "1", "--style", "2"}).code == 0);
    // --check too: the values it hands the backend are checked the same way (review of TASK-0044)
    const CliResult check = RunCli({"nr", "--check", "--backend", "stub", "--warp", "--style", "5"});
    CHECK(check.code != 0);
    CHECK(check.out.find("must be") != std::string::npos);
    CHECK(RunCli({"nr", "--check", "--backend", "stub", "--warp", "--preset", "3"}).code != 0);  // PR-2 E7: no preset option any more
}

// TASK-0044: the schema offers tiff, npz and raw for the nr pass and `process` took them, while `dlssvid nr` (the GUI's
// «Запустить отдельно») refused anything but exr / png16 - and a colour pass in raw could not be written at all.
TEST_CASE("dlssvid nr writes every pass format the schema offers, with the same colour", "[integration][cli][nr][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 24;
    spec.height = 16;
    const auto dir = Dir("formats");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    auto frameIn = [&](const char* format) {
        const auto out = dir / format;
        const CliResult r = RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", out.string(), "--frames", "1", "--format", format});
        INFO(format << ": " << r.out);
        REQUIRE(r.code == 0);
        return PassReader::Open(out / "color_nr").ReadFrame(0).ConvertNormalizedTo(PixelType::F32);
    };
    const PassImage exr = frameIn("exr");
    for (const char* format : {"png16", "tiff", "npz", "raw"}) {
        INFO(format);
        const PassImage f = frameIn(format);
        REQUIRE(f.width == exr.width);
        CHECK(PassImage::MaxAbsDiff(f, exr) <= 1e-3);
    }
}

TEST_CASE("dlssvid nr --check reports the diagnostics: stub ok, ngx on WARP fails with the instruction", "[integration][cli][nr]") {
    const auto dir = Dir("check");
    const CliResult ok = RunCli({"nr", "--check", "--warp", "--backend", "stub"});
    CHECK(ok.code == 0);
    CHECK(ok.out.find("status:       ok") != std::string::npos);
    CHECK(ok.out.find("GPU:") != std::string::npos);
    const auto json = dir / "check.json";
    const CliResult bad = RunCli({"nr", "--check", "--warp", "--backend", "ngx", "--json", json.string()});
    CHECK(bad.code == 1);
    CHECK(bad.out.find("status:       FAILED") != std::string::npos);
    const nlohmann::json j = ReadJson(json);
    CHECK(j["ok"] == false);
    CHECK(j["init_ms"].is_null());  // stopped before NGX: no time, not -1
    CHECK(!std::string(j["hint"]).empty());
    CHECK(j.contains("driver"));
    CHECK(j.contains("dll"));
}

// TASK-0053: the plumbing of the E8 / E6 measurements on the stub (WARP); the NGX numbers come from nr_ab.py on the GPU (PR-2)
TEST_CASE("dlssvid nr --check --check-repeat re-creates the feature N times and --check-features keeps N alive", "[integration][cli][nr]") {
    const auto dir = Dir("check_repeat");
    const auto json = dir / "repeat.json";
    const CliResult rep = RunCli({"nr", "--check", "--warp", "--backend", "stub", "--check-repeat", "3", "--json", json.string()});
    REQUIRE(rep.code == 0);
    CHECK(rep.out.find("recreate:     3 of 3 ok") != std::string::npos);
    const nlohmann::json j = ReadJson(json);
    REQUIRE(j["recreations"].size() == 3);
    for (const auto& r : j["recreations"]) {
        CHECK(r["ok"] == true);
        CHECK(r.contains("vram_after_release_bytes"));
        CHECK(r.contains("create_ms"));
    }
    CHECK(j["recreations"][2]["index"] == 3);
    CHECK(j["recreations"][0]["release_result"] == "stub");
    CHECK(j["recreations_stop"] == "");
    CHECK(!j.contains("features"));

    const auto fjson = dir / "features.json";
    const CliResult feat = RunCli({"nr", "--check", "--warp", "--backend", "stub", "--check-features", "3", "--json", fjson.string()});
    REQUIRE(feat.code == 0);
    CHECK(feat.out.find("features:     3 alive at once") != std::string::npos);
    const nlohmann::json f = ReadJson(fjson);
    REQUIRE(f["features"].size() == 3);
    CHECK(f["features_alive"] == 3);
    CHECK(f["features_stop"] == "");
    CHECK(f["features"][1]["ok"] == true);
    CHECK(!f.contains("recreations"));

    // one measurement per run, and only with --check
    CHECK(RunCli({"nr", "--check", "--warp", "--backend", "stub", "--check-repeat", "2", "--check-features", "2"}).code != 0);
    const auto notClip = dir / "x.mp4";
    std::ofstream(notClip) << "x";  // exists for CLI::ExistingFile: the refusal comes before the decoder opens it
    const CliResult noCheck = RunCli({"nr", "--warp", "--backend", "stub", "--check-repeat", "2", "-i", notClip.string(), "-o", dir.string()});
    CHECK(noCheck.code != 0);
    CHECK(noCheck.out.find("belong to --check") != std::string::npos);
    const CliResult noCheckF = RunCli({"nr", "--warp", "--backend", "stub", "--check-features", "2", "-i", notClip.string(), "-o", dir.string()});
    CHECK(noCheckF.out.find("belong to --check") != std::string::npos);
    // one feature asked: its row is written as for more (nr_ab E6f --check-features 1)
    const auto one = dir / "one.json";
    REQUIRE(RunCli({"nr", "--check", "--warp", "--backend", "stub", "--check-features", "1", "--json", one.string()}).code == 0);
    const nlohmann::json o = ReadJson(one);
    CHECK(o["features_alive"] == 1);
    CHECK(o["features"].size() == 1);
    CHECK(RunCli({"nr", "--check", "--warp", "--backend", "stub", "--check-features", "0"}).code != 0);
    // regression (INCIDENT 2026-10-25): a probe of up to 18 NGX features hung the machine - more than the 4 passes need is refused
    const CliResult many = RunCli({"nr", "--check", "--warp", "--backend", "stub", "--check-features", "5"});
    CHECK(many.code != 0);
    CHECK(many.out.find("feature  5") == std::string::npos);
    // ngx on WARP fails at the first Init: neither measurement starts, the JSON says why
    const auto bad = dir / "bad.json";
    const CliResult ngx = RunCli({"nr", "--check", "--warp", "--backend", "ngx", "--check-repeat", "5", "--json", bad.string()});
    CHECK(ngx.code == 1);
    const nlohmann::json b = ReadJson(bad);
    CHECK(b["recreations"].empty());
    CHECK(b["vram_usage_after_init_bytes"].is_null());
}

TEST_CASE("dlssvid nr-patch runs the patcher, hashes the result and writes the sidecar", "[integration][cli][nr]") {
    if (!HavePython()) SKIP("python not on PATH");
    const auto dir = Dir("patch");
    const auto patcher = dir / "patcher";
    std::filesystem::create_directories(patcher);
    std::ofstream(patcher / "dlssnr_patcher.py") << R"(import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("-t", "--turing", action="store_true")
p.add_argument("-A", "--ampere", action="store_true")
p.add_argument("-a", "--ada", action="store_true")
p.add_argument("-b", "--blackwell", action="store_true")
p.add_argument("-o", "--output")
p.add_argument("--cuda-bin")
p.add_argument("--force", action="store_true")
p.add_argument("--dry-run", action="store_true")
p.add_argument("--work-dir")
args = p.parse_args()
if not args.ada:
    print("expected --ada")
    sys.exit(3)
if args.dry_run:
    print("dry run ok")
    sys.exit(0)
with open(args.input, "rb") as f:
    data = f.read()
with open(args.output, "wb") as f:
    f.write(data + b"PATCHED")
print("patched", args.output)
)";
    const auto input = dir / "nvngx_dlssnr.dll";
    {
        std::ofstream f(input, std::ios::binary);
        for (int i = 0; i < 1000; ++i) f.put(static_cast<char>(i % 251));
    }
    const auto output = dir / "out" / "nvidia" / "nvngx_dlssnr.dll";
    const CliResult r = RunCli({"nr-patch", "--input", input.string(), "--patcher", patcher.string(), "--output", output.string(), "--arch", "ada", "--python", "python"});
    CHECK(r.code == 0);
    CHECK(r.out.find("status:      ok") != std::string::npos);
    REQUIRE(std::filesystem::exists(output));
    CHECK(std::filesystem::file_size(output) == 1007);
    const auto sidecar = std::filesystem::path(output.string() + ".patch.json");
    REQUIRE(std::filesystem::exists(sidecar));
    const nlohmann::json side = ReadJson(sidecar);
    CHECK(side["input_sha256"] == Sha256File(input));
    CHECK(side["output_sha256"] == Sha256File(output));
    CHECK(side["input_sha256"] != side["output_sha256"]);
    CHECK(side["command"].size() >= 6);

    const auto dry = dir / "dry.dll";
    const CliResult d = RunCli({"nr-patch", "--input", input.string(), "--patcher", patcher.string(), "--output", dry.string(), "--arch", "ada", "--dry-run"});
    CHECK(d.code == 0);
    CHECK(!std::filesystem::exists(dry));
    // the patcher refusing (wrong architecture flag) is reported, not hidden
    CHECK(RunCli({"nr-patch", "--input", input.string(), "--patcher", patcher.string(), "--output", (dir / "x.dll").string(), "--arch", "turing"}).code == 1);
    // no patcher: the instruction and exit code 1
    CHECK(RunCli({"nr-patch", "--input", input.string(), "--patcher", (dir / "nope").string()}).code == 1);
}

// TASK-0047 (spec §2f): --start F --frames N --warmup K - the model sees F-K.. for its history, only F..F+N-1 are written,
// numbered as the pass folders and the viewport count; the seek lands on the frame the index names.
TEST_CASE("dlssvid nr --start/--frames/--warmup writes only the requested range", "[integration][cli][nr][preview]") {
    ClipSpec spec;
    spec.frames = 64;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("range");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    auto run = [&](const char* name, std::vector<std::string> extra, bool ok = true) {
        std::vector<std::string> args{"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / name).string(), "--json", (dir / (std::string(name) + ".json")).string()};
        args.insert(args.end(), extra.begin(), extra.end());
        const CliResult r = RunCli(args);
        INFO(name << ": " << r.out);
        if (ok) REQUIRE(r.code == 0);
        return r;
    };
    auto frame = [&](const char* name, int64_t i) { return PassReader::Open(dir / name / "color_nr").ReadFrame(i).ConvertNormalizedTo(PixelType::F32); };

    run("full", {"--frames", "60"});
    // 52 - 2 = 50 is more than 48 frames ahead: a real seek, not decoding forward
    run("part", {"--start", "52", "--frames", "4", "--warmup", "2"});
    const Manifest part = Manifest::Load(dir / "part" / "color_nr");
    CHECK(part.firstFrame == 52);
    CHECK(part.lastFrame == 55);
    CHECK(part.frameCount == 4);
    CHECK(part.stageParams["range"] == nlohmann::json{{"first", 52}, {"last", 55}, {"warmup", 2}});
    int files = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir / "part" / "color_nr")) files += e.path().filename() != Manifest::kFileName;
    CHECK(files == 4);
    for (int64_t i = 52; i <= 55; ++i) CHECK(PassImage::MaxAbsDiff(frame("part", i), frame("full", i)) == 0.0);
    const nlohmann::json pj = ReadJson(dir / "part.json");
    CHECK(pj["stats"]["frames"] == 6);
    CHECK(pj["stats"]["frames_written"] == 4);
    CHECK(pj["stats"]["warmup_frames"] == 2);
    CHECK(pj["range"]["first"] == 52);

    // the warm-up builds the temporal filter's history: equal to a run that writes from the warm-up's start
    const std::vector<std::string> temporal{"--temporal", "0.9", "--temporal-threshold", "1"};
    auto with = [&](std::vector<std::string> a) {
        a.insert(a.end(), temporal.begin(), temporal.end());
        return a;
    };
    run("warm", with({"--start", "10", "--frames", "3", "--warmup", "4"}));
    run("from6", with({"--start", "6", "--frames", "7"}));
    run("cold", with({"--start", "10", "--frames", "3"}));
    for (int64_t i = 10; i <= 12; ++i) CHECK(PassImage::MaxAbsDiff(frame("warm", i), frame("from6", i)) == 0.0);
    CHECK(PassImage::MaxAbsDiff(frame("warm", 10), frame("cold", 10)) > 2e-3);
    CHECK(PassImage::MaxAbsDiff(frame("cold", 10), frame("full", 10)) == 0.0);  // no warm-up: the history starts at F

    // the warm-up is shortened at the clip's start; past the end is an error
    run("clamp", {"--start", "1", "--frames", "1", "--warmup", "6"});
    CHECK(Manifest::Load(dir / "clamp" / "color_nr").firstFrame == 1);
    CHECK(ReadJson(dir / "clamp.json")["stats"]["warmup_frames"] == 1);
    const CliResult past = run("past", {"--start", "70"}, false);
    CHECK(past.code != 0);
    CHECK(past.out.find("past the end") != std::string::npos);

    // a colour pass over the whole clip and --frames past its end: written to the end, not refused (review of TASK-0047)
    REQUIRE(RunCli({"export", "-i", clip.string(), "-o", (dir / "color").string()}).code == 0);
    const CliResult tail = run("tail", {"--color-dir", (dir / "color").string(), "--start", "60", "--frames", "10"});
    CHECK(Manifest::Load(dir / "tail" / "color_nr").lastFrame == 63);
    CHECK(tail.out.find("clip ends") != std::string::npos);
    // a colour pass that starts after F-K: refused before the GPU starts; from F with no warm-up: accepted
    REQUIRE(RunCli({"export", "-i", clip.string(), "-o", (dir / "color52").string(), "--range", "52-63"}).code == 0);
    const CliResult cov = run("cov_bad", {"--color-dir", (dir / "color52").string(), "--start", "52", "--frames", "2", "--warmup", "2"}, false);
    CHECK(cov.code != 0);
    CHECK(cov.out.find("the colour pass") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(dir / "cov_bad" / "color_nr"));
    run("cov_ok", {"--color-dir", (dir / "color52").string(), "--start", "52", "--frames", "2"});
    CHECK(Manifest::Load(dir / "cov_ok" / "color_nr").firstFrame == 52);
    // --frames 0 with --start writes nothing: said so, not «past the end»; --warmup without --start has nothing to warm
    const CliResult zero = run("zero", {"--start", "5", "--frames", "0"}, false);
    CHECK(zero.code != 0);
    CHECK(zero.out.find("writes nothing") != std::string::npos);
    run("warm0", {"--warmup", "3", "--frames", "2"});
    CHECK(ReadJson(dir / "warm0.json")["stats"]["warmup_frames"] == 0);
    CHECK(ReadJson(dir / "warm0.json")["stats"]["frames_written"] == 2);
    CHECK_FALSE(Manifest::Load(dir / "warm0" / "color_nr").stageParams.contains("range"));

    // the video of a range has its frames only
    run("video", {"--start", "52", "--frames", "4", "--warmup", "2", "--video", (dir / "part.mkv").string(), "--codec", "ffv1"});
    const auto video = DecodeAll(dir / "part.mkv");
    REQUIRE(video.size() == 4);
    CHECK(video[0].pts == 0);  // the range's video starts at 0, not 52 frames in

    // never over the pass of a processed project (it has a fingerprint); a whole run there is the user's call, as before
    Manifest stamped = Manifest::Load(dir / "part" / "color_nr");
    stamped.fingerprint = "sha256:processed";
    stamped.Save(dir / "part" / "color_nr");
    const CliResult guard = run("part", {"--start", "52", "--frames", "4"}, false);
    CHECK(guard.code != 0);
    CHECK(guard.out.find("processed project") != std::string::npos);
    run("part", {"--frames", "1"});
}

// TASK-0047 (spec §2h): --params-file holds the project's nr keys; the options given on the command line win, even at their
// default value; the file is validated with the flags by the schema, and a key the schema does not have is refused.
TEST_CASE("dlssvid nr --params-file; explicit flags override the file", "[integration][cli][nr]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("params_file");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    auto file = [&](const char* name, const std::string& text) {
        const auto p = dir / name;
        std::ofstream(p) << text;
        return p.string();
    };
    const std::string tuned =
        file("tuned.json", R"({"backend":"stub","intensity":0.5,"local-tone":0.3,"temporal":0.4,"model_scale":0.5,"passes":2,"format":"png","no_guides":true,"auto_mask":true})");
    auto run = [&](const char* name, std::vector<std::string> extra) {
        std::vector<std::string> args{"nr", "--warp", "-i", clip.string(), "-o", (dir / name).string(), "--frames", "1", "--json", (dir / (std::string(name) + ".json")).string()};
        args.insert(args.end(), extra.begin(), extra.end());
        return RunCli(args);
    };

    {
        const CliResult r = run("file", {"--params-file", tuned});
        INFO(r.out);
        REQUIRE(r.code == 0);
        const Manifest m = Manifest::Load(dir / "file" / "color_nr");
        CHECK(m.format == FileFormat::Png);
        CHECK(m.stageParams["backend"] == "stub");
        CHECK(m.stageParams["passes"] == 2);
        CHECK(m.stageParams["model_scale"] == 0.5);
        CHECK(m.stageParams["guides"] == false);
        const nlohmann::json p = ReadJson(dir / "file.json")["params"];
        CHECK(p["intensity"] == 0.5);
        CHECK(p["local_tone"] == 0.3);
        CHECK(p["temporal"] == 0.4);
        CHECK(p["auto_mask"] == true);
        CHECK(p["style"] == "natural");  // the run's settings: the schema's defaults filled in
        // what reached the model and the filter, not only the merged object
        const nlohmann::json b = ReadJson(dir / "file.json")["backend"];
        CHECK_FALSE(b.contains("intensity"));  // PR-7: the strength is NrCompose's, the model is not given it
        CHECK_THAT(m.stageParams["intensity"].get<double>(), WithinAbs(0.5, 1e-6));
        CHECK(m.stageParams["model_intensity"] == 1.0);
        CHECK_THAT(b["local_tone"].get<double>(), WithinAbs(0.3, 1e-6));
        CHECK_THAT(m.stageParams["temporal"].get<double>(), WithinAbs(0.4, 1e-6));
    }
    {  // given at their defaults, the options still win: count(), not «differs from the default»
        const CliResult r = run("flags", {"--params-file", tuned, "--intensity", "1.5", "--temporal", "0", "--model-scale", "1", "--format", "exr", "--guides", "--no-auto-mask"});
        INFO(r.out);
        REQUIRE(r.code == 0);
        const Manifest m = Manifest::Load(dir / "flags" / "color_nr");
        CHECK(m.format == FileFormat::Exr);
        CHECK(m.stageParams["model_scale"] == 1.0);
        CHECK(m.stageParams["guides"] == true);
        CHECK(m.stageParams["passes"] == 2);  // from the file
        const nlohmann::json p = ReadJson(dir / "flags.json")["params"];
        CHECK(p["intensity"] == 1.5);
        CHECK(p["temporal"] == 0.0);
        CHECK(p["local_tone"] == 0.3);  // from the file
        CHECK(p["auto_mask"] == false);  // --no-auto-mask over the file's true
        const nlohmann::json b = ReadJson(dir / "flags.json")["backend"];
        CHECK_FALSE(b.contains("intensity"));
        CHECK_THAT(m.stageParams["intensity"].get<double>(), WithinAbs(1.5, 1e-6));
        CHECK_THAT(b["local_tone"].get<double>(), WithinAbs(0.3, 1e-6));
        CHECK(m.stageParams["temporal"] == 0.0);
    }
    {  // --intensity 0.3 and 0.3 from a file: one value in --json, not 0.30000001192...
        const CliResult r = run("typed", {"--backend", "stub", "--intensity", "0.3"});
        INFO(r.out);
        REQUIRE(r.code == 0);
        CHECK(ReadJson(dir / "typed.json")["params"]["intensity"] == 0.3);
    }
    {  // --check creates the feature with the run's settings, the file's included: without --backend the default ngx fails on WARP
        const std::string stubOnly = file("check_stub.json", R"({"backend":"stub","local_tone":0.2,"auto_mask":true})");
        const CliResult r = RunCli({"nr", "--check", "--warp", "--params-file", stubOnly});
        INFO(r.out);
        CHECK(r.code == 0);
        CHECK(r.out.find("status:       ok") != std::string::npos);
    }
    auto refused = [&](const std::string& text, std::vector<std::string> extra, const char* says) {
        std::vector<std::string> args{"--params-file", file("bad.json", text)};
        args.insert(args.end(), extra.begin(), extra.end());
        const CliResult r = run("bad", args);
        INFO(text << ": " << r.out);
        CHECK(r.code != 0);
        CHECK(r.out.find(says) != std::string::npos);
    };
    refused(R"({"backend":"stub","intensity":5})", {}, "must be");
    refused(R"({"backend":"stub","intensity":"high"})", {}, "must be a number");
    refused(R"([1,2])", {}, "JSON object");
    refused(R"({"backend":"stub"})", {"--intensity", "5"}, "must be");
    refused(R"({"backend":"stub","intensty":1.2})", {}, "intensty");  // a typo changed nothing and said nothing
    refused(R"({"backend":"stub","start":3})", {}, "option of dlssvid nr");
    refused(R"({"backend":"stub","mv_scale":"auto"})", {}, "option of dlssvid nr");  // an A/B switch (TASK-0048): a flag, not a setting
    refused(R"({"backend":"stub","style":"4"})", {"--check"}, "must be");
    refused(R"({"backend":"stub","preset":3})", {"--check"}, "preset");  // PR-2 E7: not a setting any more
}

// PR-9: a dlssvid-preset in --params-file - a card preset's nr settings (its engineer keys dropped with a warning: backend and format
// come from the options), a pipeline preset's nr entry (with its backend and format); refused whole with a value out of range, a newer
// schema or no nr settings
TEST_CASE("dlssvid nr --params-file takes a dlssvid-preset: engineer and unknown keys dropped with a warning, flags win, a value out of range and a newer schema refused",
          "[integration][cli][nr][presets]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("params_preset");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    auto file = [&](const char* name, const nlohmann::json& j) {
        const auto p = dir / name;
        std::ofstream(p, std::ios::binary) << j.dump();
        return p.string();
    };
    auto preset = [](const char* scope, const nlohmann::json& stages, int schema = 1) {
        return nlohmann::json{{"format", "dlssvid-preset"}, {"schema", schema}, {"scope", scope}, {"name", "Soft"}, {"stages", stages}};
    };
    auto run = [&](const char* name, std::vector<std::string> extra) {
        std::vector<std::string> args{"nr", "--warp", "-i", clip.string(), "-o", (dir / name).string(), "--frames", "1", "--json", (dir / (std::string(name) + ".json")).string()};
        args.insert(args.end(), extra.begin(), extra.end());
        return RunCli(args);
    };
    nlohmann::json cardStages = nlohmann::json::object();
    cardStages["nr"] = {{"params", {{"intensity", 0.5}, {"local_tone", 0.3}, {"temporal", 0.4}, {"format", "png"}, {"foo", 1}}}};
    const std::string card = file("card.json", preset("stage", cardStages));
    {
        const CliResult r = run("card_run", {"--backend", "stub", "--params-file", card, "--temporal", "0.2"});
        INFO(r.out);
        REQUIRE(r.code == 0);
        CHECK(r.out.find("nr.format is an engineer key") != std::string::npos);
        CHECK(r.out.find("nr.foo is not a setting of this dlssvid") != std::string::npos);
        const Manifest m = Manifest::Load(dir / "card_run" / "color_nr");
        CHECK(m.format == FileFormat::Exr);  // the preset's png is not taken: --format's default
        const nlohmann::json p = ReadJson(dir / "card_run.json")["params"];
        CHECK(p["intensity"] == 0.5);
        CHECK(p["local_tone"] == 0.3);
        CHECK(p["temporal"] == 0.2);  // the flag wins
    }
    {
        nlohmann::json stages = nlohmann::json::object();
        stages["depth"] = {{"enabled", true}, {"params", {{"backend", "vda"}}}};
        stages["nr"] = {{"enabled", true}, {"params", {{"intensity", 0.6}, {"backend", "stub"}, {"format", "png"}}}};
        const CliResult r = run("pipe_run", {"--params-file", file("pipe.json", preset("pipeline", stages))});
        INFO(r.out);
        REQUIRE(r.code == 0);
        CHECK(r.out.find("only its nr settings are used here") != std::string::npos);
        const Manifest m = Manifest::Load(dir / "pipe_run" / "color_nr");
        CHECK(m.format == FileFormat::Png);  // a pipeline entry holds the pass format and the backend
        CHECK(m.stageParams["backend"] == "stub");
        CHECK(ReadJson(dir / "pipe_run.json")["params"]["intensity"] == 0.6);
    }
    auto refused = [&](const char* name, const nlohmann::json& j, const char* says) {
        const CliResult r = run("bad", {"--backend", "stub", "--params-file", file(name, j)});
        INFO(name << ": " << r.out);
        CHECK(r.code != 0);
        CHECK(r.out.find(says) != std::string::npos);
    };
    nlohmann::json five = nlohmann::json::object();
    five["nr"] = {{"params", {{"intensity", 5}}}};
    refused("five.json", preset("stage", five), "nr.intensity must be within");
    refused("newer.json", preset("stage", cardStages, 2), "newer");
    nlohmann::json depth = nlohmann::json::object();
    depth["depth"] = {{"params", {{"backend", "vda"}}}};
    refused("depth.json", preset("stage", depth), "has no settings for nr");
}

// TASK-0048 (spec §2e): the A/B switches of the standalone run - the mv scale, zero guides at CreateFeature, no depth guide -
// reported in the summary and the JSON, written into the pass (stage_params.ab); contradictions refused before the GPU starts.
TEST_CASE("dlssvid nr A/B switches on the stub: --mv-scale, --create-with-mv, --no-depth-guide; contradictions refused", "[integration][cli][nr][ab]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("ab");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    auto guides = [&](const std::filesystem::path& root) {  // half-size depth_dlss and mv_dlss under -o, picked up automatically
        for (PassKind kind : {PassKind::DepthDlss, PassKind::MvDlss}) {
            PassWriter w(root / Spec(kind).name, Manifest::ForPass(kind, 24, 16, FileFormat::Exr));
            for (int64_t f = 0; f < 2; ++f) w.WriteFrame(f, MakePassImage(kind, 24, 16));
            w.Finish();
        }
    };
    auto run = [&](const char* name, std::vector<std::string> extra) {
        guides(dir / name);
        std::vector<std::string> args{"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / name).string(), "--frames", "2", "--json",
                                      (dir / (std::string(name) + ".json")).string()};
        args.insert(args.end(), extra.begin(), extra.end());
        return RunCli(args);
    };
    {
        const CliResult r = run("auto", {"--mv-scale", "auto", "--create-with-mv"});
        INFO(r.out);
        REQUIRE(r.code == 0);
        CHECK(r.out.find("a/b:") != std::string::npos);
        CHECK(r.out.find("mv scale:") != std::string::npos);
        const nlohmann::json j = ReadJson(dir / "auto.json");
        CHECK(j["ab"]["mv_scale"] == "auto");
        CHECK(j["ab"]["create_with_mv"] == true);
        REQUIRE(j["stats"]["mv_scale"].is_array());
        CHECK(j["stats"]["mv_scale"][0] == 2.0);  // mv 24x16 -> model 48x32
        CHECK(Manifest::Load(dir / "auto" / "color_nr").stageParams["ab"]["mv_scale"] == "auto");
    }
    {
        const CliResult r = run("no_depth", {"--no-depth-guide"});
        INFO(r.out);
        REQUIRE(r.code == 0);
        const nlohmann::json j = ReadJson(dir / "no_depth.json");
        CHECK(j["stats"]["depth_used"] == false);
        CHECK(j["stats"]["mv_used"] == true);
        CHECK(j["ab"] == nlohmann::json{{"depth_guide", false}});
    }
    {
        const CliResult r = run("plain", {});
        REQUIRE(r.code == 0);
        CHECK(ReadJson(dir / "plain.json")["ab"].empty());
        CHECK_FALSE(Manifest::Load(dir / "plain" / "color_nr").stageParams.contains("ab"));
    }
    {  // never over the pass of a processed project, as a range (review of TASK-0048: an A/B run overwrote it in place)
        Manifest m = Manifest::Load(dir / "plain" / "color_nr");
        m.fingerprint = "sha256:processed";
        m.Save(dir / "plain" / "color_nr");
        const CliResult r = run("plain", {"--mv-scale", "auto"});
        CHECK(r.code != 0);
        CHECK(r.out.find("processed project") != std::string::npos);
        CHECK(Manifest::Load(dir / "plain" / "color_nr").fingerprint == "sha256:processed");  // left as it was
    }
    const CliResult pixels = run("pixels", {"--mv-scale", "pixels"});
    CHECK(pixels.code != 0);
    CHECK(pixels.out.find("must be unit | auto") != std::string::npos);
    const CliResult still = run("still", {"--no-guides", "--create-with-mv"});
    CHECK(still.code != 0);
    CHECK(still.out.find("--no-guides") != std::string::npos);
    const CliResult help = RunCli({"nr", "--help"});
    for (const char* flag : {"--mv-scale", "--create-with-mv", "--no-depth-guide"}) {
        INFO(flag);
        CHECK(help.out.find(flag) != std::string::npos);
    }
}

// TASK-0048 (spec E6, E8): `nr --check` prints what creating the feature costs - the device, NGX init and CreateFeature times
// and the feature's video memory (DXGI, this process); the same lines for the stub on WARP.
// PR-7 (E2): --intensity 0 is the upscaled frame - the model is not called, so ngx runs on WARP without an NVIDIA GPU or the DLL
TEST_CASE("dlssvid nr --intensity 0 does not call the model; the A/B switches of its guides are refused there", "[integration][cli][nr][regression]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("bypass");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto json = dir / "nr.json";
    const CliResult r = RunCli({"nr", "--backend", "ngx", "--warp", "-i", clip.string(), "-o", (dir / "passes").string(), "--intensity", "0", "--json", json.string()});
    INFO(r.out);
    REQUIRE(r.code == 0);
    CHECK(r.out.find("the model is not called") != std::string::npos);
    const nlohmann::json j = ReadJson(json);
    CHECK(j["backend"]["bypass"] == true);
    CHECK(j["stats"]["passes"] == 0);
    CHECK(j["stats"]["bypass"] == true);
    CHECK(j["diagnostics"].empty());
    const CliResult ab = RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / "ab").string(), "--intensity", "0", "--mv-scale", "auto"});
    CHECK(ab.code != 0);
    CHECK(ab.out.find("at --intensity 0") != std::string::npos);
    // --check creates the feature whatever the strength: a project's settings at 0 do not refuse its A/B switches there
    const CliResult check = RunCli({"nr", "--check", "--backend", "stub", "--warp", "--intensity", "0", "--create-with-mv", "--check-size", "64x32"});
    INFO(check.out);
    CHECK(check.code == 0);
}

TEST_CASE("dlssvid nr --check prints the device, init and create times and the feature VRAM", "[integration][cli][nr]") {
    const auto dir = Dir("check_times");
    const auto json = dir / "c.json";
    const CliResult ok = RunCli({"nr", "--check", "--warp", "--backend", "stub", "--json", json.string()});
    INFO(ok.out);
    REQUIRE(ok.code == 0);
    CHECK(ok.out.find("time:         device") != std::string::npos);
    CHECK(ok.out.find("vram:") != std::string::npos);
    const nlohmann::json j = ReadJson(json);
    CHECK(j["device_ms"].get<double>() >= 0.0);
    CHECK(j["init_ms"].get<double>() >= 0.0);
    CHECK(j["create_ms"] == 0.0);
    CHECK(j.contains("vram_feature_bytes"));
    CHECK(j["ab"].empty());
    const CliResult ab = RunCli({"nr", "--check", "--warp", "--backend", "stub", "--create-with-mv", "--mv-scale", "auto", "--json", json.string()});
    CHECK(ab.code == 0);
    CHECK(ReadJson(json)["ab"]["create_with_mv"] == true);
    const CliResult still = RunCli({"nr", "--check", "--warp", "--backend", "stub", "--no-guides", "--no-depth-guide"});
    CHECK(still.code != 0);
    CHECK(still.out.find("--no-guides") != std::string::npos);
}

// PR-13: dlssvid nr reads the clip's scene list - <output>/scenes found by itself, --scenes-dir given, none with --no-scenes / --no-auto;
// one of another source is an error when given and is skipped when found; --scene-cut is gone (the threshold is the list's)
TEST_CASE("dlssvid nr resets at the cuts of the clip's scene list", "[integration][cli][nr][scenes]") {
    const uint32_t w = 48, h = 32;
    const int cutAt = 3;
    ClipSpec spec;
    spec.frames = 6;
    spec.width = w;
    spec.height = h;
    const auto dir = Dir("scenes");
    const auto clip = WriteClipWith(dir / "clip.mkv", spec, [&](int64_t i) {
        CpuFrame f = SyntheticFrame(i, w, h);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c) y[r * w + c] = static_cast<uint8_t>(i < cutAt ? 40 + c + 4 * i : 220 - c);
        return f;
    });
    const auto root = dir / "passes";
    REQUIRE(RunCli({"scenes", "-i", clip.string(), "-o", root.string()}).code == 0);
    const std::string id = Manifest::Load(root / "scenes").stageParams["cuts_id"];
    REQUIRE(id.rfind("cuts:", 0) == 0);
    auto run = [&](const char* name, const std::vector<std::string>& extra, const std::filesystem::path& out) {
        std::vector<std::string> args{"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", out.string(), "--temporal", "0.5", "--json", (dir / (std::string(name) + ".json")).string()};
        args.insert(args.end(), extra.begin(), extra.end());
        const CliResult r = RunCli(args);
        INFO(name << ": " << r.out);
        REQUIRE(r.code == 0);
        return ReadJson(dir / (std::string(name) + ".json"))["stats"];
    };
    const nlohmann::json found = run("auto", {}, root);
    CHECK(found["scenes_used"] == true);
    CHECK(found["scene_cuts_id"] == id);
    CHECK(found["scene_cuts"] == nlohmann::json::array({cutAt}));
    CHECK(found["scene_cut_count"] == 1);
    CHECK(Manifest::Load(root / "color_nr").stageParams["scene_cuts_id"] == id);
    const nlohmann::json none = run("no_scenes", {"--no-scenes"}, root);
    CHECK(none["scenes_used"] == false);
    CHECK(none["scene_cuts"].is_null());
    CHECK_FALSE(Manifest::Load(root / "color_nr").stageParams.contains("scene_cuts_id"));
    CHECK(run("no_auto", {"--no-auto"}, root)["scenes_used"] == false);
    CHECK(run("given", {"--no-auto", "--scenes-dir", (root / "scenes").string()}, root)["scenes_used"] == true);
    CHECK(run("bypass", {"--intensity", "0"}, root)["scenes_used"] == false);  // no history to reset: the list is not read
    // the list of another source: an error when given, skipped when found under -o
    ClipSpec other = spec;
    other.frames = 4;
    const auto otherClip = WriteClip(dir / "other.mkv", other);
    const auto otherRoot = dir / "other_passes";
    REQUIRE(RunCli({"scenes", "-i", otherClip.string(), "-o", otherRoot.string()}).code == 0);
    const CliResult foreign = RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", root.string(), "--scenes-dir", (otherRoot / "scenes").string()});
    CHECK(foreign.code != 0);
    CHECK(foreign.out.find("of another source") != std::string::npos);
    CHECK(run("foreign_auto", {}, otherRoot)["scenes_used"] == false);
    // the option of PR-6 is gone; the help names the list's
    CHECK(RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", root.string(), "--scene-cut", "0.3"}).code != 0);
    const CliResult help = RunCli({"nr", "--help"});
    CHECK(help.out.find("--scenes-dir") != std::string::npos);
    CHECK(help.out.find("--no-scenes") != std::string::npos);
    CHECK(help.out.find("--scene-cut") == std::string::npos);
}
