// `dlssvid nr`, `dlssvid nr --check` and `dlssvid nr-patch` as processes (stage 6).

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "TestClips.h"
#include "passes/Manifest.h"
#include "passes/PassSequence.h"
#include "util/Sha256.h"
#include "util/Subprocess.h"

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
    // underscore spellings (the GUI passes project JSON keys as options) are accepted too
    const CliResult r2 = RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / "passes2").string(), "--frames", "1", "--model_scale", "1", "--no_guides"});
    CHECK(r2.code == 0);
    // missing input
    CHECK(RunCli({"nr", "--backend", "stub", "--warp", "-o", (dir / "passes3").string()}).code != 0);

    // TASK-0044: the standalone run checks every value against the schema (before: `--intensity 5` and `--preset 9` went
    // to NGX); an experimental style is a valid one
    for (const std::vector<std::string>& bad : std::vector<std::vector<std::string>>{
             {"--intensity", "5"}, {"--preset", "9"}, {"--passes", "5"}, {"--exposure", "20"}, {"--style", "7"}, {"--local-tone", "3"}}) {
        std::vector<std::string> args{"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / "bad").string(), "--frames", "1"};
        args.insert(args.end(), bad.begin(), bad.end());
        const CliResult b = RunCli(args);
        INFO(bad[0] << " " << bad[1]);
        CHECK(b.code != 0);
        CHECK(b.out.find("must be") != std::string::npos);
    }
    CHECK(RunCli({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", (dir / "style4").string(), "--frames", "1", "--style", "4"}).code == 0);
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
    CHECK(!std::string(j["hint"]).empty());
    CHECK(j.contains("driver"));
    CHECK(j.contains("dll"));
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
