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
