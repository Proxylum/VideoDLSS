// `dlssvid fg`, `dlssvid fg --check`, `--crash-after` (process isolation) and `compare --start/--step` as processes (stage 7).

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "TestClips.h"
#include "passes/Manifest.h"
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
    const auto d = TempDir() / "cli_fg" / name;
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

TEST_CASE("dlssvid fg --backend blend --warp writes color_fg, a preview and a JSON summary; compare --start/--step", "[integration][cli][fg]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("blend");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto passes = dir / "passes";
    const auto json = dir / "fg.json";
    const CliResult r = RunCli({"fg", "--backend", "blend", "--warp", "-i", clip.string(), "-o", passes.string(), "--frames", "4", "--multiplier", "2", "--video",
                                (dir / "fg.mkv").string(), "--codec", "ffv1", "--json", json.string()});
    CHECK(r.code == 0);
    CHECK(r.out.find("color_fg:") != std::string::npos);
    CHECK(r.out.find("backend:     blend") != std::string::npos);
    CHECK(std::filesystem::exists(passes / "color_fg" / Manifest::kFileName));
    const nlohmann::json j = ReadJson(json);
    CHECK(j["stats"]["frames"] == 4);
    CHECK(j["stats"]["generated"] == 3);
    CHECK(j["stats"]["frames_out"] == 7);
    CHECK(j["stats"]["multiplier"] == 2);
    CHECK(DecodeAll(dir / "fg.mkv").size() == 7);
    // compare the real frames of color_fg (even indices) against the clip: --step 2 walks the reference, --offset 0
    const CliResult c = RunCli({"compare", "--ref", (passes / "color_fg").string(), "--test", (passes / "color_fg").string(), "--start", "1", "--step", "2", "--frames", "3", "-q", "--json",
                                (dir / "cmp.json").string()});
    CHECK(c.code == 0);
    const nlohmann::json cj = ReadJson(dir / "cmp.json");
    CHECK(cj["summary"]["frames"] == 3);
    CHECK(cj["frames"][0]["frame"] == 1);
    CHECK(cj["frames"][2]["frame"] == 5);
    CHECK(RunCli({"compare", "--ref", clip.string(), "--test", clip.string(), "--step", "0"}).code != 0);
    // missing input
    CHECK(RunCli({"fg", "--backend", "blend", "--warp", "-o", (dir / "p2").string()}).code != 0);
}

TEST_CASE("dlssvid fg --check: blend ok, dlssg on WARP fails with the instruction", "[integration][cli][fg]") {
    const auto dir = Dir("check");
    const CliResult ok = RunCli({"fg", "--check", "--warp", "--backend", "blend"});
    CHECK(ok.code == 0);
    CHECK(ok.out.find("status:       ok") != std::string::npos);
    const auto json = dir / "check.json";
    const CliResult bad = RunCli({"fg", "--check", "--warp", "--backend", "dlssg", "--json", json.string()});
    CHECK(bad.code == 1);
    CHECK(bad.out.find("status:       FAILED") != std::string::npos);
    const nlohmann::json j = ReadJson(json);
    CHECK(j["ok"] == false);
    CHECK(!std::string(j["hint"]).empty());
}

TEST_CASE("dlssvid fg --crash-after ends the stage process only: the caller sees a crash exit code and goes on", "[integration][cli][fg]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 32;
    spec.height = 16;
    const auto dir = Dir("crash");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const CliResult immediate = RunCli({"fg", "--crash-after", "0"});
    CHECK(immediate.code == 0xC0000005u);
    const CliResult later = RunCli({"fg", "--backend", "blend", "--warp", "-i", clip.string(), "-o", (dir / "passes").string(), "--crash-after", "2"});
    CHECK(later.code == 0xC0000005u);
    // the frames written before the crash are on disk; the caller (this test) is alive
    CHECK(std::filesystem::exists(dir / "passes" / "color_fg"));
    const CliResult fine = RunCli({"fg", "--backend", "blend", "--warp", "-i", clip.string(), "-o", (dir / "passes_ok").string()});
    CHECK(fine.code == 0);
}
