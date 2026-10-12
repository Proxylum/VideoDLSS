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

// Audit fg-2 (TASK-0038): `dlssvid process` runs FG as `fg --as-stage` (pipeline/FgProcess.h) - progress and the result
// as protocol lines, the stage's parameters as JSON, the runner's source hash, a disabled backend reported, not failed.
TEST_CASE("dlssvid fg --as-stage reports progress and the result as lines; --params-json is applied; --disable-unavailable reports a disabled stage",
          "[integration][cli][fg][isolation]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("as_stage");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    auto resultOf = [](const std::string& out) {
        const size_t at = out.find("@result ");
        REQUIRE(at != std::string::npos);
        const size_t end = out.find('\n', at);
        return nlohmann::json::parse(out.substr(at + 8, end == std::string::npos ? std::string::npos : end - at - 8));
    };

    const CliResult r = RunCli({"fg", "--as-stage", "--warp", "-i", clip.string(), "-o", (dir / "p").string(), "--no-auto", "--params-json",
                                R"({"backend":"blend","multiplier":3})", "--source-hash", "sha256:00"});
    CHECK(r.code == 0);
    for (int n = 1; n <= 6; ++n) CHECK(r.out.find("@progress " + std::to_string(n) + "\n") != std::string::npos);
    const nlohmann::json res = resultOf(r.out);
    CHECK(res["disabled"] == false);
    CHECK(res["stats"]["multiplier"] == 3);
    CHECK(res["stats"]["frames"] == 6);
    CHECK(res["stats"]["frames_out"] == 16);
    CHECK(res["stats"]["ms_per_frame_wall"].get<double>() > 0.0);
    CHECK(Manifest::Load(dir / "p" / "color_fg").sourceHash == "sha256:00");  // the runner's hash, not hashed again
    CHECK(Manifest::Load(dir / "p" / "color_fg").frameCount == 16);

    const CliResult off = RunCli({"fg", "--as-stage", "--warp", "-i", clip.string(), "-o", (dir / "p_off").string(), "--no-auto", "--params-json",
                                  R"({"backend":"dlssg"})", "--disable-unavailable"});
    CHECK(off.code == 0);  // disabled is a result, not a failure
    const nlohmann::json offRes = resultOf(off.out);
    CHECK(offRes["disabled"] == true);
    CHECK(!offRes["disabled_reason"].get<std::string>().empty());

    const CliResult tiff = RunCli({"fg", "--as-stage", "--warp", "-i", clip.string(), "-o", (dir / "p_tiff").string(), "--no-auto", "--format", "tiff",
                                   "--params-json", R"({"backend":"blend"})", "--frames", "2"});
    CHECK(tiff.code == 0);  // the formats `process` accepts for the stage
    CHECK(Manifest::Load(dir / "p_tiff" / "color_fg").format == FileFormat::Tiff);

    CHECK(RunCli({"fg", "--as-stage", "--warp", "-i", clip.string(), "-o", (dir / "p_bad").string(), "--params-json", "[1]"}).code != 0);
}
