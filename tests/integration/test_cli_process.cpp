// `dlssvid process` (project-driven and with parameters) and `dlssvid bench` as processes (stage 8).

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "TestClips.h"
#include "passes/Manifest.h"
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
