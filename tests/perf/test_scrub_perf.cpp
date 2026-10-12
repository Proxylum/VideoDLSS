// Scrubbing performance (ТЗ §9(6), TASK-0040): random jumps on 1080p at >= 30 fps. A long-GOP 1080p clip (one
// keyframe), its proxy built, `dlssvid render --bench --bench-mode random`: only jumps into frames that are not
// resident are timed (a cache hit is not a jump), the threshold is the p90 of those misses. 8 and 10 bits (the 10-bit
// proxy is FFVHuff, the 8-bit one MagicYUV). The path without a proxy is printed for the record (what a user gets
// while the proxy builds or above the disk limit). Needs an NVIDIA GPU; run serially (ctest label perf).

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "TestClips.h"
#include "gpu/D3D12Device.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

std::string Q(const std::filesystem::path& p) { return "\"" + p.string() + "\""; }

int Run(const std::string& args) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + " >nul 2>&1\"";
    return std::system(cmd.c_str());
}

nlohmann::json Bench(const std::filesystem::path& clip, const std::filesystem::path& passes, const std::string& proxy, int jumps, const std::string& extra = "") {
    const auto json = passes.parent_path() / ("bench_" + proxy + ".json");
    REQUIRE(Run("render -i " + Q(clip) + " --passes " + Q(passes) + " --proxy " + proxy + " --bench " + std::to_string(jumps) +
                " --bench-mode random --bench-json " + Q(json) + " " + extra) == 0);
    nlohmann::json j;
    std::ifstream(json) >> j;
    return j["passes"]["random"];
}

void ScrubCase(const char* name, const char* pixFmt) {
    D3D12Device probe;
    if (probe.IsWarp() || !probe.IsNvidia()) SKIP("no NVIDIA GPU: the threshold is for the reference hardware");
    const auto dir = TempDir() / "perf_scrub" / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    ClipSpec spec;
    spec.width = 1920;
    spec.height = 1080;
    spec.frames = 300;
    spec.fpsNum = 30;
    spec.gop = 300;  // one keyframe: without a proxy a jump decodes from frame 0
    spec.pixFmt = pixFmt;
    const FrameDesc desc = FrameDescFor(spec);
    const auto clip = std::string(pixFmt) == "yuv420p" ? WriteClip(dir / "clip.mkv", spec)
                                                      : WriteClipWith(dir / "clip.mkv", spec, [&](int64_t i) { return SyntheticFrame(i, desc); });
    const auto passes = dir / "passes";
    REQUIRE(Run("proxy -i " + Q(clip) + " --passes " + Q(passes) + " --sources source") == 0);
    const nlohmann::json without = Bench(clip, passes, "off", 12);
    WARN(name << " without the proxy: p50 " << without["p50_ms"].get<double>() << " ms, p90 " << without["p90_ms"].get<double>() << " ms");
    const nlohmann::json with = Bench(clip, passes, "auto", 80);
    const double p90 = with["p90_ms"].get<double>(), mean = with["mean_ms"].get<double>();
    WARN(name << " with the proxy: mean " << mean << " ms, p50 " << with["p50_ms"].get<double>() << " ms, p90 " << p90 << " ms over " << with["misses"].get<int>()
              << " misses");
    CHECK(with["misses"].get<int>() == 80);
    CHECK(with["proxy_loads"].get<int>() > 0);
    CHECK(p90 <= 1000.0 / 30.0);  // >= 30 fps for 90 % of the jumps
}

}  // namespace

TEST_CASE("Random jumps on 1080p 8-bit: p90 of the misses within 33.3 ms with the proxy", "[perf][viewport][gpu]") { ScrubCase("p8", "yuv420p"); }

TEST_CASE("Random jumps on 1080p 10-bit: p90 of the misses within 33.3 ms with the proxy", "[perf][viewport][gpu]") { ScrubCase("p10", "yuv420p10le"); }
