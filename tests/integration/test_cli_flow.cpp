// CLI: `dlssvid flow --backend stub` and `dlssvid depth --mv-dir` as processes.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "TestClips.h"
#include "passes/PassSequence.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {
int Run(const std::string& args) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + " >nul 2>&1\"";
    return std::system(cmd.c_str());
}
std::string Q(const std::filesystem::path& p) { return "\"" + p.string() + "\""; }
}  // namespace

TEST_CASE("dlssvid flow --backend stub produces mv_raw and mv_dlss", "[integration][cli][flow]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 40;
    spec.height = 24;
    const auto clip = WriteClip(TempDir() / "cli_flow.mkv", spec);
    const auto out = TempDir() / "cli_flow_out";
    std::filesystem::remove_all(out);
    REQUIRE(Run("flow --backend stub --hwaccel none --warp --target 80x48 -i " + Q(clip) + " -o " + Q(out)) == 0);
    const PassReader raw = PassReader::Open(out / "mv_raw");
    raw.Validate({40, 24, 5, PassKind::MvRaw});
    const PassReader dlss = PassReader::Open(out / "mv_dlss");
    dlss.Validate({80, 48, 5, PassKind::MvDlss});
    CHECK(dlss.Man().stageParams.contains("warp_psnr_mean_db"));

    // depth with motion-compensated TAE
    const auto depthOut = TempDir() / "cli_flow_depth";
    std::filesystem::remove_all(depthOut);
    REQUIRE(Run("depth --backend stub --warp --no-dlss --mv-dir " + Q(out / "mv_dlss") + " -i " + Q(clip) + " -o " + Q(depthOut)) == 0);
    CHECK(PassReader::Open(depthOut / "depth_raw").Man().stageParams["tae_mode"] == "warped");

    CHECK(Run("flow --backend stub --hwaccel none --warp --format png16 -i " + Q(clip) + " -o " + Q(out)) != 0);
    CHECK(Run("flow --backend nope --hwaccel none --warp -i " + Q(clip) + " -o " + Q(out)) != 0);
}
