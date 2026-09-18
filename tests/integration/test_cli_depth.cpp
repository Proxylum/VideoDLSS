// CLI: `dlssvid depth --backend stub` and `dlssvid models list` as processes.

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

TEST_CASE("dlssvid depth --backend stub produces both depth passes", "[integration][cli][depth]") {
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 40;
    spec.height = 24;
    const auto clip = WriteClip(TempDir() / "cli_depth.mkv", spec);
    const auto out = TempDir() / "cli_depth_out";
    std::filesystem::remove_all(out);
    REQUIRE(Run("depth --backend stub --warp --near 0.2 --far 50 -i " + Q(clip) + " -o " + Q(out)) == 0);
    const PassReader raw = PassReader::Open(out / "depth_raw");
    raw.Validate({40, 24, 6, PassKind::DepthRaw});
    CHECK(raw.Man().depth.zFar == 50.f);
    CHECK(raw.Man().sourceHash.rfind("sha256:", 0) == 0);
    PassReader::Open(out / "depth_dlss").Validate({40, 24, 6, PassKind::DepthDlss});

    const auto out2 = TempDir() / "cli_depth_out2";
    std::filesystem::remove_all(out2);
    REQUIRE(Run("depth --backend stub --warp --no-dlss --frames 3 --format npz -i " + Q(clip) + " -o " + Q(out2)) == 0);
    const PassReader r2 = PassReader::Open(out2 / "depth_raw");
    CHECK(r2.Man().frameCount == 3);
    CHECK(r2.Man().format == FileFormat::Npz);
    CHECK_FALSE(std::filesystem::exists(out2 / "depth_dlss"));
    CHECK(Run("depth --backend bogus --warp -i " + Q(clip) + " -o " + Q(out2)) != 0);
    CHECK(Run("depth --backend stub --warp --format png16 -i " + Q(clip) + " -o " + Q(out2)) != 0);
}

TEST_CASE("dlssvid models list works from the build tree", "[integration][cli][depth]") {
    CHECK(Run("models list") == 0);
    CHECK(Run("models fetch nope") != 0);
}
