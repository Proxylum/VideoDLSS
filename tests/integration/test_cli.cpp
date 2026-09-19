// The CLI is the contract for automation (ТЗ §7, §9): run the real executable.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "TestClips.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

int Run(const std::string& args) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + "\"";  // cmd.exe quoting
    return std::system(cmd.c_str());
}

}  // namespace

TEST_CASE("dlssvid --version and info run", "[integration][cli]") {
    CHECK(Run("--version") == 0);
    ClipSpec spec;
    spec.frames = 3;
    const auto clip = WriteClip(TempDir() / "cli_info.mkv", spec);
    CHECK(Run("info --warp \"" + clip.string() + "\"") == 0);
}

TEST_CASE("dlssvid process --passthrough --codec ffv1 is lossless", "[integration][cli]") {
    ClipSpec spec;
    spec.frames = 10;
    spec.audio = true;
    const auto in = WriteClip(TempDir() / "cli_in.mkv", spec);
    const auto out = TempDir() / "cli_out.mkv";
    std::filesystem::remove(out);

    const int rc = Run("process --passthrough --codec ffv1 --warp -i \"" + in.string() + "\" -o \"" + out.string() + "\"");
    REQUIRE(rc == 0);
    REQUIRE(std::filesystem::exists(out));

    const auto a = DecodeAll(in), b = DecodeAll(out);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) REQUIRE(a[i].data == b[i].data);
    CHECK(Summarize(out).audioStreams == 1);
}

TEST_CASE("dlssvid process --frames limits the run", "[integration][cli]") {
    ClipSpec spec;
    spec.frames = 10;
    const auto in = WriteClip(TempDir() / "cli_frames_in.mkv", spec);
    const auto out = TempDir() / "cli_frames_out.mkv";
    REQUIRE(Run("process --passthrough --codec ffv1 --warp --frames 4 -i \"" + in.string() + "\" -o \"" + out.string() + "\"") == 0);
    CHECK(DecodeAll(out).size() == 4);
}

TEST_CASE("dlssvid process rejects an unknown stage (the full pipeline arrived in stage 8)", "[integration][cli]") {
    ClipSpec spec;
    spec.frames = 2;
    const auto in = WriteClip(TempDir() / "cli_reject_in.mkv", spec);
    CHECK(Run("process --codec ffv1 --warp --stages bogus -i \"" + in.string() + "\" -o \"" + (TempDir() / "cli_reject_out.mkv").string() + "\"") != 0);
}
