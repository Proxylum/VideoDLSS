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

#ifdef DLSSVID_WITH_CUDA
#include "io/VideoEncoder.h"
#include "stages/flow/OfaFlowEstimator.h"

// Regression (2026-09-29): `dlssvid process` on a 4:4:4 16-bit HEVC source failed in the flow stage — NVDEC frames of such
// streams are not NV12, and OFA must fall back to ABGR uploads from the host frames instead of the stage failing.
TEST_CASE("dlssvid flow --backend ofa on an HEVC 4:4:4 16-bit source whose NVDEC frames are not NV12", "[integration][cli][flow][gpu][regression]") {
    std::string reason;
    if (!OfaFlowEstimator::Available(&reason)) SKIP("OFA unavailable: " << reason);
    if (!VideoEncoder::EncoderAvailable("hevc_nvenc")) SKIP("hevc_nvenc unavailable");
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 320;
    spec.height = 192;
    spec.codec = "hevc_nvenc";
    spec.pixFmt = "yuv444p16le";
    std::filesystem::path clip;
    try {
        clip = WriteClip(TempDir() / "cli_flow_444_16.mp4", spec);
    } catch (const std::exception& e) {
        SKIP("hevc_nvenc cannot encode yuv444p16le on this GPU: " << e.what());
    }
    const auto out = TempDir() / "cli_flow_444_16_out";
    std::filesystem::remove_all(out);
    REQUIRE(Run("flow --backend ofa --hwaccel cuda -i " + Q(clip) + " -o " + Q(out)) == 0);
    const PassReader raw = PassReader::Open(out / "mv_raw");
    raw.Validate({spec.width, spec.height, spec.frames, PassKind::MvRaw});
}
#endif
