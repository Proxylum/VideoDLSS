// CLI: `dlssvid flow --backend stub` and `dlssvid depth --mv-dir` as processes.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "TestClips.h"
#include "passes/PassSequence.h"
#include "passes/ScenesPass.h"

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

// PR-13: dlssvid flow reads the clip's scene list - <output>/scenes found by itself, --no-scenes, an explicit --scenes-dir of another
// source refused, one found under -o of another source skipped
TEST_CASE("dlssvid flow writes zero vectors at the cuts of the clip's scene list", "[integration][cli][flow][scenes]") {
    const uint32_t w = 40, h = 24;
    ClipSpec spec;
    spec.frames = 5;
    spec.width = w;
    spec.height = h;
    const auto clip = WriteClipWith(TempDir() / "cli_flow_cut.mkv", spec, [&](int64_t i) {  // a hard cut at frame 2
        CpuFrame f = SyntheticFrame(i, w, h);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c) y[r * w + c] = static_cast<uint8_t>(i < 2 ? 40 + c + 4 * i : 220 - c);
        return f;
    });
    const auto out = TempDir() / "cli_flow_scenes";
    std::filesystem::remove_all(out);
    REQUIRE(Run("scenes -i " + Q(clip) + " -o " + Q(out)) == 0);
    const std::string id = Manifest::Load(out / "scenes").stageParams["cuts_id"];
    REQUIRE(id == SceneCutsId({2}));
    REQUIRE(Run("flow --backend stub --hwaccel none --warp -i " + Q(clip) + " -o " + Q(out)) == 0);
    CHECK(Manifest::Load(out / "mv_raw").stageParams["scene_cuts_id"] == id);
    CHECK(Manifest::Load(out / "mv_dlss").stageParams["scene_cut_count"] == 1);
    REQUIRE(Run("flow --backend stub --hwaccel none --warp --no-scenes -i " + Q(clip) + " -o " + Q(out)) == 0);
    CHECK_FALSE(Manifest::Load(out / "mv_dlss").stageParams.contains("scene_cuts_id"));
    // another source's list
    ClipSpec otherSpec = spec;
    otherSpec.frames = 3;
    const auto other = WriteClipWith(TempDir() / "cli_flow_other.mkv", otherSpec, [&](int64_t i) {  // a cut at 1: a list with a label
        CpuFrame f = SyntheticFrame(i, w, h);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c) y[r * w + c] = static_cast<uint8_t>(i < 1 ? 40 + c : 220 - c);
        return f;
    });
    const auto otherOut = TempDir() / "cli_flow_other";
    std::filesystem::remove_all(otherOut);
    REQUIRE(Run("scenes -i " + Q(other) + " -o " + Q(otherOut)) == 0);
    REQUIRE(Manifest::Load(otherOut / "scenes").stageParams["cuts_id"] == SceneCutsId({1}));  // accepted, it would leave its label
    CHECK(Run("flow --backend stub --hwaccel none --warp --scenes-dir " + Q(otherOut / "scenes") + " -i " + Q(clip) + " -o " + Q(out)) != 0);
    REQUIRE(Run("flow --backend stub --hwaccel none --warp -i " + Q(clip) + " -o " + Q(otherOut)) == 0);  // found, of another source: skipped
    CHECK_FALSE(Manifest::Load(otherOut / "mv_dlss").stageParams.contains("scene_cuts_id"));
}
