// Bit depth and chroma through the I/O (TASK-0034): 10 / 12-bit and 4:2:2 / 4:4:4 sources decode into their own layout
// (software and NVDEC alike), the encoder writes them back losslessly (FFV1), into ProRes and into 10-bit NVENC streams,
// passthrough and `process` keep the depth, the policy picks the result's layout, the fingerprints see the source layout.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "TestClips.h"
#include "convert/YuvConvert.h"
#include "gpu/D3D12Device.h"
#include "io/OutputFormat.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "passes/Manifest.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/ProcessRunner.h"
#include "util/Subprocess.h"
#include "viewport/Project.h"

#ifdef DLSSVID_WITH_CUDA
#include "gpu/CudaInterop.h"
#endif

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "formats" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// Writes `frames` synthetic frames of `desc` with the encoder and decodes the file back.
std::vector<CpuFrame> RoundTrip(const std::filesystem::path& file, const std::string& codec, const FrameDesc& desc, int frames, VideoEncoder::Options eo = {}) {
    eo.codec = codec;
    eo.frameRate = Rational{24, 1};
    VideoEncoder enc(file, desc, eo);
    enc.Open();
    for (int i = 0; i < frames; ++i) enc.WriteFrame(SyntheticFrame(i, desc));
    enc.Close();
    return DecodeAll(file);
}

// Mean and largest absolute difference over every 16-bit sample of two frames of the same layout, `shift` applied to `a`.
double Diff16(const CpuFrame& a, const CpuFrame& b, int shiftA, int* maxDiff) {
    REQUIRE(a.desc.width == b.desc.width);
    REQUIRE(a.data.size() == b.data.size());
    const size_t n = a.data.size() / 2;
    double sum = 0.0;
    int mx = 0;
    for (size_t i = 0; i < n; ++i) {
        const int d = std::abs(static_cast<int>(a.Plane16(0)[i] >> shiftA) - static_cast<int>(b.Plane16(0)[i]));
        sum += d;
        mx = std::max(mx, d);
    }
    if (maxDiff) *maxDiff = mx;
    return sum / static_cast<double>(n);
}

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

std::vector<ProcessStage> StubStages() {
    std::vector<ProcessStage> s = DefaultProcessStages();
    for (const char* spec : {"depth.backend=stub", "flow.backend=stub", "upscale.backend=nis", "upscale.scale=2", "nr.backend=stub", "fg.backend=blend", "fg.multiplier=2"})
        ApplyParamSpec(s, spec);
    return s;
}

bool HaveNvidiaEncoder(const char* name) {
    D3D12Device dev;
    return !dev.IsWarp() && dev.IsNvidia() && VideoEncoder::EncoderAvailable(name);
}

}  // namespace

TEST_CASE("Planar 10 / 12-bit and 4:2:2 / 4:4:4 clips decode into their own layout, bit-exact", "[integration][formats]") {
    const auto dir = Dir("decode");
    for (const char* pix : {"yuv420p10le", "yuv422p10le", "yuv444p10le", "yuv444p12le", "yuv422p", "yuv444p"}) {
        ClipSpec spec;
        spec.frames = 4;
        spec.width = 48;
        spec.height = 32;
        spec.pixFmt = pix;
        const FrameDesc desc = FrameDescFor(spec);
        const auto clip = WriteClip(dir / (std::string(pix) + ".mkv"), spec);
        VideoDecoder dec(clip);
        INFO(pix);
        CHECK(dec.Info().pixelFormat == pix);
        CHECK(dec.Info().decoded == desc);
        CHECK(dec.Info().bitDepth == (desc.HighBitDepth() ? desc.bitDepth : 8));
        CHECK(dec.Info().chroma == ChromaName(desc.format));
        const auto frames = DecodeAll(clip);
        REQUIRE(frames.size() == 4);
        for (int i = 0; i < 4; ++i) {
            REQUIRE(frames[i].desc == desc);
            REQUIRE(frames[i].data == SyntheticFrame(i, desc).data);
        }
    }
    CHECK(DecodedFrameDescFor("yuv420p") == FrameDesc{0, 0, PixelFormat::Yuv420p, 8});
    CHECK(DecodedFrameDescFor("p010le") == FrameDesc{0, 0, PixelFormat::Yuv420p16, 10});
    CHECK(DecodedFrameDescFor("yuva444p10le") == FrameDesc{0, 0, PixelFormat::Yuv444p16, 10});  // the alpha plane is dropped
    CHECK(DecodedFrameDescFor("yuv444p16le") == FrameDesc{0, 0, PixelFormat::Yuv444p16, 16});
    CHECK(DecodedFrameDescFor("rgb24") == FrameDesc{0, 0, PixelFormat::Yuv420p, 8});  // swscale fallback
    CHECK(DecodedFrameDescFor("gray") == FrameDesc{0, 0, PixelFormat::Yuv420p, 8});
    CHECK(DecodedFrameDescFor("yuv411p") == FrameDesc{0, 0, PixelFormat::Yuv420p, 8});
    CHECK(DecodedFrameDescFor("nope") == FrameDesc{0, 0, PixelFormat::Yuv420p, 8});
}

TEST_CASE("FFV1 keeps 10 / 12-bit frames losslessly in every chroma layout", "[integration][formats][encode]") {
    const auto dir = Dir("ffv1");
    const FrameDesc layouts[] = {FrameDesc{48, 32, PixelFormat::Yuv420p16, 10}, FrameDesc{48, 32, PixelFormat::Yuv422p16, 10}, FrameDesc{48, 32, PixelFormat::Yuv444p16, 10},
                                 FrameDesc{48, 32, PixelFormat::Yuv444p16, 12}, FrameDesc{48, 32, PixelFormat::Yuv420p16, 16}, FrameDesc{48, 32, PixelFormat::Yuv422p, 8},
                                 FrameDesc{48, 32, PixelFormat::Yuv444p, 8}};
    for (const FrameDesc& d : layouts) {
        INFO(d.Name());
        CHECK(VideoEncoder::PixelFormatFor("ffv1", d) == d.Name() + (d.HighBitDepth() ? "le" : ""));
        const auto frames = RoundTrip(dir / (d.Name() + ".mkv"), "ffv1", d, 3);
        REQUIRE(frames.size() == 3);
        for (int i = 0; i < 3; ++i) {
            REQUIRE(frames[i].desc == d);
            REQUIRE(frames[i].data == SyntheticFrame(i, d).data);
        }
    }
    // a frame of another layout than the encoder was opened for is refused
    VideoEncoder::Options eo;
    eo.codec = "ffv1";
    eo.frameRate = Rational{24, 1};
    VideoEncoder enc(dir / "mismatch.mkv", FrameDesc{48, 32, PixelFormat::Yuv420p16, 10}, eo);
    enc.Open();
    CHECK_THROWS_WITH(enc.WriteFrame(SyntheticFrame(0, FrameDesc{48, 32, PixelFormat::Yuv420p, 8})), ContainsSubstring("format mismatch"));
    CHECK_THROWS(VideoEncoder(dir / "bad.mkv", FrameDesc{48, 32, PixelFormat::Yuv420p16, 9}, eo));
}

TEST_CASE("ProRes 422 HQ and 4444 take 10-bit frames and tag the stream", "[integration][formats][encode]") {
    if (!VideoEncoder::EncoderAvailable("prores_ks")) SKIP("prores_ks unavailable");
    const auto dir = Dir("prores");
    VideoEncoder::Options eo;
    eo.color = ColorTags{1, 1, 1, 1};  // limited, BT.709
    const FrameDesc d422{64, 32, PixelFormat::Yuv422p16, 10};
    CHECK(VideoEncoder::PixelFormatFor("prores_ks", d422) == "yuv422p10le");
    const auto hq = RoundTrip(dir / "hq.mov", "prores_ks", d422, 3, eo);
    const StreamSummary s = Summarize(dir / "hq.mov");
    CHECK(s.videoCodec == "prores");
    CHECK(s.pixelFormat == "yuv422p10le");
    CHECK(s.profile == 3);  // HQ
    CHECK(s.colorSpace == 1);  // the `colr` atom carries primaries, transfer and matrix (MOV's nclc has no range flag)
    CHECK(s.colorPrimaries == 1);
    CHECK(s.colorTrc == 1);
    REQUIRE(hq.size() == 3);
    REQUIRE(hq[0].desc == d422);
    int mx = 0;
    const double mean = Diff16(hq[1], SyntheticFrame(1, d422), 0, &mx);
    CHECK(mean < 8.0);  // lossy, but close: 10-bit codes
    CHECK(mx < 256);
    // 4:4:4 -> the 4444 profile (FFmpeg decodes it as 12-bit words: the values are the 10-bit ones times 4)
    const FrameDesc d444{64, 32, PixelFormat::Yuv444p16, 10};
    const auto p4444 = RoundTrip(dir / "4444.mov", "prores_ks", d444, 2, eo);
    const StreamSummary s4 = Summarize(dir / "4444.mov");
    CHECK(s4.profile == 4);
    REQUIRE(p4444.size() == 2);
    CHECK(p4444[0].desc.format == PixelFormat::Yuv444p16);
    CHECK(p4444[0].desc.bitDepth == 12);
    const double mean4 = Diff16(p4444[1], SyntheticFrame(1, d444), 2, &mx);
    CHECK(mean4 < 8.0);
    // no 4:2:0 and no 8-bit ProRes
    CHECK(VideoEncoder::PixelFormatFor("prores_ks", FrameDesc{64, 32, PixelFormat::Yuv420p16, 10}).empty());
    CHECK_THROWS_WITH(VideoEncoder(dir / "no.mov", FrameDesc{64, 32, PixelFormat::Yuv422p, 8}, [] {
                          VideoEncoder::Options o;
                          o.codec = "prores_ks";
                          o.frameRate = Rational{24, 1};
                          return o;
                      }()),
                      ContainsSubstring("ProRes"));
}

TEST_CASE("NVENC 10-bit: HEVC Main 10 and Rext 4:4:4, AV1 10-bit; H.264 refuses", "[integration][formats][encode][gpu]") {
    if (!HaveNvidiaEncoder("hevc_nvenc")) SKIP("no NVIDIA GPU / hevc_nvenc");
    const auto dir = Dir("nvenc");
    VideoEncoder::Options eo;
    eo.color = ColorTags{1, 1, 1, 1};
    const FrameDesc d420{320, 192, PixelFormat::Yuv420p16, 10};  // NVENC refuses tiny frames: the size the other GPU tests use
    CHECK(VideoEncoder::PixelFormatFor("hevc_nvenc", d420) == "p010le");
    const auto main10 = RoundTrip(dir / "main10.mp4", "hevc_nvenc", d420, 4, eo);
    const StreamSummary s = Summarize(dir / "main10.mp4");
    CHECK(s.videoCodec == "hevc");
    CHECK(s.pixelFormat == "yuv420p10le");
    CHECK(s.profile == 2);  // Main 10
    CHECK(s.colorSpace == 1);
    REQUIRE(main10.size() == 4);
    CHECK(main10[0].desc == d420);
    int mx = 0;
    CHECK(Diff16(main10[2], SyntheticFrame(2, d420), 0, &mx) < 24.0);
    const FrameDesc d444{320, 192, PixelFormat::Yuv444p16, 10};
    CHECK(VideoEncoder::PixelFormatFor("hevc_nvenc", d444) == "yuv444p16le");
    const auto rext = RoundTrip(dir / "rext.mp4", "hevc_nvenc", d444, 2, eo);
    const StreamSummary s4 = Summarize(dir / "rext.mp4");
    CHECK(s4.pixelFormat == "yuv444p10le");
    CHECK(s4.profile == 4);  // Rext
    REQUIRE(rext.size() == 2);
    CHECK(rext[0].desc == d444);
    if (VideoEncoder::EncoderAvailable("av1_nvenc")) {
        try {
            const auto av1 = RoundTrip(dir / "av1.mp4", "av1_nvenc", d420, 2, eo);
            CHECK(Summarize(dir / "av1.mp4").pixelFormat == "yuv420p10le");
            CHECK(av1.size() == 2);
        } catch (const std::exception& e) {
            WARN("av1_nvenc 10-bit not available on this GPU: " << e.what());
        }
    }
    // H.264 NVENC lists p010le but the hardware encodes 8-bit only: the open fails with a readable message
    VideoEncoder::Options h264;
    h264.codec = "h264_nvenc";
    h264.frameRate = Rational{24, 1};
    VideoEncoder enc(dir / "h264.mp4", d420, h264);
    CHECK_THROWS_WITH(enc.Open(), ContainsSubstring("bit depth"));
}

#ifdef DLSSVID_WITH_CUDA
TEST_CASE("A 10-bit 4:2:0 source through NVDEC equals the software decode", "[integration][formats][gpu][nvdec]") {
    D3D12Device dev;
    std::string reason;
    if (dev.IsWarp() || !dev.IsNvidia() || !CudaInterop::Available(&reason)) SKIP("no NVIDIA GPU / CUDA: " << reason);
    if (!VideoEncoder::EncoderAvailable("hevc_nvenc")) SKIP("hevc_nvenc unavailable");
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 320;
    spec.height = 192;
    spec.codec = "hevc_nvenc";
    spec.pixFmt = "p010le";
    const auto clip = WriteClip(Dir("nvdec") / "p010.mp4", spec);
    VideoDecoder hw(clip, VideoDecoder::Options{HwAccel::Cuda});
    if (!hw.UsingHwAccel()) SKIP("CUDA hwaccel not available");
    VideoDecoder sw(clip);
    CHECK(hw.Info().decoded == FrameDesc{320, 192, PixelFormat::Yuv420p16, 10});
    CpuFrame cpu, ref;
    GpuFrame gpu;
    int n = 0;
    while (hw.NextFrame(cpu, &gpu, true)) {
        REQUIRE(sw.NextFrame(ref));
        CHECK_FALSE(gpu.Valid());  // P010 is not the NV12 fast path
        REQUIRE(cpu.desc == FrameDesc{320, 192, PixelFormat::Yuv420p16, 10});
        REQUIRE(cpu.desc == ref.desc);
        REQUIRE(cpu.data == ref.data);  // NVDEC's MSB-aligned P010 words shifted down equal the software decoder's 10-bit samples
        CHECK(cpu.Plane16(0)[0] <= 1023);
        ++n;
    }
    CHECK(n == 6);
}
#endif

TEST_CASE("YuvConverter: the identity is a copy, 4:4:4 10-bit to 4:2:0 8-bit keeps the picture", "[integration][formats]") {
    const FrameDesc d444{48, 32, PixelFormat::Yuv444p16, 10};
    const CpuFrame src = SyntheticFrame(3, d444);
    YuvConverter conv;
    CpuFrame same;
    conv.Convert(src, d444, same);
    CHECK(same.data == src.data);
    CHECK(same.index == 3);
    CpuFrame out;
    conv.Convert(src, FrameDesc{0, 0, PixelFormat::Yuv420p, 8}, out);
    REQUIRE(out.desc == FrameDesc{48, 32, PixelFormat::Yuv420p, 8});
    for (uint32_t i = 0; i < 48 * 32; i += 37) CHECK(std::abs(static_cast<int>(out.Plane(0)[i]) - static_cast<int>(src.Plane16(0)[i] >> 2)) <= 1);
    CpuFrame out10;
    conv.Convert(src, FrameDesc{0, 0, PixelFormat::Yuv420p16, 10}, out10);
    CHECK(out10.desc == FrameDesc{48, 32, PixelFormat::Yuv420p16, 10});
    for (uint32_t i = 0; i < 48 * 32; i += 41) CHECK(out10.Plane16(0)[i] == src.Plane16(0)[i]);  // luma untouched, only the chroma is resampled
}

TEST_CASE("Passthrough keeps a 10-bit source and converts when the result layout differs", "[integration][formats][passthrough]") {
    const auto dir = Dir("passthrough");
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    spec.pixFmt = "yuv420p10le";
    spec.audio = true;
    const auto clip = WriteClip(dir / "in10.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "out10.mkv";
    o.codec = "ffv1";
    o.warp = true;
    o.passthrough = true;
    o.progress = nullptr;
    const ProcessResult r = RunProcess(o);
    CHECK(r.outputFormat == FrameDesc{48, 32, PixelFormat::Yuv420p16, 10});
    CHECK(r.outputPixelFormat == "yuv420p10le");
    CHECK(r.codec == "ffv1");
    CHECK(r.ToJson()["output_format"]["bit_depth"] == 10);
    CHECK(r.ToJson()["output_format"]["chroma"] == "420");
    CHECK(r.ToJson()["output_format"]["name"] == "yuv420p10");
    const auto a = DecodeAll(clip), b = DecodeAll(dir / "out10.mkv");
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) REQUIRE(a[i].data == b[i].data);  // bit-exact: the same layout in and out
    CHECK(Summarize(dir / "out10.mkv").audioStreams == 1);
    // asked for 8 bits: converted on the way (the luma code divided by four)
    o.output = dir / "out8.mkv";
    o.outputFormat.bitDepth = 8;
    const ProcessResult r8 = RunProcess(o);
    CHECK(r8.outputFormat == FrameDesc{48, 32, PixelFormat::Yuv420p, 8});
    const auto c = DecodeAll(dir / "out8.mkv");
    REQUIRE(c.size() == a.size());
    CHECK(c[1].desc.format == PixelFormat::Yuv420p);
    for (uint32_t i = 0; i < 48 * 32; i += 29) CHECK(std::abs(static_cast<int>(c[1].Plane(0)[i]) - static_cast<int>(a[1].Plane16(0)[i] >> 2)) <= 1);
    // a 4:4:4 10-bit source into the NVENC-style default 4:2:0 (FFV1 here): chroma resampled, luma kept
    spec.pixFmt = "yuv444p10le";
    spec.audio = false;
    const auto clip444 = WriteClip(dir / "in444.mkv", spec);
    o.input = clip444;
    o.output = dir / "out444to420.mkv";
    o.outputFormat = OutputFormatRequest{0, "420"};
    const ProcessResult r444 = RunProcess(o);
    CHECK(r444.outputFormat == FrameDesc{48, 32, PixelFormat::Yuv420p16, 10});
    const auto d = DecodeAll(dir / "out444to420.mkv");
    REQUIRE(d.size() == 4);
    CHECK(d[0].Plane16(0)[100] == SyntheticFrame(0, FrameDescFor(spec)).Plane16(0)[100]);
    // what the codec cannot do is refused before anything is written
    o.output = dir / "no.mkv";
    o.codec = "h264_nvenc";
    o.outputFormat = OutputFormatRequest{10, "auto"};
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("8-bit only"));
    CHECK_FALSE(std::filesystem::exists(dir / "no.mkv"));
}

TEST_CASE("process keeps the bit depth through the stages, reports the format and fingerprints the source layout", "[integration][formats][process]") {
    const auto dir = Dir("process");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    spec.pixFmt = "yuv420p10le";
    const auto clip = WriteClip(dir / "in10.mkv", spec);
    ProcessOptions o;
    o.input = clip;
    o.output = dir / "result.mkv";
    o.passesRoot = dir / "passes";
    o.stages = StubStages();
    o.codec = "ffv1";
    o.warp = true;
    o.disableUnavailable = true;
    const ProcessResult r = RunProcess(o);
    CHECK(r.outputFormat == FrameDesc{96, 64, PixelFormat::Yuv420p16, 10});  // x2 upscale, the source's 10 bits
    CHECK(r.outputPixelFormat == "yuv420p10le");
    CHECK(Summarize(dir / "result.mkv").pixelFormat == "yuv420p10le");
    const nlohmann::json j = r.ToJson();
    CHECK(j["output_format"]["bit_depth"] == 10);
    CHECK(j["output_format"]["description"] == "10-bit 4:2:0");
    CHECK(j["stages"].back()["details"]["pix_fmt"] == "yuv420p10le");
    // the passes know the layout they were made from; the rerun reuses them
    const Manifest m = Manifest::Load(dir / "passes" / "color_sr");
    CHECK(m.sourceFormat == "yuv420p10");
    CHECK(Manifest::Load(dir / "passes" / "depth_raw").sourceFormat == "yuv420p10");
    const ProcessPlan again = PlanProcess(o);
    for (const auto& d : again.stages) {
        INFO(d.name);
        CHECK(d.action == "reuse");
        CHECK(d.sourceFormat == "yuv420p10");
    }
    CHECK(again.ToJson()["stages"][0]["source_format"] == "yuv420p10");
    // a pass that was made from the old 8-bit 4:2:0 reduction (no source.format in its manifest, a fingerprint without
    // the source layout) is recomputed and says why
    Manifest legacy = Manifest::Load(dir / "passes" / "depth_dlss");  // the stage's last output is the one the plan looks at
    legacy.sourceFormat.clear();
    legacy.fingerprint = ComputeFingerprint("depth", legacy.sourceHash, legacy.paramsCanonical, legacy.inputs, ToolInfo::FromJson(legacy.tool)).value;
    legacy.Save(dir / "passes" / "depth_dlss");
    const ProcessPlan after = PlanProcess(o);
    REQUIRE(after.stages.front().name == "depth");
    CHECK(after.stages.front().action == "run");
    CHECK(after.stages.front().kind == "source_changed");
    CHECK_THAT(after.stages.front().detail, ContainsSubstring("yuv420p (now yuv420p10)"));
    // an 8-bit source: the layout is recorded, the fingerprints are the ones from before (no source_format in the hash)
    ClipSpec spec8;
    spec8.frames = 3;
    spec8.width = 48;
    spec8.height = 32;
    const auto clip8 = WriteClip(dir / "in8.mkv", spec8);
    ProcessOptions o8 = o;
    o8.input = clip8;
    o8.output = dir / "result8.mkv";
    o8.passesRoot = dir / "passes8";
    const ProcessResult r8 = RunProcess(o8);
    CHECK(r8.outputFormat == FrameDesc{96, 64, PixelFormat::Yuv420p, 8});
    CHECK(Manifest::Load(dir / "passes8" / "color_sr").sourceFormat == "yuv420p");
    CHECK(PlanProcess(o8).stages.front().sourceFormat == "yuv420p");
}

TEST_CASE("dlssvid process --bit-depth / --chroma reach the encoder and refuse what the codec cannot do; info prints the layout",
          "[integration][formats][cli]") {
    const auto dir = Dir("cli");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto clip8 = WriteClip(dir / "in8.mkv", spec);
    const CliResult up = RunCli({"process", "--passthrough", "-i", clip8.string(), "-o", (dir / "up10.mkv").string(), "--codec", "ffv1", "--bit-depth", "10", "--chroma", "444", "--warp"});
    CHECK(up.code == 0);
    CHECK(up.out.find("format:      ffv1 10-bit 4:4:4 (yuv444p10le)") != std::string::npos);
    CHECK(Summarize(dir / "up10.mkv").pixelFormat == "yuv444p10le");
    const CliResult bad = RunCli({"process", "--passthrough", "-i", clip8.string(), "-o", (dir / "bad.mp4").string(), "--codec", "h264_nvenc", "--bit-depth", "10", "--warp"});
    CHECK(bad.code != 0);
    CHECK(bad.out.find("8-bit only") != std::string::npos);
    CHECK(RunCli({"process", "--passthrough", "-i", clip8.string(), "-o", (dir / "bad2.mkv").string(), "--codec", "ffv1", "--chroma", "411", "--warp"}).code != 0);
    CHECK(RunCli({"process", "--passthrough", "-i", clip8.string(), "-o", (dir / "bad3.mkv").string(), "--codec", "ffv1", "--bit-depth", "9", "--warp"}).code != 0);
    spec.pixFmt = "yuv444p10le";
    const auto clip10 = WriteClip(dir / "in10.mkv", spec);
    const CliResult info = RunCli({"info", clip10.string(), "--warp"});
    CHECK(info.code == 0);
    CHECK(info.out.find("samples:     10-bit 444, decoded as yuv444p10 (10-bit 4:4:4)") != std::string::npos);
    // the project's settings are used unless the flags say otherwise; the default result name follows the codec
    Project p = Project::Create(clip10, dir / "passes");
    for (auto& st : p.stages) st.enabled = false;
    p.codec = "ffv1";
    p.outputFormat.bitDepth = 8;
    p.Save(dir / "clip.dlssvid.json");
    const CliResult proj = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "--warp"});
    CHECK(proj.code == 0);
    REQUIRE(std::filesystem::exists(dir / "in10_result.mkv"));
    CHECK(Summarize(dir / "in10_result.mkv").pixelFormat == "yuv444p");
    const CliResult flag = RunCli({"process", "--project", (dir / "clip.dlssvid.json").string(), "-o", (dir / "flag.mkv").string(), "--bit-depth", "auto", "--chroma", "420", "--warp"});
    CHECK(flag.code == 0);
    CHECK(Summarize(dir / "flag.mkv").pixelFormat == "yuv420p10le");
}
