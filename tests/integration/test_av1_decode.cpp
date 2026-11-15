// TASK-0076: AV1 sources. This build of FFmpeg has no software AV1 decoder: before the task every AV1 clip failed on its first packet
// («avcodec_send_packet failed: Function not implemented») in the viewport, the stages and `process` without --hwaccel cuda. Now
// VideoDecoder decodes such a stream on NVDEC whatever the options say, seeks in it as in any other, and a machine without NVDEC AV1
// refuses it with the reason before anything is written. The clips are written with av1_nvenc (an NVIDIA GPU; skipped without one).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrame.h"
#include "io/NvdecSupport.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "passes/PassSequence.h"
#include "passes/formats/PngIO.h"
#include "viewport/FrameStore.h"
#include "viewport/Project.h"
#include "viewport/ViewportRenderer.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "av1" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// DLSSVID_FAKE_NVDEC_AV1 for one scope (nullptr: unset); the children dlssvid starts inherit it. The value before it comes back.
struct FakeNvdecEnv {
    std::optional<std::string> before;
    explicit FakeNvdecEnv(const char* value) {
        if (const char* e = std::getenv("DLSSVID_FAKE_NVDEC_AV1")) before = e;
        _putenv_s("DLSSVID_FAKE_NVDEC_AV1", value ? value : "");
    }
    ~FakeNvdecEnv() { _putenv_s("DLSSVID_FAKE_NVDEC_AV1", before ? before->c_str() : ""); }
};

bool HaveAv1Encoder() {
    D3D12Device dev;
    return !dev.IsWarp() && dev.IsNvidia() && VideoEncoder::EncoderAvailable("av1_nvenc");
}

constexpr uint32_t kW = 320, kH = 192;  // NVENC refuses tiny frames: the size the other GPU tests use

// An AV1 clip from av1_nvenc: 8-bit 4:2:0 in MP4, or 10-bit (p010le in, Main 10-bit out) in MKV; a keyframe every 8 frames.
// nullopt when this GPU's encoder does not take it (the test is skipped then).
std::optional<std::filesystem::path> Av1Clip(const std::filesystem::path& dir, bool tenBit, int frames) {
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = frames;
    spec.gop = 8;  // av1_nvenc: longer than its B-frame run + 1
    spec.codec = "av1_nvenc";
    spec.pixFmt = tenBit ? "p010le" : "yuv420p";
    try {
        return WriteClip(dir / (tenBit ? "av1_10.mkv" : "av1_8.mp4"), spec);
    } catch (const std::exception& e) {
        WARN("av1_nvenc did not write the clip: " << e.what());
        return std::nullopt;
    }
}

#define REQUIRE_AV1_CLIP(var, dir, tenBit, frames)                       \
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");         \
    const auto var##Opt = Av1Clip(dir, tenBit, frames);                  \
    if (!var##Opt) SKIP("av1_nvenc did not take the clip on this GPU"); \
    const std::filesystem::path var = *var##Opt

// Mean absolute difference of the luma of a decoded frame (8-bit, or high bit depth brought down to 8) from SyntheticFrame(i).
double LumaDiff(const CpuFrame& f, int64_t i) {
    const CpuFrame s = SyntheticFrame(i, f.desc.width, f.desc.height);
    const size_t n = static_cast<size_t>(f.desc.width) * f.desc.height;
    double sum = 0.0;
    for (size_t k = 0; k < n; ++k) {
        const int v = f.desc.HighBitDepth() ? static_cast<int>(f.Plane16(0)[k] >> (f.desc.bitDepth - 8)) : static_cast<int>(f.Plane(0)[k]);
        sum += std::abs(v - static_cast<int>(s.Plane(0)[k]));
    }
    return sum / static_cast<double>(n);
}

// Every frame with the default options (no --hwaccel): NVDEC, the only decoder of AV1 here.
std::vector<CpuFrame> DecodeSequential(const std::filesystem::path& clip) {
    VideoDecoder dec(clip);
    std::vector<CpuFrame> out;
    CpuFrame f;
    while (dec.NextFrame(f)) out.push_back(f);
    return out;
}

std::string Read(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct CliRun {
    int code = 0;
    std::string out;  // stdout and stderr: FFmpeg's and the log's messages go to stderr
};

CliRun RunAll(const std::vector<std::string>& args, const std::filesystem::path& log) {
    std::string cmd = "\"\"" DLSSVID_CLI_PATH "\"";
    for (const auto& a : args) cmd += " \"" + a + "\"";
    cmd += " > \"" + log.string() + "\" 2>&1\"";
    CliRun r;
    r.code = std::system(cmd.c_str());
    r.out = Read(log);
    return r;
}

// FFmpeg's words of the failure before TASK-0076 - none of them may show again
void CheckNoEnosys(const std::string& out) {
    CHECK_THAT(out, !ContainsSubstring("Function not implemented"));
    CHECK_THAT(out, !ContainsSubstring("hardware accelerated AV1"));
}

// VideoDecoder's get_format took CUDA from av1dec's list at once: FFmpeg did not have to strike the D3D formats offered before it (each
// struck one logs «Invalid / Failed setup for format ...» - a get_format answering the list's first entry gets there only by elimination)
void CheckCudaChosenAtOnce(const std::string& out) {
    CHECK_THAT(out, !ContainsSubstring("setup for format"));
}

bool AnyPartFile(const std::filesystem::path& dir) {
    for (const auto& e : std::filesystem::recursive_directory_iterator(dir))
        if (e.path().filename().string().find(".dlssvid-part") != std::string::npos) return true;
    return false;
}

}  // namespace

TEST_CASE("An AV1 clip decodes on NVDEC with the default options: no device before the first frame, host frames, as with --hwaccel cuda",
          "[integration][gpu][av1]") {
    const auto dir = Dir("decode");
    for (const bool tenBit : {false, true}) {
        DYNAMIC_SECTION((tenBit ? "10-bit MKV" : "8-bit MP4")) {
            REQUIRE_AV1_CLIP(clip, dir, tenBit, 12);
            VideoDecoder dec(clip);
            const VideoStreamInfo& info = dec.Info();
            CHECK(info.codecName == "av1");
            CHECK(info.hardwareOnly);
            CHECK(info.pixelFormat == (tenBit ? "yuv420p10le" : "yuv420p"));
            CHECK(info.bitDepth == (tenBit ? 10 : 8));
            CHECK(info.decoded == (tenBit ? FrameDesc{kW, kH, PixelFormat::Yuv420p16, 10} : FrameDesc{kW, kH, PixelFormat::Yuv420p, 8}));
            // an open that only reads Info() touches no GPU: the device comes with the first frame
            CHECK_FALSE(dec.DecodesOnNvdec());
            CHECK_FALSE(dec.UsingHwAccel());
            std::vector<CpuFrame> frames;
            CpuFrame f;
            while (dec.NextFrame(f)) {
                frames.push_back(f);
                CHECK(dec.DecodesOnNvdec());
                CHECK_FALSE(dec.UsingHwAccel());  // not asked for: its consumers stay on host frames (no GpuFrame)
            }
            REQUIRE(frames.size() == 12);
            for (size_t i = 0; i < frames.size(); ++i) {
                INFO("frame " << i);
                CHECK(frames[i].index == static_cast<int64_t>(i));
                CHECK(frames[i].desc == info.decoded);
                // NVENC is lossy: near its own synthetic frame, far from a later one (the gradient moves by 3 levels a frame)
                const double own = LumaDiff(frames[i], static_cast<int64_t>(i));
                CHECK(own < 3.0);
                CHECK(LumaDiff(frames[i], static_cast<int64_t>(i) + 4) > own + 5.0);
            }

            // --hwaccel cuda: the device at the open, GpuFrames for 8-bit NV12, the same host frames bit for bit
            VideoDecoder hw(clip, VideoDecoder::Options{HwAccel::Cuda});
            CHECK(hw.UsingHwAccel());
            CHECK(hw.DecodesOnNvdec());
            CpuFrame cpu;
            GpuFrame gpu;
            size_t n = 0;
            while (hw.NextFrame(cpu, &gpu, true)) {
                REQUIRE(n < frames.size());
                INFO("frame " << n);
                CHECK(gpu.Valid() == !tenBit);  // P010 is not the NV12 fast path
                CHECK(cpu.index == frames[n].index);
                CHECK(cpu.desc == frames[n].desc);
                CHECK(cpu.data == frames[n].data);
                ++n;
            }
            CHECK(n == frames.size());
        }
    }
}

TEST_CASE("AV1 on NVDEC seeks like a sequential decode: SeekToFrame both ways and before the first frame, SeekToKeyframeBefore",
          "[integration][gpu][av1][seek]") {
    const auto dir = Dir("seek");
    for (const bool tenBit : {false, true}) {
        DYNAMIC_SECTION((tenBit ? "10-bit MKV" : "8-bit MP4")) {
            REQUIRE_AV1_CLIP(clip, dir, tenBit, 64);  // longer than kSequentialSeekFrames: jumps go through the demuxer
            const std::vector<CpuFrame> ref = DecodeSequential(clip);
            REQUIRE(ref.size() == 64);
            static_assert(kSequentialSeekFrames < 55, "the jumps below must not be decoded forward");

            // a seek before any frame was decoded: the demuxer moves on the decoder not yet on the device, NVDEC starts with the frames
            // SeekToFrame decodes up to the target
            {
                VideoDecoder dec(clip);
                CHECK_FALSE(dec.DecodesOnNvdec());
                REQUIRE(dec.SeekToFrame(60));
                CHECK(dec.DecodesOnNvdec());
                CpuFrame f;
                REQUIRE(dec.NextFrame(f));
                CHECK(f.index == 60);
                CHECK(f.data == ref[60].data);
                // then both ways, the long jump back past the forward window too
                for (const int64_t k : {5, 6, 30, 63, 0, 50, 49, 2}) {
                    INFO("seek to " << k);
                    REQUIRE(dec.SeekToFrame(k));
                    REQUIRE(dec.NextFrame(f));
                    CHECK(f.index == k);
                    CHECK(f.data == ref[static_cast<size_t>(k)].data);
                }
                CHECK(dec.Seeks() >= 5);
            }
            // a short forward seek before the first frame: decoded forward from the start
            {
                VideoDecoder dec(clip);
                REQUIRE(dec.SeekToFrame(7));
                CpuFrame f;
                REQUIRE(dec.NextFrame(f));
                CHECK(f.index == 7);
                CHECK(f.data == ref[7].data);
            }
            // SeekToKeyframeBefore, also as the first call (it only moves the demuxer: still no device until the frame after it), then the
            // frames from that keyframe on carry their own indices and pictures
            {
                VideoDecoder dec(clip);
                for (const int64_t k : {45, 10, 62}) {
                    INFO("keyframe before " << k);
                    REQUIRE(dec.SeekToKeyframeBefore(k));
                    if (k == 45) CHECK_FALSE(dec.DecodesOnNvdec());
                    CpuFrame f;
                    REQUIRE(dec.NextFrame(f));
                    CHECK(f.index <= k);
                    CHECK(f.index >= k - 16);
                    CHECK(f.data == ref[static_cast<size_t>(f.index)].data);
                    while (f.index < k) {
                        REQUIRE(dec.NextFrame(f));
                        CHECK(f.data == ref[static_cast<size_t>(f.index)].data);
                    }
                    CHECK(f.index == k);
                }
            }
        }
    }
}

TEST_CASE("FrameStore shows an AV1 video: frames ready at any jump, texels as the NVDEC decode", "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("store");
    for (const bool tenBit : {false, true}) {
        DYNAMIC_SECTION((tenBit ? "10-bit MKV" : "8-bit MP4")) {
            REQUIRE_AV1_CLIP(clip, dir, tenBit, 64);
            const std::vector<CpuFrame> ref = DecodeSequential(clip);
            REQUIRE(ref.size() == 64);
            VideoDecoder probe(clip, VideoDecoder::Probe());
            const ColorInfo ci = ColorInfoFromStream(probe.Info());

            D3D12Device dev({true, false});
            FrameStore::Options opt;
            opt.prefetch = 1;
            FrameStore store(dev, opt);
            store.SetSources(Project::Create(clip, dir / "none").Sources());
            REQUIRE(store.Sources().size() == 1);
            const ViewportSource* src = store.FindSource("source");
            REQUIRE(src);
            CHECK_FALSE(src->undecodable);
            CHECK(store.FrameCount() == 64);
            ViewportRenderer r(dev);
            auto check = [&](int64_t f) {
                INFO("frame " << f);
                store.SetCurrentFrame(f, {"source"});
                store.WaitForCurrent();
                REQUIRE(store.GetStatus(f, "source") == FrameStore::Status::Ready);
                const FrameTextures tex = store.Textures(f);
                REQUIRE(tex.count("source") == 1);
                const LayerTexture& t = tex.at("source");
                CHECK(t.width == kW);
                const PassImage rgb = YuvToRgb(ref[static_cast<size_t>(f)], ci, PixelType::F16);
                for (auto [x, y] : {std::pair<uint32_t, uint32_t>{5, 7}, {200, 100}, {kW - 1, kH - 1}}) {
                    const auto v = r.ReadTexel(t, x, y);
                    CHECK(v[0] == rgb.Get(x, y, 0));
                    CHECK(v[1] == rgb.Get(x, y, 1));
                    CHECK(v[2] == rgb.Get(x, y, 2));
                }
            };
            check(40);  // a jump forward past the window
            check(3);   // back
            check(63);  // the last frame
            check(0);
            CHECK(store.Stats("source").decodedFrames > 0);
        }
    }
}

TEST_CASE("FrameStore opens no decoder for a source marked undecodable: its frames are missing; the same video unmarked shows",
          "[integration][viewport][av1]") {
    // no GPU: an FFV1 clip stands in for the video VideoSource found undecodable (what it marks an AV1 one with when NVDEC lacks AV1)
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("store_undecodable");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const auto found = FrameStore::VideoSource("source", clip);
    REQUIRE(found);
    CHECK_FALSE(found->undecodable);  // FFV1 decodes in software: never marked
    ViewportSource marked = *found;
    VideoDecoder probe(clip, VideoDecoder::Probe());
    marked.undecodable = RefusedDecode(probe.Info(), DecodeBlock::NoAv1, "test");

    D3D12Device dev({true, false});
    FrameStore::Options opt;
    opt.prefetch = 1;
    {
        FrameStore store(dev, opt);
        store.SetSources({marked});
        REQUIRE(store.FindSource("source"));
        CHECK(store.FindSource("source")->undecodable);
        store.SetCurrentFrame(2, {"source"});
        store.WaitForCurrent();
        CHECK(store.GetStatus(2, "source") == FrameStore::Status::Missing);
        CHECK(store.Stats("source").decodedFrames == 0);
        CHECK(store.Textures(2).count("source") == 0);
    }
    {
        FrameStore store(dev, opt);
        store.SetSources({*found});
        store.SetCurrentFrame(2, {"source"});
        store.WaitForCurrent();
        CHECK(store.GetStatus(2, "source") == FrameStore::Status::Ready);
        CHECK(store.Stats("source").decodedFrames > 0);
    }
}

TEST_CASE("dlssvid process --passthrough, render, upscale and info take an AV1 source without --hwaccel (regression TASK-0076)",
          "[integration][cli][gpu][av1][regression]") {
    const auto dir = Dir("cli");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 12);

    // process --passthrough: before the task «avcodec_send_packet failed: Function not implemented» on the first packet
    const auto pt = RunAll({"process", "-i", clip.string(), "-o", (dir / "pt.mkv").string(), "--passthrough", "--warp", "--codec", "ffv1"}, dir / "pt.log");
    INFO(pt.out);
    CHECK(pt.code == 0);
    CheckNoEnosys(pt.out);
    CheckCudaChosenAtOnce(pt.out);
    const std::vector<CpuFrame> out = DecodeAll(dir / "pt.mkv");
    REQUIRE(out.size() == ref.size());
    for (size_t i = 0; i < out.size(); ++i) CHECK(out[i].data == ref[i].data);  // 8-bit 4:2:0 into FFV1: bit-exact

    // render: a PNG of the frame (the viewport's decoder)
    const auto png = dir / "frame.png";
    const auto rd = RunAll({"render", "--warp", "-i", clip.string(), "--frame", "5", "-o", png.string()}, dir / "render.log");
    INFO(rd.out);
    CHECK(rd.code == 0);
    CheckNoEnosys(rd.out);
    CheckCudaChosenAtOnce(rd.out);
    REQUIRE(std::filesystem::exists(png));
    CHECK(ReadPng(png).width == kW);

    // upscale --backend bicubic: the color_sr pass of every frame
    const auto sr = dir / "passes";
    const auto up = RunAll({"upscale", "--backend", "bicubic", "--warp", "-i", clip.string(), "-o", sr.string()}, dir / "upscale.log");
    INFO(up.out);
    CHECK(up.code == 0);
    CheckNoEnosys(up.out);
    CheckCudaChosenAtOnce(up.out);
    const PassReader reader = PassReader::Open(sr / "color_sr");
    reader.Validate({2 * kW, 2 * kH, 12, PassKind::ColorSr});

    // info: how the stream decodes here
    const auto inf = RunAll({"info", clip.string()}, dir / "info.log");
    CHECK(inf.code == 0);
    CHECK_THAT(inf.out, ContainsSubstring("decode:      NVDEC only (av1: no software AV1 decoder in this build): NVDEC on '"));
    CHECK_THAT(inf.out, ContainsSubstring("AV1 8-bit 4:2:0 up to "));
    // a video with a software decoder says so
    ClipSpec plain;
    plain.frames = 2;
    plain.width = 48;
    plain.height = 32;
    const auto ffv1 = RunAll({"info", WriteClip(dir / "ffv1.mkv", plain).string()}, dir / "info_ffv1.log");
    CHECK(ffv1.code == 0);
    CHECK_THAT(ffv1.out, ContainsSubstring("decode:      software (ffv1), NVDEC with --hwaccel cuda"));
}

TEST_CASE("Without NVDEC AV1 every command refuses an AV1 source with the reason, before it writes anything (DLSSVID_FAKE_NVDEC_AV1)",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("refused");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);  // written on the real GPU; refused below by the stand-in for a GPU without AV1 in its NVDEC
    FakeNvdecEnv fake("no-av1");
    const std::string reason = "the GPU 'DLSSVID_FAKE_NVDEC_AV1' has no NVDEC AV1 decoder for 8-bit 4:2:0";

    // info still describes the file and says why it cannot be decoded
    const auto inf = RunAll({"info", clip.string()}, dir / "info.log");
    CHECK(inf.code == 0);
    CHECK_THAT(inf.out, ContainsSubstring("video:       av1 320x192"));
    CHECK_THAT(inf.out, ContainsSubstring("decode:      unavailable - " + reason));

    // process --passthrough: refused before the part file (no «passthrough: ...» line - the encoder was never opened)
    const auto result = dir / "pt.mkv";
    const auto pt = RunAll({"process", "-i", clip.string(), "-o", result.string(), "--passthrough", "--warp", "--codec", "ffv1"}, dir / "pt.log");
    {
        INFO(pt.out);
        CHECK(pt.code != 0);
        CHECK_THAT(pt.out, ContainsSubstring("process: " + clip.string() + ": " + reason));
        CHECK_THAT(pt.out, !ContainsSubstring("passthrough: ffv1"));
        CheckNoEnosys(pt.out);
        CHECK_FALSE(std::filesystem::exists(result));
        CHECK_FALSE(AnyPartFile(dir));
    }

    // process with stages: refused before the first stage - no passes root, no result
    const auto passes = dir / "passes";
    const auto staged = dir / "staged.mkv";
    const auto st = RunAll({"process", "-i", clip.string(), "-o", staged.string(), "--passes", passes.string(), "--warp", "--codec", "ffv1", "--stages", "upscale",
                            "--upscale-backend", "bicubic"},
                           dir / "stages.log");
    {
        INFO(st.out);
        CHECK(st.code != 0);
        CHECK_THAT(st.out, ContainsSubstring("process: " + clip.string() + ": " + reason));
        CheckNoEnosys(st.out);
        CHECK_FALSE(std::filesystem::exists(passes));
        CHECK_FALSE(std::filesystem::exists(staged));
        CHECK_FALSE(AnyPartFile(dir));
    }
    // ... and its plan (PlanProcess: the GUI's «Обработать» is planned the same way)
    const auto plan = RunAll({"process", "-i", clip.string(), "-o", staged.string(), "--passes", passes.string(), "--warp", "--stages", "upscale",
                              "--upscale-backend", "bicubic", "--plan"},
                             dir / "plan.log");
    {
        INFO(plan.out);
        CHECK(plan.code != 0);
        CHECK_THAT(plan.out, ContainsSubstring("process: " + clip.string() + ": " + reason));
    }

    // render: the reason instead of «has none of the requested sources»
    const auto png = dir / "frame.png";
    const auto rd = RunAll({"render", "--warp", "-i", clip.string(), "--frame", "2", "-o", png.string()}, dir / "render.log");
    {
        INFO(rd.out);
        CHECK(rd.code != 0);
        CHECK_THAT(rd.out, ContainsSubstring("render: 'source': " + reason));
        CHECK_THAT(rd.out, !ContainsSubstring("has none of the requested sources"));
        CheckNoEnosys(rd.out);
        CHECK_FALSE(std::filesystem::exists(png));
    }

    // upscale decodes the video itself: the decoder's refusal, not FFmpeg's
    const auto up = RunAll({"upscale", "--backend", "bicubic", "--warp", "-i", clip.string(), "-o", (dir / "sr").string()}, dir / "upscale.log");
    {
        INFO(up.out);
        CHECK(up.code != 0);
        CHECK_THAT(up.out, ContainsSubstring(reason));
        CheckNoEnosys(up.out);
        CHECK_FALSE(std::filesystem::exists(dir / "sr" / "color_sr"));
    }
}

TEST_CASE("VideoDecoder and FrameStore::VideoSource refuse an AV1 stream NVDEC cannot decode, with the reason", "[integration][gpu][av1]") {
    const auto dir = Dir("refused_inproc");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    {
        const auto source = FrameStore::VideoSource("source", clip);
        REQUIRE(source);
        CHECK_FALSE(source->undecodable);  // this GPU decodes it
    }
    for (const char* fake : {"no-av1", "no-cuda"}) {
        DYNAMIC_SECTION(fake) {
            FakeNvdecEnv env(fake);
            const DecodeBlock expected = std::string(fake) == "no-av1" ? DecodeBlock::NoAv1 : DecodeBlock::NoCuda;
            for (const HwAccel hw : {HwAccel::None, HwAccel::Cuda}) {
                INFO((hw == HwAccel::Cuda ? "--hwaccel cuda" : "default options"));
                VideoDecoder dec(clip, VideoDecoder::Options{hw});
                CHECK(dec.Info().hardwareOnly);  // the open itself is fine: Info() of an undecodable stream
                CpuFrame f;
                bool refused = false;
                try {
                    dec.NextFrame(f);
                } catch (const DecodeUnavailable& e) {
                    refused = true;
                    CHECK(e.support.block == expected);
                    CHECK_THAT(std::string(e.what()), ContainsSubstring(clip.string() + ": "));
                    CHECK_THAT(std::string(e.what()), ContainsSubstring(std::string("DLSSVID_FAKE_NVDEC_AV1=") + fake));
                }
                CHECK(refused);
                if (hw == HwAccel::None) CHECK_FALSE(dec.DecodesOnNvdec());  // refused before the device was made
            }
            const auto source = FrameStore::VideoSource("source", clip);
            REQUIRE(source);  // listed, with its size and duration
            CHECK(source->width == kW);
            CHECK(source->lastFrame == 7);
            REQUIRE(source->undecodable);
            CHECK(source->undecodable->block == expected);
            CHECK_THAT(source->undecodable->why, ContainsSubstring(std::string("DLSSVID_FAKE_NVDEC_AV1=") + fake));
        }
    }
}
