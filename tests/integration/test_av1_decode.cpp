// TASK-0076: AV1 sources. This build of FFmpeg has no software AV1 decoder: before the task every AV1 clip failed on its first packet
// («avcodec_send_packet failed: Function not implemented») in the viewport, the stages and `process` without --hwaccel cuda. Now
// VideoDecoder decodes such a stream on NVDEC whatever the options say, seeks in it as in any other, and a machine without NVDEC AV1
// refuses it with the reason before anything is written. The clips are written with av1_nvenc (an NVIDIA GPU; skipped without one).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrame.h"
#include "io/NvdecSupport.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "passes/Manifest.h"
#include "passes/PassSequence.h"
#include "passes/formats/PngIO.h"
#include "viewport/FrameStore.h"
#include "viewport/Project.h"
#include "viewport/ViewportProxy.h"
#include "viewport/ViewportRenderer.h"

#ifdef DLSSVID_WITH_CUDA
#include <cuda.h>
#endif

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

bool HaveNvidia() {
    D3D12Device dev;
    return !dev.IsWarp() && dev.IsNvidia();
}

bool HaveAv1Encoder() { return HaveNvidia() && VideoEncoder::EncoderAvailable("av1_nvenc"); }

constexpr uint32_t kW = 320, kH = 192;  // NVENC refuses tiny frames: the size the other GPU tests use

// An AV1 clip from av1_nvenc: 8-bit 4:2:0 in MP4, or 10-bit (p010le in, Main 10-bit out) in MKV; a keyframe every 8 frames.
// nullopt when this GPU's encoder says it does not support it (`why`: its words); any other failure throws - the test fails.
std::optional<std::filesystem::path> Av1Clip(const std::filesystem::path& dir, bool tenBit, int frames, std::string* why) {
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = frames;
    spec.gop = 8;  // av1_nvenc: longer than its B-frame run + 1
    spec.codec = "av1_nvenc";
    spec.pixFmt = tenBit ? "p010le" : "yuv420p";
    return WriteClipIfSupported(dir / (tenBit ? "av1_10.mkv" : "av1_8.mp4"), spec, why);
}

// SKIP only without an NVIDIA GPU with av1_nvenc, or when the encoder says the clip is not supported here
#define REQUIRE_AV1_CLIP(var, dir, tenBit, frames)                                 \
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");                   \
    std::string var##Why;                                                          \
    const auto var##Opt = Av1Clip(dir, tenBit, frames, &var##Why);                 \
    if (!var##Opt) SKIP("av1_nvenc does not support the clip here: " << var##Why); \
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

            // --hwaccel cuda: the device at the open (DecodesOnNvdec only with a frame from it), GpuFrames for 8-bit NV12, the same host
            // frames bit for bit
            VideoDecoder hw(clip, VideoDecoder::Options{HwAccel::Cuda});
            CHECK(hw.UsingHwAccel());
            CHECK_FALSE(hw.DecodesOnNvdec());
            CpuFrame cpu;
            GpuFrame gpu;
            size_t n = 0;
            while (hw.NextFrame(cpu, &gpu, true)) {
                REQUIRE(n < frames.size());
                INFO("frame " << n);
                CHECK(hw.DecodesOnNvdec());
                CHECK(gpu.Valid() == !tenBit);  // P010 is not the NV12 fast path
                CHECK(cpu.index == frames[n].index);
                CHECK(cpu.desc == frames[n].desc);
                CHECK(cpu.data == frames[n].data);
                ++n;
            }
            CHECK(n == frames.size());

            // a GpuFrame asked for without --hwaccel cuda: none is handed out, the host frame is filled even when skipping it was asked
            VideoDecoder unasked(clip);
            n = 0;
            while (unasked.NextFrame(cpu, &gpu, false)) {
                REQUIRE(n < frames.size());
                INFO("frame " << n << " without --hwaccel cuda");
                CHECK_FALSE(gpu.Valid());
                CHECK(gpu.y == 0);
                CHECK(cpu.desc == frames[n].desc);
                CHECK(cpu.data == frames[n].data);
                ++n;
            }
            CHECK(n == frames.size());
            CHECK(unasked.DecodesOnNvdec());
            CHECK_FALSE(unasked.UsingHwAccel());
        }
    }
}

TEST_CASE("DecodesOnNvdec says where the frames came from: an FFV1 clip with --hwaccel cuda decodes in software", "[integration][gpu][av1]") {
    if (!HaveNvidia()) SKIP("no NVIDIA GPU");  // a CUDA device for HwAccel::Cuda
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(Dir("ffv1_cuda") / "clip.mkv", spec);
    VideoDecoder dec(clip, VideoDecoder::Options{HwAccel::Cuda});
    CHECK_FALSE(dec.Info().hardwareOnly);
    CHECK_FALSE(dec.Info().nvdecCodec);
    if (!dec.UsingHwAccel()) SKIP("CUDA hwaccel not available");
    CHECK_NOTHROW(dec.RequireDecodable());  // a software decoder: nothing to refuse
    CpuFrame f;
    GpuFrame gpu;
    int n = 0;
    while (dec.NextFrame(f, &gpu, true)) {
        CHECK_FALSE(gpu.Valid());
        CHECK(f.data == SyntheticFrame(n, 48, 32).data);
        ++n;
    }
    CHECK(n == 3);
    CHECK(dec.UsingHwAccel());        // asked for, the device is up
    CHECK_FALSE(dec.DecodesOnNvdec());  // but FFV1 has no NVDEC: no frame came from it
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
    CHECK_THAT(ffv1.out, ContainsSubstring("decode:      software (ffv1); NVDEC does not decode this codec"));
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

namespace {
// NextFrame's DecodeUnavailable (FAIL when a frame comes or another error is thrown)
std::optional<DecodeUnavailable> RefusalOfNextFrame(VideoDecoder& dec) {
    CpuFrame f;
    try {
        if (dec.NextFrame(f)) FAIL("a frame was decoded: " << f.index);
        else FAIL("the stream ended without a refusal");
    } catch (const DecodeUnavailable& e) {
        return e;
    } catch (const std::exception& e) {
        FAIL("another error than DecodeUnavailable: " << e.what());
    }
    return std::nullopt;
}

// A FrameStore of one video source on the WARP adapter, and what the viewport would show of frame `f`
struct Av1Store {
    D3D12Device dev{D3D12Device::Options{true, false}};
    std::unique_ptr<FrameStore> store;
    ViewportRenderer renderer{dev};
    explicit Av1Store(int prefetch = 1, int loaderThreads = 0) {
        FrameStore::Options opt;
        opt.prefetch = prefetch;
        opt.loaderThreads = loaderThreads;
        store = std::make_unique<FrameStore>(dev, opt);
    }
    // The frame's settled status: WaitForCurrent returns once it is resident or marked missing, and a frame a prefetch marked missing
    // may be asked for again as the current one - its status is Loading meanwhile
    FrameStore::Status Load(int64_t f) {
        store->SetCurrentFrame(f, {"source"});
        store->WaitForCurrent();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        FrameStore::Status st = store->GetStatus(f, "source");
        while ((st == FrameStore::Status::Loading || st == FrameStore::Status::Queued) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            store->Update();
            st = store->GetStatus(f, "source");
        }
        return st;
    }
    // the frame's texels equal the colour conversion of `ref` (a decode on NVDEC)
    void CheckTexels(int64_t f, const CpuFrame& ref, const ColorInfo& ci) {
        const FrameTextures tex = store->Textures(f);
        REQUIRE(tex.count("source") == 1);
        const LayerTexture& t = tex.at("source");
        const PassImage rgb = YuvToRgb(ref, ci, PixelType::F16);
        for (auto [x, y] : {std::pair<uint32_t, uint32_t>{5, 7}, {200, 100}, {kW - 1, kH - 1}}) {
            const auto v = renderer.ReadTexel(t, x, y);
            CHECK(v[0] == rgb.Get(x, y, 0));
            CHECK(v[1] == rgb.Get(x, y, 1));
            CHECK(v[2] == rgb.Get(x, y, 2));
        }
    }
};

const std::string kNoAv1Reason = "the GPU 'DLSSVID_FAKE_NVDEC_AV1' has no NVDEC AV1 decoder for 8-bit 4:2:0";
const std::string kFirstFrameReason = "NVDEC refused this AV1 stream and this build has no software AV1 decoder (on the first frame)";
}  // namespace

TEST_CASE("NVDEC refusing the first frame (DLSSVID_FAKE_NVDEC_AV1=first-frame) is a DecodeUnavailable the decoder keeps", "[integration][gpu][av1]") {
    const auto dir = Dir("first_frame");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    std::optional<FakeNvdecEnv> fake;
    fake.emplace("first-frame");
    VideoDecoder dec(clip);
    CHECK_NOTHROW(dec.RequireDecodable());  // the driver told nothing in advance: not refused before a frame
    const auto first = RefusalOfNextFrame(dec);
    REQUIRE(first);
    CHECK(first->support.block == DecodeBlock::NvdecFailed);
    CHECK(first->support.why == kFirstFrameReason);
    CHECK(std::string(first->what()) == clip.string() + ": " + kFirstFrameReason);
    CHECK_FALSE(dec.DecodesOnNvdec());
    // the stand-in gone: what refuses now is the decoder's memory of the refusal - not FFmpeg's next error («Missing Sequence Header»)
    fake.reset();
    for (int i = 0; i < 3; ++i) {
        INFO("NextFrame after the refusal, " << i);
        const auto again = RefusalOfNextFrame(dec);
        REQUIRE(again);
        CHECK(again->support.block == DecodeBlock::NvdecFailed);
        CHECK(std::string(again->what()) == std::string(first->what()));
    }
    try {
        dec.RequireDecodable();
        FAIL("RequireDecodable passed a decoder NVDEC refused");
    } catch (const DecodeUnavailable& e) {
        CHECK(e.support.block == DecodeBlock::NvdecFailed);
        CHECK(std::string(e.what()) == std::string(first->what()));
    }
    // a new decoder of the same file decodes (nothing of the refusal is kept beyond its decoder)
    VideoDecoder fresh(clip);
    CpuFrame f;
    REQUIRE(fresh.NextFrame(f));
    CHECK(f.index == 0);
}

TEST_CASE("RequireDecodable refuses before any frame a stream NVDEC cannot decode; the refusal leaves the decoder usable", "[integration][gpu][av1]") {
    const auto dir = Dir("require");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 8);
    VideoDecoder dec(clip);
    CHECK_NOTHROW(dec.RequireDecodable());  // this GPU decodes it
    {
        FakeNvdecEnv fake("no-av1");
        try {
            dec.RequireDecodable();
            FAIL("RequireDecodable passed without NVDEC AV1");
        } catch (const DecodeUnavailable& e) {
            CHECK(e.support.block == DecodeBlock::NoAv1);
            CHECK_THAT(std::string(e.what()), ContainsSubstring(clip.string() + ": " + kNoAv1Reason));
        }
        CHECK_FALSE(dec.DecodesOnNvdec());
        const auto refused = RefusalOfNextFrame(dec);  // the same at the first frame
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NoAv1);
        CHECK_FALSE(dec.DecodesOnNvdec());
    }
    // the GPU's answer again: the decoder the refusal left behind decodes from the first frame, as a new one would
    CHECK_NOTHROW(dec.RequireDecodable());
    CpuFrame f;
    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE(dec.NextFrame(f));
        CHECK(f.index == static_cast<int64_t>(i));
        CHECK(f.data == ref[i].data);
    }
    CHECK(dec.DecodesOnNvdec());
}

TEST_CASE("FrameStore: NVDEC refusing a video in a loader marks it undecodable until the sources are set again", "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("store_first_frame");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 24);
    const ColorInfo ci = ColorInfoFromStream(VideoDecoder(clip, VideoDecoder::Probe()).Info());
    Av1Store s;
    {
        FakeNvdecEnv fake("first-frame");
        const auto found = FrameStore::VideoSource("source", clip);
        REQUIRE(found);
        CHECK_FALSE(found->undecodable);  // the probe cannot tell: the driver said nothing in advance
        s.store->SetSources({*found});
        CHECK_FALSE(s.store->Undecodable("source"));
        CHECK(s.Load(3) == FrameStore::Status::Missing);
        const auto marked = s.store->Undecodable("source");
        REQUIRE(marked);
        CHECK(marked->block == DecodeBlock::NvdecFailed);
        CHECK(marked->why == kFirstFrameReason);
        // the later frames stay missing - no frame was decoded, none is
        for (const int64_t f : {4, 15, 0}) {
            INFO("frame " << f);
            CHECK(s.Load(f) == FrameStore::Status::Missing);
            CHECK(s.store->Undecodable("source"));
        }
        CHECK(s.store->Stats("source").decodedFrames == 0);
        // set again (a reload): the refusal is forgotten, the next load asks NVDEC again - and is refused again
        s.store->SetSources({*found});
        CHECK_FALSE(s.store->Undecodable("source"));
        CHECK(s.Load(5) == FrameStore::Status::Missing);
        CHECK(s.store->Undecodable("source"));
    }
    // the stand-in gone and the sources set again: the frames show
    s.store->SetSources(Project::Create(clip, dir / "none").Sources());
    CHECK_FALSE(s.store->Undecodable("source"));
    REQUIRE(s.Load(7) == FrameStore::Status::Ready);
    s.CheckTexels(7, ref[7], ci);
}

TEST_CASE("FrameStore shows a video this machine cannot decode through its proxy", "[integration][viewport][gpu][av1][proxy]") {
    const auto dir = Dir("store_proxy");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 24);
    const ColorInfo ci = ColorInfoFromStream(VideoDecoder(clip, VideoDecoder::Probe()).Info());
    const auto passes = dir / "passes";
    BuildProxy(clip, passes, "source");  // made here, where NVDEC decodes AV1 (the proxy is FFV1: software everywhere)
    REQUIRE(ValidProxy(clip, passes, "source"));
    {
        FakeNvdecEnv fake("no-av1");
        Av1Store s;
        s.store->SetSources(Project::Create(clip, passes).Sources());
        const ViewportSource* src = s.store->FindSource("source");
        REQUIRE(src);
        REQUIRE(src->undecodable);  // still undecodable here ...
        CHECK(src->undecodable->block == DecodeBlock::NoAv1);
        CHECK_FALSE(src->proxy.empty());
        for (const int64_t f : {12, 3, 23, 4}) {  // ... and shown through the proxy, jumps and a step alike
            INFO("frame " << f);
            REQUIRE(s.Load(f) == FrameStore::Status::Ready);
            s.CheckTexels(f, ref[static_cast<size_t>(f)], ci);
        }
        CHECK(s.store->Stats("source").proxyLoads > 0);
        CHECK(s.store->Undecodable("source"));
    }
}

TEST_CASE("FrameStore::ReleaseNvdecDecoders closes a video's decoder on NVDEC; the next frames open one again", "[integration][viewport][gpu][av1]") {
    // a run queued in the GUI gets the memory the viewport's NVDEC decoders hold (AppModel::releaseNvdec)
    const auto dir = Dir("store_release");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 24);
    const ColorInfo ci = ColorInfoFromStream(VideoDecoder(clip, VideoDecoder::Probe()).Info());
    Av1Store s(0);  // no prefetch: no load runs after the current frame's
    s.store->SetSources(Project::Create(clip, dir / "none").Sources());  // no proxy: the original's decoder on NVDEC
    REQUIRE(s.Load(2) == FrameStore::Status::Ready);
    CHECK(s.store->Stats("source").decodedFrames > 0);
    s.store->ReleaseNvdecDecoders();
    CHECK(s.store->Stats("source").decodedFrames == 0);  // the stats count the decoders open: none is
    for (const int64_t f : {3, 20}) {
        INFO("frame " << f << " after the release");
        REQUIRE(s.Load(f) == FrameStore::Status::Ready);
        s.CheckTexels(f, ref[static_cast<size_t>(f)], ci);
    }
    CHECK(s.store->Stats("source").decodedFrames > 0);
}

TEST_CASE("BuildProxy refuses a video NVDEC cannot decode before it writes anything", "[integration][gpu][av1][proxy]") {
    const auto dir = Dir("proxy_refused");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    FakeNvdecEnv fake("no-av1");
    const auto passes = dir / "passes";
    try {
        BuildProxy(clip, passes, "source");
        FAIL("BuildProxy built a proxy without NVDEC AV1");
    } catch (const DecodeUnavailable& e) {
        CHECK(e.support.block == DecodeBlock::NoAv1);
    }
    CHECK_FALSE(ValidProxy(clip, passes, "source"));
    CHECK_FALSE(std::filesystem::exists(ProxyFile(passes, "source")));
    CHECK_FALSE(std::filesystem::exists(ProxyDir(passes) / "source.building.mkv"));
}

TEST_CASE("Without NVDEC AV1 the stage commands refuse an AV1 source at once: the reason, no GPU device, no pass written",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("refused_commands");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    // a scenes pass made where NVDEC decodes AV1: its scores would let scenes re-derive the cuts without decoding - the refusal comes
    // first, the pass is not retired to its history
    const auto scenesRoot = dir / "scenes_root";
    REQUIRE(RunAll({"scenes", "-i", clip.string(), "-o", scenesRoot.string()}, dir / "scenes_first.log").code == 0);
    REQUIRE(std::filesystem::exists(scenesRoot / "scenes"));
    FakeNvdecEnv fake("no-av1");
    const auto out = dir / "passes";
    struct Command {
        const char* name;
        std::vector<std::string> args;
        std::filesystem::path pass;  // what it would write
    };
    const std::string c = clip.string(), o = out.string();
    const Command commands[] = {
        {"depth", {"depth", "-i", c, "-o", o, "--backend", "stub"}, out / "depth_raw"},
        {"flow", {"flow", "-i", c, "-o", o, "--backend", "stub"}, out / "mv_raw"},
        {"scenes", {"scenes", "-i", c, "-o", o}, out / "scenes"},
        {"fg", {"fg", "-i", c, "-o", o, "--backend", "blend"}, out / "color_fg"},
        {"nr", {"nr", "--backend", "stub", "--warp", "-i", c, "-o", o}, out / "color_nr"},
        {"upscale", {"upscale", "--backend", "bicubic", "--warp", "-i", c, "-o", o}, out / "color_sr"},
        {"export", {"export", "-i", c, "-o", (dir / "exported").string()}, dir / "exported"},
        {"compare", {"compare", "--ref", c, "--test", c, "-q", "--json", (dir / "compare.json").string()}, dir / "compare.json"},
    };
    for (const Command& cmd : commands) {
        const auto r = RunAll(cmd.args, dir / (std::string(cmd.name) + ".log"));
        INFO(cmd.name << ":\n" << r.out);
        CHECK(r.code != 0);
        CHECK_THAT(r.out, ContainsSubstring(c + ": " + kNoAv1Reason));
        CHECK_THAT(r.out, !ContainsSubstring("D3D12 device"));  // refused before the GPU work, the model and the hash
        CheckNoEnosys(r.out);
        CHECK_FALSE(std::filesystem::exists(cmd.pass));
        if (std::string(cmd.name) == "compare") {  // refused at its first video: the second is not opened
            size_t opens = 0;
            for (size_t at = r.out.find("] open "); at != std::string::npos; at = r.out.find("] open ", at + 1)) ++opens;
            CHECK(opens == 1);
        }
    }
    const auto sc = RunAll({"scenes", "-i", c, "-o", scenesRoot.string()}, dir / "scenes_again.log");
    INFO("scenes over its earlier pass:\n" << sc.out);
    CHECK(sc.code != 0);
    CHECK_THAT(sc.out, ContainsSubstring(c + ": " + kNoAv1Reason));
    CHECK(std::filesystem::exists(scenesRoot / "scenes"));
    CHECK_FALSE(std::filesystem::exists(scenesRoot / "scenes.v"));
}

TEST_CASE("dlssvid proxy and render --proxy build skip a video this machine cannot decode, saying why", "[integration][cli][gpu][av1][proxy]") {
    const auto dir = Dir("proxy_cli");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    FakeNvdecEnv fake("no-av1");
    const auto passes = dir / "passes";
    const std::string said = "source: no proxy, the video cannot be decoded here: " + kNoAv1Reason;
    for (const bool force : {false, true}) {
        std::vector<std::string> args{"proxy", "-i", clip.string(), "--passes", passes.string(), "--sources", "source"};
        if (force) args.push_back("--force");
        const auto r = RunAll(args, dir / (force ? "proxy_force.log" : "proxy.log"));
        INFO((force ? "--force" : "") << "\n" << r.out);
        CHECK(r.code == 0);  // the other videos and the render go on
        CHECK_THAT(r.out, ContainsSubstring(said));
        CHECK_FALSE(ValidProxy(clip, passes, "source"));
        CHECK_FALSE(std::filesystem::exists(ProxyDir(passes) / "source.building.mkv"));
    }
    const auto png = dir / "frame.png";
    const auto rd = RunAll({"render", "--warp", "-i", clip.string(), "--passes", passes.string(), "--proxy", "build", "--frame", "2", "-o", png.string()},
                           dir / "render.log");
    INFO(rd.out);
    CHECK(rd.code != 0);
    CHECK_THAT(rd.out, ContainsSubstring(said));
    CHECK_THAT(rd.out, ContainsSubstring("render: 'source': " + kNoAv1Reason));
    CHECK_FALSE(std::filesystem::exists(png));
}

TEST_CASE("NVDEC refusing the first frame: render and process say why, nothing is written", "[integration][cli][gpu][av1]") {
    const auto dir = Dir("first_frame_cli");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    FakeNvdecEnv fake("first-frame");
    const auto png = dir / "frame.png";
    const auto rd = RunAll({"render", "--warp", "-i", clip.string(), "--frame", "2", "-o", png.string()}, dir / "render.log");
    {
        INFO(rd.out);
        CHECK(rd.code != 0);
        CHECK_THAT(rd.out, ContainsSubstring("render: 'source': " + kFirstFrameReason));
        CHECK_THAT(rd.out, !ContainsSubstring("has none of the requested sources"));
        CHECK_THAT(rd.out, !ContainsSubstring("Function not implemented"));
        CHECK_FALSE(std::filesystem::exists(png));
    }
    // process: the check before the run cannot tell (the driver said nothing) - the decoder's refusal on the first frame ends it, the
    // partial result goes with it
    const auto result = dir / "pt.mkv";
    const auto pt = RunAll({"process", "-i", clip.string(), "-o", result.string(), "--passthrough", "--warp", "--codec", "ffv1"}, dir / "pt.log");
    {
        INFO(pt.out);
        CHECK(pt.code != 0);
        CHECK_THAT(pt.out, ContainsSubstring(clip.string() + ": " + kFirstFrameReason));
        CHECK_THAT(pt.out, !ContainsSubstring("Function not implemented"));
        CHECK_FALSE(std::filesystem::exists(result));
        CHECK_FALSE(AnyPartFile(dir));
    }
}

namespace {
// Whether the primary CUDA context of device 0 is active - held by someone in this process (an NVDEC decoder's device, the CUDA runtime);
// nullopt: no CUDA here
std::optional<bool> PrimaryContextActive() {
#ifdef DLSSVID_WITH_CUDA
    CUdevice dev = 0;
    unsigned flags = 0;
    int active = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS || cuDevicePrimaryCtxGetState(dev, &flags, &active) != CUDA_SUCCESS)
        return std::nullopt;
    return active != 0;
#else
    return std::nullopt;
#endif
}
}  // namespace

TEST_CASE("FrameStore lets the CUDA device go: a decoder NVDEC refused is closed at once, ReleaseNvdecDecoders closes the one decoding",
          "[integration][viewport][gpu][av1]") {
    // The primary context goes inactive when its last holder closes - checkable only where nothing else holds it: the CUDA runtime keeps
    // it to the process's end once started (VideoDecoder starts it only to align the flags of a context already active), so the case
    // checks first that nothing in this process holds it - ctest runs each case in a process of its own
    const auto before = PrimaryContextActive();
    if (!before) SKIP("no CUDA driver here");
    if (*before) SKIP("the primary CUDA context is held by an earlier case in this process: run this case alone");
    const auto dir = Dir("store_cuda");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    CHECK(PrimaryContextActive() == false);  // av1_nvenc wrote the clip in a context of its own
    {
        // NVDEC refuses the first frame: the loader closes the refused decoder - its CUDA device - at once
        FakeNvdecEnv fake("first-frame");
        Av1Store s(0, 1);
        const auto found = FrameStore::VideoSource("source", clip);
        REQUIRE(found);
        s.store->SetSources({*found});
        REQUIRE(s.Load(3) == FrameStore::Status::Missing);
        REQUIRE(s.store->Undecodable("source"));
        CHECK(PrimaryContextActive() == false);
        s.store->ReleaseNvdecDecoders();  // nothing left to let go
        CHECK(PrimaryContextActive() == false);
    }
    // a decoder on NVDEC holds it until ReleaseNvdecDecoders; the next frame opens one again; the store closing lets it go too
    {
        Av1Store s(0, 1);
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        REQUIRE(s.Load(2) == FrameStore::Status::Ready);
        CHECK(PrimaryContextActive() == true);
        s.store->ReleaseNvdecDecoders();
        CHECK(PrimaryContextActive() == false);
        REQUIRE(s.Load(9) == FrameStore::Status::Ready);
        CHECK(PrimaryContextActive() == true);
    }
    CHECK(PrimaryContextActive() == false);
    // the same for a decoder asked for HwAccel::Cuda (the device at the open): closing it lets the context go
    {
        VideoDecoder dec(clip, VideoDecoder::Options{HwAccel::Cuda});
        CHECK(dec.HoldsCudaDevice());
        CHECK(PrimaryContextActive() == true);
    }
    CHECK(PrimaryContextActive() == false);
}

TEST_CASE("HoldsCudaDevice: from the device's creation to the decoder's end - also when NVDEC then refused the stream", "[integration][gpu][av1]") {
    const auto dir = Dir("holds_device");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    {
        VideoDecoder dec(clip);
        CHECK_FALSE(dec.HoldsCudaDevice());  // no device before the first frame
        CpuFrame f;
        REQUIRE(dec.NextFrame(f));
        CHECK(dec.HoldsCudaDevice());
        CHECK(dec.DecodesOnNvdec());
    }
    {
        FakeNvdecEnv fake("first-frame");
        VideoDecoder dec(clip);
        CHECK(RefusalOfNextFrame(dec));
        CHECK(dec.HoldsCudaDevice());  // refused, still holding it: FrameStore closes such a decoder
        CHECK_FALSE(dec.DecodesOnNvdec());
    }
    {
        FakeNvdecEnv fake("no-av1");
        VideoDecoder dec(clip);
        CHECK(RefusalOfNextFrame(dec));
        CHECK_FALSE(dec.HoldsCudaDevice());  // refused before the device was made
    }
}

TEST_CASE("dlssvid nr without NVDEC AV1: refused when its colour comes from the video, not when the colour pass serves the range",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("nr_refused");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const std::string c = clip.string();
    const auto passes = dir / "passes";
    // the colour pass, made where NVDEC decodes AV1
    REQUIRE(RunAll({"upscale", "--backend", "bicubic", "--warp", "-i", c, "-o", passes.string()}, dir / "upscale.log").code == 0);
    REQUIRE(std::filesystem::exists(passes / "color_sr"));
    FakeNvdecEnv fake("no-av1");
    // from frame 0, or a warm-up reaching back to it: NR walks the video even with the colour pass - refused before the hash, the GPU, NGX
    struct Refused {
        const char* name;
        std::vector<std::string> range;
    };
    const Refused refused[] = {{"whole", {}}, {"warmup to 0", {"--start", "2", "--frames", "2", "--warmup", "2"}}};
    for (const Refused& r : refused) {
        std::vector<std::string> args{"nr", "--backend", "stub", "--warp", "-i", c, "-o", passes.string()};
        args.insert(args.end(), r.range.begin(), r.range.end());
        const auto run = RunAll(args, dir / "nr_refused.log");
        INFO(r.name << ":\n" << run.out);
        CHECK(run.code != 0);
        CHECK_THAT(run.out, ContainsSubstring(c + ": " + kNoAv1Reason));
        CHECK_THAT(run.out, !ContainsSubstring("D3D12 device"));
        CheckNoEnosys(run.out);
        CHECK_FALSE(std::filesystem::exists(passes / "color_nr"));
    }
    // a range past frame 0 with the colour pass: its frames come from color_sr by index - the video is not decoded, nothing refuses
    const auto part = RunAll({"nr", "--backend", "stub", "--warp", "-i", c, "-o", passes.string(), "--start", "6", "--frames", "3", "--warmup", "2"},
                             dir / "nr_part.log");
    INFO(part.out);
    CHECK(part.code == 0);
    CHECK_THAT(part.out, !ContainsSubstring(kNoAv1Reason));
    CheckNoEnosys(part.out);
    REQUIRE(std::filesystem::exists(passes / "color_nr" / Manifest::kFileName));
    const Manifest m = Manifest::Load(passes / "color_nr");
    CHECK(m.firstFrame == 6);
    CHECK(m.lastFrame == 8);
}
