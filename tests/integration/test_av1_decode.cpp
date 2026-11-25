// TASK-0076: AV1 sources. This build of FFmpeg has no software AV1 decoder: before the task every AV1 clip failed on its first packet
// («avcodec_send_packet failed: Function not implemented») in the viewport, the stages and `process` without --hwaccel cuda. Now
// VideoDecoder decodes such a stream on NVDEC whatever the options say, seeks in it as in any other, and a machine without NVDEC AV1
// refuses it with the reason before anything is written. The clips are written with av1_nvenc (an NVIDIA GPU; skipped without one).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
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
#include "util/Log.h"

#include <spdlog/sinks/ostream_sink.h>
#include <sstream>

extern "C" {
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}

#ifdef DLSSVID_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;

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

// FFmpeg's words of a refused AV1 decode - ENOSYS before TASK-0076, and av1dec's own after a get_format that chose nothing («Your platform
// doesn't support hardware accelerated AV1 decoding», «Failed to get pixel format»: VideoDecoder lowers them to debug) - none may show;
// the refusal says it with its reason
void CheckNoEnosys(const std::string& out) {
    CHECK_THAT(out, !ContainsSubstring("Function not implemented"));
    CHECK_THAT(out, !ContainsSubstring("hardware accelerated AV1"));
    CHECK_THAT(out, !ContainsSubstring("Failed to get pixel format"));
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
            CHECK_FALSE(dec.HoldsCudaDevice());
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
            CHECK(hw.HoldsCudaDevice());        // the device at the open ...
            CHECK_FALSE(hw.DecodesOnNvdec());  // ... no frame from it yet
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
                CHECK_FALSE(dec.HoldsCudaDevice());
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
                    if (k == 45) CHECK_FALSE(dec.HoldsCudaDevice());
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
        // the settled status: a frame the prefetch marked missing may be asked for again as the current one (Loading meanwhile)
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        FrameStore::Status st = store.GetStatus(2, "source");
        while ((st == FrameStore::Status::Loading || st == FrameStore::Status::Queued) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            store.Update();
            st = store.GetStatus(2, "source");
        }
        CHECK(st == FrameStore::Status::Missing);
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
    CHECK_THAT(inf.out, ContainsSubstring("decode:      NVDEC only (av1: no software AV1 decoder in this build), on '"));
    CHECK_THAT(inf.out, ContainsSubstring("AV1 8-bit 4:2:0 up to "));
}

TEST_CASE("dlssvid info says how a video with a software decoder decodes - no GPU needed", "[integration][cli][av1]") {
    const auto dir = Dir("info_software");
    ClipSpec plain;
    plain.frames = 2;
    plain.width = 48;
    plain.height = 32;
    const auto ffv1 = RunAll({"info", WriteClip(dir / "ffv1.mkv", plain).string()}, dir / "info_ffv1.log");
    INFO(ffv1.out);
    CHECK(ffv1.code == 0);
    CHECK_THAT(ffv1.out, ContainsSubstring("decode:      software (ffv1); NVDEC does not decode this codec"));
}

TEST_CASE("AV1 through the stages: upscale bicubic and nr stub take an 8- and a 10-bit clip, process with stages writes the result (regression TASK-0076)",
          "[integration][cli][gpu][av1][regression]") {
    for (const bool tenBit : {false, true}) {
        INFO((tenBit ? "10-bit (p010 in, Main 10 out)" : "8-bit"));
        const auto dir = Dir(tenBit ? "stages_10" : "stages_8");
        REQUIRE_AV1_CLIP(clip, dir, tenBit, 12);
        const std::vector<CpuFrame> ref = DecodeSequential(clip);
        REQUIRE(ref.size() == 12);
        const uint8_t depth = tenBit ? 10 : 8;
        CHECK(ref[0].desc.bitDepth == depth);

        // a stage alone: upscale (the color_sr pass of every frame)
        const auto sr = dir / "sr";
        const auto up = RunAll({"upscale", "--backend", "bicubic", "--warp", "-i", clip.string(), "-o", sr.string()}, dir / "upscale.log");
        {
            INFO(up.out);
            CHECK(up.code == 0);
            CheckNoEnosys(up.out);
            CheckCudaChosenAtOnce(up.out);
            PassReader::Open(sr / "color_sr").Validate({2 * kW, 2 * kH, 12, PassKind::ColorSr});
        }
        // nr (stub: no model) on the source's frames
        const auto nrRoot = dir / "nr";
        const auto nr = RunAll({"nr", "--backend", "stub", "--warp", "-i", clip.string(), "-o", nrRoot.string()}, dir / "nr.log");
        {
            INFO(nr.out);
            CHECK(nr.code == 0);
            CheckNoEnosys(nr.out);
            CheckCudaChosenAtOnce(nr.out);
            PassReader::Open(nrRoot / "color_nr").Validate({kW, kH, 12, PassKind::ColorNr});
        }
        // process with stages (not --passthrough): upscale, then nr on its frames, then the encode of the last pass
        const auto staged = dir / "staged.mkv";
        const auto pr = RunAll({"process", "-i", clip.string(), "-o", staged.string(), "--passes", (dir / "passes").string(), "--warp", "--codec", "ffv1",
                                "--stages", "upscale,nr", "--upscale-backend", "bicubic", "--nr-backend", "stub"},
                               dir / "process.log");
        INFO(pr.out);
        CHECK(pr.code == 0);
        CheckNoEnosys(pr.out);
        CheckCudaChosenAtOnce(pr.out);
        CHECK_FALSE(AnyPartFile(dir));
        CHECK(std::filesystem::exists(dir / "passes" / "color_sr"));
        CHECK(std::filesystem::exists(dir / "passes" / "color_nr"));
        const std::vector<CpuFrame> out = DecodeAll(staged);
        REQUIRE(out.size() == ref.size());
        CHECK(out[0].desc.width == 2 * kW);
        CHECK(out[0].desc.height == 2 * kH);
        CHECK(out[0].desc.bitDepth == depth);  // the source's depth kept through the stages
    }
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
                if (hw == HwAccel::None) CHECK_FALSE(dec.HoldsCudaDevice());  // refused before the device was made
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
// `done` becomes true within `timeoutMs` (polled)
bool Eventually(const std::function<bool()>& done, int timeoutMs = 5000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

// The process's log into `out` for one scope ("<level> <message>", every level)
struct DecoderLog {
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink;
    spdlog::level::level_enum level;
    explicit DecoderLog(std::ostringstream& out) : sink(std::make_shared<spdlog::sinks::ostream_sink_mt>(out)), level(Log()->level()) {
        sink->set_pattern("%l %v");
        Log()->set_level(spdlog::level::debug);
        Log()->sinks().push_back(sink);
    }
    ~DecoderLog() {
        auto& sinks = Log()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
        Log()->set_level(level);
    }
    DecoderLog(const DecoderLog&) = delete;
    DecoderLog& operator=(const DecoderLog&) = delete;
};

// The process's log ("<level> <message>", every level) under the sink's own mutex: a case reads it (Text()) while a store's loaders
// may still log from their threads. Made before the store: the logger's sinks are not changed while it logs
class LiveLogSink : public spdlog::sinks::base_sink<std::mutex> {
public:
    std::string Text() {
        std::lock_guard<std::mutex> lock(mutex_);
        return text_;
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        text_.append(formatted.data(), formatted.size());
    }
    void flush_() override {}

private:
    std::string text_;
};
struct LiveLog {
    std::shared_ptr<LiveLogSink> sink = std::make_shared<LiveLogSink>();
    spdlog::level::level_enum level = Log()->level();
    LiveLog() {
        sink->set_pattern("%l %v");
        Log()->set_level(spdlog::level::debug);
        Log()->sinks().push_back(sink);
    }
    ~LiveLog() {
        auto& sinks = Log()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
        Log()->set_level(level);
    }
    std::string Text() const { return sink->Text(); }
    LiveLog(const LiveLog&) = delete;
    LiveLog& operator=(const LiveLog&) = delete;
};

size_t CountOf(const std::string& text, const std::string& what) {
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
    return n;
}

std::vector<CpuFrame> DecodeAllOf(VideoDecoder& dec) {
    std::vector<CpuFrame> out;
    CpuFrame f;
    while (dec.NextFrame(f)) out.push_back(f);
    return out;
}

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
    // Load, then wait for the neighbours too: the store prefetches at least one frame on each side (FrameStore::SpanOf), each load on the
    // source's decoder - after this no load runs (the CUDA context checks need a store at rest)
    FrameStore::Status Settle(int64_t f) {
        const FrameStore::Status st = Load(f);
        const ViewportSource* src = store->FindSource("source");
        const int64_t last = src ? src->lastFrame : f;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        auto busy = [&] {
            for (const int64_t n : {f - 1, f + 1}) {
                if (n < 0 || n > last) continue;
                const FrameStore::Status ns = store->GetStatus(n, "source");
                if (ns == FrameStore::Status::Loading || ns == FrameStore::Status::Queued) return true;
            }
            return false;
        };
        while (busy() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            store->Update();
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
    CHECK(dec.HoldsCudaDevice());        // the device was made ...
    CHECK_FALSE(dec.DecodesOnNvdec());  // ... and no frame came from it
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
        CHECK_FALSE(dec.HoldsCudaDevice());
        const auto refused = RefusalOfNextFrame(dec);  // the same at the first frame
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NoAv1);
        CHECK_FALSE(dec.HoldsCudaDevice());
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
        // set again with the source as it was found before the refusal: the store's own mark goes, a load is refused again
        s.store->SetSources({*found});
        CHECK_FALSE(s.store->Undecodable("source"));
        CHECK(s.Settle(5) == FrameStore::Status::Missing);  // the loaders at rest: the stand-in goes while none reads it
        CHECK(s.store->Undecodable("source"));
    }
    // the stand-in gone: the process remembers the file NVDEC refused - a new source of it is undecodable at once, no decoder opens
    const auto again = FrameStore::VideoSource("source", clip);
    REQUIRE(again);
    REQUIRE(again->undecodable);
    CHECK(again->undecodable->block == DecodeBlock::NvdecFailed);
    CHECK(again->undecodable->why == kFirstFrameReason);
    s.store->SetSources(Project::Create(clip, dir / "none").Sources());
    CHECK(s.store->Undecodable("source"));
    CHECK(s.Settle(7) == FrameStore::Status::Missing);
    CHECK(s.store->Stats("source").decodedFrames == 0);
    // the memory is the file's: a copy of the same stream decodes
    const auto copy = dir / "copy.mp4";
    std::filesystem::copy_file(clip, copy);
    s.store->SetSources(Project::Create(copy, dir / "none").Sources());
    CHECK_FALSE(s.store->Undecodable("source"));
    REQUIRE(s.Settle(7) == FrameStore::Status::Ready);
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
    Av1Store s(0, 1);
    s.store->SetSources(Project::Create(clip, dir / "none").Sources());  // no proxy: the original's decoder on NVDEC
    REQUIRE(s.Settle(2) == FrameStore::Status::Ready);  // and its neighbours: the store at rest, no load competes with the release
    CHECK(s.store->Stats("source").decodedFrames > 0);
    s.store->ReleaseNvdecDecoders();
    CHECK(s.store->Stats("source").decodedFrames == 0);  // the stats count the decoders open: none is
    for (const int64_t f : {3, 20}) {
        INFO("frame " << f << " after the release");
        REQUIRE(s.Settle(f) == FrameStore::Status::Ready);
        s.CheckTexels(f, ref[static_cast<size_t>(f)], ci);
    }
    CHECK(s.store->Stats("source").decodedFrames > 0);
}

TEST_CASE("BuildProxy refuses a video NVDEC cannot decode before it opens the partial proxy", "[integration][gpu][av1][proxy]") {
    const auto dir = Dir("proxy_refused");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    const auto passes = dir / "passes";
    // a folder where the partial proxy goes (not empty: the clean-up of a stale part leaves it): an encoder opened before the refusal would
    // fail on it with another error - the decoder's refusal (RequireDecodable) must come first; the decode itself would refuse too, later
    const auto part = ProxyDir(passes) / "source.building.mkv";
    std::filesystem::create_directories(part);
    std::ofstream(part / "keep") << "x";
    FakeNvdecEnv fake("no-av1");
    try {
        BuildProxy(clip, passes, "source");
        FAIL("BuildProxy built a proxy without NVDEC AV1");
    } catch (const DecodeUnavailable& e) {
        CHECK(e.support.block == DecodeBlock::NoAv1);
    } catch (const std::exception& e) {
        FAIL("not refused before the partial proxy was opened: " << e.what());
    }
    CHECK(std::filesystem::exists(part / "keep"));  // untouched
    CHECK_FALSE(ValidProxy(clip, passes, "source"));
    CHECK_FALSE(std::filesystem::exists(ProxyFile(passes, "source")));
}

TEST_CASE("Without NVDEC AV1 the stage commands refuse an AV1 source at once: the reason, no GPU device, no pass written",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("refused_commands");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    // a scenes pass made where NVDEC decodes AV1: with its scores scenes re-derives the cuts without decoding (at the end) - nothing to
    // refuse there
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
        // refused before the GPU work and the model (scenes, with no pass in its root at all: before the hash too; with a pass there that
        // has no scores of this video it hashes first)
        CHECK_THAT(r.out, !ContainsSubstring("D3D12 device"));
        CheckNoEnosys(r.out);
        CHECK_FALSE(std::filesystem::exists(cmd.pass));
        if (std::string(cmd.name) == "compare") {  // refused at its first video: the second is not opened
            size_t opens = 0;
            for (size_t at = r.out.find("] open "); at != std::string::npos; at = r.out.find("] open ", at + 1)) ++opens;
            CHECK(opens == 1);
        }
    }
    // scenes over its earlier pass: the cuts re-derived from the stored scores, no decoding - nothing refused; the pass before it kept
    // as a version
    const auto sc = RunAll({"scenes", "-i", c, "-o", scenesRoot.string()}, dir / "scenes_again.log");
    INFO("scenes over its earlier pass:\n" << sc.out);
    CHECK(sc.code == 0);
    CHECK_THAT(sc.out, ContainsSubstring("stored scores, no decoding"));
    CHECK_THAT(sc.out, !ContainsSubstring(kNoAv1Reason));
    CHECK(std::filesystem::exists(scenesRoot / "scenes"));
    CHECK(std::filesystem::exists(scenesRoot / "scenes.v"));
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
        CheckNoEnosys(rd.out);
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
        CheckNoEnosys(pt.out);
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

// The primary context reaching `want` and staying so for a moment: a load publishes its frame before its decoder is let go (and the
// store prefetches a neighbour or two), so a check right after a frame shows could read the moment before the release
bool ContextBecomes(bool want, int timeoutMs = 5000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
        if (PrimaryContextActive() == want) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (PrimaryContextActive() == want) return true;
        }
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
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
    CHECK(ContextBecomes(false));  // av1_nvenc wrote the clip in a context of its own
    {
        // NVDEC refuses the first frame: the loader closes the refused decoder - its CUDA device - at once
        const auto refusedCopy = dir / "refused.mp4";  // the process remembers a file NVDEC refused: the blocks below use the clip
        std::filesystem::copy_file(clip, refusedCopy);
        FakeNvdecEnv fake("first-frame");
        Av1Store s(0, 1);
        const auto found = FrameStore::VideoSource("source", refusedCopy);
        REQUIRE(found);
        s.store->SetSources({*found});
        REQUIRE(s.Settle(3) == FrameStore::Status::Missing);
        REQUIRE(s.store->Undecodable("source"));
        CHECK(ContextBecomes(false));
        s.store->ReleaseNvdecDecoders();  // nothing left to let go
        CHECK(ContextBecomes(false));
    }
    {
        // NVDEC failing to start although the driver takes the stream: the loader closes that decoder at once too
        FakeNvdecEnv fake("start-fails");
        Av1Store s(0, 1);
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        REQUIRE(s.Settle(3) == FrameStore::Status::Missing);
        CHECK_FALSE(s.store->Undecodable("source"));
        CHECK(ContextBecomes(false));
    }
    {
        // SetHoldNvdec(false) (a task runs): every load closes its decoder afterwards; true again: the decoder stays for the next frame
        Av1Store s(0, 1);
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        s.store->SetHoldNvdec(false);
        // the shown frame alone (no prefetch of an NVDEC-only video without a proxy while a task runs): its load closed its decoder after it
        REQUIRE(s.Load(4) == FrameStore::Status::Ready);
        // a loader closes its decoder after it published the frame: the count settles a moment after the frame shows
        CHECK(Eventually([&] { return s.store->Stats("source").decodedFrames == 0; }));  // the stats count the decoders open: none is
        CHECK(ContextBecomes(false));
        REQUIRE(s.Load(9) == FrameStore::Status::Ready);
        CHECK(ContextBecomes(false));
        s.store->SetHoldNvdec(true);
        REQUIRE(s.Settle(15) == FrameStore::Status::Ready);
        CHECK(s.store->Stats("source").decodedFrames > 0);
        CHECK(ContextBecomes(true));
    }
    CHECK(ContextBecomes(false));
    // a decoder on NVDEC holds it until ReleaseNvdecDecoders; the next frame opens one again; the store closing lets it go too
    {
        Av1Store s(0, 1);
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        REQUIRE(s.Settle(2) == FrameStore::Status::Ready);
        CHECK(ContextBecomes(true));
        s.store->ReleaseNvdecDecoders();  // the store at rest: closed at once
        CHECK(ContextBecomes(false));
        REQUIRE(s.Settle(9) == FrameStore::Status::Ready);
        CHECK(ContextBecomes(true));
    }
    CHECK(ContextBecomes(false));
    // the same for a decoder asked for HwAccel::Cuda (the device at the open): closing it lets the context go
    {
        VideoDecoder dec(clip, VideoDecoder::Options{HwAccel::Cuda});
        CHECK(dec.HoldsCudaDevice());
        CHECK(ContextBecomes(true));
    }
    CHECK(ContextBecomes(false));
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

namespace {
const std::string kStartFailedReason =
    "NVDEC could not start decoding this AV1 stream now - out of GPU memory or a driver error; this build has no software AV1 decoder (on the first "
    "frame)";
}  // namespace

TEST_CASE("NVDEC failing to start although the driver takes the stream (start-fails) is a transient refusal the decoder keeps", "[integration][gpu][av1]") {
    const auto dir = Dir("start_fails");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    std::optional<FakeNvdecEnv> fake;
    fake.emplace("start-fails");
    VideoDecoder dec(clip);
    CHECK_NOTHROW(dec.RequireDecodable());  // the driver says it takes the stream
    const auto first = RefusalOfNextFrame(dec);
    REQUIRE(first);
    CHECK(first->support.block == DecodeBlock::NvdecStartFailed);
    CHECK(first->support.Transient());
    CHECK(first->support.why == kStartFailedReason);
    CHECK(std::string(first->what()) == clip.string() + ": " + kStartFailedReason);
    CHECK(dec.HoldsCudaDevice());
    // this decoder is not fed again: the same refusal, also with the stand-in gone (a new decoder would decode - see FrameStore)
    fake.reset();
    const auto again = RefusalOfNextFrame(dec);
    REQUIRE(again);
    CHECK(again->support.block == DecodeBlock::NvdecStartFailed);
    CHECK(std::string(again->what()) == std::string(first->what()));
    CHECK_THROWS_AS(dec.RequireDecodable(), DecodeUnavailable);
    VideoDecoder fresh(clip);
    CpuFrame f;
    REQUIRE(fresh.NextFrame(f));
    CHECK(f.index == 0);
}

TEST_CASE("FrameStore: NVDEC failing to start is no verdict on the video - not marked, the next frame tries a new decoder", "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("store_start_fails");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 24);
    const ColorInfo ci = ColorInfoFromStream(VideoDecoder(clip, VideoDecoder::Probe()).Info());
    Av1Store s(0, 1);
    {
        FakeNvdecEnv fake("start-fails");
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        CHECK_FALSE(s.store->FindSource("source")->undecodable);  // the driver takes it
        CHECK_FALSE(s.store->StartFailure("source"));
        CHECK(s.Settle(3) == FrameStore::Status::Missing);
        CHECK_FALSE(s.store->Undecodable("source"));
        const auto failed = s.store->StartFailure("source");
        REQUIRE(failed);
        CHECK(failed->block == DecodeBlock::NvdecStartFailed);
        CHECK(failed->why == kStartFailedReason);
        CHECK(s.Settle(14) == FrameStore::Status::Missing);  // a new decoder, refused again (and the loaders at rest)
        CHECK_FALSE(s.store->Undecodable("source"));
    }
    // NVDEC starts again (the memory came back): the next frame decodes, and with it the start failure goes - no reload of the sources
    REQUIRE(s.Settle(10) == FrameStore::Status::Ready);
    s.CheckTexels(10, ref[10], ci);
    CHECK_FALSE(s.store->Undecodable("source"));
    CHECK_FALSE(s.store->StartFailure("source"));
    // the frames it left missing come again
    REQUIRE(s.Settle(3) == FrameStore::Status::Ready);
    s.CheckTexels(3, ref[3], ci);
    CHECK(s.Settle(14) == FrameStore::Status::Ready);
    CHECK_FALSE(FrameStore::VideoSource("source", clip)->undecodable);  // nothing remembered for the file
}

TEST_CASE("FrameStore::ReleaseNvdecDecoders does not wait for a load in flight: its loader closes that decoder after the load",
          "[integration][viewport][gpu][av1]") {
    const auto before = PrimaryContextActive();
    if (!before) SKIP("no CUDA driver here");
    if (*before) SKIP("the primary CUDA context is held by an earlier case in this process: run this case alone");
    // one keyframe in 3000 frames: a jump to the end walks the whole GOP on the loader, holding the source's decoder meanwhile
    const auto dir = Dir("release_busy");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 3000;
    spec.gop = 3000;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto written = WriteClipIfSupported(dir / "long_gop.mp4", spec, &why);
    if (!written) SKIP("av1_nvenc does not support the clip here: " << why);
    const std::filesystem::path clip = *written;
    Av1Store s(0, 1);
    s.store->SetSources(Project::Create(clip, dir / "none").Sources());
    REQUIRE(s.Settle(0) == FrameStore::Status::Ready);
    CHECK(ContextBecomes(true));

    using Clock = std::chrono::steady_clock;
    const int64_t target = 2999;  // the last frame: no prefetch after it, the one before is kept on the walk - no load follows the jump's
    const auto t0 = Clock::now();
    s.store->SetCurrentFrame(target, {"source"});
    // the loader has taken the frame (and the decoder's lock): a moment into the walk
    while (s.store->GetStatus(target, "source") == FrameStore::Status::Queued && Clock::now() - t0 < std::chrono::seconds(10))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const auto r0 = Clock::now();
    s.store->ReleaseNvdecDecoders();
    const auto r1 = Clock::now();
    REQUIRE(s.Settle(target) == FrameStore::Status::Ready);
    const auto t1 = Clock::now();
    const double releaseMs = std::chrono::duration<double, std::milli>(r1 - r0).count();
    const double loadSinceReleaseMs = std::chrono::duration<double, std::milli>(t1 - r0).count();
    INFO("ReleaseNvdecDecoders took " << releaseMs << " ms; the load lasted " << loadSinceReleaseMs << " ms from its call on");
    // meaningful only when the walk was still going on well after the call: a release that waited would have taken most of that
    REQUIRE(loadSinceReleaseMs > 150.0);
    CHECK(releaseMs < 50.0);
    // the release was not lost: the loader closed the decoder after the load (the store holds NVDEC otherwise)
    CHECK(ContextBecomes(false));
    CHECK(s.store->GetStatus(target - 1, "source") == FrameStore::Status::Ready);  // kept on the walk
    REQUIRE(s.Settle(10) == FrameStore::Status::Ready);  // and the next frames open one again
    CHECK(ContextBecomes(true));
}

TEST_CASE("BuildProxy and dlssvid proxy: NVDEC refusing the first frame leaves no partial proxy; dlssvid proxy says so and goes on",
          "[integration][gpu][av1][proxy]") {
    const auto dir = Dir("proxy_first_frame");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    FakeNvdecEnv fake("first-frame");  // the probe's check passes (the driver says nothing): the refusal comes with the first frame
    const auto passes = dir / "passes";
    try {
        BuildProxy(clip, passes, "source");
        FAIL("BuildProxy built a proxy NVDEC refused");
    } catch (const DecodeUnavailable& e) {
        CHECK(e.support.block == DecodeBlock::NvdecFailed);
        CHECK(e.support.why == kFirstFrameReason);
    }
    CHECK_FALSE(std::filesystem::exists(ProxyDir(passes) / "source.building.mkv"));
    CHECK_FALSE(ValidProxy(clip, passes, "source"));

    const std::string said = "source: no proxy, the video cannot be decoded here: " + kFirstFrameReason;
    const auto r = RunAll({"proxy", "-i", clip.string(), "--passes", passes.string(), "--sources", "source", "--force"}, dir / "proxy.log");
    {
        INFO(r.out);
        CHECK(r.code == 0);  // the other videos and the render go on
        CHECK_THAT(r.out, ContainsSubstring(said));
        CheckNoEnosys(r.out);
        CHECK_FALSE(std::filesystem::exists(ProxyDir(passes) / "source.building.mkv"));
        CHECK_FALSE(ValidProxy(clip, passes, "source"));
    }
    const auto png = dir / "frame.png";
    const auto rd = RunAll({"render", "--warp", "-i", clip.string(), "--passes", passes.string(), "--proxy", "build", "--frame", "2", "-o", png.string()},
                           dir / "render.log");
    INFO(rd.out);
    CHECK(rd.code != 0);
    CHECK_THAT(rd.out, ContainsSubstring("render: 'source': " + kFirstFrameReason));
    CheckNoEnosys(rd.out);
    CHECK_FALSE(std::filesystem::exists(png));
}

TEST_CASE("SaysUnsupported: only NVENC's words for a missing encoder skip a GPU test; anything else fails it", "[integration][av1]") {
    for (const char* skip : {"[av1_nvenc @ 02] Codec not supported", "[hevc_nvenc @ 03] 10 bit encode not supported",
                             "[hevc_nvenc @ 04] YUV444P not supported",
                             "open av1_nvenc (yuv420p) failed: Function not implemented\nCODEC NOT SUPPORTED"}) {
        INFO(skip);
        CHECK(SaysUnsupported(skip));
    }
    for (const char* fail : {"InitializeEncoder failed: unsupported param (12): Gop Length should be greater than number of B frames + 1",
                             "test clip pixel format unknown to FFmpeg: yuv420x", "open av1_nvenc (yuv420p) failed: Invalid argument",
                             "open av1_nvenc (yuv420p) failed: Function not implemented", "Provided device doesn't support required NVENC features",
                             // NVENC says it for a session not opened now too (memory, a busy GPU): a transient failure must not skip the AV1 set
                             "[av1_nvenc @ 0000] No capable devices found", "[h264_nvenc @ 01] No CUDA capable devices found", ""}) {
        INFO(fail);
        CHECK_FALSE(SaysUnsupported(fail));
    }
    // a test's own mistake in the spec is never a skip: WriteClipIfSupported throws it
    ClipSpec bad;
    bad.pixFmt = "yuv420x";
    CHECK_THROWS(WriteClipIfSupported(Dir("says_unsupported") / "bad.mkv", bad));
}

namespace {
const std::string kDeviceFailedReason =
    "NVDEC could not start decoding this AV1 stream now - out of GPU memory or a driver error; this build has no software AV1 decoder (the CUDA "
    "device could not be created: DLSSVID_FAKE_NVDEC_AV1=device-fails)";
}  // namespace

TEST_CASE("The CUDA device not created for a stream the driver takes (device-fails) is a transient refusal; the decoder is left as it was",
          "[integration][gpu][av1]") {
    const auto dir = Dir("device_fails");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 8);
    std::optional<FakeNvdecEnv> fake;
    fake.emplace("device-fails");
    VideoDecoder dec(clip);
    CHECK_NOTHROW(dec.RequireDecodable());  // the driver takes the stream
    for (int i = 0; i < 2; ++i) {
        INFO("attempt " << i);
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NvdecStartFailed);
        CHECK(refused->support.Transient());
        CHECK(refused->support.why == kDeviceFailedReason);
        CHECK(std::string(refused->what()) == clip.string() + ": " + kDeviceFailedReason);
        CHECK_FALSE(dec.HoldsCudaDevice());
    }
    // the device comes now: the same decoder decodes from the first frame (StartNvdec changed nothing when it failed)
    fake.reset();
    CpuFrame f;
    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE(dec.NextFrame(f));
        CHECK(f.index == static_cast<int64_t>(i));
        CHECK(f.data == ref[i].data);
    }
    CHECK(dec.HoldsCudaDevice());
    // asked for HwAccel::Cuda: the open does not get the device either - no GpuFrames, the first frame says why
    {
        FakeNvdecEnv cuda("device-fails");
        VideoDecoder hw(clip, VideoDecoder::Options{HwAccel::Cuda});
        CHECK_FALSE(hw.UsingHwAccel());
        CHECK_FALSE(hw.HoldsCudaDevice());
        const auto refused = RefusalOfNextFrame(hw);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NvdecStartFailed);
        CHECK(refused->support.why == kDeviceFailedReason);
    }
}

TEST_CASE("FrameStore: the CUDA device not created is no verdict - StartFailure says it, the next frame tries again", "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("store_device_fails");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 24);
    const ColorInfo ci = ColorInfoFromStream(VideoDecoder(clip, VideoDecoder::Probe()).Info());
    Av1Store s(0, 1);
    {
        FakeNvdecEnv fake("device-fails");
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        CHECK_FALSE(s.store->FindSource("source")->undecodable);
        CHECK(s.Settle(3) == FrameStore::Status::Missing);  // the loaders at rest before the stand-in goes
        CHECK_FALSE(s.store->Undecodable("source"));
        const auto failed = s.store->StartFailure("source");
        REQUIRE(failed);
        CHECK(failed->block == DecodeBlock::NvdecStartFailed);
        CHECK(failed->why == kDeviceFailedReason);
    }
    CHECK_FALSE(FrameStore::VideoSource("source", clip)->undecodable);  // nothing remembered for the file
    REQUIRE(s.Settle(12) == FrameStore::Status::Ready);
    s.CheckTexels(12, ref[12], ci);
    CHECK_FALSE(s.store->Undecodable("source"));
    CHECK_FALSE(s.store->StartFailure("source"));  // a frame decoded: the failure is over
    CHECK(s.Settle(3) == FrameStore::Status::Ready);  // the frame it left missing comes again
}

TEST_CASE("FrameStore::RetryStartFailures before the next Update: the missing marks the failed loads left go too, the frames load again",
          "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("store_retry_unpublished");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    Av1Store s(0, 1);
    std::optional<FakeNvdecEnv> fake;  // changed only while the loaders are at rest
    fake.emplace("start-fails");
    s.store->SetSources(Project::Create(clip, dir / "none").Sources());
    s.store->SetCurrentFrame(5, {"source"});
    // no Update: the loads' missing marks wait for it in the store (a frame reads Missing once its load is over) - as when a new
    // request of frames (AppModel::requestFrames, render's second try) comes between a loader's refusal and the next paint
    const std::vector<int64_t> window{4, 5, 6};  // the current frame and the neighbours every source prefetches
    auto all = [&](FrameStore::Status want) {
        for (const int64_t n : window)
            if (s.store->GetStatus(n, "source") != want) return false;
        return true;
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!all(FrameStore::Status::Missing) && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    REQUIRE(all(FrameStore::Status::Missing));  // every load of the window refused, none running
    REQUIRE(s.store->StartFailure("source"));
    fake.reset();  // the memory came back
    s.store->RetryStartFailures();
    CHECK_FALSE(s.store->StartFailure("source"));
    const auto again = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!all(FrameStore::Status::Ready) && std::chrono::steady_clock::now() < again) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        s.store->Update();
    }
    for (const int64_t n : window) {
        INFO("frame " << n);
        CHECK(s.store->GetStatus(n, "source") == FrameStore::Status::Ready);  // a stale missing mark would make the retried load drop itself
    }
    CHECK_FALSE(s.store->Undecodable("source"));
}

TEST_CASE("FrameStore opens the original's decoder again quietly: the log says open once, not at every reopen while a task runs",
          "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("reopen_log");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    std::ostringstream text;
    {
        // declared before the store: the store's loaders end before the sink and the level go back (also on a failed REQUIRE)
        struct LogCapture {
            std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink;
            spdlog::level::level_enum level;
            explicit LogCapture(std::ostringstream& out) : sink(std::make_shared<spdlog::sinks::ostream_sink_mt>(out)), level(Log()->level()) {
                sink->set_pattern("%l %v");
                Log()->set_level(spdlog::level::debug);
                Log()->sinks().push_back(sink);
            }
            ~LogCapture() {
                auto& sinks = Log()->sinks();
                sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
                Log()->set_level(level);
            }
        } capture(text);
        Av1Store s(0, 1);
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        s.store->SetHoldNvdec(false);  // a task runs: every load closes the decoder after it - each frame reopens it
        for (const int64_t f : {2, 9, 16}) REQUIRE(s.Load(f) == FrameStore::Status::Ready);  // the shown frames alone: no prefetch meanwhile
    }
    const std::string log = text.str();
    const std::string opened = "open " + clip.string() + ": ";
    auto count = [&](const std::string& what) {
        size_t n = 0;
        for (size_t at = log.find(what); at != std::string::npos; at = log.find(what, at + 1)) ++n;
        return n;
    };
    INFO(log);
    CHECK(count("info " + opened) == 1);   // the first open is news
    CHECK(count("debug " + opened) >= 2);  // the reopens are not
    const std::string onNvdec = clip.string() + ": av1 decodes on NVDEC only";  // VideoDecoder::StartNvdec, at each decoder's start
    CHECK(count("info " + onNvdec) == 1);
    CHECK(count("debug " + onNvdec) >= 2);
}

TEST_CASE("FrameStore prefetches no frame of an NVDEC-only video while a task runs - the shown frame alone opens a decoder; the neighbours come after",
          "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("held_no_prefetch");
    REQUIRE_AV1_CLIP(clip, dir, false, 64);  // a keyframe every 8: frame 56 is beyond FrameStore's walk from frame 0
    LiveLog log;  // before the store: its loaders log while it lives
    Av1Store s(3, 1);  // three frames on either side - while the decoders are held
    s.store->SetSources(Project::Create(clip, dir / "none").Sources());
    const ViewportSource* src = s.store->FindSource("source");
    REQUIRE(src);
    REQUIRE(src->hardwareOnly);
    REQUIRE(src->proxy.empty());
    s.store->SetHoldNvdec(false);  // a task runs: each load opens a decoder (the CUDA device, a walk from the keyframe) and closes it after
    const size_t mark = log.Text().size();
    REQUIRE(s.Load(56) == FrameStore::Status::Ready);
    const std::vector<int64_t> around = {53, 54, 55, 57, 58, 59};
    auto anyAround = [&] {
        s.store->Update();
        for (const int64_t n : around) {
            const FrameStore::Status st = s.store->GetStatus(n, "source");
            if (st == FrameStore::Status::Ready || st == FrameStore::Status::Loading) return true;
        }
        return false;
    };
    CHECK_FALSE(Eventually(anyAround, 1500));  // none of them is loaded meanwhile
    {
        const std::string since = log.Text().substr(mark);
        INFO(since);
        CHECK(CountOf(since, "open " + clip.string() + ": ") == 1);  // the shown frame's decoder alone
    }
    // the task over: the prefetch comes back at once - no frame asked for again
    s.store->SetHoldNvdec(true);
    CHECK(Eventually(
        [&] {
            s.store->Update();
            for (const int64_t n : around)
                if (s.store->GetStatus(n, "source") != FrameStore::Status::Ready) return false;
            return true;
        },
        30000));
}

TEST_CASE("FrameStore prefetches an NVDEC-only video that has a proxy while a task runs - the neighbours come from the proxy",
          "[integration][viewport][gpu][av1][proxy]") {
    const auto dir = Dir("held_proxy_prefetch");
    REQUIRE_AV1_CLIP(clip, dir, false, 64);  // a keyframe every 8: frame 56 is beyond FrameStore's walk from frame 0
    const auto passes = dir / "passes";
    BuildProxy(clip, passes, "source");  // where NVDEC decodes AV1 (the proxy is FFV1: software everywhere)
    REQUIRE(ValidProxy(clip, passes, "source"));
    LiveLog log;  // before the store: its loaders log while it lives
    Av1Store s(3, 1);  // three frames on either side
    s.store->SetSources(Project::Create(clip, passes).Sources());
    const ViewportSource* src = s.store->FindSource("source");
    REQUIRE(src);
    REQUIRE(src->hardwareOnly);
    REQUIRE_FALSE(src->proxy.empty());
    s.store->SetHoldNvdec(false);  // a task runs: the proxy takes no NVDEC - its prefetch stays
    const size_t mark = log.Text().size();
    REQUIRE(s.Load(56) == FrameStore::Status::Ready);
    CHECK(Eventually(
        [&] {
            s.store->Update();
            for (const int64_t n : {53, 54, 55, 57, 58, 59})
                if (s.store->GetStatus(n, "source") != FrameStore::Status::Ready) return false;
            return true;
        },
        20000));
    // all of them from the proxy: the original's NVDEC decoder never opened while the task ran
    const std::string since = log.Text().substr(mark);
    INFO(since);
    CHECK(CountOf(since, "open " + clip.string() + ": ") == 0);
    CHECK(s.store->Stats("source").proxyLoads > 0);
}

TEST_CASE("The driver not answering for a CUDA error (query-fails), NVDEC refusing the first frame then is a start failure, not a verdict",
          "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("query_fails");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const std::vector<CpuFrame> ref = DecodeSequential(clip);
    REQUIRE(ref.size() == 24);
    const ColorInfo ci = ColorInfoFromStream(VideoDecoder(clip, VideoDecoder::Probe()).Info());
    {
        FakeNvdecEnv fake("query-fails");
        VideoDecoder dec(clip);
        CHECK_NOTHROW(dec.RequireDecodable());  // unchecked: goes on
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NvdecStartFailed);
        CHECK(refused->support.Transient());
        CHECK(refused->support.why == kStartFailedReason);
    }
    Av1Store s(0, 1);
    {
        FakeNvdecEnv fake("query-fails");
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        CHECK_FALSE(s.store->FindSource("source")->undecodable);
        CHECK(s.Settle(3) == FrameStore::Status::Missing);  // the loaders at rest before the stand-in goes
        CHECK_FALSE(s.store->Undecodable("source"));
        const auto failed = s.store->StartFailure("source");
        REQUIRE(failed);
        CHECK(failed->block == DecodeBlock::NvdecStartFailed);
    }
    CHECK_FALSE(FrameStore::VideoSource("source", clip)->undecodable);  // nothing remembered for the file
    REQUIRE(s.Settle(10) == FrameStore::Status::Ready);
    s.CheckTexels(10, ref[10], ci);
    CHECK_FALSE(s.store->StartFailure("source"));
}

namespace {
constexpr const char* kFlagsChild = "CUDA primary context active with flags beside BLOCKING_SYNC - the child of the case that starts it";
}  // namespace

// Hidden ([.]): run by the case below in a process of its own - the CUDA runtime keeps the primary context it starts to the process's end
TEST_CASE(kFlagsChild, "[.][av1-child]") {
#ifdef DLSSVID_WITH_CUDA
    const char* clip = std::getenv("DLSSVID_TEST_AV1_CLIP");
    REQUIRE(clip);
    // as a stage whose CUDA runtime came first: the primary context active with the blocking scheduling FFmpeg wants and another flag
    // (local memory kept at its largest; cudaDeviceMapHost would not do: the driver no longer reports it, every context maps host memory)
    REQUIRE(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceLmemResizeToMax) == cudaSuccess);
    REQUIRE(cudaFree(nullptr) == cudaSuccess);
    CUdevice dev = 0;
    unsigned flags = 0;
    int active = 0;
    REQUIRE(cuDeviceGet(&dev, 0) == CUDA_SUCCESS);
    REQUIRE(cuDevicePrimaryCtxGetState(dev, &flags, &active) == CUDA_SUCCESS);
    REQUIRE(active);
    INFO("primary context flags " << flags);
    REQUIRE((flags & CU_CTX_SCHED_MASK) == CU_CTX_SCHED_BLOCKING_SYNC);  // the scheduling bits as FFmpeg wants them
    if (flags == CU_CTX_SCHED_BLOCKING_SYNC) SKIP("the driver keeps no flag beside the scheduling: nothing to tell apart here");
    // a decoder with the default options: NVDEC, or FFmpeg's «Primary context already active with incompatible flags» - and no frame
    VideoDecoder dec(clip);
    CpuFrame f;
    int frames = 0;
    while (dec.NextFrame(f)) ++frames;
    CHECK(frames == 12);
    CHECK(dec.DecodesOnNvdec());
#else
    SKIP("built without CUDA");
#endif
}

TEST_CASE("NVDEC takes an AV1 stream after a stage's CUDA runtime made the primary context with another flag beside blocking sync (regression of round 3)",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("primary_flags");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    char* self = nullptr;
    REQUIRE(_get_pgmptr(&self) == 0);
    REQUIRE(self);
    const auto log = dir / "child.log";
    _putenv_s("DLSSVID_TEST_AV1_CLIP", clip.string().c_str());
    const std::string cmd = std::string("\"\"") + self + "\" \"" + kFlagsChild + "\" -r compact > \"" + log.string() + "\" 2>&1\"";
    const int code = std::system(cmd.c_str());
    _putenv_s("DLSSVID_TEST_AV1_CLIP", "");
    const std::string out = Read(log);
    INFO(out);
    if (out.find("skipped") != std::string::npos && out.find("failed") == std::string::npos) SKIP("the child could not set the case up: see its output");
    CHECK(code == 0);
    CHECK_THAT(out, ContainsSubstring("All tests passed"));
    CHECK_THAT(out, !ContainsSubstring("incompatible flags"));
}

TEST_CASE("A part of the stream in other parameters the GPU takes is judged as the probe was: NVDEC not starting there is a start failure that names the part",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("other_part");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    // two streams one after the other in one IVF file: the probe sees the first part's 320x192, av1dec meets 384x224 at frame 16
    ClipSpec first;
    first.width = kW;
    first.height = kH;
    first.frames = 16;
    first.gop = 8;
    first.codec = "av1_nvenc";
    ClipSpec second = first;
    second.width = 384;
    second.height = 224;
    second.frames = 16;
    std::string why;
    const auto a = WriteClipIfSupported(dir / "a.ivf", first, &why);
    if (!a) SKIP("av1_nvenc does not support the clip here: " << why);
    const auto b = WriteClipIfSupported(dir / "b.ivf", second, &why);
    if (!b) SKIP("av1_nvenc does not support the clip here: " << why);
    const auto parts = ConcatIvf(*a, *b, dir / "parts.ivf");
    const VideoStreamInfo info = VideoDecoder(parts, VideoDecoder::Probe()).Info();
    REQUIRE(info.codedWidth == kW);
    REQUIRE(info.codedHeight == kH);
    // NVDEC decodes both parts (the stand-in aside): the second part's frames are there
    {
        VideoDecoder dec(parts);
        REQUIRE(dec.SeekToKeyframeBefore(20));
        CpuFrame f;
        REQUIRE(dec.NextFrame(f));
        CHECK(f.desc.width == 384);
        CHECK(f.desc.height == 224);
    }
    // a decoder whose first frame is in the first part: NVDEC not starting is a start failure - no verdict
    {
        FakeNvdecEnv fake("start-fails");
        VideoDecoder dec(parts);
        REQUIRE(dec.SeekToKeyframeBefore(10));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NvdecStartFailed);
        CHECK(refused->support.Transient());
        CHECK_THAT(refused->support.why, !ContainsSubstring("other parameters"));
    }
    // a decoder whose first frame is in the second part (a seek): av1dec has parsed its sequence header - other parameters than the
    // probe's, judged against the GPU as the probe was: taken (4:2:0, a size the driver takes), so NVDEC not starting there is a start
    // failure too - and its reason names the part
    {
        FakeNvdecEnv fake("start-fails");
        VideoDecoder dec(parts);
        REQUIRE(dec.SeekToKeyframeBefore(20));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NvdecStartFailed);
        CHECK(refused->support.Transient());
        CHECK_THAT(refused->support.why, ContainsSubstring("other parameters"));
        // the frame's place in the stream (the keyframe 16 the seek landed on), not this decoder's count
        CHECK_THAT(refused->support.why, ContainsSubstring("(at frame 16, on a part of the stream in other parameters (yuv420p 384x224))"));
    }
}

TEST_CASE("A part of the stream NVDEC does not take (12-bit after 8-bit) is refused with the part's own reason - the layout, not a start failure",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("twelve_bit_part");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 16;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto a = WriteClipIfSupported(dir / "a.ivf", spec, &why);
    if (!a) SKIP("av1_nvenc does not support the clip here: " << why);
    const auto b = WriteClipIfSupported(dir / "b.ivf", spec, &why);
    REQUIRE(b);
    // the second part's sequence headers say 12-bit 4:2:0: av1dec meets yuv420p12 at frame 16, which FFmpeg offers to no NVDEC
    const auto parts = ConcatIvf(*a, *b, dir / "parts.ivf", TwelveBitTemporalUnit);
    const VideoStreamInfo info = VideoDecoder(parts, VideoDecoder::Probe()).Info();
    REQUIRE(info.pixelFormat == "yuv420p");
    REQUIRE(CheckDecodeSupport(info).Ok());  // the probe sees the first part: the GPU takes it
    const std::string layout = "AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here";
    const std::string part = "on a part of the stream in other parameters (yuv420p12le 320x192))";
    // decoded from the start: the first part's frames, then the part's own verdict - its layout - after them
    {
        FfmpegLogCapture ffmpeg;  // av1dec's words after a refused format go to debug (RefuseQuietly): the refusal says why itself
        VideoDecoder dec(parts);
        CpuFrame f;
        int frames = 0;
        std::optional<DecodeUnavailable> refused;
        try {
            while (dec.NextFrame(f)) ++frames;
            FAIL("the 12-bit part decoded");
        } catch (const DecodeUnavailable& e) {
            refused = e;
        }
        REQUIRE(refused);
        CHECK(frames >= 8);
        CHECK(refused->support.block == DecodeBlock::Layout);
        CHECK(refused->support.part);  // the verdict is on a part of the stream
        CHECK(refused->support.bitDepth == 12);  // that part's
        CHECK_FALSE(refused->support.Transient());
        CHECK_THAT(refused->support.why, ContainsSubstring(layout));
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 16, " + part));  // the part's first frame
        CHECK_THAT(ffmpeg.Text(), !ContainsSubstring("hardware accelerated AV1"));
        CHECK_THAT(ffmpeg.Text(), !ContainsSubstring("Failed to get pixel format"));
    }
    // a decoder whose first frame is in that part (a seek): the same verdict on its first frame - also where NVDEC would not start (the
    // stand-in): a part the GPU does not take is no start failure
    for (const char* value : {static_cast<const char*>(nullptr), "start-fails"}) {
        INFO((value ? value : "no stand-in"));
        FakeNvdecEnv fake(value);
        FfmpegLogCapture ffmpeg;
        VideoDecoder dec(parts);
        REQUIRE(dec.SeekToKeyframeBefore(20));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::Layout);
        CHECK(refused->support.part);
        CHECK_FALSE(refused->support.Transient());
        CHECK_THAT(refused->support.why, ContainsSubstring(layout));
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 16, " + part));  // its place in the stream, the same
        CHECK_THAT(ffmpeg.Text(), !ContainsSubstring("hardware accelerated AV1"));
        CHECK_THAT(ffmpeg.Text(), !ContainsSubstring("Failed to get pixel format"));
    }
}

TEST_CASE("A layout the probe did not learn, refused after frames under another sequence header, is a part; a seek straight into it is not",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("unknown_then_part");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 16;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto a = WriteClipIfSupported(dir / "a.ivf", spec, &why);
    if (!a) SKIP("av1_nvenc does not support the clip here: " << why);
    // as unknown.ivf: no sequence header before frame 160 - the probe does not learn the layout; 8-bit 4:2:0 from there, and from the
    // keyframe 184 a part with its own sequence header - 12-bit 4:2:0 of the same size, or 8-bit of 191 rows
    constexpr int kFirstHeader = 160;
    constexpr int kPart = 184;
    const auto twelve = RepeatIvf(*a, 13, dir / "unknown_part12.ivf", [](int i, const std::string& u) {
        return i < kFirstHeader ? WithoutSequenceHeaders(u) : i < kPart ? u : TwelveBitTemporalUnit(u);
    });
    const auto odd = RepeatIvf(*a, 13, dir / "unknown_part_odd.ivf", [](int i, const std::string& u) {
        return i < kFirstHeader ? WithoutSequenceHeaders(u) : i < kPart ? u : ShorterFrameTemporalUnit(u, 1);
    });
    for (const auto& file : {twelve, odd}) {
        INFO(file.filename().string());
        REQUIRE(VideoDecoder(file, VideoDecoder::Probe()).Info().pixelFormat == "unknown");
    }
    // decoded from the first sequence header on: the 8-bit frames, then the part - refused as a part (frames under another sequence
    // header came first), in the part's own words
    auto decodedThrough = [&](const std::filesystem::path& file) {
        VideoDecoder dec(file);
        REQUIRE(dec.SeekToKeyframeBefore(kFirstHeader + 2));  // the keyframe at or before 162: 160
        CpuFrame f;
        int frames = 0;
        std::optional<DecodeUnavailable> refused;
        try {
            while (dec.NextFrame(f)) ++frames;
            FAIL("the part decoded: " << frames << " frames");
        } catch (const DecodeUnavailable& e) {
            refused = e;
        }
        REQUIRE(refused);
        CHECK(frames == kPart - kFirstHeader);
        return *refused;
    };
    {
        const DecodeUnavailable refused = decodedThrough(twelve);
        CHECK(refused.support.block == DecodeBlock::Layout);
        CHECK(refused.support.part);
        CHECK(refused.support.bitDepth == 12);
        CHECK_THAT(refused.support.why, ContainsSubstring("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here"));
        CHECK_THAT(refused.support.why, Catch::Matchers::EndsWith("(at frame 184, on a part of the stream in other parameters (yuv420p12le 320x192))"));
    }
    {
        const DecodeUnavailable refused = decodedThrough(odd);
        CHECK(refused.support.block == DecodeBlock::OddHeight);
        CHECK(refused.support.part);
        CHECK(refused.support.height == kH - 1);
    }
    // a seek straight into the part: no frame under another sequence header - the stream's layout, learned late (as before)
    {
        VideoDecoder dec(twelve);
        REQUIRE(dec.SeekToKeyframeBefore(kPart + 6));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::Layout);
        CHECK_FALSE(refused->support.part);
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 184, in yuv420p12le 320x192 - a layout the probe did not learn)"));
    }
    {
        VideoDecoder dec(odd);
        REQUIRE(dec.SeekToKeyframeBefore(kPart + 6));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::OddHeight);
        CHECK_FALSE(refused->support.part);
    }
}

TEST_CASE("dlssvid render --bench on videos this machine cannot decode says why before it times anything; a source that shows still measures",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("bench_refused");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    ClipSpec plain;
    plain.frames = 12;
    plain.width = kW;
    plain.height = kH;
    const auto ffv1 = WriteClip(dir / "result.mkv", plain);
    const auto passes = dir / "passes";
    const auto proxied = dir / "proxied.mp4";  // the AV1 video again, with a proxy made where NVDEC decodes it
    std::filesystem::copy_file(clip, proxied);
    const auto proxiedPasses = dir / "proxied_passes";
    BuildProxy(proxied, proxiedPasses, "source");
    REQUIRE(ValidProxy(proxied, proxiedPasses, "source"));
    FakeNvdecEnv fake("no-av1");
    // the source alone, not decodable here and no proxy: the reason, no timing, no report
    const auto json = dir / "bench.json";
    const auto rd = RunAll({"render", "--warp", "-i", clip.string(), "--passes", passes.string(), "--bench", "4", "--bench-json", json.string()}, dir / "bench.log");
    {
        INFO(rd.out);
        CHECK(rd.code != 0);
        CHECK_THAT(rd.out, ContainsSubstring("render: 'source': " + kNoAv1Reason));
        CHECK_FALSE(std::filesystem::exists(json));
    }
    // a grid with the result (FFV1 decodes): measured - the source's cells empty
    const auto grid = dir / "bench_grid.json";
    const auto gd = RunAll({"render", "--warp", "-i", clip.string(), "--passes", passes.string(), "--result", ffv1.string(), "--mode", "grid", "--sources",
                            "source,result,source,result", "--bench", "4", "--bench-json", grid.string()},
                           dir / "bench_grid.log");
    {
        INFO(gd.out);
        CHECK(gd.code == 0);
        CHECK(std::filesystem::exists(grid));
    }
    // the AV1 video shown by its proxy: measured
    const auto byProxy = dir / "bench_proxy.json";
    const auto px = RunAll({"render", "--warp", "-i", proxied.string(), "--passes", proxiedPasses.string(), "--bench", "4", "--bench-json", byProxy.string()},
                           dir / "bench_proxy.log");
    {
        INFO(px.out);
        CHECK(px.code == 0);
        CHECK(std::filesystem::exists(byProxy));
    }
}

TEST_CASE("dlssvid proxy --check says a video this machine cannot decode gets no proxy and why - not whether it needs one", "[integration][cli][gpu][av1][proxy]") {
    const auto dir = Dir("proxy_check_refused");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    ClipSpec plain;
    plain.frames = 4;
    plain.width = 48;
    plain.height = 32;
    const auto ffv1 = WriteClip(dir / "result.mkv", plain);
    FakeNvdecEnv fake("no-av1");
    const std::string reason = CheckDecodeSupport(VideoDecoder(clip, VideoDecoder::Probe()).Info()).why;
    REQUIRE_THAT(reason, StartsWith(kNoAv1Reason));
    const auto r = RunAll({"proxy", "--check", "-i", clip.string(), "--passes", (dir / "passes").string(), "--result", ffv1.string()}, dir / "check.log");
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK_THAT(r.out, ContainsSubstring("source   " + clip.filename().string() + ": cannot be decoded here, no proxy: " + reason));
    CHECK_THAT(r.out, !ContainsSubstring("source   " + clip.filename().string() + ": 24 frames"));  // no plan of a proxy for it
    CHECK_THAT(r.out, ContainsSubstring("result   " + ffv1.filename().string() + ": 4 frames"));     // the other video as before
    CHECK_FALSE(std::filesystem::exists(dir / "passes" / ".proxy"));                                // nothing built
}

TEST_CASE("FrameStore keeps a software decoder open while a task runs and through ReleaseNvdecDecoders - only NVDEC is let go", "[integration][viewport][av1]") {
    const auto dir = Dir("software_hold");
    ClipSpec plain;
    plain.frames = 24;
    plain.width = kW;
    plain.height = kH;
    const auto clip = WriteClip(dir / "clip.mkv", plain);
    Av1Store s(0, 1);
    s.store->SetSources(Project::Create(clip, dir / "none").Sources());
    REQUIRE_FALSE(s.store->FindSource("source")->hardwareOnly);
    s.store->SetHoldNvdec(false);  // a task runs: an NVDEC decoder would close after every load
    int64_t decoded = 0;
    for (const int64_t f : {2, 9, 16}) {
        INFO("frame " << f);
        REQUIRE(s.Settle(f) == FrameStore::Status::Ready);
        // its neighbours too: a video with a software path keeps its prefetch while a task runs (only NVDEC-only ones lose it)
        CHECK(s.store->GetStatus(f - 1, "source") == FrameStore::Status::Ready);
        CHECK(s.store->GetStatus(f + 1, "source") == FrameStore::Status::Ready);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));  // a loader closing its decoder does it right after the load
        const int64_t now = s.store->Stats("source").decodedFrames;  // the frames of the decoders open
        CHECK(now > decoded);  // the same decoder, going on: none closed and opened anew
        decoded = now;
    }
    s.store->ReleaseNvdecDecoders();
    CHECK(s.store->Stats("source").decodedFrames == decoded);  // not on NVDEC: kept
    REQUIRE(s.Settle(22) == FrameStore::Status::Ready);
    CHECK(s.store->Stats("source").decodedFrames > decoded);  // and it went on
}

TEST_CASE("The layout the probe did not learn is judged when NVDEC meets it: taken - NVDEC not starting is a start failure; 12-bit - the layout's verdict",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("unknown_layout");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 16;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto a = WriteClipIfSupported(dir / "a.ivf", spec, &why);
    if (!a) SKIP("av1_nvenc does not support the clip here: " << why);
    // 13 copies (208 frames, a keyframe every 8) with no sequence header before frame 160: the probe (some 5 s of the stream) meets
    // none - the layout stays unknown; av1dec learns it at the keyframe a seek lands on (168), before it asks for a pixel format
    constexpr int kFirstHeader = 160;
    const auto plain = RepeatIvf(*a, 13, dir / "unknown.ivf", [](int i, const std::string& u) { return i < kFirstHeader ? WithoutSequenceHeaders(u) : u; });
    const auto twelve = RepeatIvf(*a, 13, dir / "unknown12.ivf",
                                  [](int i, const std::string& u) { return i < kFirstHeader ? WithoutSequenceHeaders(u) : TwelveBitTemporalUnit(u); });
    for (const auto& file : {plain, twelve}) {
        INFO(file.filename().string());
        REQUIRE(VideoDecoder(file, VideoDecoder::Probe()).Info().pixelFormat == "unknown");
    }
    {
        // dlssvid info: not the defaults the probe has for the layout
        const auto info = RunAll({"info", plain.string()}, dir / "info.log");
        INFO(info.out);
        CHECK(info.code == 0);
        CHECK_THAT(info.out, ContainsSubstring("samples:     not known before the first frame (the probe did not learn the pixel format)"));
        CHECK_THAT(info.out, !ContainsSubstring("decoded as"));
    }
    // refused here (no CUDA, no AV1 in NVDEC): no decoder starts - no first frame to learn the layout from; the decode line says why
    for (const char* value : {"no-cuda", "no-av1"}) {
        INFO(value);
        FakeNvdecEnv fake(value);
        const auto info = RunAll({"info", plain.string()}, dir / (std::string("info_") + value + ".log"));
        INFO(info.out);
        CHECK(info.code == 0);
        CHECK_THAT(info.out, ContainsSubstring("samples:     not known (the probe did not learn the pixel format; not decoded here, see decode)"));
        CHECK_THAT(info.out, !ContainsSubstring("before the first frame"));  // the samples line, nor NoAv1's words on the decode line
        CHECK_THAT(info.out, ContainsSubstring("decode:      unavailable - "));
        CHECK_THAT(info.out, !ContainsSubstring("decoded as"));
    }
    const std::string learned = "a layout the probe did not learn)";
    {
        // the stand-in aside: NVDEC decodes it from there
        VideoDecoder dec(plain);
        REQUIRE(dec.SeekToKeyframeBefore(170));
        CpuFrame f;
        REQUIRE(dec.NextFrame(f));
        CHECK(f.desc.width == kW);
    }
    {
        // 8-bit 4:2:0, which the driver takes: NVDEC not starting (the stand-in) is a start failure
        FakeNvdecEnv fake("start-fails");
        VideoDecoder dec(plain);
        REQUIRE(dec.SeekToKeyframeBefore(170));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NvdecStartFailed);
        CHECK(refused->support.Transient());
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 168, in yuv420p 320x192 - " + learned));
    }
    {
        // the driver not answering (first-frame): no word of it taking the stream - NVDEC's refusal stands as the stream's verdict
        FakeNvdecEnv fake("first-frame");
        VideoDecoder dec(plain);
        REQUIRE(dec.SeekToKeyframeBefore(170));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NvdecFailed);
        CHECK_FALSE(refused->support.Transient());
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 168, in yuv420p 320x192 - " + learned));
    }
    // 12-bit: a layout the GPU does not take - its own verdict, whatever the stand-in
    for (const char* value : {static_cast<const char*>(nullptr), "start-fails"}) {
        INFO((value ? value : "no stand-in"));
        FakeNvdecEnv fake(value);
        FfmpegLogCapture ffmpeg;  // av1dec's words after a refused format go to debug (RefuseQuietly)
        VideoDecoder dec(twelve);
        REQUIRE(dec.SeekToKeyframeBefore(170));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::Layout);
        CHECK_FALSE(refused->support.part);  // the stream's own layout, learned late
        CHECK_FALSE(refused->support.Transient());
        CHECK_THAT(refused->support.why, ContainsSubstring("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here"));
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 168, in yuv420p12le 320x192 - " + learned));
        CHECK_THAT(ffmpeg.Text(), !ContainsSubstring("hardware accelerated AV1"));
        CHECK_THAT(ffmpeg.Text(), !ContainsSubstring("Failed to get pixel format"));
    }
}

TEST_CASE("dlssvid proxy --check and --force keep a proxy built earlier for a video this machine cannot decode: shown from it, not rebuilt",
          "[integration][cli][gpu][av1][proxy]") {
    const auto dir = Dir("proxy_kept");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const auto passes = dir / "passes";
    BuildProxy(clip, passes, "source");  // where NVDEC decodes AV1
    REQUIRE(ValidProxy(clip, passes, "source"));
    const std::string proxy = ValidProxy(std::filesystem::absolute(clip), std::filesystem::absolute(passes), "source")->string();  // as the CLI says it
    const auto builtAt = std::filesystem::last_write_time(proxy);
    auto stays = [&] {
        CHECK(ValidProxy(clip, passes, "source"));
        CHECK(std::filesystem::last_write_time(proxy) == builtAt);
    };
    {
        // the driver's answer (no-av1): --check names the proxy that shows it, --force keeps it
        FakeNvdecEnv fake("no-av1");
        const std::string reason = CheckDecodeSupport(VideoDecoder(clip, VideoDecoder::Probe()).Info()).why;
        const auto check = RunAll({"proxy", "--check", "-i", clip.string(), "--passes", passes.string()}, dir / "check.log");
        INFO(check.out);
        CHECK(check.code == 0);
        CHECK_THAT(check.out, ContainsSubstring("source   " + clip.filename().string() + ": cannot be decoded here, shown from its proxy " + proxy + ": " + reason));
        CHECK_THAT(check.out, !ContainsSubstring("no proxy"));
        const auto force = RunAll({"proxy", "--force", "-i", clip.string(), "--passes", passes.string(), "--sources", "source"}, dir / "force_no_av1.log");
        INFO(force.out);
        CHECK(force.code == 0);
        CHECK_THAT(force.out,
                   ContainsSubstring("source: proxy not rebuilt, the video cannot be decoded here (the proxy built earlier stays: " + proxy + "): " + reason));
        CHECK_THAT(force.out, !ContainsSubstring("source: no proxy"));
        stays();
    }
    // NVDEC refusing (first-frame) or not starting (start-fails) on the rebuild's first frame: the proxy built earlier stays
    for (const auto& [value, words] : {std::pair<const char*, const char*>{"first-frame", "the video cannot be decoded here"},
                                       std::pair<const char*, const char*>{"start-fails", "NVDEC did not start (run it again later)"}}) {
        INFO(value);
        FakeNvdecEnv fake(value);
        const auto force = RunAll({"proxy", "--force", "-i", clip.string(), "--passes", passes.string(), "--sources", "source"},
                                  dir / (std::string("force_") + value + ".log"));
        INFO(force.out);
        CHECK(force.code == 0);
        CHECK_THAT(force.out, ContainsSubstring(std::string("source: proxy not rebuilt, ") + words + " (the proxy built earlier stays: " + proxy + "): "));
        CHECK_THAT(force.out, ContainsSubstring("(on the first frame)"));
        CheckNoEnosys(force.out);
        CHECK_THAT(force.out, !ContainsSubstring("source: no proxy"));
        stays();
    }
}

TEST_CASE("dlssvid render and render --bench: NVDEC refusing or not starting on the first frame is said before timing; beside a shown source a refused one is shown empty and said",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("render_shown");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    ClipSpec plain;
    plain.frames = 12;
    plain.width = kW;
    plain.height = kH;
    const auto ffv1 = WriteClip(dir / "result.mkv", plain);
    const auto passes = dir / "passes";
    for (const auto& [value, words] : {std::pair<const char*, const char*>{"first-frame", "NVDEC refused this AV1 stream"},
                                       std::pair<const char*, const char*>{"start-fails", "NVDEC could not start decoding this AV1 stream now"}}) {
        INFO(value);
        FakeNvdecEnv fake(value);
        const auto json = dir / (std::string(value) + ".json");
        const auto rd = RunAll({"render", "--warp", "-i", clip.string(), "--passes", passes.string(), "--bench", "4", "--bench-json", json.string()},
                               dir / (std::string("bench_") + value + ".log"));
        INFO(rd.out);
        CHECK(rd.code != 0);
        CHECK_THAT(rd.out, ContainsSubstring(std::string("render: 'source': ") + words));
        CheckNoEnosys(rd.out);
        CHECK_FALSE(std::filesystem::exists(json));  // nothing timed
    }
    // a grid with the result (FFV1 decodes): measured / rendered - the source's cells empty, and said
    FakeNvdecEnv fake("first-frame");
    const std::string emptyWords = "render: 'source': NVDEC refused this AV1 stream and this build has no software AV1 decoder (on the first frame) - shown empty";
    const std::vector<std::string> grid{"--mode", "grid", "--sources", "source,result,source,result"};
    std::vector<std::string> bench{"render", "--warp", "-i", clip.string(), "--passes", passes.string(), "--result", ffv1.string()};
    bench.insert(bench.end(), grid.begin(), grid.end());
    std::vector<std::string> png = bench;
    const auto json = dir / "grid.json";
    for (const auto& a : {std::string("--bench"), std::string("4"), std::string("--bench-json"), json.string()}) bench.push_back(a);
    const auto gd = RunAll(bench, dir / "bench_grid.log");
    {
        INFO(gd.out);
        CHECK(gd.code == 0);
        CHECK(std::filesystem::exists(json));
        CHECK_THAT(gd.out, ContainsSubstring(emptyWords));
        CheckNoEnosys(gd.out);
    }
    const auto image = dir / "grid.png";
    for (const auto& a : {std::string("--frame"), std::string("2"), std::string("-o"), image.string()}) png.push_back(a);
    const auto pg = RunAll(png, dir / "render_grid.log");
    {
        INFO(pg.out);
        CHECK(pg.code == 0);
        CHECK(std::filesystem::exists(image));
        CHECK_THAT(pg.out, ContainsSubstring(emptyWords));
        CheckNoEnosys(pg.out);
    }
}

TEST_CASE("A stream whose sequence header changes (another size, another depth) decodes on NVDEC from end to end - each part on a decoder of its own",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("sequence_change");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec first;
    first.width = kW;
    first.height = kH;
    first.frames = 16;
    first.gop = 8;
    first.codec = "av1_nvenc";
    std::string why;
    const auto a = WriteClipIfSupported(dir / "a.ivf", first, &why);
    if (!a) SKIP("av1_nvenc does not support the clip here: " << why);
    ClipSpec bigger = first;
    bigger.width = 384;
    bigger.height = 224;
    const auto b = WriteClipIfSupported(dir / "b.ivf", bigger, &why);
    REQUIRE(b);
    struct Joined {
        std::filesystem::path file, second;
        uint32_t width, height;
        uint8_t depth;
    };
    const Joined sizes{ConcatIvf(*a, *b, dir / "sizes.ivf"), *b, 384, 224, 8};
    std::vector<Joined> joined;  // the depth first: a failure of the size's decode ends the case
    ClipSpec ten = first;  // 8-bit, then Main 10 (p010 in) of the same size
    ten.pixFmt = "p010le";
    if (const auto c = WriteClipIfSupported(dir / "c.ivf", ten, &why)) joined.push_back({ConcatIvf(*a, *c, dir / "depths.ivf"), *c, kW, kH, 10});
    else WARN("av1_nvenc writes no 10-bit clip here: " << why);
    joined.push_back(sizes);
    for (const Joined& j : joined) {
        INFO(j.file.filename().string());
        VideoDecoder alone(j.second);
        const std::vector<CpuFrame> reference = DecodeAllOf(alone);  // the second part decoded as a video of its own
        REQUIRE(reference.size() == 16);
        std::ostringstream text;
        std::vector<CpuFrame> frames;
        {
            DecoderLog capture(text);
            VideoDecoder dec(j.file);  // one decoder from frame 0 through the change
            frames = DecodeAllOf(dec);
        }
        REQUIRE(frames.size() == 32);
        for (size_t i = 0; i < frames.size(); ++i) {
            INFO("frame " << i);
            const bool second = i >= 16;
            CHECK(frames[i].index == static_cast<int64_t>(i));
            CHECK(frames[i].desc.width == (second ? j.width : kW));
            CHECK(frames[i].desc.height == (second ? j.height : kH));
            CHECK(frames[i].desc.bitDepth == (second ? j.depth : 8));
            if (second) CHECK(frames[i].data == reference[i - 16].data);  // the same pixels as the part alone
        }
        // said once, at info: another sequence header after the frames of the first
        const std::string log = text.str();
        INFO(log);
        CHECK(CountOf(log, "info " + j.file.string() + ": another AV1 sequence header (at frame 16) - its own NVDEC decoder") == 1);
        CHECK(CountOf(log, "another AV1 sequence header") == 1);
        // one decoder seeking into the second part, back into the first, forward again: each frame of its part
        VideoDecoder seeker(j.file);
        for (const int64_t target : {20, 5, 28, 10, 17}) {
            INFO("seek to " << target);
            REQUIRE(seeker.SeekToFrame(target));
            CpuFrame f;
            REQUIRE(seeker.NextFrame(f));
            CHECK(f.index == target);
            CHECK(f.desc.width == (target >= 16 ? j.width : kW));
            CHECK(f.desc.bitDepth == (target >= 16 ? j.depth : 8));
            CHECK(f.data == frames[static_cast<size_t>(target)].data);
        }
    }
    {
        // a stream of one sequence: nothing of the kind
        std::ostringstream text;
        DecoderLog capture(text);
        VideoDecoder dec(*a);
        CHECK(DecodeAllOf(dec).size() == 16);
        CHECK(CountOf(text.str(), "another AV1 sequence header") == 0);
    }
    {
        // NVDEC not starting for the second part's decoder (the stand-in comes after the first part's frames): that decoder's first frame
        // - a start failure, not a refusal of the stream, though the first part's decoder had decoded 16 frames
        VideoDecoder dec(sizes.file);
        CpuFrame f;
        for (int i = 0; i < 16; ++i) REQUIRE(dec.NextFrame(f));
        FakeNvdecEnv fake("start-fails");
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::NvdecStartFailed);
        CHECK(refused->support.Transient());
        CHECK_THAT(refused->support.why, ContainsSubstring("(at frame 16, on a part of the stream in other parameters (yuv420p 384x224))"));
    }
}

namespace {
// The temporal units of an IVF file, in order
std::vector<std::string> IvfUnits(const std::filesystem::path& file, const std::filesystem::path& scratch) {
    std::vector<std::string> units;
    RepeatIvf(file, 1, scratch, [&](int, const std::string& u) {
        units.push_back(u);
        return u;
    });
    return units;
}

std::string SequenceHeaderOf(const std::string& unit) {
    return std::string(Av1SequenceHeader(reinterpret_cast<const uint8_t*>(unit.data()), unit.size()));
}
}  // namespace

TEST_CASE("Av1SequenceKeyOf on av1_nvenc's own sequence headers, read from the packets: 8 and 10 bits as written, 12 bits as rewritten",
          "[integration][gpu][av1]") {
    const auto dir = Dir("sequence_key");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 8;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto eight = WriteClipIfSupported(dir / "eight.ivf", spec, &why);
    if (!eight) SKIP("av1_nvenc does not support the clip here: " << why);
    const std::string header8 = SequenceHeaderOf(IvfUnits(*eight, dir / "copy8.ivf")[0]);
    REQUIRE_FALSE(header8.empty());
    const auto k8 = Av1SequenceKeyOf(header8);
    REQUIRE(k8);
    CHECK(k8->profile == 0);
    CHECK(k8->bitDepth == 8);
    CHECK(k8->subsamplingX == 1);
    CHECK(k8->subsamplingY == 1);
    CHECK(k8->maxWidth == kW);
    CHECK(k8->maxHeight == kH);
    CHECK_FALSE(k8->mono);
    CHECK_FALSE(k8->rgb);
    spec.pixFmt = "p010le";  // Main 10
    if (const auto ten = WriteClipIfSupported(dir / "ten.ivf", spec, &why)) {
        const auto k10 = Av1SequenceKeyOf(SequenceHeaderOf(IvfUnits(*ten, dir / "copy10.ivf")[0]));
        REQUIRE(k10);
        CHECK(k10->bitDepth == 10);
        CHECK(k10->maxWidth == kW);
        CHECK(k10->maxHeight == kH);
        CHECK(k10->subsamplingX == 1);
        CHECK(k10->subsamplingY == 1);
        CHECK_FALSE(*k10 == *k8);
    } else {
        WARN("av1_nvenc writes no 10-bit clip here: " << why);
    }
    const auto k12 = Av1SequenceKeyOf(SequenceHeaderOf(TwelveBitTemporalUnit(IvfUnits(*eight, dir / "copy12.ivf")[0])));
    REQUIRE(k12);
    CHECK(k12->profile == 2);
    CHECK(k12->bitDepth == 12);
    CHECK(k12->subsamplingX == 1);
    CHECK(k12->subsamplingY == 1);
    CHECK(k12->maxWidth == kW);
}

TEST_CASE("A repeated sequence header that differs outside what NVDEC's decoder is made for (an initial display delay) keeps the decoder: the same frames",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("sequence_repeat");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 24;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto clip = WriteClipIfSupported(dir / "clip.ivf", spec, &why);
    if (!clip) SKIP("av1_nvenc does not support the clip here: " << why);
    // the keyframe 8's sequence header says an initial display delay: other bytes, the same key (size, depth, chroma)
    const auto delayed = RepeatIvf(*clip, 1, dir / "delayed.ivf", [](int i, const std::string& u) { return i == 8 ? InitialDisplayDelayTemporalUnit(u) : u; });
    const std::vector<std::string> units = IvfUnits(delayed, dir / "copy.ivf");
    REQUIRE(units.size() == 24);
    const std::string first = SequenceHeaderOf(units[0]), repeat = SequenceHeaderOf(units[8]);
    REQUIRE_FALSE(first.empty());
    REQUIRE_FALSE(repeat.empty());
    REQUIRE(first != repeat);
    REQUIRE(Av1SequenceKeyOf(first));
    REQUIRE(Av1SequenceKeyOf(repeat));
    REQUIRE(*Av1SequenceKeyOf(first) == *Av1SequenceKeyOf(repeat));
    VideoDecoder original(*clip);
    const std::vector<CpuFrame> reference = DecodeAllOf(original);
    REQUIRE(reference.size() == 24);
    std::ostringstream text;
    std::vector<CpuFrame> frames;
    {
        DecoderLog capture(text);
        VideoDecoder dec(delayed);
        frames = DecodeAllOf(dec);
    }
    REQUIRE(frames.size() == reference.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        INFO("frame " << i);
        CHECK(frames[i].data == reference[i].data);  // bit-identical to the clip as written
    }
    INFO(text.str());
    CHECK(CountOf(text.str(), "another AV1 sequence header") == 0);  // no decoder of its own: the same one goes on
}

TEST_CASE("dlssvid render --bench with a bench mode it does not know says that first - before any video is opened or refused", "[integration][cli][gpu][av1]") {
    const auto dir = Dir("bench_mode");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    FakeNvdecEnv fake("no-av1");  // the video would be refused here
    const auto json = dir / "bench.json";
    const auto r = RunAll({"render", "--warp", "-i", clip.string(), "--bench", "10", "--bench-mode", "foo", "--bench-json", json.string()}, dir / "bench.log");
    INFO(r.out);
    CHECK(r.code != 0);
    CHECK_THAT(r.out, ContainsSubstring("render: --bench-mode must be all | cold | warm | random"));
    CHECK_THAT(r.out, !ContainsSubstring(kNoAv1Reason));        // not the refusal of the video
    CHECK_THAT(r.out, !ContainsSubstring("D3D12 device"));      // nothing set up
    CHECK_THAT(r.out, !ContainsSubstring("decodes on NVDEC"));  // nothing decoded
    CHECK_FALSE(std::filesystem::exists(json));
}

TEST_CASE("NVDEC refusing a frame after a few decoded ones (later-frame) is the stream's verdict: kept, marked, remembered for the file",
          "[integration][viewport][gpu][av1][regression]") {
    const auto dir = Dir("later_frame");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const std::string atFrame = "NVDEC refused this AV1 stream and this build has no software AV1 decoder (at frame ";
    {
        std::optional<FakeNvdecEnv> fake;
        fake.emplace("later-frame");
        VideoDecoder dec(clip);
        CHECK_NOTHROW(dec.RequireDecodable());  // the driver takes the stream
        CpuFrame f;
        int decoded = 0;
        std::optional<DecodeUnavailable> refused;
        try {
            while (dec.NextFrame(f)) ++decoded;
            FAIL("all frames decoded under later-frame");
        } catch (const DecodeUnavailable& e) {
            refused = e;
        }
        REQUIRE(refused);
        CHECK(decoded >= 3);
        CHECK(dec.DecodesOnNvdec());  // NVDEC did start: the refusal is the stream's
        CHECK(refused->support.block == DecodeBlock::NvdecFailed);
        CHECK_FALSE(refused->support.Transient());
        // decoded from frame 0: the frame refused is the next one in the stream - its index, as many as came before it
        CHECK(refused->support.why == atFrame + std::to_string(decoded) + ")");
        fake.reset();
        const auto again = RefusalOfNextFrame(dec);  // kept
        REQUIRE(again);
        CHECK(again->support.block == DecodeBlock::NvdecFailed);
    }
    Av1Store s(0, 1);
    {
        FakeNvdecEnv fake("later-frame");
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        CHECK_FALSE(s.store->FindSource("source")->undecodable);  // the driver takes it
        // a fresh decoder walks to frame 20 from frame 0 (within FrameStore's kWalkLimit, no seek): 0, 1, 2, then the refusal
        CHECK(s.Settle(20) == FrameStore::Status::Missing);
        const auto marked = s.store->Undecodable("source");
        REQUIRE(marked);
        CHECK(marked->block == DecodeBlock::NvdecFailed);
        CHECK(marked->why == atFrame + "3)");  // the frame refused, by its place in the stream
        CHECK_FALSE(s.store->StartFailure("source"));
    }
    // remembered for the file: a new source of it is undecodable at once, whatever the stand-in now
    const auto again = FrameStore::VideoSource("source", clip);
    REQUIRE(again);
    REQUIRE(again->undecodable);
    CHECK(again->undecodable->block == DecodeBlock::NvdecFailed);
}

TEST_CASE("A refusal remembered for a file is found however the path is written: forward slashes, a sub/.. segment, another letter case",
          "[integration][viewport][gpu][av1][regression]") {
    const auto dir = Dir("refused_spelling");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    const auto twin = dir / "twin.mp4";  // the same stream in another file: not refused
    std::filesystem::copy_file(clip, twin);
    const VideoStreamInfo info = VideoDecoder(clip, VideoDecoder::Probe()).Info();
    REQUIRE(info.hardwareOnly);
    const DecodeSupport refusal = RefusedDecode(info, DecodeBlock::NvdecFailed, "a spelling of the path");
    REQUIRE_FALSE(FrameStore::VideoSource("source", clip)->undecodable);
    // remembered under one spelling: forward slashes throughout
    const std::filesystem::path forward = clip.generic_string();
    REQUIRE(forward.string() != clip.string());
    REQUIRE(FrameStore::RememberRefusedFile(forward, refusal));
    // found under others: back slashes with a «sub/..» segment, the name in other letters (Windows: the same file)
    const std::filesystem::path dotted = std::filesystem::path(dir.string() + "\\sub\\..\\" + clip.filename().string()).make_preferred();
    std::string upper = clip.string();
    for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    REQUIRE(std::filesystem::exists(upper));
    for (const std::filesystem::path& spelling : {clip, dotted, std::filesystem::path(upper)}) {
        INFO(spelling.string());
        const auto again = FrameStore::VideoSource("source", spelling);
        REQUIRE(again);
        REQUIRE(again->undecodable);
        CHECK(again->undecodable->block == DecodeBlock::NvdecFailed);
        CHECK(again->undecodable->why == refusal.why);
    }
    CHECK_FALSE(FrameStore::VideoSource("source", twin)->undecodable);  // another file is not
}

TEST_CASE("FrameStore::RememberRefusedFile says whether it remembered: a file not there is not, and a file made there later is not refused",
          "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("refused_missing");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    const DecodeSupport refusal = RefusedDecode(VideoDecoder(clip, VideoDecoder::Probe()).Info(), DecodeBlock::NvdecFailed, "remembered or not");
    CHECK(FrameStore::RememberRefusedFile(clip, refusal));  // a file there: remembered
    CHECK(FrameStore::VideoSource("source", clip)->undecodable);
    const auto later = dir / "later.mp4";
    REQUIRE_FALSE(std::filesystem::exists(later));
    CHECK_FALSE(FrameStore::RememberRefusedFile(later, refusal));  // nothing to read its size and time from: not remembered
    std::filesystem::copy_file(clip, later);
    CHECK_FALSE(FrameStore::VideoSource("source", later)->undecodable);  // a file made there afterwards is not refused
}

TEST_CASE("A file NVDEC refused is remembered with its size and time: written again on the same path (another size, or only touched), the verdict goes",
          "[integration][viewport][gpu][av1][regression]") {
    const auto dir = Dir("refused_stamp");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    const auto otherDir = Dir("refused_stamp_other");
    REQUIRE_AV1_CLIP(other, otherDir, false, 16);  // another stream: another size
    REQUIRE(std::filesystem::file_size(other) != std::filesystem::file_size(clip));
    // NVDEC refuses the file in a loader (later-frame: a verdict on the stream) - the process remembers it for the path
    auto refuse = [&] {
        Av1Store s(0, 1);
        FakeNvdecEnv fake("later-frame");
        s.store->SetSources(Project::Create(clip, dir / "none").Sources());
        REQUIRE(s.Settle(20) == FrameStore::Status::Missing);  // the loaders at rest before the stand-in goes
        REQUIRE(s.store->Undecodable("source"));
    };
    refuse();
    const auto refusedAt = std::filesystem::last_write_time(clip);
    REQUIRE(FrameStore::VideoSource("source", clip)->undecodable);  // the same file: remembered
    REQUIRE(FrameStore::VideoSource("source", clip)->undecodable);  // and asked again: still
    // only touched: the same bytes, another time - another file as far as the process knows
    std::filesystem::last_write_time(clip, refusedAt + std::chrono::seconds(2));
    {
        const auto again = FrameStore::VideoSource("source", clip);
        REQUIRE(again);
        CHECK_FALSE(again->undecodable);  // asked anew: the driver takes it
    }
    std::filesystem::last_write_time(clip, refusedAt);  // the old time back: the record went with the first look, nothing comes back
    CHECK_FALSE(FrameStore::VideoSource("source", clip)->undecodable);
    // refused again, then written over by another stream of another size - with the time of the refusal
    refuse();
    const auto refusedAgainAt = std::filesystem::last_write_time(clip);
    REQUIRE(FrameStore::VideoSource("source", clip)->undecodable);
    std::filesystem::copy_file(other, clip, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::last_write_time(clip, refusedAgainAt);  // only the size tells it from the refused one
    const auto replaced = FrameStore::VideoSource("source", clip);
    REQUIRE(replaced);
    CHECK_FALSE(replaced->undecodable);
}

TEST_CASE("NVDEC not starting now (start-fails, device-fails): render tries once more and says why; the log and dlssvid proxy say it is not a verdict",
          "[integration][cli][gpu][av1][proxy]") {
    const auto dir = Dir("start_fails_cli");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);
    struct Case {
        const char* fake;
        std::string reason;
    };
    const Case cases[] = {{"start-fails", kStartFailedReason}, {"device-fails", kDeviceFailedReason}};
    for (const Case& c : cases) {
        INFO(c.fake);
        FakeNvdecEnv fake(c.fake);  // the children inherit it
        const auto png = dir / (std::string(c.fake) + ".png");
        const auto rd = RunAll({"render", "--warp", "-i", clip.string(), "--frame", "2", "-o", png.string()}, dir / (std::string(c.fake) + "_render.log"));
        {
            INFO(rd.out);
            CHECK(rd.code != 0);
            CHECK_THAT(rd.out, ContainsSubstring("render: 'source': " + c.reason));  // the start failure's reason
            CHECK_THAT(rd.out, !ContainsSubstring("has none of the requested sources"));
            // the loader's log: a start failure is no verdict
            CHECK_THAT(rd.out, ContainsSubstring("'source': NVDEC did not start now (its frames are asked for again later): "));  // the path as render made it absolute
            CHECK_THAT(rd.out, !ContainsSubstring("cannot be decoded here"));
            CheckNoEnosys(rd.out);
            CHECK_FALSE(std::filesystem::exists(png));
        }
        const auto passes = dir / (std::string(c.fake) + "_passes");
        const auto px = RunAll({"proxy", "-i", clip.string(), "--passes", passes.string(), "--sources", "source", "--force"},
                               dir / (std::string(c.fake) + "_proxy.log"));
        {
            INFO(px.out);
            CHECK(px.code == 0);  // the other videos and the render go on
            CHECK_THAT(px.out, ContainsSubstring("source: no proxy now, NVDEC did not start (run it again later): " + c.reason));
            CHECK_THAT(px.out, !ContainsSubstring("the video cannot be decoded here"));
            CheckNoEnosys(px.out);
            CHECK_FALSE(ValidProxy(clip, passes, "source"));
            CHECK_FALSE(std::filesystem::exists(ProxyDir(passes) / "source.building.mkv"));
        }
    }
}

TEST_CASE("An AV1 stream of an odd frame height is refused before any frame; a part of one after an even part is refused when NVDEC meets it",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("odd_height");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 16;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto a = WriteClipIfSupported(dir / "a.ivf", spec, &why);
    if (!a) SKIP("av1_nvenc does not support the clip here: " << why);
    const auto b = WriteClipIfSupported(dir / "b.ivf", spec, &why);
    REQUIRE(b);
    // av1_nvenc's 320x192 said as 320x191 (the sequence header's maximum, the IVF header): the frames' headers stay valid - a stream NVDEC
    // decodes, the chroma of its frames a row off in FFmpeg 7.1.1
    const auto shorter = [](int, const std::string& u) { return ShorterFrameTemporalUnit(u, 1); };
    const auto odd = RepeatIvf(*a, 1, dir / "odd.ivf", shorter);
    SetIvfFrameSize(odd, static_cast<uint16_t>(kW), static_cast<uint16_t>(kH - 1));
    const VideoStreamInfo info = VideoDecoder(odd, VideoDecoder::Probe()).Info();
    REQUIRE(info.hardwareOnly);
    REQUIRE(info.codedHeight == kH - 1);
    REQUIRE(info.pixelFormat == "yuv420p");
    const std::string oddWords = "AV1 320x191 (the frame as stored) has an odd frame height - FFmpeg's NVDEC reads the chroma of such a frame a row off";
    {
        // the driver's answer first (this GPU decodes AV1: av1_nvenc wrote the clip), then the height - the caps as for the even stream
        const DecodeSupport s = CheckDecodeSupport(info);
        CHECK(s.block == DecodeBlock::OddHeight);
        CHECK(s.caps.checked == CheckDecodeSupport(VideoDecoder(*a, VideoDecoder::Probe()).Info()).caps.checked);
        CHECK_THAT(s.why, StartsWith(oddWords));
    }
    {
        // the whole stream: refused before its first frame, no CUDA device made for it
        VideoDecoder dec(odd);
        std::optional<DecodeUnavailable> refused;
        try {
            dec.RequireDecodable();
        } catch (const DecodeUnavailable& e) {
            refused = e;
        }
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::OddHeight);
        CHECK(refused->support.width == kW);
        CHECK(refused->support.height == kH - 1);
        CHECK_FALSE(refused->support.part);
        CHECK_FALSE(refused->support.Transient());
        CHECK_THAT(refused->support.why, StartsWith(oddWords));
        const auto first = RefusalOfNextFrame(dec);  // the first frame asked for: the same
        REQUIRE(first);
        CHECK(first->support.block == DecodeBlock::OddHeight);
        CHECK_FALSE(dec.HoldsCudaDevice());
    }
    {
        // HwAccel::Cuda: a hardware-only decoder holds the device from the open whatever its height (no software path to take instead),
        // also once refused for it
        VideoDecoder::Options cuda;
        cuda.hwaccel = HwAccel::Cuda;
        VideoDecoder dec(odd, cuda);
        CHECK(dec.HoldsCudaDevice());
        const auto first = RefusalOfNextFrame(dec);
        REQUIRE(first);
        CHECK(first->support.block == DecodeBlock::OddHeight);
        CHECK(dec.HoldsCudaDevice());
    }
    {
        const auto source = FrameStore::VideoSource("source", odd);
        REQUIRE(source);
        REQUIRE(source->undecodable);
        CHECK(source->undecodable->block == DecodeBlock::OddHeight);
    }
    // 192 rows, then a part of 191 (its own sequence header, its own NVDEC decoder): the probe sees the first part - refused when NVDEC
    // gives the part's first frame, as a part
    const auto parts = ConcatIvf(*a, *b, dir / "parts.ivf", [](const std::string& u) { return ShorterFrameTemporalUnit(u, 1); });
    const VideoStreamInfo partsInfo = VideoDecoder(parts, VideoDecoder::Probe()).Info();
    REQUIRE(partsInfo.codedHeight == kH);
    REQUIRE(CheckDecodeSupport(partsInfo).Ok());
    {
        VideoDecoder dec(parts);
        CpuFrame f;
        int frames = 0;
        std::optional<DecodeUnavailable> refused;
        try {
            while (dec.NextFrame(f)) {
                INFO("frame " << frames << ": " << f.desc.width << "x" << f.desc.height);
                ++frames;
            }
            FAIL("the part of an odd height decoded: " << frames << " frames");
        } catch (const DecodeUnavailable& e) {
            refused = e;
        }
        REQUIRE(refused);
        CHECK(frames == 16);  // the even part whole
        CHECK(refused->support.block == DecodeBlock::OddHeight);
        CHECK(refused->support.part);  // the probe's height was even
        CHECK(refused->support.height == kH - 1);
        CHECK_FALSE(refused->support.Transient());
        CHECK_THAT(refused->support.why, StartsWith(oddWords));
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 16)"));
        CHECK(RefusalOfNextFrame(dec));  // kept
    }
    {
        // a decoder whose first frame is in that part (a seek): the same verdict on its first frame
        VideoDecoder dec(parts);
        REQUIRE(dec.SeekToKeyframeBefore(20));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::OddHeight);
        CHECK(refused->support.part);
        CHECK(refused->support.height == kH - 1);
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 16)"));  // its place in the stream: the part's first frame
    }
}

TEST_CASE("FrameStore::RememberRefusedFile with keep: a verdict remembered for the very file stays; a file written again since takes the new one",
          "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("refused_keep");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    const VideoStreamInfo info = VideoDecoder(clip, VideoDecoder::Probe()).Info();
    const DecodeSupport loader = RefusedDecode(info, DecodeBlock::NvdecFailed, "on the first frame");    // a loader's own, more exact
    const DecodeSupport proxy = RefusedDecode(info, DecodeBlock::NvdecFailed, "the proxy build: at frame 3");  // another process's
    auto remembered = [&] {
        const auto s = FrameStore::VideoSource("source", clip);
        REQUIRE(s);
        return s->undecodable ? s->undecodable->why : std::string();
    };
    // nothing remembered yet: keep remembers it as any other
    CHECK(FrameStore::RememberRefusedFile(clip, loader, true));
    CHECK(remembered() == loader.why);
    // the same file as it was: kept - and said remembered
    CHECK(FrameStore::RememberRefusedFile(clip, proxy, true));
    CHECK(remembered() == loader.why);
    // without keep: the new verdict
    CHECK(FrameStore::RememberRefusedFile(clip, proxy));
    CHECK(remembered() == proxy.why);
    CHECK(FrameStore::RememberRefusedFile(clip, loader));
    // the file written again (another time) after the verdict: keep does not keep the old one - the new verdict is the file's
    std::filesystem::last_write_time(clip, std::filesystem::last_write_time(clip) + std::chrono::seconds(2));
    CHECK(FrameStore::RememberRefusedFile(clip, proxy, true));
    CHECK(remembered() == proxy.why);
    // another size, the same time: the same
    CHECK(FrameStore::RememberRefusedFile(clip, loader));
    const auto time = std::filesystem::last_write_time(clip);
    {
        std::ofstream grow(clip, std::ios::binary | std::ios::app);  // an MP4 'free' box of 16 bytes: the file still a valid MP4
        grow << std::string("\0\0\0\x10" "free", 8) << std::string(8, '\0');
    }
    std::filesystem::last_write_time(clip, time);
    CHECK(FrameStore::RememberRefusedFile(clip, proxy, true));
    CHECK(remembered() == proxy.why);
    // no file: nothing remembered, keep or not
    CHECK_FALSE(FrameStore::RememberRefusedFile(dir / "none.mp4", proxy, true));
}

TEST_CASE("An AV1 stream of an odd frame width decodes on NVDEC at its own width: frames cut to it, no GpuFrame; process --passthrough and BuildProxy take it",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("odd_width");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 16;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto even = WriteClipIfSupported(dir / "even.ivf", spec, &why);
    if (!even) SKIP("av1_nvenc does not support the clip here: " << why);
    // av1_nvenc's 320x192 said as 319x192 (the sequence header's maximum, the IVF header): the 8-pixel block columns (MiCols) are the same,
    // the frames' headers valid - NVDEC decodes it into an even-sided surface (320)
    const auto odd = RepeatIvf(*even, 1, dir / "odd.ivf", [](int, const std::string& u) { return NarrowerFrameTemporalUnit(u, 1); });
    SetIvfFrameSize(odd, static_cast<uint16_t>(kW - 1), static_cast<uint16_t>(kH));
    const VideoStreamInfo info = VideoDecoder(odd, VideoDecoder::Probe()).Info();
    REQUIRE(info.width == kW - 1);
    REQUIRE(info.codedWidth == kW - 1);
    CHECK(CheckDecodeSupport(info).Ok());  // the driver takes it
    std::vector<CpuFrame> evenFrames, oddFrames;
    {
        VideoDecoder dec(*even);
        evenFrames = DecodeAllOf(dec);
    }
    {
        VideoDecoder dec(odd);
        oddFrames = DecodeAllOf(dec);  // no refusal, no start failure
    }
    REQUIRE(evenFrames.size() == 16);
    REQUIRE(oddFrames.size() == 16);
    for (size_t i = 0; i < oddFrames.size(); ++i) {
        INFO("frame " << i);
        const CpuFrame& o = oddFrames[i];
        const CpuFrame& e = evenFrames[i];
        CHECK(o.desc.width == kW - 1);  // the video's own width, not the surface's
        CHECK(o.desc.height == kH);
        REQUIRE(o.data.size() == o.desc.ByteSize());
        // the chroma (160 wide for 319 and 320) as the even stream's, row for row; the luma too, away from the right edge (the frame's own
        // edge filtering differs in its last columns)
        const size_t ew = e.desc.width, ow = o.desc.width, ecw = e.desc.PlaneWidth(1), ocw = o.desc.PlaneWidth(1);
        REQUIRE(ocw == 160);
        int lumaRows = 0, uRows = 0, vRows = 0;
        for (size_t r = 0; r < kH; ++r) lumaRows += std::memcmp(e.Plane(0) + r * ew, o.Plane(0) + r * ow, 256) == 0;
        for (size_t r = 0; r < kH / 2; ++r) {
            uRows += std::memcmp(e.Plane(1) + r * ecw, o.Plane(1) + r * ocw, 160) == 0;
            vRows += std::memcmp(e.Plane(2) + r * ecw, o.Plane(2) + r * ocw, 160) == 0;
        }
        CHECK(lumaRows == static_cast<int>(kH));
        CHECK(uRows == static_cast<int>(kH / 2));
        CHECK(vRows == static_cast<int>(kH / 2));
    }
    // CUDA asked for and a GpuFrame with it: an odd width goes through the host (the device's NV12 consumers take even rows); an even one
    // has its GpuFrame
    for (const auto& [file, oddWidth] : {std::pair{*even, false}, std::pair{odd, true}}) {
        INFO(file.filename().string());
        VideoDecoder::Options o;
        o.hwaccel = HwAccel::Cuda;
        VideoDecoder dec(file, o);
        CpuFrame f;
        GpuFrame g;
        REQUIRE(dec.NextFrame(f, &g));
        REQUIRE(dec.UsingHwAccel());
        CHECK(g.Valid() == !oddWidth);
        if (oddWidth) {
            CHECK(f.desc.width == kW - 1);
            CHECK(f.data == oddFrames[0].data);
        } else {
            CHECK(g.width == kW);
        }
    }
    // process --passthrough: the frames into FFV1 as decoded (it used to fail: «frame format mismatch»)
    const auto pt = RunAll({"process", "-i", odd.string(), "-o", (dir / "pt.mkv").string(), "--passthrough", "--warp", "--codec", "ffv1"}, dir / "pt.log");
    INFO(pt.out);
    CHECK(pt.code == 0);
    CHECK_THAT(pt.out, !ContainsSubstring("frame format mismatch"));
    const std::vector<CpuFrame> written = DecodeAll(dir / "pt.mkv");
    REQUIRE(written.size() == oddFrames.size());
    for (size_t i = 0; i < written.size(); ++i) CHECK(written[i].data == oddFrames[i].data);
    // the scrubbing proxy: built, of the video's own width
    const auto passes = dir / "passes";
    const auto proxy = BuildProxy(odd, passes, "source");
    CHECK(ValidProxy(odd, passes, "source"));
    const std::vector<CpuFrame> proxied = DecodeAll(proxy);
    REQUIRE(proxied.size() == oddFrames.size());
    CHECK(proxied[0].desc.width == kW - 1);
}

TEST_CASE("NVDEC with a software path: an odd frame height (MPEG-4 320x191) decodes in software, as the software decode; an even one on NVDEC",
          "[integration][gpu][nvdec][regression]") {
    const auto dir = Dir("odd_height_software_codec");
    if (!HaveNvidia()) SKIP("no NVIDIA GPU");
    if (!VideoEncoder::EncoderAvailable("mpeg4")) SKIP("no MPEG-4 encoder in this build");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH - 1;
    spec.frames = 6;
    spec.codec = "mpeg4";  // FFmpeg's own encoder: the frame's height as it is (H.264 / HEVC code 16 / 8-row aligned heights)
    const auto odd = WriteClip(dir / "odd.mkv", spec);
    spec.height = kH;
    const auto even = WriteClip(dir / "even.mkv", spec);
    VideoDecoder::Options cuda;
    cuda.hwaccel = HwAccel::Cuda;
    // the odd one: software from the open (no CUDA device, no GpuFrame), said once; the frames as the software decode's
    std::ostringstream text;
    std::vector<CpuFrame> hard;
    {
        DecoderLog capture(text);
        VideoDecoder dec(odd, cuda);
        CHECK_FALSE(dec.UsingHwAccel());
        CHECK_FALSE(dec.HoldsCudaDevice());
        CpuFrame f;
        GpuFrame g;
        while (dec.NextFrame(f, &g)) {
            CHECK_FALSE(g.Valid());
            hard.push_back(f);
        }
        CHECK_FALSE(dec.DecodesOnNvdec());
        CHECK_FALSE(dec.HoldsCudaDevice());
    }
    VideoDecoder plain(odd);
    const std::vector<CpuFrame> soft = DecodeAllOf(plain);
    REQUIRE(hard.size() == 6);
    REQUIRE(soft.size() == hard.size());
    for (size_t i = 0; i < hard.size(); ++i) {
        INFO("frame " << i);
        CHECK(hard[i].desc.height == kH - 1);
        CHECK((hard[i].data == soft[i].data));  // (not decomposed: no dump of the frames)
    }
    const std::string log = text.str();
    INFO(log);
    CHECK(CountOf(log, odd.string() + ": an odd height (191): FFmpeg's NVDEC gives its chroma wrong, decoding in software") == 1);
    CHECK(CountOf(log, "a frame of an odd height") == 0);  // get_format's warning: for a size met later only
    {
        // the even one: NVDEC where this GPU decodes MPEG-4 (last: the odd-height checks above hold on any NVIDIA GPU)
        VideoDecoder dec(even, cuda);
        CpuFrame f;
        GpuFrame g;
        REQUIRE(dec.NextFrame(f, &g));
        if (!dec.DecodesOnNvdec()) SKIP("NVDEC does not decode MPEG-4 here");
        CHECK(dec.UsingHwAccel());
        CHECK(g.Valid());
    }
}

TEST_CASE("NVDEC with a software path: an H.264 stream of an odd height (cropped from 192) with --hwaccel cuda gives the software decode's frames, the chroma's last row too",
          "[integration][gpu][nvdec][regression]") {
    const auto dir = Dir("odd_height_h264");
    if (!HaveNvidia() || !VideoEncoder::EncoderAvailable("h264_nvenc")) SKIP("no NVIDIA GPU with h264_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH - 1;
    spec.frames = 6;
    spec.codec = "h264_nvenc";
    std::string why;
    const auto clip = WriteClipIfSupported(dir / "odd.mkv", spec, &why);
    if (!clip) SKIP("h264_nvenc does not support the clip here: " << why);
    VideoDecoder plain(*clip);
    const std::vector<CpuFrame> soft = DecodeAllOf(plain);
    VideoDecoder::Options cuda;
    cuda.hwaccel = HwAccel::Cuda;
    std::ostringstream text;
    std::vector<CpuFrame> hard;
    std::optional<VideoDecoder> decoder;
    {
        DecoderLog capture(text);
        decoder.emplace(*clip, cuda);
        CpuFrame f;
        while (decoder->NextFrame(f)) hard.push_back(f);
    }
    const VideoDecoder& dec = *decoder;
    {
        // the height as shown (191; H.264 codes it as 192 and crops): software from the open, said once with that height
        const std::string log = text.str();
        INFO(log);
        CHECK_FALSE(dec.DecodesOnNvdec());
        CHECK_FALSE(dec.UsingHwAccel());
        CHECK_FALSE(dec.HoldsCudaDevice());
        CHECK(CountOf(log, clip->string() + ": an odd height (191): FFmpeg's NVDEC gives its chroma wrong, decoding in software") == 1);
        CHECK(CountOf(log, "a frame of an odd height") == 0);
    }
    REQUIRE(soft.size() == 6);
    REQUIRE(hard.size() == soft.size());
    for (size_t i = 0; i < hard.size(); ++i) {
        INFO("frame " << i << (dec.DecodesOnNvdec() ? " (NVDEC)" : " (software)"));
        CHECK(hard[i].desc.height == kH - 1);
        // the chroma rows: (191 + 1) / 2 = 96 - FFmpeg's CUDA transfer copies 191 >> 1 = 95 of them
        const size_t cw = soft[i].desc.PlaneWidth(1), ch = soft[i].desc.PlaneHeight(1);
        REQUIRE(ch == 96);
        CHECK(std::memcmp(hard[i].Plane(1) + (ch - 1) * cw, soft[i].Plane(1) + (ch - 1) * cw, cw) == 0);
        CHECK((hard[i].data == soft[i].data));  // (not decomposed: no dump of the frames)
    }
}

TEST_CASE("An AV1 stream whose layout the probe did not learn: the container's odd height judges nothing; the first frame's odd height is refused, not as a part",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("unknown_odd_height");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 16;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto a = WriteClipIfSupported(dir / "a.ivf", spec, &why);
    if (!a) SKIP("av1_nvenc does not support the clip here: " << why);
    // 320x191 in its sequence headers, none before frame 160: the probe does not learn the layout; its size is the IVF header's - 192
    // (even: a stream judged by it would call the odd frames a part), then 191
    constexpr int kFirstHeader = 160;
    const auto evenHeader = RepeatIvf(*a, 13, dir / "unknown_odd.ivf", [](int i, const std::string& u) {
        const std::string shorter = ShorterFrameTemporalUnit(u, 1);
        return i < kFirstHeader ? WithoutSequenceHeaders(shorter) : shorter;
    });
    const auto oddHeader = dir / "unknown_odd191.ivf";
    std::filesystem::copy_file(evenHeader, oddHeader);
    SetIvfFrameSize(oddHeader, static_cast<uint16_t>(kW), static_cast<uint16_t>(kH - 1));
    for (const auto& [file, containerHeight] : {std::pair{evenHeader, kH}, std::pair{oddHeader, kH - 1}}) {
        INFO(file.filename().string());
        const VideoStreamInfo info = VideoDecoder(file, VideoDecoder::Probe()).Info();
        REQUIRE(info.pixelFormat == "unknown");
        REQUIRE(info.codedHeight == containerHeight);
        CHECK(CheckDecodeSupport(info).Ok());  // the container's height is no verdict
        VideoDecoder dec(file);
        CHECK_NOTHROW(dec.RequireDecodable());
        REQUIRE(dec.SeekToKeyframeBefore(170));
        const auto refused = RefusalOfNextFrame(dec);  // the frame's own height, 191
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::OddHeight);
        CHECK(refused->support.height == kH - 1);
        CHECK_FALSE(refused->support.part);  // the stream's own height: the probe knew no layout of it
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 168)"));  // the keyframe the seek landed on
    }
    // NVDEC refusing the first frame (the stand-in): CheckDecode judges the layout learned there by the sequence header's size (320x191)
    // - the odd height, the stream's own: the IVF header's size (the probe's, which learned no layout) is no evidence of a part, also
    // where it (192) differs
    for (const auto& file : {evenHeader, oddHeader}) {
        INFO(file.filename().string());
        FakeNvdecEnv fake("start-fails");
        VideoDecoder dec(file);
        REQUIRE(dec.SeekToKeyframeBefore(170));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::OddHeight);
        CHECK(refused->support.height == kH - 1);
        CHECK_FALSE(refused->support.Transient());
        CHECK_FALSE(refused->support.part);
        CHECK_THAT(refused->support.why, StartsWith("AV1 320x191 (the frame as stored) has an odd frame height"));
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 168, in yuv420p 320x191 - a layout the probe did not learn)"));
    }
    // 12-bit of 191 rows in its sequence headers, the IVF header 192, no stand-in: the layout's verdict, the stream's own
    const auto twelve = RepeatIvf(*a, 13, dir / "unknown_odd12.ivf", [](int i, const std::string& u) {
        const std::string part = TwelveBitTemporalUnit(ShorterFrameTemporalUnit(u, 1));
        return i < kFirstHeader ? WithoutSequenceHeaders(part) : part;
    });
    {
        const VideoStreamInfo info = VideoDecoder(twelve, VideoDecoder::Probe()).Info();
        REQUIRE(info.pixelFormat == "unknown");
        REQUIRE(info.codedHeight == kH);
        VideoDecoder dec(twelve);
        REQUIRE(dec.SeekToKeyframeBefore(170));
        const auto refused = RefusalOfNextFrame(dec);
        REQUIRE(refused);
        CHECK(refused->support.block == DecodeBlock::Layout);
        CHECK_FALSE(refused->support.part);
        CHECK_THAT(refused->support.why, Catch::Matchers::EndsWith("(at frame 168, in yuv420p12le 320x191 - a layout the probe did not learn)"));
    }
}

TEST_CASE("dlssvid process --project P -i <an AV1 video not decodable here> refuses it before hashing it: the reason, no word of P's source (also --plan)",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("process_project_other");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    ClipSpec plain;
    plain.frames = 6;
    plain.width = 64;
    plain.height = 48;
    const auto own = WriteClip(dir / "own.mkv", plain);
    Project::Create(own, dir / "passes").Save(dir / "own.dlssvid.json");
    FakeNvdecEnv fake("no-av1");
    for (const bool plan : {false, true}) {
        INFO((plan ? "--plan" : "a run"));
        std::vector<std::string> args{"process", "--project", (dir / "own.dlssvid.json").string(), "-i", clip.string(), "-o", (dir / "out.mkv").string(),
                                      "--warp", "--codec", "ffv1"};
        if (plan) args.push_back("--plan");
        const auto r = RunAll(args, dir / (plan ? "plan.log" : "run.log"));
        INFO(r.out);
        CHECK(r.code != 0);
        CHECK_THAT(r.out, ContainsSubstring(kNoAv1Reason));
        CHECK_THAT(r.out, !ContainsSubstring("is not the source of the project"));
        CHECK_FALSE(std::filesystem::exists(dir / "out.mkv"));
    }
}

TEST_CASE("dlssvid info of an AV1 video not decodable here: its samples without decoded-as - no AV1 decoder, a layout refused; one that decodes says it",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("info_not_decoded");
    if (!HaveAv1Encoder()) SKIP("no NVIDIA GPU with av1_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 16;
    spec.gop = 8;
    spec.codec = "av1_nvenc";
    std::string why;
    const auto a = WriteClipIfSupported(dir / "a.ivf", spec, &why);
    if (!a) SKIP("av1_nvenc does not support the clip here: " << why);
    const auto twelve = RepeatIvf(*a, 1, dir / "twelve.ivf", [](int, const std::string& u) { return TwelveBitTemporalUnit(u); });  // 12-bit: a layout refused
    REQUIRE(VideoDecoder(twelve, VideoDecoder::Probe()).Info().bitDepth == 12);
    {
        // decodes here: «decoded as»
        const auto r = RunAll({"info", a->string()}, dir / "a.log");
        INFO(r.out);
        CHECK(r.code == 0);
        CHECK_THAT(r.out, ContainsSubstring("samples:     8-bit 420, decoded as "));
        CHECK_THAT(r.out, !ContainsSubstring("not decoded here"));
    }
    {
        const auto r = RunAll({"info", twelve.string()}, dir / "twelve.log");
        INFO(r.out);
        CHECK(r.code == 0);
        CHECK_THAT(r.out, ContainsSubstring("samples:     12-bit 420 (not decoded here, see decode)"));
        CHECK_THAT(r.out, !ContainsSubstring("decoded as"));
        CHECK_THAT(r.out, ContainsSubstring("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here"));
    }
    {
        FakeNvdecEnv fake("no-av1");
        const auto r = RunAll({"info", a->string()}, dir / "noav1.log");
        INFO(r.out);
        CHECK(r.code == 0);
        CHECK_THAT(r.out, ContainsSubstring("samples:     8-bit 420 (not decoded here, see decode)"));
        CHECK_THAT(r.out, !ContainsSubstring("decoded as"));
        CHECK_THAT(r.out, ContainsSubstring(kNoAv1Reason));
    }
}

TEST_CASE("NVDEC with a software path: a frame decoded in software, asked for on the device without a host copy (cpu = false), comes on the host - FFV1, an odd-height MPEG-4",
          "[integration][gpu][nvdec][regression]") {
    const auto dir = Dir("software_no_cpu");
    if (!HaveNvidia()) SKIP("no NVIDIA GPU");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 4;
    std::vector<std::filesystem::path> files{WriteClip(dir / "ffv1.mkv", spec)};  // no NVDEC for FFV1: software with the device held
    if (VideoEncoder::EncoderAvailable("mpeg4")) {
        spec.height = kH - 1;
        spec.codec = "mpeg4";
        files.push_back(WriteClip(dir / "odd.mkv", spec));  // an odd height: software from the open
    }
    for (const auto& file : files) {
        INFO(file.filename().string());
        VideoDecoder plain(file);
        const std::vector<CpuFrame> soft = DecodeAllOf(plain);
        REQUIRE(soft.size() == 4);
        VideoDecoder::Options cuda;
        cuda.hwaccel = HwAccel::Cuda;
        VideoDecoder dec(file, cuda);
        if (file.filename() == "ffv1.mkv") REQUIRE(dec.UsingHwAccel());  // GpuFrames may come: the path where cpu = false is kept
        for (size_t i = 0; i < soft.size(); ++i) {
            INFO("frame " << i);
            CpuFrame f;
            GpuFrame g;
            REQUIRE(dec.NextFrame(f, &g, false));  // a consumer that takes GpuFrames and copies nothing to the host itself
            CHECK_FALSE(g.Valid());                 // no frame on the device: software
            REQUIRE_FALSE(f.data.empty());          // so on the host, whatever cpu said
            CHECK(f.data.size() == f.desc.ByteSize());
            CHECK((f.data == soft[i].data));
        }
    }
}

TEST_CASE("dlssvid info of an MPEG-4 video of an odd height: software even with --hwaccel cuda, and why; an even one may use NVDEC", "[integration][cli][nvdec]") {
    const auto dir = Dir("info_odd_height");
    if (!VideoEncoder::EncoderAvailable("mpeg4")) SKIP("no MPEG-4 encoder in this build");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH - 1;
    spec.frames = 3;
    spec.codec = "mpeg4";
    const auto odd = WriteClip(dir / "odd.mkv", spec);
    spec.height = kH;
    const auto even = WriteClip(dir / "even.mkv", spec);
    const auto o = RunAll({"info", odd.string()}, dir / "odd.log");
    INFO(o.out);
    CHECK(o.code == 0);
    CHECK_THAT(o.out, ContainsSubstring("decode:      software (mpeg4); not NVDEC even with --hwaccel cuda: an odd frame height (320x191, the frame as "
                                        "stored) - FFmpeg's NVDEC gives its chroma wrong"));
    const auto e = RunAll({"info", even.string()}, dir / "even.log");
    INFO(e.out);
    CHECK(e.code == 0);
    CHECK_THAT(e.out, ContainsSubstring("decode:      software (mpeg4); NVDEC with --hwaccel cuda where FFmpeg offers it for this layout and the GPU decodes it"));
}

namespace {
// Every file under `dir` with its bytes (relative paths)
std::map<std::string, std::string> FilesOf(const std::filesystem::path& dir) {
    std::map<std::string, std::string> out;
    if (!std::filesystem::exists(dir)) return out;
    for (const auto& e : std::filesystem::recursive_directory_iterator(dir))
        if (e.is_regular_file()) out[std::filesystem::relative(e.path(), dir).generic_string()] = Read(e.path());
    return out;
}
}  // namespace

TEST_CASE("dlssvid scenes without NVDEC AV1 refuses only when it decodes: --threshold 0 and stored scores go on; --reanalyze and no scores refuse before the pass is kept as a version",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("scenes_no_av1");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    const std::string c = clip.string();
    // a pass with this video's scores, made where NVDEC decodes AV1; a pass of another video (no scores for this one)
    const auto scored = dir / "scored";
    REQUIRE(RunAll({"scenes", "-i", c, "-o", scored.string()}, dir / "scored.log").code == 0);
    ClipSpec plain;
    plain.frames = 8;
    plain.width = 64;
    plain.height = 48;
    const auto other = WriteClip(dir / "other.mkv", plain);
    const auto foreign = dir / "foreign";
    REQUIRE(RunAll({"scenes", "-i", other.string(), "-o", foreign.string()}, dir / "foreign.log").code == 0);
    FakeNvdecEnv fake("no-av1");
    {
        // --threshold 0: the pass written off - nothing to decode
        const auto off = dir / "off";
        const auto r = RunAll({"scenes", "-i", c, "-o", off.string(), "--threshold", "0"}, dir / "off.log");
        INFO(r.out);
        CHECK(r.code == 0);
        CHECK_THAT(r.out, ContainsSubstring("threshold 0.00 (off"));
        CHECK_THAT(r.out, !ContainsSubstring(kNoAv1Reason));
        CHECK(std::filesystem::exists(off / "scenes" / "manifest.json"));
    }
    {
        // --reanalyze over the scored pass: refused; that pass as it was, no version of it made
        const auto before = FilesOf(scored);
        const auto r = RunAll({"scenes", "-i", c, "-o", scored.string(), "--reanalyze"}, dir / "reanalyze.log");
        INFO(r.out);
        CHECK(r.code != 0);
        CHECK_THAT(r.out, ContainsSubstring(c + ": " + kNoAv1Reason));
        CHECK_FALSE(std::filesystem::exists(scored / "scenes.v"));
        CHECK((FilesOf(scored) == before));
    }
    {
        // no scores for this video (another one's pass there): refused before that pass goes to the history
        const auto before = FilesOf(foreign);
        const auto r = RunAll({"scenes", "-i", c, "-o", foreign.string()}, dir / "foreign_again.log");
        INFO(r.out);
        CHECK(r.code != 0);
        CHECK_THAT(r.out, ContainsSubstring(c + ": " + kNoAv1Reason));
        CHECK_FALSE(std::filesystem::exists(foreign / "scenes.v"));
        CHECK((FilesOf(foreign) == before));
    }
    {
        // no pass at all: refused, nothing made
        const auto none = dir / "none";
        const auto r = RunAll({"scenes", "-i", c, "-o", none.string()}, dir / "none.log");
        INFO(r.out);
        CHECK(r.code != 0);
        CHECK_THAT(r.out, ContainsSubstring(c + ": " + kNoAv1Reason));
        CHECK_FALSE(std::filesystem::exists(none / "scenes"));
        CHECK_FALSE(std::filesystem::exists(none / "scenes.v"));
    }
    {
        // the stored scores: the cuts re-derived, no decoding - the earlier pass kept as a version
        const auto r = RunAll({"scenes", "-i", c, "-o", scored.string()}, dir / "rederive.log");
        INFO(r.out);
        CHECK(r.code == 0);
        CHECK_THAT(r.out, ContainsSubstring("stored scores, no decoding"));
        CHECK(std::filesystem::exists(scored / "scenes.v"));
    }
}

TEST_CASE("dlssvid process --project P whose own source is an AV1 video not decodable here refuses it at once: the reason, no GPU device, nothing written (also --plan)",
          "[integration][cli][gpu][av1]") {
    const auto dir = Dir("process_project_own");
    REQUIRE_AV1_CLIP(clip, dir, false, 12);
    const auto passes = dir / "passes";
    Project::Create(clip, passes).Save(dir / "clip.dlssvid.json");
    FakeNvdecEnv fake("no-av1");
    for (const bool plan : {false, true}) {
        INFO((plan ? "--plan" : "a run"));
        const auto out = dir / "out.mkv";
        std::vector<std::string> args{"process", "--project", (dir / "clip.dlssvid.json").string(), "-o", out.string(), "--warp", "--codec", "ffv1"};
        if (plan) args.push_back("--plan");
        const auto r = RunAll(args, dir / (plan ? "plan.log" : "run.log"));
        INFO(r.out);
        CHECK(r.code != 0);
        CHECK_THAT(r.out, ContainsSubstring("] process: "));  // (the project's path, as it stores it)
        CHECK_THAT(r.out, ContainsSubstring(clip.filename().string() + ": " + kNoAv1Reason));
        CHECK_THAT(r.out, !ContainsSubstring("D3D12 device"));
        CHECK_FALSE(std::filesystem::exists(out));
    }
}

TEST_CASE("A codec NVDEC does not decode (ProRes) of an odd height with --hwaccel cuda: software, and no word of an odd height - NVDEC was not offered",
          "[integration][gpu][nvdec]") {
    const auto dir = Dir("prores_odd_height");
    if (!HaveNvidia()) SKIP("no NVIDIA GPU");
    if (!VideoEncoder::EncoderAvailable("prores_ks")) SKIP("no ProRes encoder in this build");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH - 1;
    spec.frames = 3;
    spec.codec = "prores_ks";
    spec.pixFmt = "yuv422p10le";
    const auto odd = WriteClip(dir / "odd.mov", spec);
    VideoDecoder plain(odd);
    const std::vector<CpuFrame> soft = DecodeAllOf(plain);
    REQUIRE(soft.size() == 3);
    std::ostringstream text;
    std::vector<CpuFrame> frames;
    {
        DecoderLog capture(text);
        VideoDecoder::Options cuda;
        cuda.hwaccel = HwAccel::Cuda;
        VideoDecoder dec(odd, cuda);
        frames = DecodeAllOf(dec);
        CHECK_FALSE(dec.DecodesOnNvdec());
    }
    const std::string log = text.str();
    INFO(log);
    REQUIRE(frames.size() == soft.size());
    for (size_t i = 0; i < frames.size(); ++i) CHECK((frames[i].data == soft[i].data));
    CHECK(CountOf(log, "odd height") == 0);  // no CUDA in the decoder's list: not the height's doing
}

namespace {
// `in`'s video stream through the bitstream filter `bsf` with `options` ("key=value", ...) into `out` (the same container kind); false: the
// filter is not in this build
bool WithBitstreamFilter(const std::filesystem::path& in, const std::filesystem::path& out, const char* bsf,
                         const std::vector<std::pair<const char*, const char*>>& options) {
    const AVBitStreamFilter* filter = av_bsf_get_by_name(bsf);
    if (!filter) return false;
    AVFormatContext* ic = nullptr;
    REQUIRE(avformat_open_input(&ic, in.string().c_str(), nullptr, nullptr) == 0);
    REQUIRE(avformat_find_stream_info(ic, nullptr) >= 0);
    const int vi = av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    REQUIRE(vi >= 0);
    AVBSFContext* ctx = nullptr;
    REQUIRE(av_bsf_alloc(filter, &ctx) == 0);
    REQUIRE(avcodec_parameters_copy(ctx->par_in, ic->streams[vi]->codecpar) >= 0);
    ctx->time_base_in = ic->streams[vi]->time_base;
    for (const auto& [key, value] : options) REQUIRE(av_opt_set(ctx->priv_data, key, value, 0) == 0);
    REQUIRE(av_bsf_init(ctx) == 0);
    AVFormatContext* oc = nullptr;
    REQUIRE(avformat_alloc_output_context2(&oc, nullptr, nullptr, out.string().c_str()) >= 0);
    AVStream* os = avformat_new_stream(oc, nullptr);
    REQUIRE(os);
    REQUIRE(avcodec_parameters_copy(os->codecpar, ctx->par_out) >= 0);
    os->codecpar->codec_tag = 0;
    os->time_base = ctx->time_base_out;
    REQUIRE(avio_open(&oc->pb, out.string().c_str(), AVIO_FLAG_WRITE) >= 0);
    REQUIRE(avformat_write_header(oc, nullptr) >= 0);
    AVPacket* pkt = av_packet_alloc();
    auto drain = [&] {
        while (av_bsf_receive_packet(ctx, pkt) == 0) {
            av_packet_rescale_ts(pkt, ctx->time_base_out, os->time_base);
            pkt->stream_index = 0;
            REQUIRE(av_interleaved_write_frame(oc, pkt) >= 0);
        }
    };
    while (av_read_frame(ic, pkt) >= 0) {
        if (pkt->stream_index != vi) {
            av_packet_unref(pkt);
            continue;
        }
        REQUIRE(av_bsf_send_packet(ctx, pkt) == 0);
        drain();
    }
    REQUIRE(av_bsf_send_packet(ctx, nullptr) == 0);
    drain();
    REQUIRE(av_write_trailer(oc) == 0);
    av_packet_free(&pkt);
    avio_closep(&oc->pb);
    avformat_free_context(oc);
    av_bsf_free(&ctx);
    avformat_close_input(&ic);
    return true;
}
}  // namespace

TEST_CASE("NVDEC: an H.264 stream cropped at the top (its SPS) with --hwaccel cuda is not cut at the top-left corner - its frame as before the cut, the picture where it was",
          "[integration][gpu][nvdec][regression]") {
    const auto dir = Dir("h264_crop_top");
    if (!HaveNvidia() || !VideoEncoder::EncoderAvailable("h264_nvenc")) SKIP("no NVIDIA GPU with h264_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 6;
    spec.codec = "h264_nvenc";
    std::string why;
    const auto plain = WriteClipIfSupported(dir / "plain.mkv", spec, &why);
    if (!plain) SKIP("h264_nvenc does not support the clip here: " << why);
    // its SPS rewritten to crop 8 rows at the top: shown 320x184, coded 320x192 - FFmpeg crops a CUDA frame only on the right and the
    // bottom (the pointers stay): crop_top stays on the frame
    const auto cropped = dir / "cropped.mkv";
    if (!WithBitstreamFilter(*plain, cropped, "h264_metadata", {{"crop_top", "8"}})) SKIP("no h264_metadata bitstream filter in this build");
    const VideoStreamInfo info = VideoDecoder(cropped, VideoDecoder::Probe()).Info();
    REQUIRE(info.height == kH - 8);
    VideoDecoder plainDec(cropped);
    const std::vector<CpuFrame> soft = DecodeAllOf(plainDec);  // software: cropped as it should be
    REQUIRE(soft.size() == 6);
    REQUIRE(soft[0].desc.height == kH - 8);
    VideoDecoder::Options cuda;
    cuda.hwaccel = HwAccel::Cuda;
    VideoDecoder dec(cropped, cuda);
    std::vector<CpuFrame> hard;
    CpuFrame f;
    while (dec.NextFrame(f)) hard.push_back(f);
    REQUIRE(dec.DecodesOnNvdec());
    REQUIRE(hard.size() == soft.size());
    for (size_t i = 0; i < hard.size(); ++i) {
        INFO("frame " << i);
        // not cut to 184 keeping the top rows (that would move the picture 8 rows down): the surface's 192 rows, as before the task -
        // the shown picture in its rows 8..191, as the software decode's
        REQUIRE(hard[i].desc.width == kW);
        REQUIRE(hard[i].desc.height == kH);
        const size_t w = kW, cw = hard[i].desc.PlaneWidth(1);
        int luma = 0, chroma = 0;
        for (size_t r = 0; r < kH - 8; ++r) luma += std::memcmp(hard[i].Plane(0) + (r + 8) * w, soft[i].Plane(0) + r * w, w) == 0;
        for (size_t r = 0; r < (kH - 8) / 2; ++r) {
            chroma += std::memcmp(hard[i].Plane(1) + (r + 4) * cw, soft[i].Plane(1) + r * cw, cw) == 0;
            chroma += std::memcmp(hard[i].Plane(2) + (r + 4) * cw, soft[i].Plane(2) + r * cw, cw) == 0;
        }
        CHECK(luma == static_cast<int>(kH - 8));
        CHECK(chroma == static_cast<int>(kH - 8));
    }
    // cropped 64 columns on the left (shown 256x192; 64 keeps the software decode's pointers aligned, so it really crops - FFmpeg rounds a
    // smaller left crop down to 0): the NVDEC frame not cut to 256 keeping the left columns - its 320 columns, the picture in 64..319
    const auto croppedLeft = dir / "cropped_left.mkv";
    REQUIRE(WithBitstreamFilter(*plain, croppedLeft, "h264_metadata", {{"crop_left", "64"}}));
    REQUIRE(VideoDecoder(croppedLeft, VideoDecoder::Probe()).Info().width == kW - 64);
    VideoDecoder leftPlain(croppedLeft);
    const std::vector<CpuFrame> softLeft = DecodeAllOf(leftPlain);
    REQUIRE(softLeft.size() == 6);
    REQUIRE(softLeft[0].desc.width == kW - 64);
    REQUIRE(softLeft[0].desc.height == kH);
    VideoDecoder leftDec(croppedLeft, cuda);
    std::vector<CpuFrame> hardLeft;
    while (leftDec.NextFrame(f)) hardLeft.push_back(f);
    REQUIRE(leftDec.DecodesOnNvdec());
    REQUIRE(hardLeft.size() == softLeft.size());
    for (size_t i = 0; i < hardLeft.size(); ++i) {
        INFO("cropped left, frame " << i);
        REQUIRE(hardLeft[i].desc.width == kW);
        REQUIRE(hardLeft[i].desc.height == kH);
        const size_t hw = kW, sw = kW - 64, hcw = hardLeft[i].desc.PlaneWidth(1), scw = softLeft[i].desc.PlaneWidth(1);
        REQUIRE(scw == 128);
        int luma = 0, chroma = 0;
        for (size_t r = 0; r < kH; ++r) luma += std::memcmp(hardLeft[i].Plane(0) + r * hw + 64, softLeft[i].Plane(0) + r * sw, sw) == 0;
        for (size_t r = 0; r < kH / 2; ++r) {
            chroma += std::memcmp(hardLeft[i].Plane(1) + r * hcw + 32, softLeft[i].Plane(1) + r * scw, scw) == 0;
            chroma += std::memcmp(hardLeft[i].Plane(2) + r * hcw + 32, softLeft[i].Plane(2) + r * scw, scw) == 0;
        }
        CHECK(luma == static_cast<int>(kH));
        CHECK(chroma == static_cast<int>(kH));
    }
}

TEST_CASE("A refusal names the frame's place in the stream: after seeks back and forth, the index of the frame refused - not the frames this decoder decoded",
          "[integration][gpu][av1][regression]") {
    const auto dir = Dir("refusal_place");
    REQUIRE_AV1_CLIP(clip, dir, false, 24);  // a keyframe every 8 frames
    std::optional<FakeNvdecEnv> fake;
    VideoDecoder dec(clip);
    CpuFrame f;
    // a walk back and forth: frames decoded here and there - more than any index the refusal below may name
    for (const int64_t target : {10, 2, 13, 5}) {
        INFO("seek to " << target);
        REQUIRE(dec.SeekToFrame(target));
        REQUIRE(dec.NextFrame(f));
        REQUIRE(f.index == target);
    }
    // the driver refusing the stream's frames from now on (later-frame: at the next packet, past the 3 frames it lets through): a seek
    // back to the keyframe before frame 17 - the keyframe 16 is refused
    fake.emplace("later-frame");
    REQUIRE(dec.SeekToKeyframeBefore(17));
    const auto refused = RefusalOfNextFrame(dec);
    fake.reset();
    REQUIRE(refused);
    CHECK(refused->support.block == DecodeBlock::NvdecFailed);
    CHECK(refused->support.why == "NVDEC refused this AV1 stream and this build has no software AV1 decoder (at frame 16)");
    CHECK_THAT(refused->support.why, !ContainsSubstring("after"));
    CHECK_THAT(refused->support.why, !ContainsSubstring("first frame"));
    // a fresh decoder, three frames from the start, then that seek: the same place (not «at frame 3», its count)
    {
        FakeNvdecEnv lateFake("later-frame");
        VideoDecoder fresh(clip);
        for (int i = 0; i < 3; ++i) REQUIRE(fresh.NextFrame(f));  // the 3 frames the stand-in lets through
        REQUIRE(fresh.SeekToKeyframeBefore(17));
        const auto again = RefusalOfNextFrame(fresh);
        REQUIRE(again);
        CHECK(again->support.why == "NVDEC refused this AV1 stream and this build has no software AV1 decoder (at frame 16)");
    }
}

TEST_CASE("NVDEC gives an HEVC 4:4:4 frame of an odd height (cropped from 192) right: bit for bit as the software decode - it stays on NVDEC",
          "[integration][gpu][nvdec][regression]") {
    const auto dir = Dir("hevc444_odd_height");
    if (!HaveNvidia() || !VideoEncoder::EncoderAvailable("hevc_nvenc")) SKIP("no NVIDIA GPU with hevc_nvenc");
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = 6;
    spec.codec = "hevc_nvenc";
    spec.pixFmt = "yuv444p";
    std::string why;
    const auto plain = WriteClipIfSupported(dir / "plain.mkv", spec, &why);
    if (!plain) SKIP("hevc_nvenc does not support 4:4:4 here: " << why);
    // its conformance window cut by one row at the bottom (SubHeightC 1 in 4:4:4): shown 320x191, coded 320x192
    const auto odd = dir / "odd.mkv";
    if (!WithBitstreamFilter(*plain, odd, "hevc_metadata", {{"crop_bottom", "1"}})) SKIP("no hevc_metadata bitstream filter in this build");
    const VideoStreamInfo info = VideoDecoder(odd, VideoDecoder::Probe()).Info();
    REQUIRE(info.height == kH - 1);
    REQUIRE(info.pixelFormat == "yuv444p");
    VideoDecoder plainDec(odd);
    const std::vector<CpuFrame> soft = DecodeAllOf(plainDec);
    REQUIRE(soft.size() == 6);
    VideoDecoder::Options cuda;
    cuda.hwaccel = HwAccel::Cuda;
    VideoDecoder dec(odd, cuda);
    CHECK(dec.UsingHwAccel());
    std::vector<CpuFrame> hard;
    std::string said;
    {
        FfmpegLogCapture ffmpeg;
        CpuFrame f;
        while (dec.NextFrame(f)) hard.push_back(f);
        said = ffmpeg.Text();
    }
    // an NVDEC without HEVC 4:4:4 (the driver's answer, nvdec.c): FFmpeg decodes in software - not this case. Nothing else skips: a fall
    // back to software for any other reason (the odd height judged against it) fails
    if (!dec.DecodesOnNvdec() && said.find("Hardware is lacking required capabilities") != std::string::npos)
        SKIP("this GPU's NVDEC does not decode HEVC 4:4:4: " << said);
    CHECK(dec.DecodesOnNvdec());
    REQUIRE(hard.size() == soft.size());
    for (size_t i = 0; i < hard.size(); ++i) {
        INFO("frame " << i);
        CHECK(hard[i].desc.width == kW);
        CHECK(hard[i].desc.height == kH - 1);
        CHECK(hard[i].desc == soft[i].desc);
        for (size_t p = 0; p < 3; ++p) {
            INFO("plane " << p);
            const size_t w = soft[i].desc.PlaneWidth(p), h = soft[i].desc.PlaneHeight(p);
            int rows = 0;
            for (size_t r = 0; r < h; ++r) rows += std::memcmp(hard[i].Plane(p) + r * w, soft[i].Plane(p) + r * w, w) == 0;
            CHECK(rows == static_cast<int>(h));
        }
        CHECK((hard[i].data == soft[i].data));
    }
    const auto r = RunAll({"info", odd.string()}, dir / "info.log");
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK_THAT(r.out, ContainsSubstring("decode:      software (hevc); NVDEC with --hwaccel cuda where FFmpeg offers it for this layout and the GPU decodes it"));
}

TEST_CASE("NVDEC gives the delayed frames from before an H.264 resolution change at their own size - none cut to the new geometry",
          "[integration][gpu][nvdec][regression]") {
    const auto dir = Dir("h264_resolution_change");
    if (!HaveNvidia() || !VideoEncoder::EncoderAvailable("h264_nvenc")) SKIP("no NVIDIA GPU with h264_nvenc");
    // two raw (Annex B) H.264 streams with B-frames one after the other - 320x240, then 256x144, each with its own SPS and IDR: the
    // decoder's output delay gives the last frames of the first part after codec_ took the second's size
    auto part = [&](const char* name, uint32_t w, uint32_t h) {
        ClipSpec spec;
        spec.width = w;
        spec.height = h;
        spec.frames = 24;
        spec.gop = 12;
        spec.bframes = 2;
        spec.codec = "h264_nvenc";
        spec.format = "h264";
        std::string why;
        const auto clip = WriteClipIfSupported(dir / name, spec, &why);
        if (!clip) SKIP("h264_nvenc does not write the clip here: " << why);
        return *clip;
    };
    const auto a = part("a.h264", 320, 240);
    const auto b = part("b.h264", 256, 144);
    const auto parts = dir / "parts.h264";
    {
        std::ofstream out(parts, std::ios::binary);
        for (const auto& p : {a, b}) {
            std::ifstream in(p, std::ios::binary);
            out << in.rdbuf();
        }
    }
    VideoDecoder plain(parts);
    const std::vector<CpuFrame> soft = DecodeAllOf(plain);
    VideoDecoder::Options cuda;
    cuda.hwaccel = HwAccel::Cuda;
    VideoDecoder dec(parts, cuda);
    const std::vector<CpuFrame> hard = DecodeAllOf(dec);
    CHECK(dec.DecodesOnNvdec());
    REQUIRE(soft.size() == 48);
    REQUIRE(hard.size() == soft.size());
    for (size_t n = 0; n < hard.size(); ++n) {
        INFO("frame " << n);
        const uint32_t w = n < 24 ? 320 : 256, h = n < 24 ? 240 : 144;
        CHECK(soft[n].desc.width == w);
        CHECK(soft[n].desc.height == h);
        CHECK(hard[n].desc.width == w);  // the first part's frames at their own size under NVDEC too, none cut to 256x144
        CHECK(hard[n].desc.height == h);
        if (hard[n].desc.width != soft[n].desc.width || hard[n].desc.height != soft[n].desc.height) continue;
        int rows = 0;  // the luma as the software decode's (H.264 decodes bit-exactly)
        for (size_t r = 0; r < h; ++r) rows += std::memcmp(hard[n].Plane(0) + r * w, soft[n].Plane(0) + r * w, w) == 0;
        CHECK(rows == static_cast<int>(h));
    }
}

TEST_CASE("FrameStore: NVDEC not starting for a video is said once per file at warn - not once per place a load met it (a seek elsewhere)",
          "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("start_said_once");
    REQUIRE_AV1_CLIP(clip, dir, false, 64);  // a keyframe every 8: frame 56 is beyond FrameStore's walk from frame 0
    const auto copy = dir / ("copy" + clip.extension().string());  // another file, the same source name
    std::filesystem::copy_file(clip, copy, std::filesystem::copy_options::overwrite_existing);
    std::ostringstream text;
    {
        DecoderLog capture(text);  // before the store: its loaders log while it lives
        {
            FakeNvdecEnv fake("start-fails");  // before the store: its loaders read it while the store lives
            Av1Store s(0, 1);
            s.store->SetSources(Project::Create(clip, dir / "none").Sources());
            CHECK(s.Load(0) == FrameStore::Status::Missing);   // «on the first frame»
            CHECK(s.Load(56) == FrameStore::Status::Missing);  // a seek: «at frame 56» - other words, the same file
            REQUIRE(s.store->StartFailure("source"));
            s.Settle(56);
            // the copy under the same name: its own file - said at warn; the first file again: said before - quietly
            s.store->SetSources(Project::Create(copy, dir / "none_copy").Sources());
            CHECK(s.Load(0) == FrameStore::Status::Missing);
            s.Settle(0);
            s.store->SetSources(Project::Create(clip, dir / "none").Sources());
            CHECK(s.Load(0) == FrameStore::Status::Missing);
            s.Settle(0);
        }
    }
    const std::string log = text.str();
    INFO(log);
    size_t warned = 0, debugged = 0;
    for (size_t at = log.find("'source': NVDEC did not start now"); at != std::string::npos; at = log.find("'source': NVDEC did not start now", at + 1)) {
        const size_t lineStart = log.rfind('\n', at) == std::string::npos ? 0 : log.rfind('\n', at) + 1;
        const std::string level = log.substr(lineStart, at - lineStart);
        if (level.rfind("warning", 0) == 0) ++warned;
        else if (level.rfind("debug", 0) == 0) ++debugged;
    }
    CHECK(warned == 2);    // the file and its copy, once each
    CHECK(debugged >= 2);  // the later ones (a seek, the first file again), said quietly
    CHECK_THAT(log, ContainsSubstring("(at frame 56)"));  // the second load did meet another place
}

TEST_CASE("FrameStore: a start failure is marked with its missing frame - seen at once when the frame is, no poll after", "[integration][viewport][gpu][av1]") {
    const auto dir = Dir("start_mark_at_once");
    REQUIRE_AV1_CLIP(clip, dir, false, 8);
    FakeNvdecEnv fake("start-fails");  // before the store: its loaders read it while the store lives
    Av1Store s(0, 1);
    const auto sources = Project::Create(clip, dir / "none").Sources();
    for (int i = 0; i < 20; ++i) {
        INFO("round " << i);
        s.store->SetSources(sources);  // marks and frames forgotten
        REQUIRE_FALSE(s.store->StartFailure("source"));
        s.store->SetCurrentFrame(0, {"source"});
        s.store->WaitForCurrent();  // returns once the frame is missing
        REQUIRE(s.store->GetStatus(0, "source") == FrameStore::Status::Missing);
        CHECK(s.store->StartFailure("source"));  // already: marked in the same critical section
    }
}
