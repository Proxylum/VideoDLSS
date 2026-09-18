// Acceptance for stage 0 (ТЗ §8, §9): the passthrough path yields frames that are bit-identical
// to what FFmpeg decodes, frame count and timing are preserved, audio is stream-copied.

#include <catch2/catch_test_macros.hpp>

#include <memory>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "pipeline/Pipeline.h"
#include "stages/passthrough/PassthroughStage.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

RunStats Passthrough(const std::filesystem::path& in, const std::filesystem::path& out, bool warp, HwAccel hw = HwAccel::None) {
    VideoDecoder decoder(in, VideoDecoder::Options{hw});
    const auto& info = decoder.Info();
    VideoEncoder::Options eopt;
    eopt.codec = "ffv1";
    eopt.frameRate = info.frameRate;
    eopt.timeBase = info.timeBase;
    VideoEncoder encoder(out, FrameDesc{info.width, info.height, PixelFormat::Yuv420p}, eopt);
    D3D12Device device({warp, false});
    Pipeline pipeline(device, 4);
    pipeline.AddStage(std::make_unique<PassthroughStage>());
    pipeline.Init();
    const RunStats stats = RunPipeline(pipeline, decoder, encoder);
    pipeline.Shutdown();
    return stats;
}

}  // namespace

TEST_CASE("Synthetic clip decodes back to the frames that were written", "[integration][clip]") {
    ClipSpec spec;
    spec.frames = 12;
    const auto clip = WriteClip(TempDir() / "synthetic.mkv", spec);
    const auto frames = DecodeAll(clip);
    REQUIRE(frames.size() == 12);
    const Rational tb = VideoDecoder(clip).Info().timeBase;  // Matroska stores pts in 1/1000 s
    for (int i = 0; i < 12; ++i) {
        const CpuFrame ref = SyntheticFrame(i, spec.width, spec.height);
        CHECK(frames[i].index == i);
        const double seconds = static_cast<double>(frames[i].pts) * tb.num / tb.den;
        CHECK(std::abs(seconds - i / 24.0) <= 0.5 * tb.num / static_cast<double>(tb.den) + 1e-9);
        REQUIRE(frames[i].data == ref.data);
    }
}

TEST_CASE("Passthrough output is bit-identical to FFmpeg decode of the source", "[integration][passthrough]") {
    ClipSpec spec;
    spec.frames = 24;
    const auto in = WriteClip(TempDir() / "pt_in.mkv", spec);
    const auto out = TempDir() / "pt_out.mkv";

    const RunStats stats = Passthrough(in, out, /*warp*/ false);
    CHECK(stats.framesIn == 24);
    CHECK(stats.framesOut == 24);

    const auto a = DecodeAll(in);
    const auto b = DecodeAll(out);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].pts == b[i].pts);
        REQUIRE(a[i].data == b[i].data);
    }

    VideoDecoder od(out);
    CHECK(od.Info().frameRate.num == 24);
    CHECK(od.Info().frameRate.den == 1);
    CHECK(od.Info().width == spec.width);
    CHECK(od.Info().height == spec.height);
}

TEST_CASE("Passthrough copies the audio stream untouched", "[integration][passthrough][audio]") {
    ClipSpec spec;
    spec.frames = 24;
    spec.audio = true;
    const auto in = WriteClip(TempDir() / "pt_audio_in.mkv", spec);
    const auto out = TempDir() / "pt_audio_out.mkv";

    Passthrough(in, out, /*warp*/ true);

    const StreamSummary si = Summarize(in);
    const StreamSummary so = Summarize(out);
    REQUIRE(si.audioStreams == 1);
    REQUIRE(so.audioStreams == 1);
    CHECK(so.videoStreams == 1);
    CHECK(so.audioCodec == si.audioCodec);
    CHECK(so.audioPackets == si.audioPackets);
    CHECK(so.audioSamples == si.audioSamples);
    CHECK(si.audioSamples > 0);  // ~1 s of 48 kHz audio; exact packet durations are container-specific
}

TEST_CASE("Passthrough works with odd frame sizes", "[integration][passthrough]") {
    ClipSpec spec;
    spec.width = 322;
    spec.height = 182;
    spec.frames = 5;
    const auto in = WriteClip(TempDir() / "pt_odd_in.mkv", spec);
    const auto out = TempDir() / "pt_odd_out.mkv";
    Passthrough(in, out, /*warp*/ true);
    const auto a = DecodeAll(in), b = DecodeAll(out);
    REQUIRE(a.size() == 5);
    REQUIRE(b.size() == 5);
    for (size_t i = 0; i < 5; ++i) REQUIRE(a[i].data == b[i].data);
}

TEST_CASE("NVDEC decode matches software decode bit for bit", "[integration][nvdec]") {
    // Needs an NVIDIA GPU and an NVDEC-capable codec. h264_nvenc is only available on NVIDIA
    // hardware, so use it to produce the H.264 clip and skip elsewhere.
    if (!VideoEncoder::EncoderAvailable("h264_nvenc")) SKIP("h264_nvenc encoder not available");
    ClipSpec spec;
    spec.frames = 30;
    const auto src = WriteClip(TempDir() / "nvdec_src.mkv", spec);

    // Transcode to H.264 with NVENC through the same pipeline (lossy, but deterministic input for the decoder test).
    const auto h264 = TempDir() / "nvdec_src.mp4";
    {
        VideoDecoder decoder(src);
        VideoEncoder::Options eopt;
        eopt.codec = "h264_nvenc";
        eopt.frameRate = decoder.Info().frameRate;
        eopt.timeBase = decoder.Info().timeBase;
        eopt.codecOptions["preset"] = "p1";
        VideoEncoder encoder(h264, FrameDesc{spec.width, spec.height, PixelFormat::Yuv420p}, eopt);
        D3D12Device device;
        Pipeline pipeline(device, 2);
        StageConfig cfg;
        cfg.params["gpu_roundtrip"] = false;
        pipeline.AddStage(std::make_unique<PassthroughStage>(), cfg);
        pipeline.Init();
        const RunStats st = RunPipeline(pipeline, decoder, encoder);
        REQUIRE(st.framesOut == 30);
        pipeline.Shutdown();
    }

    VideoDecoder sw(h264);
    VideoDecoder hw(h264, VideoDecoder::Options{HwAccel::Cuda});
    if (!hw.UsingHwAccel()) SKIP("CUDA hwaccel not available in this FFmpeg build / driver");
    CpuFrame a, b;
    int n = 0;
    while (sw.NextFrame(a)) {
        REQUIRE(hw.NextFrame(b));
        REQUIRE(a.desc == b.desc);
        REQUIRE(a.data == b.data);
        ++n;
    }
    CHECK_FALSE(hw.NextFrame(b));
    CHECK(n == 30);
}
