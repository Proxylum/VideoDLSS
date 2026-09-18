// NVDEC frames that stay on the GPU: bit-exact with the CPU path, and copied device-to-device
// into a shared D3D12 buffer (no host round trip). Skipped without an NVIDIA GPU.

#include <catch2/catch_test_macros.hpp>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "pipeline/Pipeline.h"
#include "stages/passthrough/PassthroughStage.h"

#ifdef DLSSVID_WITH_CUDA
#include "gpu/CudaInterop.h"
#endif

using namespace dlssvid;
using namespace dlssvid::test;

#ifdef DLSSVID_WITH_CUDA

namespace {
std::filesystem::path H264Clip() {
    ClipSpec spec;
    spec.frames = 12;
    spec.width = 320;
    spec.height = 192;
    const auto src = WriteClip(TempDir() / "gpu_src.mkv", spec);
    const auto h264 = TempDir() / "gpu_src.mp4";
    VideoDecoder decoder(src);
    VideoEncoder::Options eopt;
    eopt.codec = "h264_nvenc";
    eopt.frameRate = decoder.Info().frameRate;
    eopt.timeBase = decoder.Info().timeBase;
    VideoEncoder encoder(h264, FrameDesc{spec.width, spec.height, PixelFormat::Yuv420p}, eopt);
    D3D12Device device;
    Pipeline pipeline(device, 2);
    StageConfig cfg;
    cfg.params["gpu_roundtrip"] = false;
    pipeline.AddStage(std::make_unique<PassthroughStage>(), cfg);
    pipeline.Init();
    RunPipeline(pipeline, decoder, encoder);
    pipeline.Shutdown();
    return h264;
}
}  // namespace

TEST_CASE("GPU frames from NVDEC match the CPU decode and reach D3D12 without a host copy", "[integration][gpu][nvdec]") {
    D3D12Device dev;
    std::string reason;
    if (dev.IsWarp() || !dev.IsNvidia() || !CudaInterop::Available(&reason)) SKIP("no NVIDIA GPU / CUDA: " << reason);
    if (!VideoEncoder::EncoderAvailable("h264_nvenc")) SKIP("h264_nvenc unavailable");
    const auto clip = H264Clip();

    VideoDecoder hw(clip, VideoDecoder::Options{HwAccel::Cuda});
    if (!hw.UsingHwAccel()) SKIP("CUDA hwaccel not available");
    VideoDecoder sw(clip);
    CudaInterop cuda(dev);
    GpuFrameCache cache(dev, 2);
    cache.AttachCuda(&cuda);
    REQUIRE(cache.CudaAttached());

    CpuFrame cpu, ref, gpuOnly;
    GpuFrame gpu;
    int n = 0;
    while (hw.NextFrame(cpu, &gpu, true)) {
        REQUIRE(sw.NextFrame(ref));
        REQUIRE(gpu.Valid());
        CHECK(gpu.width == 320);
        CHECK(gpu.height == 192);
        CHECK(gpu.pitch >= 320);
        CHECK(gpu.index == cpu.index);
        REQUIRE(cpu.data == ref.data);  // CPU path still bit-exact

        // device -> shared D3D12 buffer -> readback == CPU decode
        GpuFrameCache::Slot& slot = cache.Acquire(cpu.index, cpu.desc);
        cache.UploadFromGpuFrame(slot, gpu);
        CHECK(slot.layout == GpuFrameCache::Layout::Nv12);
        CpuFrame back;
        cache.Download(slot, back);
        REQUIRE(back.data == ref.data);
        ++n;
    }
    CHECK(n == 12);

    // cpu=false leaves the CpuFrame empty but still numbers frames
    VideoDecoder hw2(clip, VideoDecoder::Options{HwAccel::Cuda});
    REQUIRE(hw2.NextFrame(gpuOnly, &gpu, false));
    CHECK(gpuOnly.data.empty());
    CHECK(gpuOnly.index == 0);
    CHECK(gpu.Valid());
    CHECK(hw2.NextFrame(gpuOnly, nullptr, false));  // skip a frame without any copy
    CHECK(gpuOnly.index == 1);
}

#else
TEST_CASE("GPU decode path not built", "[integration][gpu][nvdec]") { SUCCEED("DLSSVID_WITH_CUDA is off"); }
#endif
