// dlssvid — command line front-end. Everything the GUI can do must be reachable from here (ТЗ §7).
// Stage 0: `info` and `process --passthrough`.

#include <CLI/CLI.hpp>

#include <cstdio>
#include <string>

#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "pipeline/Pipeline.h"
#include "stages/passthrough/PassthroughStage.h"
#include "util/Error.h"
#include "util/Log.h"

#ifdef DLSSVID_WITH_CUDA
#include "gpu/CudaInterop.h"
#endif

using namespace dlssvid;

namespace {

int CmdInfo(const std::string& input, bool warp) {
    VideoDecoder dec(input);
    const auto& i = dec.Info();
    std::printf("file:        %s\n", input.c_str());
    std::printf("video:       %s %ux%u %s\n", i.codecName.c_str(), i.width, i.height, i.pixelFormat.c_str());
    std::printf("frame rate:  %d/%d (%.3f fps)\n", i.frameRate.num, i.frameRate.den, i.frameRate.ToDouble());
    std::printf("time base:   %d/%d\n", i.timeBase.num, i.timeBase.den);
    std::printf("frames:      %lld%s\n", static_cast<long long>(i.frameCount), i.frameCount ? "" : " (unknown)");
    std::printf("duration:    %.3f s\n", i.durationUs / 1e6);
    std::printf("audio:       %s\n", i.hasAudio ? i.audioCodecName.c_str() : "none");

    D3D12Device device({warp, false});
    std::printf("d3d12:       %s%s, VRAM %llu MiB\n", device.AdapterName().c_str(), device.IsWarp() ? " [WARP]" : "",
                static_cast<unsigned long long>(device.DedicatedVideoMemory() / (1024 * 1024)));
#ifdef DLSSVID_WITH_CUDA
    std::string reason;
    if (!device.IsWarp() && CudaInterop::Available(&reason)) {
        try {
            CudaInterop cuda(device);
            std::printf("cuda:        device %d (%s)\n", cuda.CudaDeviceId(), cuda.CudaDeviceName().c_str());
        } catch (const std::exception& e) {
            std::printf("cuda:        unavailable (%s)\n", e.what());
        }
    } else {
        std::printf("cuda:        unavailable (%s)\n", device.IsWarp() ? "WARP adapter" : reason.c_str());
    }
#else
    std::printf("cuda:        not built\n");
#endif
    for (const char* name : {"h264_nvenc", "hevc_nvenc", "av1_nvenc", "ffv1"}) {
        std::printf("encoder:     %-10s %s\n", name, VideoEncoder::EncoderAvailable(name) ? "yes" : "no");
    }
    return 0;
}

struct ProcessArgs {
    std::string input;
    std::string output;
    bool passthrough = false;
    std::string codec = "h264_nvenc";
    std::vector<std::string> codecOptions;  // key=value
    std::string hwaccel = "none";
    int64_t frames = -1;
    bool warp = false;
    bool noGpuRoundTrip = false;
};

int CmdProcess(const ProcessArgs& a) {
    if (!a.passthrough) {
        Log()->error("only --passthrough is implemented in stage 0 (SR/NR/FG stages arrive in stages 5-7)");
        return 2;
    }
    VideoDecoder::Options dopt;
    if (a.hwaccel == "cuda") dopt.hwaccel = HwAccel::Cuda;
    else if (a.hwaccel != "none") Throw("--hwaccel must be none|cuda");

    VideoDecoder decoder(a.input, dopt);
    const auto& info = decoder.Info();

    VideoEncoder::Options eopt;
    eopt.codec = a.codec;
    eopt.frameRate = info.frameRate;
    eopt.timeBase = info.timeBase;
    for (const auto& kv : a.codecOptions) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) Throw("--codec-opt expects key=value, got: " + kv);
        eopt.codecOptions[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    VideoEncoder encoder(a.output, FrameDesc{info.width, info.height, PixelFormat::Yuv420p}, eopt);

    D3D12Device device({a.warp, false});
    Pipeline pipeline(device, 4);
    StageConfig cfg;
    cfg.name = "passthrough";
    cfg.params["gpu_roundtrip"] = !a.noGpuRoundTrip;
    pipeline.AddStage(std::make_unique<PassthroughStage>(), cfg);
    pipeline.Init();

    const int64_t total = a.frames > 0 ? a.frames : info.frameCount;
    const RunStats stats = RunPipeline(
        pipeline, decoder, encoder,
        [total](int64_t n) {
            if (n % 50 == 0 || n == total) {
                if (total > 0) std::fprintf(stderr, "\r%lld/%lld frames", static_cast<long long>(n), static_cast<long long>(total));
                else std::fprintf(stderr, "\r%lld frames", static_cast<long long>(n));
            }
        },
        a.frames);
    std::fprintf(stderr, "\n");
    pipeline.Shutdown();

    Log()->info("done: {} frames in {:.2f} s ({:.1f} fps)", stats.framesOut, stats.seconds,
                stats.seconds > 0 ? stats.framesOut / stats.seconds : 0.0);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"dlssvid — offline DLSS video pipeline (stage 0: passthrough)"};
    app.set_version_flag("--version", DLSSVID_VERSION);
    app.require_subcommand(1);
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "debug logging");

    std::string infoInput;
    bool infoWarp = false;
    auto* info = app.add_subcommand("info", "print stream and device information");
    info->add_option("input", infoInput, "input video file")->required()->check(CLI::ExistingFile);
    info->add_flag("--warp", infoWarp, "use the WARP software adapter");

    ProcessArgs pa;
    auto* process = app.add_subcommand("process", "run the pipeline on a video file");
    process->add_option("-i,--input", pa.input, "input video file")->required()->check(CLI::ExistingFile);
    process->add_option("-o,--output", pa.output, "output video file (container by extension)")->required();
    process->add_flag("--passthrough", pa.passthrough, "decode -> GPU -> encode without processing");
    process->add_option("--codec", pa.codec, "video encoder: h264_nvenc | hevc_nvenc | av1_nvenc | ffv1")
        ->default_val("h264_nvenc");
    process->add_option("--codec-opt", pa.codecOptions, "encoder private option key=value (repeatable)");
    process->add_option("--hwaccel", pa.hwaccel, "decoder hardware acceleration: none | cuda")->default_val("none");
    process->add_option("--frames", pa.frames, "process at most N frames")->default_val(-1);
    process->add_flag("--warp", pa.warp, "use the WARP software adapter");
    process->add_flag("--no-gpu-roundtrip", pa.noGpuRoundTrip, "skip the GPU upload/readback in passthrough");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    if (verbose) SetLogLevel(spdlog::level::debug);

    try {
        if (info->parsed()) return CmdInfo(infoInput, infoWarp);
        if (process->parsed()) return CmdProcess(pa);
    } catch (const std::exception& e) {
        Log()->error("{}", e.what());
        return 1;
    }
    return 0;
}
