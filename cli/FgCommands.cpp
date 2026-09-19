#include "FgCommands.h"

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <fstream>

#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "stages/fg/FgStage.h"
#include "stages/fg/IFrameGenerator.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

namespace dlssvid::cli {

namespace {

void Progress(int64_t done, int64_t total) {
    if (done % 10 == 0 || done == total) {
        if (total > 0) std::fprintf(stderr, "\r%lld/%lld frames", static_cast<long long>(done), static_cast<long long>(total));
        else std::fprintf(stderr, "\r%lld frames", static_cast<long long>(done));
    }
}

void ParseSize(const std::string& s, uint32_t& w, uint32_t& h) {
    const size_t x = s.find('x');
    if (x == std::string::npos) Throw("--check-size expects WxH (got '" + s + "')");
    w = static_cast<uint32_t>(std::stoul(s.substr(0, x)));
    h = static_cast<uint32_t>(std::stoul(s.substr(x + 1)));
}

void PrintDiagnostics(const FgDiagnostics& d, const std::string& error) {
    std::printf("GPU:          %s\n", d.gpu.c_str());
    std::printf("architecture: %s\n", d.architecture.empty() ? "?" : d.architecture.c_str());
    std::printf("driver:       %s\n", d.driver.ToString().c_str());
    if (!d.dll.empty()) std::printf("dll:          %s (%llu bytes, sha256 %s)\n", d.dll.string().c_str(), static_cast<unsigned long long>(d.dllSize), d.dllSha256.c_str());
    std::printf("available:    %s (MultiFrameCountMax %d)\n", d.available ? "yes" : "no", d.multiFrameMax);
    if (!d.createResult.empty()) std::printf("create:       %s [%s]\n", d.createResult.c_str(), d.backbufferFormat.empty() ? d.model.c_str() : d.backbufferFormat.c_str());
    if (d.ok) std::printf("status:       ok\n");
    else std::printf("status:       FAILED: %s\n", !d.hint.empty() ? d.hint.c_str() : error.c_str());
}

int CmdFgCheck(const FgCommands::FgArgs& a) {
    D3D12Device device({a.warp, false});
    std::unique_ptr<IFrameGenerator> gen = CreateFrameGenerator(a.backend);
    FgConfig cfg;
    cfg.backend = a.backend;
    ParseSize(a.checkSize, cfg.width, cfg.height);
    cfg.guideWidth = cfg.width;
    cfg.guideHeight = cfg.height;
    cfg.multiplier = a.multiplier;
    cfg.backbufferFormat = a.backbufferFormat;
    cfg.dllDir = a.dllDir;
    cfg.model = a.model;
    cfg.fp16 = !a.fp32;
    cfg.modelsDir = a.modelsDir;
    cfg.requireDriver = !a.skipDriverCheck;
    std::string error;
    try {
        gen->Init(device, cfg);
    } catch (const std::exception& e) {
        error = e.what();
    }
    const FgDiagnostics d = gen->Diagnostics();
    gen->Shutdown();
    PrintDiagnostics(d, error);
    if (!a.json.empty()) {
        nlohmann::json j = d.ToJson();
        j["error"] = error;
        std::ofstream(a.json) << j.dump(2) << "\n";
    }
    return d.ok ? 0 : 1;
}

int CmdFg(const FgCommands::FgArgs& a) {
    if (a.crashAfter == 0) {
        // isolation test: die before touching any input, the way a runtime fault would
        Log()->error("fg: --crash-after 0: terminating the process (isolation test)");
        TerminateProcess(GetCurrentProcess(), 0xC0000005u);
    }
    if (a.check) return CmdFgCheck(a);
    if (a.input.empty()) Throw("fg: -i/--input is required (or --check)");
    if (a.output.empty()) Throw("fg: -o/--output is required");
    VideoDecoder decoder(a.input);
    const auto& info = decoder.Info();
    FgStageOptions o;
    o.backend = a.backend;
    o.multiplier = a.multiplier;
    auto autoDir = [&](const std::string& given, std::initializer_list<const char*> names) -> std::filesystem::path {
        if (!given.empty()) return given;
        if (a.noAuto) return {};
        for (const char* name : names) {
            const auto d = std::filesystem::path(a.output) / name;
            if (std::filesystem::exists(d / Manifest::kFileName)) {
                Log()->info("fg: using {}", d.string());
                return d;
            }
        }
        return {};
    };
    o.colorDir = autoDir(a.colorDir, {"color_nr", "color_sr"});
    o.depthDir = autoDir(a.depthDir, {"depth_dlss"});
    o.mvDir = autoDir(a.mvDir, {"mv_dlss"});
    o.model = a.model;
    o.fp16 = !a.fp32;
    o.modelsDir = a.modelsDir;
    o.backbufferFormat = a.backbufferFormat;
    o.dllDir = a.dllDir;
    o.requireDriver = !a.skipDriverCheck;
    o.outputDir = a.output;
    const auto fmt = ParseFileFormat(a.format);
    if (!fmt || (*fmt != FileFormat::Exr && *fmt != FileFormat::Png)) Throw("--format must be exr|png16");
    o.format = *fmt;
    o.videoOut = a.video;
    o.videoCodec = a.codec;
    o.crashAfter = a.crashAfter;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
    const int64_t total = a.frames > 0 ? a.frames : info.frameCount;
    o.onFrame = [](int64_t frame, double ms) {
        if (frame % 10 == 0) std::fprintf(stderr, "  frame %lld %.1f ms\n", static_cast<long long>(frame), ms);
    };
    D3D12Device device({a.warp, false});
    const auto t0 = std::chrono::steady_clock::now();
    const FgRunResult r = RunFg(decoder, device, o, a.frames, [total](int64_t n) { Progress(n, total); });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "\n");
    if (r.stats.disabled) {
        std::printf("fg:          disabled — %s\n", r.stats.disabledReason.c_str());
        return 1;
    }
    Log()->info("fg done: {} frames -> {} in {:.1f} s ({:.1f} ms/frame wall, {:.1f} ms/frame backend), {}", r.stats.frames, FgFrameCount(r.stats.frames, r.stats.multiplier), sec,
                r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0, r.stats.MeanMs(), r.backend.dump());
    std::printf("color_fg:    %s\n", r.outDir.string().c_str());
    std::printf("backend:     %s\n", r.stats.backend.c_str());
    std::printf("colour from: %s\n", r.stats.colorSource.c_str());
    std::printf("size:        %ux%u, x%d\n", r.stats.width, r.stats.height, r.stats.multiplier);
    std::printf("guides:      depth %s, mv %s\n", r.stats.depthUsed ? "yes" : "no", r.stats.mvUsed ? "yes" : "no");
    std::printf("frames:      %lld in -> %lld out (%lld generated)\nms/frame:    %.1f (backend %.1f)\n", static_cast<long long>(r.stats.frames),
                static_cast<long long>(FgFrameCount(r.stats.frames, r.stats.multiplier)), static_cast<long long>(r.stats.generated), r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0,
                r.stats.MeanMs());
    if (!a.video.empty()) std::printf("video:       %s\n", a.video.c_str());
    if (!a.json.empty()) {
        nlohmann::json j = {{"color_fg", r.outDir.string()},
                            {"backend", r.backend},
                            {"diagnostics", r.diagnostics},
                            {"stats",
                             {{"frames", r.stats.frames},
                              {"generated", r.stats.generated},
                              {"frames_out", FgFrameCount(r.stats.frames, r.stats.multiplier)},
                              {"multiplier", r.stats.multiplier},
                              {"ms_per_frame_backend", r.stats.MeanMs()},
                              {"ms_per_frame_wall", r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0},
                              {"size", {r.stats.width, r.stats.height}},
                              {"depth_used", r.stats.depthUsed},
                              {"mv_used", r.stats.mvUsed},
                              {"color_source", r.stats.colorSource}}}};
        std::ofstream(a.json) << j.dump(2) << "\n";
    }
    return 0;
}

}  // namespace

void FgCommands::Register(CLI::App& app) {
    fg_ = app.add_subcommand("fg", "DLSS Frame Generation (NGX, no swapchain) | RIFE | blend -> color_fg with x2..x4 frames; --check prints the diagnostics only");
    fg_->add_option("-i,--input", fa_.input, "input video file")->check(CLI::ExistingFile);
    fg_->add_option("-o,--output", fa_.output, "pass root folder (writes color_fg/; color_nr / color_sr, depth_dlss, mv_dlss found here are used)");
    fg_->add_option("--color-dir,--color_dir", fa_.colorDir, "colour input pass folder (default: <output>/color_nr, else color_sr, else the video)");
    fg_->add_option("--depth-dir,--depth_dir", fa_.depthDir, "depth_dlss pass folder (guide, dlssg)");
    fg_->add_option("--mv-dir,--mv_dir", fa_.mvDir, "mv_dlss pass folder (guide, dlssg)");
    fg_->add_flag("--no-auto,--no_auto", fa_.noAuto, "do not pick pass folders under --output automatically");
    fg_->add_option("--backend", fa_.backend, "dlssg (DLSS Frame Generation, RTX 40+) | rife (TensorRT baseline) | blend (naive)")->default_val("dlssg");
    fg_->add_option("--multiplier", fa_.multiplier, "output frames per input frame: 2 | 3 | 4 (dlssg > 2 needs Multi Frame Generation)")->default_val(2);
    fg_->add_option("--model", fa_.model, "rife: registry model rife49 | rife48 | rife47")->default_val("rife49");
    fg_->add_flag("--fp32", fa_.fp32, "rife: fp32 engine");
    fg_->add_option("--models-dir,--models_dir", fa_.modelsDir, "folder with registry.json (default: auto)");
    fg_->add_option("--backbuffer-format,--backbuffer_format", fa_.backbufferFormat, "dlssg: rgba16f | rgba8 (rgba8 is tried automatically when rgba16f is refused)")->default_val("rgba16f");
    fg_->add_option("--format", fa_.format, "exr | png16")->default_val("exr");
    fg_->add_option("--video", fa_.video, "also encode the result into this video file at fps x multiplier (no audio)");
    fg_->add_option("--codec", fa_.codec, "encoder for --video: h264_nvenc | hevc_nvenc | av1_nvenc | ffv1")->default_val("h264_nvenc");
    fg_->add_option("--frames", fa_.frames, "process at most N input frames")->default_val(-1);
    fg_->add_flag("--warp", fa_.warp, "use the WARP software adapter (blend)");
    fg_->add_option("--dll-dir,--dll_dir", fa_.dllDir, "folder with nvngx_dlssg.dll (default: bin/nvidia next to the executable)");
    fg_->add_flag("--skip-driver-check,--skip_driver_check", fa_.skipDriverCheck, "ignore the driver version check");
    fg_->add_flag("--check", fa_.check, "no processing: initialise the backend, print GPU / driver / DLL / feature diagnostics and exit 0/1");
    fg_->add_option("--check-size,--check_size", fa_.checkSize, "resolution for --check")->default_val("1280x720");
    fg_->add_option("--json", fa_.json, "write the run summary / diagnostics to this JSON file");
    fg_->add_option("--crash-after,--crash_after", fa_.crashAfter, "development: terminate the process after N input frames (0 = immediately) — isolation test")->group("");
}

int FgCommands::Dispatch() {
    if (fg_ && fg_->parsed()) return CmdFg(fa_);
    return -1;
}

}  // namespace dlssvid::cli
