#include "FgCommands.h"

#include "ScenesArg.h"

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "passes/ScenesPass.h"
#include "pipeline/FgProcess.h"
#include "pipeline/PassFingerprint.h"
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

// One protocol line on stdout in a single write: the runner parses it, and the log (spdlog on the same handle) must not
// split it.
void EmitLine(const std::string& text) {
    const std::string s = text + "\n";
    std::fflush(stdout);
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), s.data(), static_cast<DWORD>(s.size()), &written, nullptr);
}

nlohmann::json Summary(const FgRunResult& r, double sec) {
    return {{"color_fg", r.outDir.string()},
            {"backend", r.backend},
            {"diagnostics", r.diagnostics},
            {"disabled", r.stats.disabled},
            {"disabled_reason", r.stats.disabledReason},
            {"seconds", sec},
            {"stats",
             {{"frames", r.stats.frames},
              {"generated", r.stats.generated},
              {"duplicated", r.stats.duplicated},  // held at a cut of the scene list (PR-13)
              {"scene_cut_count", r.stats.sceneCutCount},
              {"frames_out", FgFrameCount(r.stats.frames, r.stats.multiplier)},
              {"multiplier", r.stats.multiplier},
              {"ms_per_frame_backend", r.stats.MeanMs()},
              {"ms_per_frame_wall", r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0},
              {"size", {r.stats.width, r.stats.height}},
              {"depth_used", r.stats.depthUsed},
              {"mv_used", r.stats.mvUsed},
              {"color_source", r.stats.colorSource}}}};
}

// `fg --as-stage`: the FG stage of `dlssvid process` in its own process (audit fg-2). The options are assembled exactly as
// the runner does in-process (defaults + the stage's parameters through FgStage::ApplyParams), so color_fg is the same.
int CmdFgAsStage(const FgCommands::FgArgs& a) {
    if (a.input.empty() || a.output.empty()) Throw("fg --as-stage: -i and -o are required");
    if (a.stall) {
        Log()->error("fg: --stall: hanging before the input (isolation test)");
        for (;;) Sleep(1000);
    }
    nlohmann::json params = a.paramsJson.empty() ? nlohmann::json::object() : nlohmann::json::parse(a.paramsJson);
    if (!params.is_object()) Throw("fg --as-stage: --params-json must be a JSON object");
    VideoDecoder decoder(a.input);
    FgStageOptions o;
    o.outputDir = a.output;
    o.colorDir = a.colorDir;
    o.depthDir = a.depthDir;
    o.mvDir = a.mvDir;
    o.disableWhenUnavailable = a.disableUnavailable;
    const auto fmt = ParseFileFormat(a.format);  // any pass format, as `process` accepts for the stage
    if (!fmt) Throw("fg --as-stage: unknown --format '" + a.format + "'");
    o.format = *fmt;
    o.crashAfter = a.crashAfter;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = a.sourceHash.rfind("sha256:", 0) == 0 ? a.sourceHash : "sha256:" + Sha256File(a.input);
    // PR-13: the list the runner planned with, exactly that one - another on disk now fails the stage before its first frame
    if (!a.sceneCutsId.empty()) o.sceneCuts = ReadSceneListExpect(a.scenesDir, a.sceneCutsId, o.sourceHash);
    D3D12Device device({a.warp, false});
    const auto t0 = std::chrono::steady_clock::now();
    // the result goes out as soon as color_fg is written - before the backend is released, where a driver fault ends
    // the process most often: the runner then keeps the complete pass (it checks the pass on a non-zero exit code)
    RunFg(decoder, device, o, a.frames, [](int64_t n) { EmitLine(std::string(kFgProgressPrefix) + std::to_string(n)); }, params,
          [&](const FgRunResult& r) {
              const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
              EmitLine(std::string(kFgResultPrefix) + Summary(r, sec).dump(-1, ' ', true, nlohmann::json::error_handler_t::replace));
              if (a.crashOnExit) {  // isolation test: a crash while the backend is released
                  Log()->error("fg: --crash-on-exit: terminating the process after the result (isolation test)");
                  TerminateProcess(GetCurrentProcess(), 0xC0000005u);
              }
          });
    return 0;
}

int CmdFg(const FgCommands::FgArgs& a) {
    if (a.asStage) {
        // A crashed child must exit, not wait on a Windows Error Reporting dialog (the runner's job object also sets
        // this for unhandled exceptions); abort() still prints its message, without the report dialog.
        SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
        _set_abort_behavior(0, _CALL_REPORTFAULT);
    }
    if (a.crashAfter == 0) {
        // isolation test: die before touching any input, the way a runtime fault would
        Log()->error("fg: --crash-after 0: terminating the process (isolation test)");
        TerminateProcess(GetCurrentProcess(), 0xC0000005u);
    }
    if (a.check) return CmdFgCheck(a);
    if (a.asStage) return CmdFgAsStage(a);
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
    if (!fmt) Throw("--format must be exr | png16 | tiff | npz | raw");  // what the schema offers (TASK-0044: exr | png16 only)
    o.format = *fmt;
    o.videoOut = a.video;
    o.videoCodec = a.codec;
    o.crashAfter = a.crashAfter;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
    // PR-13: the clip's scene-cut list, <output>/scenes unless told otherwise - x2..x4 interpolates, never across a cut
    o.sceneCuts = ResolveScenes(a.scenesDir, a.noScenes, a.noAuto, a.output, o.sourceHash, StageReadsSceneCuts("fg", {{"multiplier", a.multiplier}}), "fg",
                                a.frames > 0 ? a.frames : 0);
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
    if (r.stats.sceneCutCount > 0)
        std::printf("scene cuts:  %lld (%lld frames held instead of interpolated)\n", static_cast<long long>(r.stats.sceneCutCount), static_cast<long long>(r.stats.duplicated));
    if (!a.video.empty()) std::printf("video:       %s\n", a.video.c_str());
    if (!a.json.empty()) std::ofstream(a.json) << Summary(r, sec).dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
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
    fg_->add_option("--scenes-dir,--scenes_dir", fa_.scenesDir,
                    "the clip's scene-cut list (the pass of `dlssvid scenes`; default: <output>/scenes when present): no interpolation across "
                    "its cuts - the last frame of a shot is held");
    fg_->add_flag("--no-scenes,--no_scenes", fa_.noScenes, "no scene-cut list: interpolation across cuts too");
    fg_->add_option("--scene-cuts-id,--scene_cuts_id", fa_.sceneCutsId, "the id of the scene list the runner planned with (--as-stage)")->group("");
    fg_->add_option("--backend", fa_.backend, "dlssg (DLSS Frame Generation, RTX 40+) | rife (TensorRT baseline) | blend (naive)")->default_val("dlssg");
    fg_->add_option("--multiplier", fa_.multiplier, "output frames per input frame: 2 | 3 | 4 (dlssg > 2 needs Multi Frame Generation)")->default_val(2);
    fg_->add_option("--model", fa_.model, "rife: registry model rife49 | rife48 | rife47")->default_val("rife49");
    fg_->add_flag("--fp32", fa_.fp32, "rife: fp32 engine");
    fg_->add_option("--models-dir,--models_dir", fa_.modelsDir, "folder with registry.json (default: auto)");
    fg_->add_option("--backbuffer-format,--backbuffer_format", fa_.backbufferFormat, "dlssg: rgba16f | rgba8 (rgba8 is tried automatically when rgba16f is refused)")->default_val("rgba16f");
    fg_->add_option("--format", fa_.format, "exr | png16 | tiff | npz | raw")->default_val("exr");
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
    // the stage of `dlssvid process` in its own process (pipeline/FgProcess.h) - hidden
    fg_->add_flag("--as-stage,--as_stage", fa_.asStage, "run as the fg stage of `dlssvid process`: @progress / @result lines on stdout")->group("");
    fg_->add_option("--params-json,--params_json", fa_.paramsJson, "the stage's parameters as a JSON object (--as-stage)")->group("");
    fg_->add_option("--source-hash,--source_hash", fa_.sourceHash, "sha256:<hex> of the input, instead of hashing it again (--as-stage)")->group("");
    fg_->add_flag("--disable-unavailable,--disable_unavailable", fa_.disableUnavailable, "report a disabled stage when the backend is unavailable (--as-stage)")
        ->group("");
    fg_->add_flag("--crash-on-exit,--crash_on_exit", fa_.crashOnExit,
                  "development: terminate after printing the result, before the backend is released (--as-stage) — isolation test")
        ->group("");
    fg_->add_flag("--stall", fa_.stall, "development: hang before reading the input (--as-stage) — isolation test")->group("");
}

int FgCommands::Dispatch() {
    if (fg_ && fg_->parsed()) return CmdFg(fa_);
    return -1;
}

}  // namespace dlssvid::cli
