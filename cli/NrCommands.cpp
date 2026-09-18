#include "NrCommands.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "stages/nr/INrBackend.h"
#include "stages/nr/NrPatch.h"
#include "stages/nr/NrStage.h"
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

void PrintDiagnostics(const NrDiagnostics& d, const std::string& error) {
    std::printf("GPU:          %s\n", d.gpu.c_str());
    std::printf("architecture: %s\n", d.architecture.empty() ? "?" : d.architecture.c_str());
    std::printf("driver:       %s%s\n", d.driver.ToString().c_str(), !d.driver.valid ? "" : d.driverOk ? " (>= 616.56)" : " (< 616.56 required)");
    if (d.dll.empty()) std::printf("dll:          not found (%s)\n", kNrDllName);
    else std::printf("dll:          %s (%llu bytes, sha256 %s)\n", d.dll.string().c_str(), static_cast<unsigned long long>(d.dllSize), d.dllSha256.c_str());
    std::printf("forwarder:    %s\n", d.forwarder.empty() ? "not found" : d.forwarder.string().c_str());
    if (!d.initResult.empty()) std::printf("init:         %s%s\n", d.initResult.c_str(), d.initAbi >= 0 ? (" (abi " + std::to_string(d.initAbi) + ")").c_str() : "");
    if (!d.createResult.empty()) std::printf("create(18):   %s [%s block]\n", d.createResult.c_str(), d.paramsBlock.c_str());
    if (d.ok) std::printf("status:       ok\n");
    else std::printf("status:       FAILED: %s\n", !d.hint.empty() ? d.hint.c_str() : error.c_str());
}

int CmdNrCheck(const NrCommands::NrArgs& a) {
    D3D12Device device({a.warp, false});
    std::unique_ptr<INrBackend> backend = CreateNrBackend(a.backend);
    NrConfig cfg;
    cfg.backend = a.backend;
    ParseSize(a.checkSize, cfg.width, cfg.height);
    cfg.dllDir = a.dllDir;
    cfg.paramsBlock = a.paramsBlock;
    cfg.requireDriver = !a.skipDriverCheck;
    cfg.style = ParseNrStyle(a.style);
    cfg.intensity = a.intensity;
    cfg.preset = a.preset;
    std::string error;
    try {
        backend->Init(device, cfg);
    } catch (const std::exception& e) {
        error = e.what();
    }
    const NrDiagnostics d = backend->Diagnostics();
    backend->Shutdown();
    PrintDiagnostics(d, error);
    if (!a.json.empty()) {
        nlohmann::json j = d.ToJson();
        j["error"] = error;
        std::ofstream(a.json) << j.dump(2) << "\n";
    }
    return d.ok ? 0 : 1;
}

int CmdNr(const NrCommands::NrArgs& a) {
    if (a.check) return CmdNrCheck(a);
    if (a.input.empty()) Throw("nr: -i/--input is required (or --check)");
    if (a.output.empty()) Throw("nr: -o/--output is required");
    VideoDecoder decoder(a.input);
    const auto& info = decoder.Info();
    NrStageOptions o;
    o.backend = a.backend;
    // pass folders produced by the earlier stages live under the same root: pick them up unless told otherwise
    auto autoDir = [&](const std::string& given, const char* name) -> std::filesystem::path {
        if (!given.empty()) return given;
        if (a.noAuto) return {};
        const auto d = std::filesystem::path(a.output) / name;
        if (std::filesystem::exists(d / Manifest::kFileName)) {
            Log()->info("nr: using {}", d.string());
            return d;
        }
        return {};
    };
    o.colorDir = autoDir(a.colorDir, "color_sr");
    o.depthDir = autoDir(a.depthDir, "depth_dlss");
    o.mvDir = autoDir(a.mvDir, "mv_dlss");
    o.masksDir = !a.masksDir.empty() ? std::filesystem::path(a.masksDir) : a.noAuto ? std::filesystem::path() : std::filesystem::path(a.output);
    o.intensity = a.intensity;
    o.style = a.style;
    o.preset = a.preset;
    o.localTone = a.localTone;
    o.localStructure = a.localStructure;
    o.skinStructure = a.skinStructure;
    o.autoMask = a.autoMask;
    o.passes = a.passes;
    o.modelScale = a.modelScale;
    o.transfer = a.transfer;
    o.maxRatio = a.maxRatio;
    o.useGuides = !a.noGuides;
    o.temporal = a.temporal;
    o.temporalThreshold = a.temporalThreshold;
    o.skinBlend = a.skinBlend;
    o.tonemap.curve = a.tonemap;
    o.tonemap.inputTransfer = a.inputTransfer;
    o.tonemap.exposure = a.exposure;
    o.paramsBlock = a.paramsBlock;
    o.requireDriver = !a.skipDriverCheck;
    o.dllDir = a.dllDir;
    o.outputDir = a.output;
    const auto fmt = ParseFileFormat(a.format);
    if (!fmt || (*fmt != FileFormat::Exr && *fmt != FileFormat::Png)) Throw("--format must be exr|png16");
    o.format = *fmt;
    o.videoOut = a.video;
    o.videoCodec = a.codec;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
    const int64_t total = a.frames > 0 ? a.frames : info.frameCount;
    o.onFrame = [](int64_t frame, double ms) {
        if (frame % 10 == 0) std::fprintf(stderr, "  frame %lld %.1f ms\n", static_cast<long long>(frame), ms);
    };
    D3D12Device device({a.warp, false});
    const auto t0 = std::chrono::steady_clock::now();
    const NrRunResult r = RunNr(decoder, device, o, a.frames, [total](int64_t n) { Progress(n, total); });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "\n");
    if (r.stats.disabled) {
        std::printf("nr:          disabled — %s\n", r.stats.disabledReason.c_str());
        return 1;
    }
    Log()->info("nr done: {} frames in {:.1f} s ({:.1f} ms/frame wall, {:.1f} ms/frame GPU), backend {}", r.stats.frames, sec,
                r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0, r.stats.MeanMs(), r.backend.dump());
    std::printf("color_nr:    %s\n", r.outDir.string().c_str());
    std::printf("backend:     %s\n", r.stats.backend.c_str());
    std::printf("colour from: %s\n", r.stats.colorSource.c_str());
    std::printf("size:        %ux%u (model at %ux%u)\n", r.stats.width, r.stats.height, r.stats.workWidth, r.stats.workHeight);
    std::printf("passes:      %d\nguides:      depth %s, mv %s\n", r.stats.passes, r.stats.depthUsed ? "yes" : "no", r.stats.mvUsed ? "yes" : "no");
    std::string masks;
    for (const auto& m : r.stats.masks) masks += (masks.empty() ? "" : ", ") + m;
    std::printf("masks:       %s\n", masks.empty() ? "none" : masks.c_str());
    std::printf("frames:      %lld\nms/frame:    %.1f (GPU %.1f)\n", static_cast<long long>(r.stats.frames), r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0,
                r.stats.MeanMs());
    if (!a.video.empty()) std::printf("video:       %s\n", a.video.c_str());
    if (!a.json.empty()) {
        nlohmann::json j = {{"color_nr", r.outDir.string()},
                            {"backend", r.backend},
                            {"diagnostics", r.diagnostics},
                            {"stats",
                             {{"frames", r.stats.frames},
                              {"ms_per_frame_gpu", r.stats.MeanMs()},
                              {"ms_per_frame_wall", r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0},
                              {"size", {r.stats.width, r.stats.height}},
                              {"work_size", {r.stats.workWidth, r.stats.workHeight}},
                              {"passes", r.stats.passes},
                              {"depth_used", r.stats.depthUsed},
                              {"mv_used", r.stats.mvUsed},
                              {"masks", r.stats.masks},
                              {"color_source", r.stats.colorSource}}}};
        std::ofstream(a.json) << j.dump(2) << "\n";
    }
    return 0;
}

int CmdPatch(const NrCommands::PatchArgs& a) {
    NrPatchOptions o;
    o.input = a.input;
    o.output = a.output;
    o.patcher = a.patcher;
    o.cudaBin = a.cudaBin;
    o.python = a.python;
    o.dryRun = a.dryRun;
    if (a.arch != "all") {
        std::stringstream ss(a.arch);
        std::string part;
        while (std::getline(ss, part, ','))
            if (!part.empty()) o.archs.push_back(part);
    }
    const NrPatchResult r = RunNrPatch(o);
    std::printf("patcher:     %s\n", r.script.string().c_str());
    std::printf("input:       %s (sha256 %s)\n", a.input.c_str(), r.inputSha256.c_str());
    if (r.ok && !a.dryRun) {
        std::printf("output:      %s (sha256 %s)\n", r.output.string().c_str(), r.outputSha256.c_str());
        std::printf("status:      ok — run `dlssvid nr --check` to verify CreateFeature(18)\n");
    } else if (r.ok) {
        std::printf("status:      dry run ok\n");
    } else {
        std::printf("status:      FAILED (patcher exit code %u) — see the patcher output above\n", r.exitCode);
    }
    return r.ok ? 0 : 1;
}

}  // namespace

void NrCommands::Register(CLI::App& app) {
    nr_ = app.add_subcommand("nr", "DLSS 5 Neural Rendering (NGX Feature 18) -> color_nr; --check prints the GPU / driver / DLL diagnostics only");
    nr_->add_option("-i,--input", na_.input, "input video file")->check(CLI::ExistingFile);
    nr_->add_option("-o,--output", na_.output, "pass root folder (writes color_nr/; color_sr, depth_dlss, mv_dlss and mask_* found here are used)");
    nr_->add_option("--color-dir,--color_dir", na_.colorDir, "colour input pass folder (default: <output>/color_sr when present, else the video)");
    nr_->add_option("--depth-dir,--depth_dir", na_.depthDir, "depth_dlss pass folder (guide)");
    nr_->add_option("--mv-dir,--mv_dir", na_.mvDir, "mv_dlss pass folder (guide)");
    nr_->add_option("--masks-dir,--masks_dir", na_.masksDir, "root with mask_ui / mask_ignore / mask_face / mask_skin pass folders (default: <output>)");
    nr_->add_flag("--no-auto,--no_auto", na_.noAuto, "do not pick pass folders under --output automatically");
    nr_->add_option("--backend", na_.backend, "ngx (nvngx_dlssnr.dll) | stub (deterministic test edit)")->default_val("ngx");
    nr_->add_option("--intensity", na_.intensity, "DLSSNR.Intensity 0..2")->default_val(1.f);
    nr_->add_option("--style", na_.style, "natural | cinematic | default | 0..15")->default_val("natural");
    nr_->add_option("--preset", na_.preset, "DLSSNR.Hint.Render.Preset 0..3")->default_val(3);
    nr_->add_option("--local-tone,--local_tone", na_.localTone, "DLSSNR.LocalToneStrength 0..2")->default_val(1.f);
    nr_->add_option("--local-structure,--local_structure", na_.localStructure, "DLSSNR.LocalStructureStrength 0..2")->default_val(1.f);
    nr_->add_option("--skin-structure,--skin_structure", na_.skinStructure, "DLSSNR.SkinStructureStrength (-1 = model default)")->default_val(-1.f);
    nr_->add_flag("--auto-mask,--auto_mask", na_.autoMask, "DLSSNR.UseAutoMask (the model's automatic skin mask)");
    nr_->add_option("--passes", na_.passes, "model passes 1 | 2")->default_val(1);
    nr_->add_option("--model-scale,--model_scale", na_.modelScale, "model resolution relative to the colour (0.25..2); != 1 uses the ratio transfer resolve")->default_val(1.0);
    nr_->add_option("--transfer", na_.transfer, "ratio transfer strength 0..1 (model-scale != 1)")->default_val(1.f);
    nr_->add_option("--max-ratio,--max_ratio", na_.maxRatio, "ratio transfer guard")->default_val(4.f);
    nr_->add_flag("--no-guides,--no_guides", na_.noGuides, "A/B: no depth / motion vectors to the model (still-image mode)");
    nr_->add_option("--temporal", na_.temporal, "temporal filter weight 0..1 (0 = off): blend with the previous frame warped by mv_dlss")->default_val(0.f);
    nr_->add_option("--temporal-threshold,--temporal_threshold", na_.temporalThreshold, "colour difference that rejects the temporal blend")->default_val(0.1f);
    nr_->add_option("--skin-blend,--skin_blend", na_.skinBlend, "edit strength inside mask_face / mask_skin (0 = keep the input)")->default_val(1.f);
    nr_->add_option("--tonemap", na_.tonemap, "passthrough | aces | reinhard")->default_val("passthrough");
    nr_->add_option("--input-transfer,--input_transfer", na_.inputTransfer, "srgb | linear | pq | hlg")->default_val("srgb");
    nr_->add_option("--exposure", na_.exposure, "exposure multiplier on linear light")->default_val(1.f);
    nr_->add_option("--format", na_.format, "exr | png16")->default_val("exr");
    nr_->add_option("--video", na_.video, "also encode the result into this video file (no audio)");
    nr_->add_option("--codec", na_.codec, "encoder for --video: h264_nvenc | hevc_nvenc | av1_nvenc | ffv1")->default_val("h264_nvenc");
    nr_->add_option("--frames", na_.frames, "process at most N frames")->default_val(-1);
    nr_->add_flag("--warp", na_.warp, "use the WARP software adapter (stub)");
    nr_->add_option("--dll-dir,--dll_dir", na_.dllDir, "folder with nvngx_dlssnr.dll (default: bin/nvidia next to the executable)");
    nr_->add_option("--params-block,--params_block", na_.paramsBlock, "NGX parameter block: capability | alloc")->default_val("capability");
    nr_->add_flag("--skip-driver-check,--skip_driver_check", na_.skipDriverCheck, "try even with a driver older than 616.56");
    nr_->add_flag("--check", na_.check, "no processing: initialise the backend, print GPU / driver / DLL / CreateFeature(18) diagnostics and exit 0/1");
    nr_->add_option("--check-size,--check_size", na_.checkSize, "resolution for --check")->default_val("1280x720");
    nr_->add_option("--json", na_.json, "write the run summary / diagnostics to this JSON file");

    patch_ = app.add_subcommand("nr-patch", "patch your nvngx_dlssnr.dll for RTX 20/30/40 with dlssnr-patcher (external tool, CUDA 13.3) into bin/nvidia/");
    patch_->add_option("-i,--input", pa_.input, "the original nvngx_dlssnr.dll")->required()->check(CLI::ExistingFile);
    patch_->add_option("-o,--output", pa_.output, "output DLL (default: bin/nvidia/nvngx_dlssnr.dll next to the executable)");
    patch_->add_option("--patcher", pa_.patcher, "dlssnr_patcher.py or its folder (default: DLSSNR_PATCHER_ROOT, tools/dlssnr-patcher)");
    patch_->add_option("--cuda-bin,--cuda_bin", pa_.cudaBin, "folder with ptxas / fatbinary / cuobjdump of CUDA Toolkit 13.3 (default: CUDA_PATH_V13_3/bin)");
    patch_->add_option("--python", pa_.python, "python interpreter (default: DLSSVID_PYTHON or python)");
    patch_->add_option("--arch", pa_.arch, "ada | ampere | turing | blackwell | all, comma separated")->default_val("all");
    patch_->add_flag("--dry-run,--dry_run", pa_.dryRun, "compile and validate without writing the output");
}

int NrCommands::Dispatch() {
    if (nr_ && nr_->parsed()) return CmdNr(na_);
    if (patch_ && patch_->parsed()) return CmdPatch(pa_);
    return -1;
}

}  // namespace dlssvid::cli
