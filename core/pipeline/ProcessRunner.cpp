#include "pipeline/ProcessRunner.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoEncoder.h"
#include "passes/PassSequence.h"
#include "pipeline/Pipeline.h"
#include "stages/depth/DepthStage.h"
#include "stages/fg/FgStage.h"
#include "stages/flow/FlowStage.h"
#include "stages/nr/NrStage.h"
#include "stages/passthrough/PassthroughStage.h"
#include "stages/upscale/UpscaleStage.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"
#include "viewport/Project.h"

namespace dlssvid {

namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

template <typename T>
T Get(const nlohmann::json& p, const char* key, const T& def) {
    if (!p.contains(key)) return def;
    try {
        return p[key].get<T>();
    } catch (...) {
        Throw(std::string("process: stage parameter '") + key + "' has the wrong type");
    }
}

HwAccel ParseHwaccel(const std::string& s) {
    if (s == "cuda") return HwAccel::Cuda;
    if (s == "none" || s.empty()) return HwAccel::None;
    Throw("--hwaccel must be none|cuda");
}

FileFormat ParsePassFormat(const std::string& s, bool allowPng) {
    const auto fmt = ParseFileFormat(s);
    if (!fmt || (!allowPng && *fmt == FileFormat::Png)) Throw("process: format '" + s + "' is not valid for this stage");
    return *fmt;
}

DepthStageOptions DepthOptions(const nlohmann::json& p, const std::filesystem::path& passesRoot) {
    DepthStageOptions o;
    o.backend = Get<std::string>(p, "backend", "da3");
    o.estimator.modelId = Get<std::string>(p, "model", "");
    o.estimator.inputSize = Get<int>(p, "input_size", 518);
    o.estimator.maxInputRes = Get<int>(p, "max_res", 1080);
    o.estimator.fp16 = !Get<bool>(p, "fp32", false);
    o.estimator.modelsDir = Get<std::string>(p, "models_dir", "");
    if (p.contains("python")) o.estimator.extra["python"] = p["python"];
    o.outputDir = passesRoot;
    o.writeDlss = !Get<bool>(p, "no_dlss", false);
    o.dlss.zNear = Get<float>(p, "z_near", 0.1f);
    o.dlss.zFar = Get<float>(p, "z_far", 1000.f);
    const std::string stabilize = Get<std::string>(p, "stabilize", "auto");
    if (stabilize == "none") o.stabilize = TemporalStabilizer::Mode::None;
    else if (stabilize == "scale") o.stabilize = TemporalStabilizer::Mode::ScaleOnly;
    else if (stabilize == "scale_shift" || stabilize == "auto") o.stabilize = TemporalStabilizer::Mode::ScaleShift;
    else Throw("process: depth.stabilize must be auto|none|scale|scale_shift");
    o.stabilizeMetric = stabilize == "scale" || stabilize == "scale_shift";
    o.stabilizeWindow = Get<int>(p, "stabilize_window", 8);
    o.fillHoles = !Get<bool>(p, "no_fill", false);
    o.format = ParsePassFormat(Get<std::string>(p, "format", "exr"), false);
    return o;
}

FlowStageOptions FlowOptions(const nlohmann::json& p, const std::filesystem::path& passesRoot) {
    FlowStageOptions o;
    o.backend = Get<std::string>(p, "backend", "ofa");
    o.estimator.modelId = Get<std::string>(p, "model", "");
    o.estimator.maxRes = Get<int>(p, "max_res", 720);
    o.estimator.fp16 = !Get<bool>(p, "fp32", false);
    o.estimator.modelsDir = Get<std::string>(p, "models_dir", "");
    o.estimator.extra["perf_level"] = Get<std::string>(p, "perf", "slow");
    o.estimator.extra["grid"] = Get<int>(p, "grid", 1);
    if (p.contains("dx")) o.estimator.extra["dx"] = p["dx"];
    if (p.contains("dy")) o.estimator.extra["dy"] = p["dy"];
    o.outputDir = passesRoot;
    o.writeDlss = !Get<bool>(p, "no_dlss", false);
    const std::string target = Get<std::string>(p, "target", "");
    if (!target.empty()) {
        const size_t x = target.find('x');
        if (x == std::string::npos) Throw("process: flow.target expects WxH");
        o.targetWidth = static_cast<uint32_t>(std::stoul(target.substr(0, x)));
        o.targetHeight = static_cast<uint32_t>(std::stoul(target.substr(x + 1)));
    }
    o.convert.dilateRadius = Get<int>(p, "dilate", 1);
    if (std::filesystem::exists(passesRoot / "depth_raw" / Manifest::kFileName)) o.depthDir = passesRoot / "depth_raw";
    o.format = ParsePassFormat(Get<std::string>(p, "format", "exr"), false);
    return o;
}

std::filesystem::path ExistingPass(const std::filesystem::path& root, std::initializer_list<const char*> names) {
    for (const char* n : names)
        if (std::filesystem::exists(root / n / Manifest::kFileName)) return root / n;
    return {};
}

}  // namespace

nlohmann::json ProcessStageReport::ToJson() const {
    return {{"name", name},     {"status", status},   {"reason", reason},          {"out_dir", outDir.string()},
            {"frames", frames}, {"seconds", seconds}, {"ms_per_frame", msPerFrame}, {"details", details}};
}

nlohmann::json ProcessResult::ToJson() const {
    nlohmann::json st = nlohmann::json::array();
    for (const auto& s : stages) st.push_back(s.ToJson());
    return {{"stages", st},         {"final_pass", finalPass.string()}, {"output", output.string()}, {"frames_in", framesIn}, {"frames_out", framesOut},
            {"output_fps", outputFps.ToDouble()}, {"audio", audio},   {"seconds", seconds}};
}

const std::vector<std::string>& ProcessStageOrder() {
    static const std::vector<std::string> order{"depth", "flow", "upscale", "nr", "fg"};
    return order;
}

std::vector<ProcessStage> DefaultProcessStages() {
    std::vector<ProcessStage> out;
    for (const StageEntry& e : Project::DefaultStages()) out.push_back(ProcessStage{e.name, e.enabled, e.params});
    return out;
}

std::vector<ProcessStage> StagesFromProject(const Project& project) {
    std::vector<ProcessStage> out;
    for (const StageEntry& e : project.stages) out.push_back(ProcessStage{e.name, e.enabled, e.params});
    return out;
}

void ApplyParamSpec(std::vector<ProcessStage>& stages, const std::string& spec) {
    const size_t dot = spec.find('.'), eq = spec.find('=');
    if (dot == std::string::npos || eq == std::string::npos || dot > eq || dot == 0 || eq == dot + 1) Throw("--param expects stage.key=value (got '" + spec + "')");
    const std::string stage = spec.substr(0, dot), key = spec.substr(dot + 1, eq - dot - 1), value = spec.substr(eq + 1);
    const auto& order = ProcessStageOrder();
    if (std::find(order.begin(), order.end(), stage) == order.end()) Throw("--param: unknown stage '" + stage + "' (depth | flow | upscale | nr | fg)");
    nlohmann::json v;
    try {
        v = nlohmann::json::parse(value);
        if (!(v.is_number() || v.is_boolean())) v = value;
    } catch (...) {
        v = value;
    }
    for (auto& s : stages)
        if (s.name == stage) {
            s.params[key] = v;
            s.enabled = true;
            return;
        }
    stages.push_back(ProcessStage{stage, true, {{key, v}}});
}

bool PassComplete(const std::filesystem::path& dir, int64_t neededFrames, const std::string& sourceHash) {
    if (!std::filesystem::exists(dir / Manifest::kFileName)) return false;
    try {
        const PassReader r = PassReader::Open(dir);
        const Manifest& m = r.Man();
        if (neededFrames > 0) return m.frameCount >= neededFrames && r.HasFrame(m.firstFrame + neededFrames - 1);
        return m.frameCount > 0 && r.HasFrame(m.firstFrame + m.frameCount - 1) && (sourceHash.empty() || m.sourceHash == sourceHash);
    } catch (...) {
        return false;
    }
}

EncodeReport EncodePassToVideo(const std::filesystem::path& passDir, const std::filesystem::path& source, const std::filesystem::path& output, const std::string& codec,
                               const std::map<std::string, std::string>& codecOptions, int64_t maxSourceFrames, const std::function<void(int64_t, int64_t)>& progress,
                               const std::string& hwaccel) {
    const auto t0 = Clock::now();
    const PassReader pass = PassReader::Open(passDir);
    const Manifest& m = pass.Man();
    if (m.channels.size() < 3) Throw("process: " + passDir.string() + " is not a colour pass");
    VideoDecoder::Options dopt;
    dopt.hwaccel = ParseHwaccel(hwaccel);
    VideoDecoder decoder(source, dopt);
    const VideoStreamInfo& info = decoder.Info();
    const Rational sourceFps = info.frameRate;
    Rational fps = m.fps.num > 0 ? m.fps : sourceFps;
    int mult = 1;
    if (sourceFps.num > 0 && fps.num > 0) mult = std::max(1, static_cast<int>(std::lround(fps.ToDouble() / sourceFps.ToDouble())));
    VideoEncoder::Options eo;
    eo.codec = codec;
    eo.frameRate = fps.num > 0 ? fps : Rational{30, 1};
    eo.timeBase = Rational{eo.frameRate.den, eo.frameRate.num};
    eo.codecOptions = codecOptions;
    VideoEncoder encoder(output, FrameDesc{m.width, m.height, PixelFormat::Yuv420p}, eo);
    EncodeReport r;
    r.fps = eo.frameRate;
    if (decoder.HasAudio()) {
        encoder.AddAudioStreamCopy(decoder.AudioCodecParameters(), decoder.AudioTimeBase());
        decoder.SetAudioPacketSink([&encoder](AVPacket* pkt) { encoder.WriteAudioPacket(pkt); });
        r.audio = true;
    }
    encoder.Open();
    const ColorInfo ci = ColorInfoFromStream(info);
    const int64_t total = maxSourceFrames > 0 ? std::min<int64_t>(maxSourceFrames, info.frameCount) : info.frameCount;
    CpuFrame src;
    int64_t s = 0;
    while ((maxSourceFrames < 0 || s < maxSourceFrames) && decoder.NextFrame(src)) {  // decoding the source pumps its audio packets
        for (int k = 0; k < mult; ++k) {
            const int64_t idx = s * mult + k;
            if (!pass.HasFrame(m.firstFrame + idx)) break;
            const PassImage rgb = pass.ReadFrame(m.firstFrame + idx);
            CpuFrame yuv;
            RgbToYuv420p(rgb, ci, yuv);
            yuv.index = idx;
            yuv.pts = idx;
            encoder.WriteFrame(yuv);
            ++r.frames;
        }
        ++s;
        if (progress) progress(s, total);
    }
    encoder.Close();
    r.sourceFrames = s;
    r.seconds = Seconds(t0);
    return r;
}

ProcessResult RunProcess(const ProcessOptions& options) {
    const auto tAll = Clock::now();
    ProcessResult result;
    result.output = options.output;
    if (options.input.empty() || !std::filesystem::exists(options.input)) Throw("process: input video not found: " + options.input.string());
    if (options.output.empty()) Throw("process: -o/--output is required");
    std::filesystem::create_directories(std::filesystem::absolute(options.output).parent_path());

    VideoStreamInfo info;
    {
        VideoDecoder probe(options.input);
        info = probe.Info();
    }
    // frame count of the source: containers without one (MKV) report 0 -> unknown (-1)
    const int64_t sourceFrames = info.frameCount > 0 ? info.frameCount : -1;
    const int64_t needed = options.frames > 0 ? (sourceFrames > 0 ? std::min<int64_t>(options.frames, sourceFrames) : options.frames) : sourceFrames;

    // ---- passthrough (stage 0) ----
    if (options.passthrough) {
        VideoDecoder::Options dopt;
        dopt.hwaccel = ParseHwaccel(options.hwaccel);
        VideoDecoder decoder(options.input, dopt);
        VideoEncoder::Options eopt;
        eopt.codec = options.codec;
        eopt.frameRate = info.frameRate;
        eopt.timeBase = info.timeBase;
        eopt.codecOptions = options.codecOptions;
        VideoEncoder encoder(options.output, FrameDesc{info.width, info.height, PixelFormat::Yuv420p}, eopt);
        D3D12Device device({options.warp, false});
        Pipeline pipeline(device, 4);
        StageConfig cfg;
        cfg.name = "passthrough";
        cfg.params["gpu_roundtrip"] = options.gpuRoundtrip;
        pipeline.AddStage(std::make_unique<PassthroughStage>(), cfg);
        pipeline.Init();
        const auto t0 = Clock::now();
        const RunStats stats = RunPipeline(
            pipeline, decoder, encoder, [&](int64_t n) { if (options.progress) options.progress("passthrough", n, needed); }, options.frames);
        pipeline.Shutdown();
        ProcessStageReport rep;
        rep.name = "passthrough";
        rep.status = "passthrough";
        rep.frames = stats.framesOut;
        rep.seconds = Seconds(t0);
        rep.msPerFrame = rep.frames ? 1000.0 * rep.seconds / static_cast<double>(rep.frames) : 0.0;
        result.stages.push_back(rep);
        result.framesOut = stats.framesOut;
        result.framesIn = stats.framesOut;
        result.outputFps = info.frameRate;
        result.audio = decoder.HasAudio();
        result.seconds = Seconds(tAll);
        return result;
    }

    // ---- stages over the pass cache ----
    const std::filesystem::path root = options.passesRoot.empty() ? options.output.parent_path() / (options.output.stem().string() + "_passes") : options.passesRoot;
    std::filesystem::create_directories(root);
    std::map<std::string, ProcessStage> wanted;
    for (const auto& s : options.stages) {
        if (std::find(ProcessStageOrder().begin(), ProcessStageOrder().end(), s.name) == ProcessStageOrder().end())
            Throw("process: unknown stage '" + s.name + "' (depth | flow | upscale | nr | fg)");
        if (s.enabled) wanted[s.name] = s;
    }
    const std::string sourceFile = options.input.filename().string();
    const std::string sourceHash = "sha256:" + Sha256File(options.input);
    D3D12Device device({options.warp, false});
    auto progressFor = [&](const std::string& stage) {
        return [&, stage](int64_t n) {
            if (options.progress) options.progress(stage, n, needed);
        };
    };

    for (const std::string& name : ProcessStageOrder()) {
        const auto it = wanted.find(name);
        if (it == wanted.end()) continue;
        const nlohmann::json& p = it->second.params;
        ProcessStageReport rep;
        rep.name = name;
        const int fgMult = name == "fg" ? Get<int>(p, "multiplier", 2) : 1;
        const std::filesystem::path checkDir = root / (name == "depth" ? "depth_dlss" : name == "flow" ? "mv_dlss" : name == "upscale" ? "color_sr" : name == "nr" ? "color_nr" : "color_fg");
        const int64_t neededHere = needed > 0 ? (name == "fg" ? FgFrameCount(needed, fgMult) : needed) : -1;
        if (options.skipExisting && PassComplete(checkDir, neededHere, sourceHash)) {
            rep.status = "reused";
            rep.outDir = checkDir;
            rep.frames = neededHere > 0 ? neededHere : PassReader::Open(checkDir).Man().frameCount;
            Log()->info("process: {} — {} is complete, reused", name, checkDir.string());
            result.stages.push_back(rep);
            continue;
        }
        const auto t0 = Clock::now();
        try {
            if (name == "depth") {
                VideoDecoder decoder(options.input);
                DepthStageOptions o = DepthOptions(p, root);
                o.sourceFile = sourceFile;
                o.sourceHash = sourceHash;
                const DepthRunResult r = RunDepth(decoder, device, o, options.frames, progressFor(name));
                rep.frames = r.stats.frames;
                rep.outDir = o.writeDlss ? r.dlssDir : r.rawDir;
                rep.details = {{"estimator", r.estimator}, {"mean_tae", r.stats.MeanTae()}};
            } else if (name == "flow") {
                VideoDecoder::Options dopt;
                const std::string backend = Get<std::string>(p, "backend", "ofa");
                dopt.hwaccel = ParseHwaccel(Get<std::string>(p, "hwaccel", backend == "ofa" ? "cuda" : "none"));
                VideoDecoder decoder(options.input, dopt);
                FlowStageOptions o = FlowOptions(p, root);
                o.sourceFile = sourceFile;
                o.sourceHash = sourceHash;
                const FlowRunResult r = RunFlow(decoder, device, o, options.frames, progressFor(name));
                rep.frames = r.stats.frames;
                rep.outDir = o.writeDlss ? r.dlssDir : r.rawDir;
                rep.details = {{"estimator", r.estimator}, {"warp_psnr", r.stats.MeanWarpPsnr()}};
            } else if (name == "upscale") {
                VideoDecoder decoder(options.input);
                UpscaleStageOptions o;
                o.outputDir = root;
                o.depthDir = ExistingPass(root, {"depth_dlss"});
                o.mvDir = ExistingPass(root, {"mv_dlss"});
                o.allowFallback = !Get<bool>(p, "no_fallback", false);
                o.format = ParsePassFormat(Get<std::string>(p, "format", "exr"), true);
                o.sourceFile = sourceFile;
                o.sourceHash = sourceHash;
                const UpscaleRunResult r = RunUpscale(decoder, device, o, options.frames, progressFor(name), p);
                rep.frames = r.stats.frames;
                rep.outDir = r.outDir;
                rep.details = {{"upscaler", r.upscaler}, {"fallback", r.stats.fellBack}, {"size", {r.stats.outputWidth, r.stats.outputHeight}}};
            } else if (name == "nr") {
                VideoDecoder decoder(options.input);
                NrStageOptions o;
                o.outputDir = root;
                o.colorDir = ExistingPass(root, {"color_sr"});
                o.depthDir = ExistingPass(root, {"depth_dlss"});
                o.mvDir = ExistingPass(root, {"mv_dlss"});
                o.masksDir = root;
                o.disableWhenUnavailable = options.disableUnavailable;
                o.format = ParsePassFormat(Get<std::string>(p, "format", "exr"), true);
                o.sourceFile = sourceFile;
                o.sourceHash = sourceHash;
                const NrRunResult r = RunNr(decoder, device, o, options.frames, progressFor(name), p);
                if (r.stats.disabled) {
                    rep.status = "disabled";
                    rep.reason = r.stats.disabledReason;
                } else {
                    rep.frames = r.stats.frames;
                    rep.outDir = r.outDir;
                }
                rep.details = {{"backend", r.backend}, {"diagnostics", r.diagnostics}};
            } else if (name == "fg") {
                VideoDecoder decoder(options.input);
                FgStageOptions o;
                o.outputDir = root;
                o.colorDir = ExistingPass(root, {"color_nr", "color_sr"});
                o.depthDir = ExistingPass(root, {"depth_dlss"});
                o.mvDir = ExistingPass(root, {"mv_dlss"});
                o.disableWhenUnavailable = options.disableUnavailable;
                o.format = ParsePassFormat(Get<std::string>(p, "format", "exr"), true);
                o.sourceFile = sourceFile;
                o.sourceHash = sourceHash;
                const FgRunResult r = RunFg(decoder, device, o, options.frames, progressFor(name), p);
                if (r.stats.disabled) {
                    rep.status = "disabled";
                    rep.reason = r.stats.disabledReason;
                } else {
                    rep.frames = r.stats.frames;
                    rep.outDir = r.outDir;
                }
                rep.details = {{"backend", r.backend}, {"diagnostics", r.diagnostics}, {"generated", r.stats.generated}};
            }
        } catch (const std::exception& e) {
            Throw("process: stage '" + name + "' failed: " + e.what());
        }
        rep.seconds = Seconds(t0);
        if (rep.status.empty()) rep.status = "ran";
        rep.msPerFrame = rep.frames ? 1000.0 * rep.seconds / static_cast<double>(rep.frames) : 0.0;
        if (rep.status == "ran") Log()->info("process: {} done — {} frames in {:.1f} s ({:.1f} ms/frame)", name, rep.frames, rep.seconds, rep.msPerFrame);
        result.stages.push_back(rep);
    }

    // ---- encode the last colour pass, or pass the source through ----
    result.finalPass = ExistingPass(root, {"color_fg", "color_nr", "color_sr"});
    ProcessStageReport enc;
    enc.name = "encode";
    const auto t0 = Clock::now();
    if (result.finalPass.empty()) {
        Log()->warn("process: no colour pass was produced — encoding the source frames unchanged");
        ProcessOptions pt = options;
        pt.passthrough = true;
        pt.progress = nullptr;
        const ProcessResult r = RunProcess(pt);
        enc.status = "passthrough";
        enc.frames = r.framesOut;
        result.framesOut = r.framesOut;
        result.framesIn = r.framesIn;
        result.outputFps = r.outputFps;
        result.audio = r.audio;
    } else {
        const EncodeReport r = EncodePassToVideo(result.finalPass, options.input, options.output, options.codec, options.codecOptions, options.frames,
                                                 [&](int64_t n, int64_t total) {
                                                     if (options.progress) options.progress("encode", n, total);
                                                 },
                                                 options.hwaccel);
        enc.status = "ran";
        enc.outDir = result.finalPass;
        enc.frames = r.frames;
        result.framesOut = r.frames;
        result.framesIn = r.sourceFrames;
        result.outputFps = r.fps;
        result.audio = r.audio;
    }
    enc.seconds = Seconds(t0);
    enc.msPerFrame = enc.frames ? 1000.0 * enc.seconds / static_cast<double>(enc.frames) : 0.0;
    result.stages.push_back(enc);
    result.seconds = Seconds(tAll);
    Log()->info("process: {} -> {} ({} frames at {:.3f} fps{}) in {:.1f} s", options.input.filename().string(), options.output.string(), result.framesOut,
                result.outputFps.ToDouble(), result.audio ? ", audio copied" : "", result.seconds);
    return result;
}

}  // namespace dlssvid
