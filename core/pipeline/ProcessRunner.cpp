#include "pipeline/ProcessRunner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <set>

#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/EncodeDefaults.h"
#include "io/VideoEncoder.h"
#include "passes/PassSequence.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "pipeline/Pipeline.h"
#include "stages/ParamSchema.h"
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

// ---- planning (stage 9, MR A) ----------------------------------------------------------------------------------

struct RunContext {
    std::filesystem::path root;
    std::string sourceHash;
    int64_t needed = -1;  // source frames the run covers (-1: unknown)
    bool skipExisting = true;
    std::set<std::string> force;  // stages recomputed on request
};

// The passes on disk (or, while planning, as they will be after the stages decided so far): pass -> fingerprint;
// legacy passes get a stable pseudo-fingerprint. `ran` lists the passes this plan (re)writes.
struct PlanState {
    std::map<std::string, std::string> fp;
    std::set<std::string> ran;
};

std::string LegacyId(const Manifest& m) { return "legacy:" + m.sourceHash + ":" + std::to_string(m.frameCount); }

void RefreshFromDisk(PlanState& s, const std::filesystem::path& root) {
    s.fp.clear();
    for (const auto& name : PassesUnderRoot(root)) {
        try {
            const Manifest m = Manifest::Load(root / name);
            s.fp[name] = m.fingerprint.empty() ? LegacyId(m) : m.fingerprint;
        } catch (const std::exception& e) {
            Log()->warn("process: {} skipped: {}", (root / name).string(), e.what());
        }
    }
}

// After a decision, what the passes look like: the stage's outputs carry the planned fingerprint.
void Advance(PlanState& s, const StageDecision& d) {
    if (d.action == "reuse") {
        if (d.kind == "legacy")
            for (const auto& out : d.outputs)
                if (s.fp.count(out)) s.fp[out] = d.fingerprint;  // adopted: stamped with this fingerprint
        return;
    }
    for (const auto& fam : StageFamilyPasses(d.name)) s.fp.erase(fam);  // retired or rewritten
    for (const auto& out : d.outputs) {
        s.fp[out] = d.fingerprint;
        s.ran.insert(out);
    }
}

int64_t NeededFor(const std::string& name, const nlohmann::json& p, int64_t needed) {
    const int fgMult = name == "fg" ? Get<int>(p, "multiplier", 2) : 1;
    return needed > 0 ? (name == "fg" ? FgFrameCount(needed, fgMult) : needed) : -1;
}

std::string Brief(const nlohmann::json& v) {
    if (v.is_null()) return "(none)";
    return v.is_string() ? v.get<std::string>() : v.dump();
}

std::string Join(const std::vector<std::string>& v) {
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
    return s;
}

// A complete history version of `pass` with this fingerprint, if any.
std::optional<PassVersionInfo> RestorableVersion(const RunContext& ctx, const std::string& pass, const std::string& fp, int64_t frames) {
    for (const auto& v : ListPassVersions(ctx.root, pass))
        if (!v.current && v.fingerprint == fp && PassComplete(v.dir, frames, ctx.sourceHash)) return v;
    return std::nullopt;
}

StageDecision DecideStage(const RunContext& ctx, const ProcessStage& stage, const PlanState& state) {
    const std::string& name = stage.name;
    const nlohmann::json p = EffectiveStageParams(name, stage.params);  // schema defaults filled in: the run's real parameters
    StageDecision d;
    d.name = name;
    d.frames = NeededFor(name, p, ctx.needed);
    d.outputs = StageOutputPasses(name, p);
    d.outDir = ctx.root / d.outputs.back();
    std::vector<std::string> inputsRan;
    for (const auto& group : StageInputCandidates(name))
        for (const auto& cand : group) {
            const auto it = state.fp.find(cand);
            if (it == state.fp.end()) continue;
            d.inputs[cand] = it->second;
            if (state.ran.count(cand)) inputsRan.push_back(cand);
            break;
        }
    const PassFingerprint fp = ComputeFingerprint(name, ctx.sourceHash, p, d.inputs, ToolForStage(name, p));
    d.fingerprint = fp.value;
    d.params = fp.params;
    d.tool = fp.tool.ToJson();

    Manifest m;
    bool have = false;
    std::string unreadable;
    if (Manifest::Exists(d.outDir)) {
        try {
            m = Manifest::Load(d.outDir);
            have = true;
        } catch (const std::exception& e) {
            unreadable = e.what();
        }
    }
    d.current = have;
    if (have) d.currentFingerprint = m.fingerprint;
    const bool complete = have && PassComplete(d.outDir, d.frames, ctx.sourceHash);
    auto run = [&](const char* kind, std::string detail, bool retire) {
        d.action = "run";
        d.kind = kind;
        d.detail = std::move(detail);
        d.retire = retire;
    };
    auto restore = [&]() {
        const auto v = RestorableVersion(ctx, d.outputs.back(), fp.value, d.frames);
        if (!v) return false;
        d.action = "restore";
        d.kind = "restored";
        d.version = v->id;
        d.detail = "version " + v->id + " from the history";
        d.retire = have;
        return true;
    };
    const std::string incomplete = (have ? std::to_string(m.frameCount) : std::string("0")) + " of " + (d.frames > 0 ? std::to_string(d.frames) : std::string("all")) + " frames";

    if (!ctx.skipExisting || ctx.force.count(name)) {
        run("forced", ctx.skipExisting ? "recomputed on request (--force)" : "every stage recomputed (--no-skip-existing)", have && m.fingerprint != fp.value);
        return d;
    }
    if (!have) {
        if (restore()) return d;
        run("missing", unreadable.empty() ? "no pass yet" : "manifest unreadable: " + unreadable, std::filesystem::exists(d.outDir));
        return d;
    }
    if (m.fingerprint == fp.value) {
        if (complete) {
            d.action = "reuse";
            d.kind = "exact";
            d.detail = "fingerprint matches";
        } else {
            run("incomplete", incomplete, false);
        }
        return d;
    }
    if (m.fingerprint.empty()) {  // made before fingerprints (or by a standalone `dlssvid <stage>`)
        if (!inputsRan.empty()) {
            d.diff["inputs"] = inputsRan;
            run("input_changed", "input recomputed: " + Join(inputsRan), true);
        } else if (m.sourceHash != ctx.sourceHash) {
            run("source_changed", m.sourceHash.empty() ? "the pass has no source hash" : "the pass was made from another source", true);
        } else if (!complete) {
            run("incomplete", incomplete, false);
        } else {
            d.action = "reuse";
            d.kind = "legacy";
            d.detail = "pass predates fingerprints; adopted with the current parameters";
        }
        return d;
    }
    // another fingerprinted version is current: restore a matching one, else say what changed
    if (restore()) return d;
    if (m.sourceHash != ctx.sourceHash) {
        run("source_changed", "the pass was made from another source", true);
        return d;
    }
    const nlohmann::json oldP = CanonicalizeJson(m.paramsCanonical);
    nlohmann::json pdiff = nlohmann::json::object();
    std::vector<std::string> ptext;
    std::set<std::string> keys;
    for (const auto& [k, v] : oldP.items()) keys.insert(k);
    for (const auto& [k, v] : fp.params.items()) keys.insert(k);
    for (const auto& k : keys) {
        const nlohmann::json o = oldP.contains(k) ? oldP[k] : nlohmann::json(), n = fp.params.contains(k) ? fp.params[k] : nlohmann::json();
        if (o != n) {
            pdiff[k] = {o, n};
            ptext.push_back(k + ": " + Brief(o) + " -> " + Brief(n));
        }
    }
    if (!pdiff.empty()) {
        d.diff["params"] = pdiff;
        run("params_changed", Join(ptext), true);
        return d;
    }
    std::vector<std::string> changedInputs;
    std::set<std::string> inputNames;
    for (const auto& [k, v] : m.inputs) inputNames.insert(k);
    for (const auto& [k, v] : d.inputs) inputNames.insert(k);
    for (const auto& k : inputNames) {
        const auto o = m.inputs.find(k), n = d.inputs.find(k);
        if ((o == m.inputs.end()) != (n == d.inputs.end()) || (o != m.inputs.end() && o->second != n->second)) changedInputs.push_back(k);
    }
    if (!changedInputs.empty()) {
        d.diff["inputs"] = changedInputs;
        run("input_changed", "input changed: " + Join(changedInputs), true);
        return d;
    }
    const ToolInfo oldT = ToolInfo::FromJson(m.tool);
    nlohmann::json tdiff = nlohmann::json::object();
    std::vector<std::string> tfields;
    auto tool = [&](const char* field, const std::string& o, const std::string& n) {
        if (o == n) return;
        tdiff[field] = {o, n};
        tfields.push_back(field);
    };
    tool("app", oldT.app, fp.tool.app);
    tool("backend", oldT.backend, fp.tool.backend);
    tool("model", oldT.model, fp.tool.model);
    tool("model_hash", oldT.modelHash, fp.tool.modelHash);
    tool("dll_file", oldT.dllFile, fp.tool.dllFile);
    tool("dll", oldT.dll, fp.tool.dll);
    if (!tdiff.empty()) {
        d.diff["tool"] = tdiff;
        run("tool_changed", "tool changed: " + Join(tfields), true);
        return d;
    }
    run("changed", "fingerprint differs", true);
    return d;
}

void StampPass(const std::filesystem::path& dir, const StageDecision& d, const std::string& created, double msPerFrame = 0.0) {
    Manifest m = Manifest::Load(dir);
    if (msPerFrame > 0) m.stageParams["ms_per_frame"] = msPerFrame;  // the last run on this machine: the estimates use it
    m.fingerprint = d.fingerprint;
    m.inputs = d.inputs;
    m.tool = d.tool;
    m.paramsCanonical = d.params;
    m.created = created;
    m.Save(dir);
}

struct Prepared {
    VideoStreamInfo info;
    int64_t sourceFrames = -1;
    RunContext ctx;
    std::map<std::string, ProcessStage> wanted;
    std::string sourceFile;
};

Prepared Prepare(const ProcessOptions& options) {
    Prepared r;
    if (options.input.empty() || !std::filesystem::exists(options.input)) Throw("process: input video not found: " + options.input.string());
    {
        VideoDecoder probe(options.input, VideoDecoder::Probe());
        r.info = probe.Info();
    }
    // frame count of the source: containers without one (MKV) report 0 -> unknown (-1)
    r.sourceFrames = r.info.frameCount > 0 ? r.info.frameCount : -1;
    r.ctx.needed = options.frames > 0 ? (r.sourceFrames > 0 ? std::min<int64_t>(options.frames, r.sourceFrames) : options.frames) : r.sourceFrames;
    r.ctx.skipExisting = options.skipExisting;
    r.ctx.force = options.forceStages;
    if (options.passesRoot.empty() && options.output.empty()) Throw("process: --passes or -o/--output is needed to locate the passes");
    r.ctx.root = options.passesRoot.empty() ? options.output.parent_path() / (options.output.stem().string() + "_passes") : options.passesRoot;
    for (const auto& s : options.stages) {
        if (std::find(ProcessStageOrder().begin(), ProcessStageOrder().end(), s.name) == ProcessStageOrder().end())
            Throw("process: unknown stage '" + s.name + "' (depth | flow | upscale | nr | fg)");
        if (s.enabled) r.wanted[s.name] = s;
    }
    std::vector<std::string> problems;
    for (const auto& [name, s] : r.wanted)
        for (const auto& p : ValidateStageParams(name, s.params)) problems.push_back(p);
    if (!problems.empty()) Throw("process: " + Join(problems));
    for (const auto& f : options.forceStages)
        if (std::find(ProcessStageOrder().begin(), ProcessStageOrder().end(), f) == ProcessStageOrder().end())
            Throw("process: --force: unknown stage '" + f + "' (depth | flow | upscale | nr | fg)");
    r.sourceFile = options.input.filename().string();
    r.ctx.sourceHash = options.sourceHash.rfind("sha256:", 0) == 0 ? options.sourceHash : "sha256:" + Sha256FileCached(options.input);
    return r;
}

std::string ProjectedFinalPass(const PlanState& s) {
    for (const char* n : {"color_fg", "color_nr", "color_sr"})
        if (s.fp.count(n)) return n;
    return {};
}

}  // namespace

nlohmann::json ProcessStageReport::ToJson() const {
    return {{"name", name},         {"status", status},         {"reason", reason},     {"out_dir", outDir.string()}, {"frames", frames},
            {"seconds", seconds},   {"ms_per_frame", msPerFrame}, {"details", details}, {"fingerprint", fingerprint},  {"retired", retired},
            {"restored", restored}, {"decision", decision}};
}

nlohmann::json ProcessResult::ToJson() const {
    nlohmann::json st = nlohmann::json::array();
    for (const auto& s : stages) st.push_back(s.ToJson());
    return {{"stages", st},         {"final_pass", finalPass.string()}, {"output", output.string()}, {"frames_in", framesIn}, {"frames_out", framesOut},
            {"output_fps", outputFps.ToDouble()}, {"audio", audio},   {"seconds", seconds}};
}

nlohmann::json StageDecision::ToJson() const {
    return {{"name", name},       {"action", action},   {"kind", kind},         {"detail", detail},   {"diff", diff},
            {"fingerprint", fingerprint}, {"current", current}, {"current_fingerprint", currentFingerprint}, {"version", version}, {"outputs", outputs},
            {"out_dir", outDir.string()}, {"frames", frames}, {"retire", retire}, {"params", params}, {"inputs", inputs}, {"tool", tool}};
}

nlohmann::json ProcessPlan::ToJson() const {
    nlohmann::json st = nlohmann::json::array();
    for (const auto& s : stages) st.push_back(s.ToJson());
    return {{"passes_root", passesRoot.string()}, {"source_hash", sourceHash}, {"frames", frames},       {"stages", st},
            {"final_pass", finalPass},            {"run_count", runCount},    {"reuse_count", reuseCount}};
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
    // no `b` given: the bitrate for this output (NVENC's own default is far too low for 4K)
    eo.codecOptions = EffectiveCodecOptions(codec, codecOptions, m.width, m.height, eo.frameRate.ToDouble());
    if (!codecOptions.count("b") && eo.codecOptions.count("b"))
        Log()->info("encode: bitrate {} chosen for {}x{} at {:.3g} fps ({}); --codec-opt b=... overrides", eo.codecOptions.at("b"), m.width, m.height,
                    eo.frameRate.ToDouble(), codec);
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

ProcessPlan PlanProcess(const ProcessOptions& options) {
    ProcessPlan plan;
    if (options.passthrough) return plan;  // nothing to decide: the source goes straight to the encoder
    const Prepared pr = Prepare(options);
    plan.passesRoot = pr.ctx.root;
    plan.sourceHash = pr.ctx.sourceHash;
    plan.frames = pr.ctx.needed;
    PlanState state;
    RefreshFromDisk(state, pr.ctx.root);
    for (const std::string& name : ProcessStageOrder()) {
        const auto it = pr.wanted.find(name);
        if (it == pr.wanted.end()) continue;
        StageDecision d = DecideStage(pr.ctx, it->second, state);
        Advance(state, d);
        (d.action == "run" ? plan.runCount : plan.reuseCount)++;
        plan.stages.push_back(std::move(d));
    }
    plan.finalPass = ProjectedFinalPass(state);
    return plan;
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
        VideoDecoder probe(options.input, VideoDecoder::Probe());
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
        eopt.codecOptions = EffectiveCodecOptions(options.codec, options.codecOptions, info.width, info.height, info.frameRate.ToDouble());
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
    const Prepared pr = Prepare(options);
    const RunContext& ctx = pr.ctx;
    const std::filesystem::path& root = ctx.root;
    std::filesystem::create_directories(root);
    const std::string& sourceFile = pr.sourceFile;
    const std::string& sourceHash = ctx.sourceHash;
    D3D12Device device({options.warp, false});
    auto progressFor = [&](const std::string& stage) {
        return [&, stage](int64_t n) {
            if (options.progress) options.progress(stage, n, needed);
        };
    };

    PlanState state;
    bool retiredAny = false;
    for (const std::string& name : ProcessStageOrder()) {
        const auto it = pr.wanted.find(name);
        if (it == pr.wanted.end()) continue;
        const nlohmann::json& p = it->second.params;
        RefreshFromDisk(state, root);  // the decision looks at what is really on disk now (a stage may have been disabled)
        const StageDecision d = DecideStage(ctx, it->second, state);
        ProcessStageReport rep;
        rep.name = name;
        rep.decision = d.ToJson();
        rep.fingerprint = d.fingerprint;
        auto framesOf = [&]() { return d.frames > 0 ? d.frames : PassReader::Open(d.outDir).Man().frameCount; };
        if (d.action == "reuse") {
            rep.status = "reused";
            rep.outDir = d.outDir;
            rep.frames = framesOf();
            if (d.kind == "legacy") {
                rep.reason = "adopted: " + d.detail;
                for (const auto& pass : StageFamilyPasses(name))
                    if (Manifest::Exists(root / pass)) StampPass(root / pass, d, FileTimeIso8601(root / pass / Manifest::kFileName));
            }
            Log()->info("process: {} — {} is complete, reused{}", name, d.outDir.string(), d.kind == "legacy" ? " (adopted)" : "");
            result.stages.push_back(rep);
            continue;
        }
        if (d.action == "restore") {
            for (const auto& pass : StageFamilyPasses(name)) {
                bool has = false;
                for (const auto& v : ListPassVersions(root, pass)) has = has || (!v.current && v.id == d.version);
                if (has && UsePassVersion(root, pass, d.version)) retiredAny = true;
            }
            rep.status = "reused";
            rep.outDir = d.outDir;
            rep.restored = d.version;
            rep.reason = "restored: " + d.detail;
            rep.frames = framesOf();
            state.ran.insert(d.outputs.begin(), d.outputs.end());
            Log()->info("process: {} — version {} restored from the history", name, d.version);
            result.stages.push_back(rep);
            continue;
        }
        if (d.retire)
            for (const auto& pass : StageFamilyPasses(name))
                if (const auto id = RetirePassVersion(root, pass)) {
                    if (pass == d.outputs.back() || rep.retired.empty()) rep.retired = *id;
                    retiredAny = true;
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
        rep.msPerFrame = rep.frames ? 1000.0 * rep.seconds / static_cast<double>(rep.frames) : 0.0;
        if (rep.status.empty()) {
            rep.status = "ran";
            rep.reason = d.kind + ": " + d.detail;
            const std::string created = NowIso8601();
            for (const auto& pass : StageFamilyPasses(name))
                if (Manifest::Exists(root / pass)) StampPass(root / pass, d, created, rep.msPerFrame);
            state.ran.insert(d.outputs.begin(), d.outputs.end());
        } else {
            rep.fingerprint.clear();  // disabled: nothing was written
        }
        if (rep.status == "ran")
            Log()->info("process: {} done — {} frames in {:.1f} s ({:.1f} ms/frame), {} ({})", name, rep.frames, rep.seconds, rep.msPerFrame, ShortFingerprint(d.fingerprint),
                        rep.reason);
        result.stages.push_back(rep);
    }
    if (retiredAny && options.keepVersions > 0) {
        const PassGcReport gc = GcPassVersions(root, options.keepVersions);
        if (!gc.removed.empty()) Log()->info("process: {} old pass version(s) removed ({} MB), {} kept per pass", gc.removed.size(), gc.bytes / (1024 * 1024), options.keepVersions);
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
        try {
            enc.fingerprint = Manifest::Load(result.finalPass).fingerprint;
        } catch (...) {
        }
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
