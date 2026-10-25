#include "pipeline/ProcessRunner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <set>

#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/EncodeDefaults.h"
#include "io/OutputFormat.h"
#include "io/VideoEncoder.h"
#include "passes/PassSequence.h"
#include "pipeline/FgProcess.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "pipeline/Pipeline.h"
#include "stages/ParamSchema.h"
#include "ml/ModelRegistry.h"
#include "stages/depth/DepthModels.h"
#include "stages/depth/DepthPreprocess.h"
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
    // "" is the schema's default: the interpreter is found as without the key (DLSSVID_PYTHON, the venv, python)
    if (const std::string python = Get<std::string>(p, "python", ""); !python.empty()) o.estimator.extra["python"] = python;
    // a project keeps depth.window after a switch from VDA: only a VDA backend reads it (the plan's parameters drop it too)
    o.estimator.window = DepthFamilyOf(o.backend) == "vda" ? Get<int>(p, "window", 0) : 0;
    o.outputDir = passesRoot;
    o.writeDlss = !Get<bool>(p, "no_dlss", false);
    o.dlss.zNear = Get<float>(p, "z_near", 0.1f);
    o.dlss.zFar = Get<float>(p, "z_far", 1000.f);
    o.stabilize = ParseDepthStabilize(Get<std::string>(p, "stabilize", "auto"));
    o.stabilizeWindow = std::clamp(Get<int>(p, "stabilize_window", 8), 1, 64);  // the schema's range: the buffer holds N+1 frames
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

// A pass a standalone run made with the A/B switches of dlssvid nr (stage_params.ab, TASK-0048): not the project's run.
bool AbPassManifest(const Manifest& m) { return m.stageParams.is_object() && m.stageParams.contains("ab"); }

// A pass that stands for the clip's run of its stage: it starts at frame 0 - a part of the clip (dlssvid nr --start, export
// --range) does not (review of TASK-0047: fg read frames F..L, «has no frame 0», the encode put frame F at t=0) - and it is
// no A/B run (review of TASK-0048). Anything else is no stage's input and not the result.
bool PassOfClip(const Manifest& m) { return m.firstFrame == 0 && !AbPassManifest(m); }
bool PassOfClip(const std::filesystem::path& dir) {
    try {
        return PassOfClip(Manifest::Load(dir));
    } catch (const std::exception&) {
        return true;  // unreadable: the stage that reads it says so, as before
    }
}

std::filesystem::path ExistingPass(const std::filesystem::path& root, std::initializer_list<const char*> names) {
    for (const char* n : names)
        if (std::filesystem::exists(root / n / Manifest::kFileName) && PassOfClip(root / n)) return root / n;
    return {};
}

// ---- planning (stage 9, MR A) ----------------------------------------------------------------------------------

struct RunContext {
    std::filesystem::path root;
    std::string sourceHash;
    std::string sourceFormat;  // the layout the source decodes into (FrameDesc::Name()): part of the fingerprints
    int64_t needed = -1;  // source frames the run covers (-1: unknown)
    uint32_t sourceWidth = 0, sourceHeight = 0;  // the model input of depth follows from them (VRAM advice)
    uint64_t vramBytes = 0;                      // 0: unknown, no VRAM error
    bool skipExisting = true;
    std::set<std::string> force;  // stages recomputed on request
};

// The passes on disk (or, while planning, as they will be after the stages decided so far): pass -> fingerprint;
// legacy passes get a stable pseudo-fingerprint. `ran` lists the passes this plan (re)writes.
struct PlanState {
    std::map<std::string, std::string> fp;
    std::set<std::string> ran;
    std::set<std::string> staleRan;  // passes rerun because they were stale (StageDecision::stale): a later stage built on
                                     // them is stale too although every fingerprint matches
    std::set<std::string> discarded;  // stale passes removed (StageDecision::stale): a pass made from one is discarded too, whether
                                      // its stage ran again or was disabled after the removal (review of PR-7)
};

// A pass without a fingerprint stands for its source and length; a y-up motion-vector pass also for its orientation:
// PassReader negates v on read, so toggling mv.y_up changes what the stages below read and must recompute them.
std::string LegacyId(const Manifest& m) {
    std::string id = "legacy:" + m.sourceHash + ":" + std::to_string(m.frameCount);
    if (m.IsYUpMv()) id += ":y_up";
    return id;
}

void RefreshFromDisk(PlanState& s, const std::filesystem::path& root) {
    s.fp.clear();
    for (const auto& name : PassesUnderRoot(root)) {
        try {
            const Manifest m = Manifest::Load(root / name);
            if (!PassOfClip(m)) {  // a part of the clip or an A/B run: no input of the stages below (its own stage still sees it)
                Log()->info("process: {} is not used as the clip's pass ({})", (root / name).string(),
                            m.firstFrame != 0 ? "frames " + std::to_string(m.firstFrame) + ".." + std::to_string(m.lastFrame) + " only" : std::string("A/B switches, stage_params.ab"));
                continue;
            }
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
    for (const auto& fam : StageFamilyPasses(d.name)) {
        s.fp.erase(fam);  // retired or rewritten
        if (d.stale && d.current) s.discarded.insert(fam);
    }
    for (const auto& out : d.outputs) {
        s.fp[out] = d.fingerprint;
        s.ran.insert(out);
        if (d.stale) s.staleRan.insert(out);
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

// A colour pass written as PNG16 before TASK-0042 - unscaled, so black: in PNG on disk while its tool lacks the "+png.2"
// revision every PNG colour pass made since carries. Judged by the file, not by the parameters (a standalone
// `--format png16` pass adopted under exr parameters is PNG all the same), and also when the pass has no fingerprint
// (adopting it would stamp the black frames with the current revision). Not reused, restored or adopted.
bool OldPngColourPass(const std::string& stage, const Manifest& m) {
    if ((stage != "upscale" && stage != "nr" && stage != "fg") || m.format != FileFormat::Png) return false;
    return m.tool.value("app", std::string()).find("+png.2") == std::string::npos;
}
constexpr const char* kOldPng = "PNG16 colour pass written before TASK-0042 (unscaled, black)";

// An nr pass made while exposure was a linear multiplier (before TASK-0043) with a value other than 1. An explicit 0 -
// the default the schema showed - gave black frames under the fingerprint of the identity run: such a pass is not
// reused, restored or adopted. Passes made with the old default (multiplier 1) stay valid.
bool OldExposurePass(const std::string& stage, const Manifest& m) {
    if (stage != "nr" || !m.stageParams.is_object() || !m.stageParams.contains("tonemap")) return false;
    const nlohmann::json& t = m.stageParams["tonemap"];
    if (!t.is_object() || t.contains("exposure_ev") || !t.contains("exposure") || !t["exposure"].is_number()) return false;
    return t["exposure"].get<double>() != 1.0;
}
constexpr const char* kOldExposure = "made with the old linear exposure (TASK-0043: exposure is in stops now)";

// An nr pass made before PR-7, when nr.intensity went to NGX as DLSSNR.Intensity: any value but 1 is another picture now (0 was a
// black frame, PR-2 E2). A pass of PR-7 carries model_intensity or bypass. Discarded like the old exposure: not reused, restored,
// adopted or kept as a version (its fingerprint differs - +nr.gain / +nr.i0 - and DecideStage discards it in that branch too).
bool OldIntensityPass(const std::string& stage, const Manifest& m) {
    if (stage != "nr" || !m.stageParams.is_object()) return false;
    const nlohmann::json& sp = m.stageParams;
    if (sp.contains("model_intensity") || sp.contains("bypass") || !sp.contains("intensity") || !sp["intensity"].is_number()) return false;
    return sp["intensity"].get<double>() != 1.0;
}
constexpr const char* kOldIntensity = "made before PR-7, when nr.intensity went to the model (it is the edit strength after the model now)";

// An nr pass a standalone `dlssvid nr` made with its A/B switches (spec §2e: --mv-scale auto, --create-with-mv, --no-depth-guide):
// the switches are not parameters, so adopting it - it has no fingerprint - would claim the default run (TASK-0048). It is a
// real result of an experiment, not a broken pass: it goes to the history (kind ab_pass), the stages below are not stale.
bool AbNrPass(const std::string& stage, const Manifest& m) { return stage == "nr" && AbPassManifest(m); }
constexpr const char* kAbPass = "made with the A/B switches of a standalone dlssvid nr (stage_params.ab)";

// A pass wrong under its own fingerprint: discarded (not kept as a version), and what was made from it is stale too.
const char* BrokenPassReason(const std::string& stage, const Manifest& m) {
    if (OldPngColourPass(stage, m)) return kOldPng;
    if (OldExposurePass(stage, m)) return kOldExposure;
    if (OldIntensityPass(stage, m)) return kOldIntensity;
    return nullptr;
}

// A pass the planner must not reuse, restore or adopt although its fingerprint may match: the reason, else nullptr.
const char* StalePassReason(const std::string& stage, const Manifest& m) {
    if (const char* broken = BrokenPassReason(stage, m)) return broken;
    if (AbNrPass(stage, m)) return kAbPass;
    return nullptr;
}

// A complete history version of `pass` with this fingerprint, if any.
std::optional<PassVersionInfo> RestorableVersion(const RunContext& ctx, const std::string& stage, const std::string& pass, const std::string& fp, int64_t frames) {
    for (const auto& v : ListPassVersions(ctx.root, pass)) {
        if (v.current || v.fingerprint != fp || !PassComplete(v.dir, frames, ctx.sourceHash)) continue;
        try {
            if (StalePassReason(stage, Manifest::Load(v.dir))) continue;
        } catch (const std::exception&) {
            continue;
        }
        return v;
    }
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
    for (const auto& group : StageInputCandidates(name, p))
        for (const auto& cand : group) {
            const auto it = state.fp.find(cand);
            if (it == state.fp.end()) continue;
            d.inputs[cand] = it->second;
            if (state.ran.count(cand)) inputsRan.push_back(cand);
            break;
        }
    std::vector<std::string> inputsStale;  // inputs rerun in this plan because they were stale (PlanState::staleRan)
    for (const auto& [input, inputFp] : d.inputs)
        if (state.staleRan.count(input)) inputsStale.push_back(input);
    const PassFingerprint fp = ComputeFingerprint(name, ctx.sourceHash, p, d.inputs, ToolForStage(name, p), ctx.sourceFormat);
    d.fingerprint = fp.value;
    d.sourceFormat = ctx.sourceFormat;
    d.params = fp.params;
    d.tool = fp.tool.ToJson();
    // the layout a pass on disk was made from: manifests from before the field come from the 8-bit 4:2:0 decode
    const std::string sourceFormatNow = ctx.sourceFormat.empty() ? kLegacySourceFormat : ctx.sourceFormat;

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
    // the current pass was made from a broken pass this run removed (PlanState::discarded): it is broken too, even when that stage
    // was disabled since and no longer feeds this one (no nr DLL: fg reads color_sr - the color_fg of the black nr went to the history)
    std::vector<std::string> inputsDiscarded;
    if (have)
        for (const auto& [input, inputFp] : m.inputs)
            if (state.discarded.count(input)) inputsDiscarded.push_back(input);
    const bool complete = have && PassComplete(d.outDir, d.frames, ctx.sourceHash);
    auto run = [&](const char* kind, std::string detail, bool retire) {
        d.action = "run";
        d.kind = kind;
        d.detail = std::move(detail);
        d.retire = retire;
    };
    auto restore = [&]() {
        const auto v = RestorableVersion(ctx, name, d.outputs.back(), fp.value, d.frames);
        if (!v) return false;
        d.action = "restore";
        d.kind = "restored";
        d.version = v->id;
        d.detail = "version " + v->id + " from the history";
        d.retire = have;
        return true;
    };
    const std::string incomplete =
        have && m.firstFrame != 0 ? "frames " + std::to_string(m.firstFrame) + ".." + std::to_string(m.lastFrame) + " only, a part of the clip"
                                  : (have ? std::to_string(m.frameCount) : std::string("0")) + " of " + (d.frames > 0 ? std::to_string(d.frames) : std::string("all")) + " frames";

    if (!ctx.skipExisting || ctx.force.count(name)) {
        // a stale pass is discarded, never kept as a version: the card would promise «vN останется на диске» (review of PR-7)
        const bool stale = !inputsStale.empty() || !inputsDiscarded.empty() || (have && BrokenPassReason(name, m));
        run("forced", ctx.skipExisting ? "recomputed on request (--force)" : "every stage recomputed (--no-skip-existing)", have && m.fingerprint != fp.value && !stale);
        d.stale = stale;
        return d;
    }
    // Every fingerprint may match, yet an input was stale when this pass (or a history version under the same fingerprint)
    // was made from it: a black color_fg from a black color_nr stayed after nr was recomputed (review of TASK-0043).
    if (!inputsStale.empty()) {
        d.diff["inputs"] = inputsStale;
        run("input_changed", "built on a stale pass that is recomputed: " + Join(inputsStale), false);
        d.stale = true;
        return d;
    }
    if (!inputsDiscarded.empty()) {
        d.diff["inputs"] = inputsDiscarded;
        run("input_changed", "built on a discarded pass: " + Join(inputsDiscarded), false);
        d.stale = true;
        return d;
    }
    if (!have) {
        if (restore()) return d;
        run("missing", unreadable.empty() ? "no pass yet" : "manifest unreadable: " + unreadable, std::filesystem::exists(d.outDir));
        return d;
    }
    if (m.fingerprint == fp.value) {
        if (const char* broken = BrokenPassReason(name, m)) {
            run("tool_changed", broken, false);
            d.stale = true;
        } else if (AbNrPass(name, m)) {
            run("ab_pass", kAbPass, true);
        } else if (complete) {
            d.action = "reuse";
            d.kind = "exact";
            d.detail = "fingerprint matches";
        } else {
            run("incomplete", incomplete, false);
        }
        return d;
    }
    const std::string madeFrom = m.sourceFormat.empty() ? kLegacySourceFormat : m.sourceFormat;
    if (m.fingerprint.empty()) {  // made before fingerprints (or by a standalone `dlssvid <stage>`)
        if (!inputsRan.empty()) {
            d.diff["inputs"] = inputsRan;
            run("input_changed", "input recomputed: " + Join(inputsRan), true);
        } else if (m.sourceHash != ctx.sourceHash) {
            run("source_changed", m.sourceHash.empty() ? "the pass has no source hash" : "the pass was made from another source", true);
        } else if (madeFrom != sourceFormatNow) {
            run("source_changed", "the pass was made from the source decoded as " + madeFrom + " (now " + sourceFormatNow + ")", true);
        } else if (name == "depth" && p.value("stabilize", std::string("auto")) != "none" && !m.stageParams.contains("stabilize_requested")) {
            // a standalone `dlssvid depth` from before the centred alignment (TASK-0038): metric depth was not aligned
            // then - adopting it would stamp the old depth with the depth.2 revision
            d.diff["tool"]["app"] = nlohmann::json::array({kPassToolVersion, PassToolVersionFor(name, p)});
            run("tool_changed", "depth made before the temporal alignment of TASK-0038", true);
        } else if (AbNrPass(name, m)) {  // kept in the history to compare; the run of the project is restored or recomputed
            if (restore()) return d;
            run("ab_pass", kAbPass, true);
        } else if (const char* broken = BrokenPassReason(name, m)) {
            run("tool_changed", broken, false);
            d.stale = true;
        } else if (!complete) {
            run("incomplete", incomplete, false);
        } else {
            d.action = "reuse";
            d.kind = "legacy";
            d.detail = "pass predates fingerprints; adopted with the current parameters";
        }
        return d;
    }
    // another fingerprinted version is current. A broken one is discarded, not kept as a version: a revision that fixes it changes the
    // fingerprint (+png.2, +nr.gain / +nr.i0), so it lands here, and «Вернуть» in the versions panel would bring the black frames back
    // (review of PR-7). The card names the added revision.
    if (const char* broken = BrokenPassReason(name, m)) {
        const std::string oldApp = ToolInfo::FromJson(m.tool).app;
        if (oldApp != fp.tool.app) d.diff["tool"]["app"] = nlohmann::json::array({oldApp, fp.tool.app});
        run("tool_changed", broken, false);
        d.stale = true;
        return d;
    }
    // restore a matching one, else say what changed
    if (restore()) return d;
    if (m.sourceHash != ctx.sourceHash) {
        run("source_changed", "the pass was made from another source", true);
        return d;
    }
    if (madeFrom != sourceFormatNow) {
        run("source_changed", "the pass was made from the source decoded as " + madeFrom + " (now " + sourceFormatNow + ")", true);
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
    // the wall time of this run (decode, pass I/O, init and the stage) on this machine: the only time the estimates use; a stage's
    // own timer is ms_per_frame_gpu (nr, D8)
    if (msPerFrame > 0) m.stageParams["ms_per_frame"] = msPerFrame;
    m.fingerprint = d.fingerprint;
    m.inputs = d.inputs;
    m.tool = d.tool;
    m.paramsCanonical = d.params;
    m.sourceFormat = d.sourceFormat;
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
    // the depth model against the registry (TASK-0039): a foreign or unknown id, a bad window - before anything runs
    if (const auto it = r.wanted.find("depth"); it != r.wanted.end() && problems.empty()) {
        const nlohmann::json p = EffectiveStageParams("depth", it->second.params);
        const std::string backend = Get<std::string>(p, "backend", "da3");
        if (!DepthFamilyOf(backend).empty()) {
            std::optional<ModelRegistry> registry;
            try {
                registry.emplace(ModelRegistry::Load(DepthRegistryPath(Get<std::string>(p, "models_dir", ""))));
            } catch (const std::exception&) {  // no registry here: the estimator checks the model when it runs
            }
            if (registry) {
                try {
                    ResolveDepthModel(*registry, backend, Get<std::string>(p, "model", ""), Get<int>(p, "window", 0), backend == "worker:vda");
                } catch (const std::exception& e) {
                    problems.push_back(e.what());
                }
            }
        }
    }
    if (!problems.empty()) Throw("process: " + Join(problems));
    for (const auto& f : options.forceStages)
        if (std::find(ProcessStageOrder().begin(), ProcessStageOrder().end(), f) == ProcessStageOrder().end())
            Throw("process: --force: unknown stage '" + f + "' (depth | flow | upscale | nr | fg)");
    r.sourceFile = options.input.filename().string();
    r.ctx.sourceHash = options.sourceHash.rfind("sha256:", 0) == 0 ? options.sourceHash : "sha256:" + Sha256FileCached(options.input);
    r.ctx.sourceFormat = r.info.decoded.Name();
    r.ctx.sourceWidth = r.info.width;
    r.ctx.sourceHeight = r.info.height;
    r.ctx.vramBytes = options.vramBytes;
    return r;
}

// Licence and VRAM advice of a depth decision (TASK-0039). An error only for a run: a reused or restored pass needs no GPU.
void AdviseStage(const RunContext& ctx, StageDecision& d) {
    if ((d.name == "upscale" || d.name == "nr" || d.name == "fg") && d.params.contains("dll_dir"))
        d.warnings.push_back(d.name + ".dll_dir is ignored by process: the DLL comes from DLSSVID_NVIDIA_DLL_DIR, <exe>/nvidia or <exe>");
    if (d.name == "nr") {
        // PR-5 (spec §2c): above 2 the model reworks its own output again - the picture turns synthetic; each pass is a feature of its own
        const int passes = Get<int>(d.params, "passes", 1);
        if (passes >= 3 && !NrBypassParams(d.params))
            d.info = std::to_string(passes) + " прохода: модель повторно обрабатывает свой результат — картинка заметно синтетичнее; каждый проход — "
                     "своя копия модели (≈ 0.5 ГБ видеопамяти на 1440p, 0.9 ГБ на 4K) и её время";
        return;
    }
    if (d.name != "depth") return;
    const std::string backend = Get<std::string>(d.params, "backend", "da3");
    if (DepthFamilyOf(backend).empty()) return;
    DepthModelChoice choice;
    try {
        choice = ResolveDepthModel(Get<std::string>(d.params, "models_dir", ""), backend, Get<std::string>(d.params, "model", ""),
                                   Get<int>(d.params, "window", 0), backend == "worker:vda");
    } catch (const std::exception&) {  // no registry: nothing to advise (a bad model was refused by Prepare)
        return;
    }
    const ModelInputSize in = ComputeModelInputSize(ctx.sourceWidth, ctx.sourceHeight, EffectiveDepthInputSize(choice, Get<int>(d.params, "input_size", 518)));
    const auto need = EstimateDepthVramMiB(choice, backend == "worker:vda", in.width, in.height, !Get<bool>(d.params, "fp32", false));
    DepthAdvice a = AdviseDepth(choice, need, ctx.vramBytes, in.width, in.height);
    d.warnings = std::move(a.warnings);
    d.info = std::move(a.info);
    if (d.action == "run") d.error = std::move(a.error);
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
            {"output_fps", outputFps.ToDouble()}, {"audio", audio},   {"seconds", seconds}, {"warnings", warnings},
            {"output_format", {{"codec", codec}, {"pix_fmt", outputPixelFormat}, {"name", outputFormat.Name()}, {"bit_depth", outputFormat.HighBitDepth() ? outputFormat.bitDepth : 8},
                               {"chroma", std::string(ChromaName(outputFormat.format))}, {"description", outputFormat.Describe()}}}};
}

nlohmann::json StageDecision::ToJson() const {
    return {{"name", name},       {"action", action},   {"kind", kind},         {"detail", detail},   {"diff", diff},
            {"fingerprint", fingerprint}, {"source_format", sourceFormat}, {"current", current}, {"current_fingerprint", currentFingerprint}, {"version", version},
            {"outputs", outputs}, {"out_dir", outDir.string()}, {"frames", frames}, {"retire", retire}, {"stale", stale}, {"params", params}, {"inputs", inputs}, {"tool", tool},
            {"warnings", warnings}, {"info", info}, {"error", error}};
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
    // a key the schema knows as text stays text: --param nr.style=4 is the style "4", not the number 4 (TASK-0044)
    const StageSchema* schema = FindStageSchema(stage);
    const ParamSpec* keySpec = schema ? schema->Find(key) : nullptr;
    if (keySpec && (keySpec->type == ParamType::Enum || keySpec->type == ParamType::String)) {
        v = value;
    } else {
        try {
            v = nlohmann::json::parse(value);
            if (!(v.is_number() || v.is_boolean())) v = value;
        } catch (...) {
            v = value;
        }
    }
    for (auto& s : stages)
        if (s.name == stage) {
            // another backend: a model from the old backend's list goes (vda + metric-vda-large -> da3)
            if (key == "backend" && v.is_string()) s.params = ParamsForBackendChange(stage, s.params, v.get<std::string>());
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
        // a pass of the clip starts at frame 0 - a part of it (dlssvid nr --start, export --range) never stands for the clip:
        // with the source's frame count unknown (mkv) only the last frame was looked at, and frames F..L were adopted (TASK-0047)
        if (m.firstFrame != 0) return false;
        if (neededFrames > 0) return m.frameCount >= neededFrames && r.HasFrame(m.firstFrame + neededFrames - 1);
        return m.frameCount > 0 && r.HasFrame(m.firstFrame + m.frameCount - 1) && (sourceHash.empty() || m.sourceHash == sourceHash);
    } catch (...) {
        return false;
    }
}

ColorTags ColorTagsFor(const ColorInfo& ci, const VideoStreamInfo& info) {
    ColorTags t;
    t.range = ci.range == ColorRange::Full ? 2 : 1;  // AVCOL_RANGE_JPEG / MPEG
    if (ci.matrix == ColorMatrix::Bt709) t.matrix = 1;  // AVCOL_SPC_BT709
    else t.matrix = info.colorSpace == 5 ? 5 : 6;        // BT.601: BT470BG as the source said, else SMPTE170M
    t.primaries = info.colorPrimaries;
    t.transfer = info.colorTrc;
    return t;
}

EncodeReport EncodePassToVideo(const std::filesystem::path& passDir, const std::filesystem::path& source, const std::filesystem::path& output, const EncodeSettings& encode,
                               int64_t maxSourceFrames, const std::function<void(int64_t, int64_t)>& progress, const std::string& hwaccel) {
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
    const OutputFormat of = ResolveOutputFormat(encode.codec, encode.format, info.decoded);
    FrameDesc target = of.desc;
    target.width = m.width;
    target.height = m.height;
    const ColorInfo ci = ColorInfoFromStream(info);
    VideoEncoder::Options eo;
    eo.codec = encode.codec;
    eo.frameRate = fps.num > 0 ? fps : Rational{30, 1};
    eo.timeBase = Rational{eo.frameRate.den, eo.frameRate.num};
    eo.color = ColorTagsFor(ci, info);
    // no `b` given: the bitrate for this output (NVENC's own default is far too low for 4K)
    eo.codecOptions = EffectiveCodecOptions(encode.codec, encode.codecOptions, m.width, m.height, eo.frameRate.ToDouble());
    if (!encode.codecOptions.count("b") && eo.codecOptions.count("b"))
        Log()->info("encode: bitrate {} chosen for {}x{} at {:.3g} fps ({}); --codec-opt b=... overrides", eo.codecOptions.at("b"), m.width, m.height,
                    eo.frameRate.ToDouble(), encode.codec);
    VideoEncoder encoder(output, target, eo);
    Log()->info("encode: {} as {} ({}) from a {} source{}", encode.codec, target.Describe(), encoder.PixelFormatName(), info.decoded.Describe(),
                of.reason.empty() ? "" : " — " + of.reason);
    EncodeReport r;
    r.fps = eo.frameRate;
    r.format = target;
    r.pixelFormat = encoder.PixelFormatName();
    if (decoder.HasAudio()) {
        encoder.AddAudioStreamCopy(decoder.AudioCodecParameters(), decoder.AudioTimeBase());
        decoder.SetAudioPacketSink([&encoder](AVPacket* pkt) { encoder.WriteAudioPacket(pkt); });
        r.audio = true;
    }
    encoder.Open();
    const int64_t total = maxSourceFrames > 0 ? std::min<int64_t>(maxSourceFrames, info.frameCount) : info.frameCount;
    CpuFrame src;
    int64_t s = 0;
    while ((maxSourceFrames < 0 || s < maxSourceFrames) && decoder.NextFrame(src)) {  // decoding the source pumps its audio packets
        for (int k = 0; k < mult; ++k) {
            const int64_t idx = s * mult + k;
            if (!pass.HasFrame(m.firstFrame + idx)) break;
            const PassImage rgb = pass.ReadFrame(m.firstFrame + idx);
            CpuFrame yuv;
            RgbToYuv(rgb, ci, target, yuv);
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
        AdviseStage(pr.ctx, d);
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
        eopt.color = ColorTagsFor(ColorInfoFromStream(info), info);
        const OutputFormat of = ResolveOutputFormat(options.codec, options.outputFormat, info.decoded);
        FrameDesc target = of.desc;
        target.width = info.width;
        target.height = info.height;
        VideoEncoder encoder(options.output, target, eopt);
        Log()->info("passthrough: {} as {} ({}) from a {} source{}{}", options.codec, target.Describe(), encoder.PixelFormatName(), info.decoded.Describe(),
                    target == info.decoded ? ", bit-exact" : ", converted", of.reason.empty() ? "" : " — " + of.reason);
        result.codec = options.codec;
        result.outputFormat = target;
        result.outputPixelFormat = encoder.PixelFormatName();
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
    Prepared pr = Prepare(options);
    D3D12Device device({options.warp, false});
    if (!pr.ctx.vramBytes && !device.IsWarp()) pr.ctx.vramBytes = device.DedicatedVideoMemory();
    const RunContext& ctx = pr.ctx;
    const std::filesystem::path& root = ctx.root;
    std::filesystem::create_directories(root);
    const std::string& sourceFile = pr.sourceFile;
    const std::string& sourceHash = ctx.sourceHash;
    auto progressFor = [&](const std::string& stage) {
        return [&, stage](int64_t n) {
            if (options.progress) options.progress(stage, n, needed);
        };
    };

    PlanState state;
    bool retiredAny = false;
    bool fgSkipped = false;           // fg was wanted but failed or is disabled: the result comes from its input, not color_fg
    std::filesystem::path fgColorDir;  // that input (color_nr / color_sr; empty: the source)
    for (const std::string& name : ProcessStageOrder()) {
        const auto it = pr.wanted.find(name);
        if (it == pr.wanted.end()) continue;
        // what the fingerprint hashes is what the stage runs with: defaults filled, aliases normalised (TASK-0043)
        const nlohmann::json p = EffectiveStageParams(name, it->second.params);
        RefreshFromDisk(state, root);  // the decision looks at what is really on disk now (a stage may have been disabled)
        StageDecision d = DecideStage(ctx, it->second, state);
        AdviseStage(ctx, d);
        if (!d.error.empty()) Throw("process: " + d.error);  // before anything is retired
        for (const auto& w : d.warnings) {
            Log()->warn("process: {}", w);
            result.warnings.push_back(w);
        }
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
        if (d.stale) {  // wrong under its own fingerprint: discarded, not kept as a version a later run would restore
            for (const auto& pass : StageFamilyPasses(name)) {
                if (Manifest::Exists(root / pass)) state.discarded.insert(pass);  // what was made from it goes too, however this stage ends
                std::error_code ec;
                std::filesystem::remove_all(root / pass, ec);
                if (ec) Throw("process: cannot remove the stale pass " + (root / pass).string() + ": " + ec.message());
            }
        } else if (d.retire)
            for (const auto& pass : StageFamilyPasses(name))
                if (const auto id = RetirePassVersion(root, pass)) {
                    if (pass == d.outputs.back() || rep.retired.empty()) rep.retired = *id;
                    retiredAny = true;
                }
        const auto t0 = Clock::now();
        double childMsPerFrame = 0.0, childSeconds = 0.0;  // fg in its own process: its own timing, without the start-up
        try {
            if (name == "depth") {
                VideoDecoder decoder(options.input);
                DepthStageOptions o = DepthOptions(p, root);
                o.sourceFile = sourceFile;
                o.sourceHash = sourceHash;
                const DepthRunResult r = RunDepth(decoder, device, o, options.frames, progressFor(name));
                rep.frames = r.stats.frames;
                rep.outDir = o.writeDlss ? r.dlssDir : r.rawDir;
                rep.details = {{"estimator", r.estimator}, {"mean_tae", r.stats.MeanTae()}, {"stabilize", r.stabilize}};
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
                FgStageOptions o;
                o.outputDir = root;
                o.colorDir = ExistingPass(root, {"color_nr", "color_sr"});
                o.depthDir = ExistingPass(root, {"depth_dlss"});
                o.mvDir = ExistingPass(root, {"mv_dlss"});
                o.disableWhenUnavailable = options.disableUnavailable;
                o.format = ParsePassFormat(Get<std::string>(p, "format", "exr"), true);
                o.sourceFile = sourceFile;
                o.sourceHash = sourceHash;
                o.crashAfter = options.fgCrashAfter;
                fgColorDir = o.colorDir;
                // What is on disk before the run: after a failure, the debris of this run is removed, while a signed version
                // the run did not touch (an in-place completion that died early) stays current.
                const std::filesystem::path fgDir = root / "color_fg";
                std::optional<std::filesystem::file_time_type> manifestBefore;
                if (Manifest::Exists(fgDir)) {
                    std::error_code ec;
                    const auto t = std::filesystem::last_write_time(fgDir / Manifest::kFileName, ec);
                    if (!ec) manifestBefore = t;
                }
                std::optional<std::string> failure;
                nlohmann::json failDetails = nlohmann::json::object();
                if (!options.fgExecutable.empty()) {
                    // audit fg-2: DLSS-G (a driver DLL) in its own process - its crash ends only that process
                    FgProcessSpec spec;
                    spec.exe = options.fgExecutable;
                    spec.input = options.input;
                    spec.passesRoot = root;
                    spec.colorDir = o.colorDir;
                    spec.depthDir = o.depthDir;
                    spec.mvDir = o.mvDir;
                    spec.params = p;
                    spec.format = Get<std::string>(p, "format", "exr");
                    spec.sourceHash = sourceHash;
                    spec.frames = options.frames;
                    spec.warp = options.warp;
                    spec.disableUnavailable = options.disableUnavailable;
                    spec.verbose = Log()->should_log(spdlog::level::debug);
                    spec.crashAfter = options.fgCrashAfter;
                    spec.crashOnExit = options.fgCrashOnExit;
                    spec.stall = options.fgStall;
                    const FgProcessOutcome c = RunFgProcess(spec, progressFor(name));
                    const nlohmann::json& res = c.result;
                    const nlohmann::json stats = res.contains("stats") && res["stats"].is_object() ? res["stats"] : nlohmann::json::object();
                    if (c.ok) {
                        if (c.disabled) {
                            rep.status = "disabled";
                            rep.reason = c.disabledReason;
                        } else {
                            rep.frames = stats.value("frames", int64_t{0});
                            rep.outDir = fgDir;
                            childMsPerFrame = stats.value("ms_per_frame_wall", 0.0);
                            childSeconds = res.value("seconds", 0.0);
                        }
                        rep.details = {{"backend", res.value("backend", nlohmann::json::object())},
                                       {"diagnostics", res.value("diagnostics", nlohmann::json::object())},
                                       {"generated", stats.value("generated", int64_t{0})},
                                       {"isolated", true},
                                       {"pid", c.pid}};
                        if (!c.warning.empty()) {
                            rep.details["warning"] = c.warning;
                            result.warnings.push_back("fg: " + c.warning);
                            Log()->warn("process: fg — {}", c.warning);
                        }
                    } else {
                        failure = c.error;
                        failDetails = {{"isolated", true},          {"pid", c.pid},         {"exit_code", c.exitCode}, {"exit", DescribeExitCode(c.exitCode)},
                                       {"crashed", c.crashed},      {"frames_done", c.framesDone}, {"error", c.error}};
                    }
                } else {
                    try {
                        VideoDecoder decoder(options.input);
                        const FgRunResult r = RunFg(decoder, device, o, options.frames, progressFor(name), p);
                        if (r.stats.disabled) {
                            rep.status = "disabled";
                            rep.reason = r.stats.disabledReason;
                        } else {
                            rep.frames = r.stats.frames;
                            rep.outDir = r.outDir;
                        }
                        rep.details = {{"backend", r.backend}, {"diagnostics", r.diagnostics}, {"generated", r.stats.generated}, {"isolated", false}};
                    } catch (const std::exception& e) {
                        failure = e.what();
                        failDetails = {{"isolated", false}, {"error", *failure}};
                    }
                }
                if (failure) {
                    // Debris: no manifest (the process died before Finish), a manifest this run wrote (no fingerprint, or
                    // not the one that was there) - it must not stay the current color_fg nor go to the history.
                    bool debris = std::filesystem::exists(fgDir);
                    if (debris && manifestBefore && Manifest::Exists(fgDir)) {
                        std::error_code ec;
                        const auto now = std::filesystem::last_write_time(fgDir / Manifest::kFileName, ec);
                        try {
                            debris = ec || now != *manifestBefore || Manifest::Load(fgDir).fingerprint.empty();
                        } catch (const std::exception&) {
                            debris = true;
                        }
                        // a rerun in place (forced, incomplete) rewrites the frames of the signed version: one written
                        // by this run (possibly cut short by the crash) makes the folder debris as well
                        for (auto entry = std::filesystem::directory_iterator(fgDir, ec); !debris && !ec && entry != std::filesystem::directory_iterator(); entry.increment(ec)) {
                            std::error_code tec;
                            const auto t = std::filesystem::last_write_time(entry->path(), tec);
                            if (!tec && t > *manifestBefore) debris = true;
                        }
                    }
                    if (debris) {
                        const uint64_t bytes = DirectoryBytes(fgDir);
                        std::error_code ec;
                        std::filesystem::remove_all(fgDir, ec);
                        if (ec) Log()->warn("process: cannot remove the partial color_fg {}: {}", fgDir.string(), ec.message());
                        failDetails["partial"] = ec ? "kept" : "removed";
                        failDetails["removed_bytes"] = ec ? 0 : bytes;
                    } else if (std::filesystem::exists(fgDir)) {
                        failDetails["partial"] = "kept";  // the previous signed version, untouched by this run
                    }
                    if (!options.disableUnavailable) Throw(*failure);  // -> "process: stage 'fg' failed: ..."
                    rep.status = "failed";
                    rep.reason = *failure;
                    rep.details = failDetails;
                    const std::string from = fgColorDir.empty() ? "the source" : fgColorDir.filename().string();
                    result.warnings.push_back("fg failed: " + *failure + "; the result is encoded from " + from + " without frame generation");
                    Log()->warn("process: fg failed — {}; the result is encoded from {}", *failure, from);  // TaskQueue parses this line
                }
                fgSkipped = rep.status == "failed" || rep.status == "disabled";
            }
        } catch (const std::exception& e) {
            Throw("process: stage '" + name + "' failed: " + e.what());
        }
        rep.seconds = Seconds(t0);
        rep.msPerFrame = rep.frames ? 1000.0 * rep.seconds / static_cast<double>(rep.frames) : 0.0;
        if (childMsPerFrame > 0.0) {
            // the fg process timed itself: the estimates (Estimates.cpp) must not count its start-up per frame
            rep.msPerFrame = childMsPerFrame;
            rep.details["startup_seconds"] = std::max(0.0, rep.seconds - childSeconds);
        }
        if (rep.status.empty()) {
            rep.status = "ran";
            rep.reason = d.kind + ": " + d.detail;
            const std::string created = NowIso8601();
            for (const auto& pass : StageFamilyPasses(name))
                if (Manifest::Exists(root / pass)) StampPass(root / pass, d, created, rep.msPerFrame);
            state.ran.insert(d.outputs.begin(), d.outputs.end());
            if (d.stale) state.staleRan.insert(d.outputs.begin(), d.outputs.end());
        } else {
            rep.fingerprint.clear();  // disabled / failed: no pass of this run
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
    // fg failed or disabled: its input, not a color_fg left from an older run (an incomplete one would cut the result short)
    result.finalPass = fgSkipped ? fgColorDir : ExistingPass(root, {"color_fg", "color_nr", "color_sr"});
    ProcessStageReport enc;
    enc.name = "encode";
    if (fgSkipped) enc.reason = "fg skipped: encoded from " + (result.finalPass.empty() ? std::string("the source") : result.finalPass.filename().string());
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
        result.codec = r.codec;
        result.outputFormat = r.outputFormat;
        result.outputPixelFormat = r.outputPixelFormat;
    } else {
        EncodeSettings es;
        es.codec = options.codec;
        es.codecOptions = options.codecOptions;
        es.format = options.outputFormat;
        const EncodeReport r = EncodePassToVideo(result.finalPass, options.input, options.output, es, options.frames,
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
        result.codec = options.codec;
        result.outputFormat = r.format;
        result.outputPixelFormat = r.pixelFormat;
    }
    enc.details = {{"codec", result.codec}, {"pix_fmt", result.outputPixelFormat}, {"format", result.outputFormat.Describe()}};
    enc.seconds = Seconds(t0);
    enc.msPerFrame = enc.frames ? 1000.0 * enc.seconds / static_cast<double>(enc.frames) : 0.0;
    result.stages.push_back(enc);
    result.seconds = Seconds(tAll);
    Log()->info("process: {} -> {} ({} frames at {:.3f} fps, {} {}{}) in {:.1f} s", options.input.filename().string(), options.output.string(), result.framesOut,
                result.outputFps.ToDouble(), result.codec, result.outputFormat.Describe(), result.audio ? ", audio copied" : "", result.seconds);
    return result;
}

}  // namespace dlssvid
