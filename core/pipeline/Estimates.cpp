#include "pipeline/Estimates.h"

#include <algorithm>
#include <cmath>

#include "passes/Manifest.h"
#include "pipeline/PassFingerprint.h"
#include "stages/ParamSchema.h"
#include "stages/fg/IFrameGenerator.h"
#include "stages/nr/NrStage.h"

namespace dlssvid {

namespace {

constexpr double kReferenceMegapixels = 1920.0 * 800.0 / 1e6;  // the face clip of docs/benchmarks.md
constexpr double kReferenceOutputMegapixels = 3840.0 * 1600.0 / 1e6;
constexpr double kVdaBaseMs = 310.0;   // TASK-0039, 2026-10-12: face 1920x800 (910x378), window 32: 306 ms/frame
constexpr double kVdaLargeMs = 420.0;  // the same: 416 ms/frame

double Baseline(const std::string& stage, const std::string& backend, const std::string& model) {
    if (stage == "depth") {
        // VDA by model: Small measured (docs/benchmarks.md); Base and Large - TASK-0039 measurement on the reference GPU
        const double vda = model == "metric-vda-large" ? kVdaLargeMs : model == "metric-vda-base" ? kVdaBaseMs : 300.0;
        if (backend == "vda") return vda;
        if (backend == "worker:vda") return vda * 1.4;  // PyTorch fp16 + SDPA against TensorRT: 1.3x (Small), 1.45x (Large)
        return backend == "worker" ? 600.0 : backend == "stub" ? 5.0 : 350.0;
    }
    if (stage == "flow") return backend == "searaft" ? 580.0 : backend == "stub" ? 5.0 : 500.0;
    if (stage == "upscale") return backend == "nis" ? 260.0 : backend == "bicubic" ? 250.0 : backend == "trt" ? 900.0 : backend == "worker" ? 1500.0 : 420.0;
    if (stage == "nr") return backend == "stub" ? 30.0 : 480.0;
    if (stage == "fg") return backend == "rife" ? 900.0 : backend == "blend" ? 400.0 : 780.0;
    if (stage == "encode") return 150.0;
    return 0.0;
}

// PR-2 E6 (docs/benchmarks.md «Результаты A/B на NGX», Ada RTX 4070 Ti SUPER): the GPU step of `passes 2` over `passes 1`,
// per 2560x1440 of work pixels - 5.7 ms at 2560x1440, 16.7 at 3840x2160 (7.4), 10.3 at 3840x1600 on 120 frames (6.2); the
// September 17.6 ms was one 12-frame run
constexpr double kNrModelMsPerPass = 6.4;
constexpr double kNrModelPixels = 2560.0 * 1440.0;
// One more NR feature in a run (PR-2 E6f / E8): with NGX already up its Init is ~1 ms and CreateFeature 110-120 ms at 4K
// (the first feature of a process: init ~1.4 s, create 0.2-0.3 s - every run pays that once); plus its output texture and the
// first Evaluate: ~0.2 s
constexpr double kNrFeatureCreateMs = 200.0;

bool HasNumber(const nlohmann::json& j, const char* key) { return j.is_object() && j.contains(key) && j[key].is_number(); }

// The wall time a `process` run recorded (StampPass), never a stage's own timer (D8, see RecordedMsPerFrame)
std::optional<double> WallMsPerFrame(const Manifest& m, const std::string& stage) {
    const nlohmann::json& sp = m.stageParams;
    if (!HasNumber(sp, "ms_per_frame")) return std::nullopt;
    const double ms = sp["ms_per_frame"].get<double>();
    const bool gpu = HasNumber(sp, "ms_per_frame_gpu");
    if (ms <= 0 || (stage == "nr" && !gpu) || (gpu && ms < sp["ms_per_frame_gpu"].get<double>())) return std::nullopt;
    return ms;
}

// nr: the record moved to the run's passes and model_scale. The run's model resolution is the recorded frame at its
// model_scale, as NrStage::Setup computes it. A changed frame (another upscale) is not compared, as for every stage.
double NrMovedRecord(const Manifest& m, double wall, const nlohmann::json& params, int64_t frames) {
    const nlohmann::json& sp = m.stageParams;
    const int passes0 = HasNumber(sp, "passes") ? sp["passes"].get<int>() : 1;
    auto [w0, h0] = NrWorkSize(m.width, m.height, HasNumber(sp, "model_scale") ? sp["model_scale"].get<double>() : 1.0);
    if (sp.contains("work_size") && sp["work_size"].is_array() && sp["work_size"].size() == 2) {
        w0 = sp["work_size"][0].get<uint32_t>();
        h0 = sp["work_size"][1].get<uint32_t>();
    }
    const int passes1 = HasNumber(params, "passes") ? params["passes"].get<int>() : passes0;
    const auto [w1, h1] = HasNumber(params, "model_scale") ? NrWorkSize(m.width, m.height, params["model_scale"].get<double>()) : std::pair{w0, h0};
    if (passes1 == passes0 && w1 == w0 && h1 == h0) return wall;  // the same run: the record as it is
    return NrMovedMsPerFrame(wall, passes0, static_cast<double>(w0) * h0, passes1, static_cast<double>(w1) * h1, frames > 0 ? frames : m.frameCount,
                             sp.value("backend", std::string("ngx")));
}

double Scale(const nlohmann::json& p) {
    const double s = p.value("scale", 2.0);
    return s > 0 ? s : 1.0;
}

}  // namespace

double BaselineMsPerFrame(const std::string& stage, const std::string& backend, double sourceMegapixels, const std::string& model) {
    const double ref = stage == "encode" ? kReferenceOutputMegapixels : kReferenceMegapixels;
    const double factor = sourceMegapixels > 0 ? sourceMegapixels / ref : 1.0;
    return Baseline(stage, backend, model) * factor;
}

double NrModelMsPerPass(const std::string& backend, double workPixels) {
    return backend == "stub" || workPixels <= 0 ? 0.0 : kNrModelMsPerPass * workPixels / kNrModelPixels;  // the stub edit: ~0.1 ms
}

double NrMovedMsPerFrame(double recordedMs, int recordedPasses, double recordedWorkPixels, int passes, double workPixels, int64_t frames, const std::string& backend) {
    const double perPass = NrModelMsPerPass(backend, workPixels);
    double ms = recordedMs - recordedPasses * NrModelMsPerPass(backend, recordedWorkPixels) + passes * perPass;
    // each added pass creates its feature once; a removed one is not taken back: the record spread it over its own frames
    if (passes > recordedPasses && frames > 0 && backend != "stub") ms += (passes - recordedPasses) * kNrFeatureCreateMs / static_cast<double>(frames);
    return std::max(ms, passes * perPass);  // a wall time is never below the GPU time of its passes
}

std::optional<double> RecordedMsPerFrame(const std::filesystem::path& passesRoot, const std::string& stage, const nlohmann::json& tool, const nlohmann::json& params,
                                         int64_t frames) {
    const auto& family = StageFamilyPasses(stage);
    for (auto it = family.rbegin(); it != family.rend(); ++it) {
        const auto dir = passesRoot / *it;
        if (!Manifest::Exists(dir)) continue;
        try {
            const Manifest m = Manifest::Load(dir);
            bool other = false;
            for (const char* field : {"backend", "model"}) {
                const std::string want = tool.is_object() ? tool.value(field, std::string()) : std::string();
                const std::string made = m.tool.is_object() ? m.tool.value(field, std::string()) : std::string();
                other = other || (!want.empty() && !made.empty() && want != made);
            }
            if (other) continue;  // another backend / model: its time says nothing about this one
            if (const auto wall = WallMsPerFrame(m, stage)) return stage == "nr" ? NrMovedRecord(m, *wall, params, frames) : *wall;
        } catch (...) {
        }
    }
    return std::nullopt;
}

uint64_t BytesPerFrame(const std::string& stage, uint32_t sourceW, uint32_t sourceH, double scale, int multiplier) {
    const uint64_t src = static_cast<uint64_t>(sourceW) * sourceH;
    const uint64_t out = static_cast<uint64_t>(std::llround(sourceW * scale)) * static_cast<uint64_t>(std::llround(sourceH * scale));
    if (stage == "depth") return src * 4 * 2;           // depth_raw + depth_dlss, R32F
    if (stage == "flow") return src * 8 * 2;            // mv_raw + mv_dlss, RG32F
    if (stage == "upscale" || stage == "nr") return out * 8;  // RGBA16F
    if (stage == "fg") return out * 8 * static_cast<uint64_t>(std::max(1, multiplier));
    return 0;
}

RunOutcome EstimateRun(const ProcessPlan& plan, const std::vector<ProcessStage>& stages, const SourceInfo& source) {
    RunOutcome o;
    o.width = source.width;
    o.height = source.height;
    o.fps = source.fps;
    o.audio = source.audio;
    const double megapixels = static_cast<double>(source.width) * source.height / 1e6;
    double scale = 1.0;
    int multiplier = 1;
    nlohmann::json params[5];
    const char* order[] = {"depth", "flow", "upscale", "nr", "fg"};
    for (const auto& s : stages) {
        if (!s.enabled) continue;
        const nlohmann::json p = EffectiveStageParams(s.name, s.params);
        for (int i = 0; i < 5; ++i)
            if (s.name == order[i]) params[i] = p;
        if (s.name == "upscale") scale = Scale(p);
        if (s.name == "fg") multiplier = std::clamp(p.value("multiplier", 2), 1, 4);
    }
    if (!params[2].is_null()) {
        o.width = static_cast<uint32_t>(std::llround(source.width * scale));
        o.height = static_cast<uint32_t>(std::llround(source.height * scale));
    }
    if (!params[4].is_null()) o.fps = source.fps * multiplier;
    const int64_t frames = std::max<int64_t>(0, source.frames);
    for (const StageDecision& d : plan.stages) {
        StageEstimate e;
        e.stage = d.name;
        e.action = d.action;
        const nlohmann::json p = EffectiveStageParams(d.name, d.params);
        const std::string backend = p.value("backend", std::string());
        if (const auto recorded = RecordedMsPerFrame(plan.passesRoot, d.name, d.tool, p, frames)) {
            e.msPerFrame = *recorded;
            e.recorded = true;
        } else {
            e.msPerFrame = BaselineMsPerFrame(d.name, backend, megapixels, d.tool.is_object() ? d.tool.value("model", std::string()) : std::string());
            if (d.name == "nr" && o.width && o.height) {
                // the baseline is one model pass at the colour's resolution (docs/benchmarks.md): the run's passes and model_scale
                // move it (D8)
                const auto [ww, wh] = NrWorkSize(o.width, o.height, p.value("model_scale", 1.0));
                e.msPerFrame = NrMovedMsPerFrame(e.msPerFrame, 1, static_cast<double>(o.width) * o.height, p.value("passes", 1), static_cast<double>(ww) * wh, frames,
                                                 backend);
            }
        }
        e.frames = d.name == "fg" ? FgFrameCount(frames, multiplier) : frames;
        const double seconds = e.msPerFrame * static_cast<double>(frames) / 1000.0;  // the baselines are per source frame
        o.fullSeconds += seconds;
        ++o.stagesTotal;
        if (d.action == "run") {
            e.seconds = seconds;
            e.bytes = BytesPerFrame(d.name, source.width, source.height, d.name == "depth" || d.name == "flow" ? 1.0 : scale, multiplier) * static_cast<uint64_t>(frames);
            o.seconds += seconds;
            o.newBytes += e.bytes;
            ++o.stagesToRun;
        } else {
            o.reused.push_back(d.name);
        }
        o.stages.push_back(e);
    }
    // the encode always runs, over the output frames
    StageEstimate enc;
    enc.stage = "encode";
    enc.action = "run";
    enc.frames = static_cast<int64_t>(std::llround(static_cast<double>(frames) * (o.fps > 0 && source.fps > 0 ? o.fps / source.fps : 1.0)));
    enc.msPerFrame = BaselineMsPerFrame("encode", "", static_cast<double>(o.width) * o.height / 1e6);
    enc.seconds = enc.msPerFrame * static_cast<double>(enc.frames) / 1000.0;
    o.seconds += enc.seconds;
    o.fullSeconds += enc.seconds;
    ++o.stagesTotal;
    ++o.stagesToRun;
    o.stages.push_back(enc);
    return o;
}

}  // namespace dlssvid
