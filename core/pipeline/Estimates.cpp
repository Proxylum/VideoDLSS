#include "pipeline/Estimates.h"

#include <algorithm>
#include <cmath>

#include "passes/Manifest.h"
#include "pipeline/PassFingerprint.h"
#include "stages/ParamSchema.h"
#include "stages/fg/IFrameGenerator.h"

namespace dlssvid {

namespace {

constexpr double kReferenceMegapixels = 1920.0 * 800.0 / 1e6;  // the face clip of docs/benchmarks.md
constexpr double kReferenceOutputMegapixels = 3840.0 * 1600.0 / 1e6;

double Baseline(const std::string& stage, const std::string& backend) {
    if (stage == "depth") return backend == "vda" ? 300.0 : backend == "worker" ? 600.0 : backend == "stub" ? 5.0 : 350.0;
    if (stage == "flow") return backend == "searaft" ? 580.0 : backend == "stub" ? 5.0 : 500.0;
    if (stage == "upscale") return backend == "nis" ? 260.0 : backend == "bicubic" ? 250.0 : 420.0;
    if (stage == "nr") return backend == "stub" ? 30.0 : 480.0;
    if (stage == "fg") return backend == "rife" ? 900.0 : backend == "blend" ? 400.0 : 780.0;
    if (stage == "encode") return 150.0;
    return 0.0;
}

double Scale(const nlohmann::json& p) {
    const double s = p.value("scale", 2.0);
    return s > 0 ? s : 1.0;
}

}  // namespace

double BaselineMsPerFrame(const std::string& stage, const std::string& backend, double sourceMegapixels) {
    const double ref = stage == "encode" ? kReferenceOutputMegapixels : kReferenceMegapixels;
    const double factor = sourceMegapixels > 0 ? sourceMegapixels / ref : 1.0;
    return Baseline(stage, backend) * factor;
}

std::optional<double> RecordedMsPerFrame(const std::filesystem::path& passesRoot, const std::string& stage) {
    const auto& family = StageFamilyPasses(stage);
    for (auto it = family.rbegin(); it != family.rend(); ++it) {
        const auto dir = passesRoot / *it;
        if (!Manifest::Exists(dir)) continue;
        try {
            const Manifest m = Manifest::Load(dir);
            if (m.stageParams.contains("ms_per_frame") && m.stageParams["ms_per_frame"].is_number()) {
                const double ms = m.stageParams["ms_per_frame"].get<double>();
                if (ms > 0) return ms;
            }
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
        if (const auto recorded = RecordedMsPerFrame(plan.passesRoot, d.name)) {
            e.msPerFrame = *recorded;
            e.recorded = true;
        } else {
            e.msPerFrame = BaselineMsPerFrame(d.name, backend, megapixels);
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
