#include "viewport/PipelinePreset.h"

#include <cmath>

#include "io/EncodeDefaults.h"
#include "pipeline/PassFingerprint.h"

namespace dlssvid {

namespace {

bool TakesBitrate(const std::string& codec) { return LossyCodec(codec) && HasBitrateControl(codec); }

const StageEntry* FindStage(const Project& p, const std::string& name) {
    for (const auto& s : p.stages)
        if (s.name == name) return &s;
    return nullptr;
}

// a stage the project lacks, as its card shows it: off, the built-in defaults
StageEntry AsCardShows(const Project& p, const std::string& name) {
    if (const StageEntry* e = FindStage(p, name)) return *e;
    for (StageEntry e : Project::DefaultStages())
        if (e.name == name) {
            e.enabled = false;
            return e;
        }
    return StageEntry{name, false, nlohmann::json::object()};
}

}  // namespace

PresetEncode EncodeOf(const Project& p) {
    PresetEncode e;
    e.codec = p.codec;
    e.format = p.outputFormat;
    if (const auto b = p.codecOptions.find("b"); b != p.codecOptions.end() && TakesBitrate(p.codec)) {
        // the form the reader gives ("50000k" -> "50M", CheckEncodeSection): a project's own preset then compares equal to it after a
        // save and a read; a value the card cannot hold stays as it is for the save to refuse and a slot to keep
        e.bitrate = b->second;
        if (const auto mbps = ParseBitrateMbps(b->second)) {
            const double whole = std::round(*mbps);
            if (std::fabs(*mbps - whole) <= 0.01 && whole >= 1 && whole <= 400) e.bitrate = BitrateOption(whole);
        }
    }
    return e;
}

Preset MakePipelinePreset(const std::string& name, const Project& p) {
    Preset out;
    out.name = TrimPresetName(name);
    out.created = NowIso8601();
    out.app = DLSSVID_VERSION;
    out.scope = PresetScope::Pipeline;
    for (const auto& s : PipelineStageNames()) {
        const StageEntry e = AsCardShows(p, s);
        PresetStage& ps = out.stages[s];
        ps.enabled = e.enabled;
        ps.params = PresetParams(s, e.params, PresetScope::Pipeline);
    }
    out.encode = EncodeOf(p);
    return out;
}

void ApplyPipelinePreset(Project& p, const Preset& preset, std::vector<PresetIssue>* notes) {
    for (const auto& s : PipelineStageNames()) {
        const auto it = preset.stages.find(s);
        if (it == preset.stages.end()) continue;  // the stage stays as it is
        StageEntry* e = p.EnsureStage(s);
        if (!e) continue;
        std::vector<std::string> ignored;
        e->params = ApplyPreset(s, e->params, it->second.params, preset.scope, &ignored);
        if (preset.scope == PresetScope::Pipeline) e->enabled = it->second.enabled;
        if (notes)
            for (const auto& k : ignored) {
                PresetIssue i;
                i.kind = PresetIssueKind::ForeignKey;
                i.stage = s;
                i.key = k;
                i.got = it->second.params.contains(k) ? it->second.params[k] : nlohmann::json();
                i.to = StageBackend(s, e->params);
                const ParamSpec* spec = FindStageSchema(s) ? FindStageSchema(s)->Find(k) : nullptr;
                i.text = spec && IsListlessKey(*spec, i.to)
                             ? s + "." + k + " has no list for the backend " + i.to + " - presets do not set it, the project's value stays"
                             : s + "." + k + " does not apply to the backend " + i.to + " - skipped";
                notes->push_back(std::move(i));
            }
    }
    if (preset.scope != PresetScope::Pipeline || !preset.encode) return;  // no encode: the project's encoder stays
    const PresetEncode& enc = *preset.encode;
    p.codec = enc.codec;
    p.outputFormat = enc.format;
    p.codecOptions.erase("b");  // «Авто»; the other options stay
    if (!enc.bitrate.empty()) p.codecOptions["b"] = enc.bitrate;
    p.resultVideo = ResultPathFor(p.resultVideo, enc.codec);  // .mp4 -> .mov for ProRes; an empty path stays empty
}

PipelineDiff DiffPipeline(const Project& before, const Project& after) {
    PipelineDiff d;
    for (const auto& s : PipelineStageNames()) {
        const StageEntry b = AsCardShows(before, s), a = AsCardShows(after, s);
        StageDiff sd;
        sd.stage = s;
        sd.enabledWas = b.enabled;
        sd.enabledNow = a.enabled;
        sd.params = DiffStageParams(s, b.params, a.params, PresetScope::Pipeline);
        if (!sd.empty()) d.stages.push_back(std::move(sd));
    }
    const PresetEncode eb = EncodeOf(before), ea = EncodeOf(after);
    auto field = [&](const char* name, const std::string& was, const std::string& now) {
        if (was != now) d.encode.push_back({name, was, now});
    };
    field("codec", eb.codec, ea.codec);
    field("bit_depth", BitDepthChoiceName(eb.format.bitDepth), BitDepthChoiceName(ea.format.bitDepth));
    field("chroma", eb.format.chroma, ea.format.chroma);
    field("bitrate", eb.bitrate, ea.bitrate);
    return d;
}

std::vector<LeftOut> PipelinePresetLeftOut(const Project& p) {
    std::vector<LeftOut> out;
    for (const auto& s : PipelineStageNames())
        if (const StageEntry* e = FindStage(p, s))
            for (LeftOut& l : StagePresetLeftOut(s, e->params, PresetScope::Pipeline)) out.push_back(std::move(l));
    for (const auto& [k, v] : p.codecOptions)
        if (k != "b" || !TakesBitrate(p.codec)) out.push_back({LeftOut::Why::CodecOption, "encode", k, v});
    if (!p.settingsNote) out.push_back({LeftOut::Why::SettingsNote, "encode", "settings_note", false});
    return out;
}

}  // namespace dlssvid
