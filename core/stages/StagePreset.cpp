#include "stages/StagePreset.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

#include "io/EncodeDefaults.h"
#include "pipeline/PassFingerprint.h"

namespace dlssvid {

namespace {

using J = nlohmann::json;
using Kind = PresetIssueKind;

// a list key without a list for the backend has no row in the card (StageCard: the form neither shows nor counts it)
bool NoRow(const ParamSpec& spec, const std::string& backend) { return !spec.formChoices.empty() && !FormChoicesFor(spec, backend); }

bool OfBackend(const ParamSpec& spec, const std::string& backend) {
    return spec.backends.empty() || std::find(spec.backends.begin(), spec.backends.end(), backend) != spec.backends.end();
}

bool SameValue(const J& a, const J& b) {
    if (a.is_number() && b.is_number()) return std::fabs(a.get<double>() - b.get<double>()) <= 1e-9;
    return a == b;
}

bool IsPathKey(const std::string& key) { return key == "models_dir" || key == "python"; }

size_t CodePoints(const std::string& s) {
    size_t n = 0;
    for (const unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

std::string Join(const std::vector<std::string>& v, const char* sep = ", ") {
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : sep) + x;
    return s;
}

std::string Brief(const J& v) { return v.is_string() ? v.get<std::string>() : v.dump(); }

// The parameters a preset keeps over the current ones: unknown keys, the backend, keys outside the scope's class and a list key
// without a list for the backend `t`.
J Keep(const std::string& stage, const J& cur, PresetScope scope, const std::string& t) {
    const StageSchema* schema = FindStageSchema(stage);
    J out = J::object();
    for (const auto& [key, value] : cur.items()) {
        const ParamSpec* spec = schema ? schema->Find(key) : nullptr;
        if (!spec || key == "backend" || !PresetKeyClass(*spec, scope) || NoRow(*spec, t)) out[key] = value;
    }
    return out;
}

// the file of a preset or of a settings object: size, read, parse (the texts without the file's name - the callers add it)
struct JsonFile {
    J json;
    std::optional<PresetIssue> issue;  // Unreadable | TooLarge | NotJson
};
JsonFile ReadJsonFile(const std::filesystem::path& file) {
    JsonFile f;
    auto fail = [&](Kind k, std::string text) {
        PresetIssue i;
        i.kind = k;
        i.text = std::move(text);
        f.issue = i;
        return f;
    };
    std::error_code ec;
    const auto size = std::filesystem::file_size(file, ec);
    if (ec) return fail(Kind::Unreadable, "cannot read the file");
    if (size > kPresetMaxBytes) return fail(Kind::TooLarge, "larger than 1 MiB - not a settings file");
    std::ifstream in(file, std::ios::binary);
    if (!in) return fail(Kind::Unreadable, "cannot read the file");
    std::stringstream text;
    text << in.rdbuf();
    try {
        // depth-limited: the parser is iterative, but copying and dumping a value are recursive - a file nested 100 000 deep (in
        // 1 MiB) would overflow the stack when the menu lists it (review of PR-9)
        f.json = J::parse(text.str(), [](int depth, J::parse_event_t, J&) {
            if (depth > kPresetMaxDepth) throw std::runtime_error("nested deeper than " + std::to_string(kPresetMaxDepth) + " levels");
            return true;
        });
    } catch (const std::exception& e) {
        return fail(Kind::NotJson, std::string("not valid JSON (") + e.what() + ")");
    }
    return f;
}

}  // namespace

const char* PresetScopeName(PresetScope s) { return s == PresetScope::Pipeline ? "pipeline" : "stage"; }

const std::string& Preset::stage() const {
    static const std::string none;
    return stages.empty() ? none : stages.begin()->first;
}

bool PresetIssue::fatal() const { return kind <= Kind::NoStage; }

bool PresetLoad::ok() const {
    return std::none_of(issues.begin(), issues.end(), [](const PresetIssue& i) { return i.fatal(); });
}

bool PresetLoad::Has(PresetIssueKind kind) const {
    return std::any_of(issues.begin(), issues.end(), [&](const PresetIssue& i) { return i.kind == kind; });
}

std::string PresetLoad::problemsText() const {
    std::vector<std::string> texts;
    for (const auto& i : issues)
        if (i.fatal()) texts.push_back(i.text);
    return Join(texts, "; ");
}

std::string PathText(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

bool PresetKeyClass(const ParamSpec& spec, PresetScope scope) { return !spec.advanced || (scope == PresetScope::Pipeline && spec.tuning); }

bool IsPresetKey(const ParamSpec& spec, const std::string& backend, PresetScope scope) {
    return PresetKeyClass(spec, scope) && OfBackend(spec, backend) && !NoRow(spec, backend);
}

std::vector<std::string> PresetKeys(const std::string& stage, const std::string& backend, PresetScope scope) {
    std::vector<std::string> keys;
    if (const StageSchema* schema = FindStageSchema(stage))
        for (const ParamSpec& p : schema->params)
            if (IsPresetKey(p, backend, scope)) keys.push_back(p.key);
    return keys;
}

bool CardResetKey(const ParamSpec& spec) { return !spec.advanced && spec.key != "backend"; }

bool IsMachineKey(const ParamSpec& spec) {
    static const std::set<std::string> machine = {"models_dir", "python", "max_res", "input_size", "tile", "fp32"};
    return spec.advanced && !spec.tuning && machine.count(spec.key) > 0;
}

nlohmann::json PresetParams(const std::string& stage, const nlohmann::json& current, PresetScope scope) {
    const J eff = EffectiveStageParams(stage, current.is_object() ? current : J::object());
    J out = J::object();
    for (const std::string& k : PresetKeys(stage, StageBackend(stage, eff), scope))
        if (eff.contains(k)) out[k] = CanonicalizeJson(eff[k]);  // an omitDefault key at its default is not in eff
    return out;
}

nlohmann::json ApplyPreset(const std::string& stage, const nlohmann::json& current, const nlohmann::json& presetParams, PresetScope scope,
                           std::vector<std::string>* ignored) {
    J cur = current.is_object() ? NormalizeStageParams(stage, current) : J::object();
    const J P = presetParams.is_object() ? NormalizeStageParams(stage, presetParams) : J::object();
    const StageSchema* schema = FindStageSchema(stage);
    const std::string b = StageBackend(stage, cur);
    std::string t = b;
    if (schema && P.contains("backend") && P["backend"].is_string())
        if (const ParamSpec* spec = schema->Find("backend"); spec && PresetKeyClass(*spec, scope)) t = P["backend"].get<std::string>();
    if (t != b) {  // before the merge: ParamsForBackendChange takes the old backend from the parameters themselves
        cur = ParamsForBackendChange(stage, cur, t);
        cur["backend"] = t;
    }
    J out = Keep(stage, cur, scope, t);
    for (const auto& [key, value] : P.items()) {
        const ParamSpec* spec = schema ? schema->Find(key) : nullptr;
        if (!spec || !IsPresetKey(*spec, t, scope)) {
            if (ignored) ignored->push_back(key);
            continue;
        }
        if (key != "backend" && SameValue(value, spec->def)) {
            out.erase(key);  // sparse: the default runs the same and the JSON editor does not grow
            continue;
        }
        out[key] = value;
    }
    return out;
}

nlohmann::json ResetStageParams(const std::string& stage, const nlohmann::json& current) {
    const J cur = current.is_object() ? NormalizeStageParams(stage, current) : J::object();
    return Keep(stage, cur, PresetScope::Stage, StageBackend(stage, cur));
}

std::vector<ParamChange> DiffStageParams(const std::string& stage, const nlohmann::json& before, const nlohmann::json& after, PresetScope scope) {
    std::vector<ParamChange> out;
    const StageSchema* schema = FindStageSchema(stage);
    if (!schema) return out;
    const J eb = EffectiveStageParams(stage, before.is_object() ? before : J::object());
    const J ea = EffectiveStageParams(stage, after.is_object() ? after : J::object());
    const std::string wb = StageBackend(stage, eb), wa = StageBackend(stage, ea);
    for (const ParamSpec& p : schema->params) {
        if (!PresetKeyClass(p, scope)) continue;
        const J vb = eb.contains(p.key) ? eb[p.key] : J(), va = ea.contains(p.key) ? ea[p.key] : J();
        if (!SameValue(vb, va)) out.push_back({p.key, vb, va, wb, wa});
    }
    return out;
}

std::string TrimPresetName(const std::string& name) {
    const size_t a = name.find_first_not_of(' ');
    if (a == std::string::npos) return {};
    return name.substr(a, name.find_last_not_of(' ') - a + 1);
}

std::string PresetNameProblem(const std::string& raw) {
    const std::string name = TrimPresetName(raw);
    const size_t n = CodePoints(name);
    if (n == 0 || n > kPresetNameMax) return "name must be 1.." + std::to_string(kPresetNameMax) + " characters (got " + std::to_string(n) + ")";
    for (const unsigned char c : name)
        if (c < 0x20 || c == 0x7F) return "name must not hold control characters";
    return {};
}

std::vector<PresetIssue> CheckEncodeSection(const nlohmann::json& encode, PresetEncode* out, PresetRead read) {
    std::vector<PresetIssue> issues;
    PresetEncode e;
    auto add = [&](Kind k, const std::string& key, const J& got, std::string text) -> PresetIssue& {
        PresetIssue i;
        i.kind = k;
        i.stage = "encode";
        i.key = key;
        i.got = got;
        i.codec = e.codec;
        i.text = std::move(text);
        issues.push_back(std::move(i));
        return issues.back();
    };
    // a value the card cannot hold: refused, or in a slot kept with a note (the project held it)
    auto bad = [&](const std::string& key, const J& got, const std::string& text) {
        if (read == PresetRead::Slot) add(Kind::SlotValue, key, got, "the slot keeps a value with an error: " + text);
        else add(Kind::BadValue, key, got, text);
    };
    if (!encode.is_object()) {
        add(Kind::BadStages, "", encode, "encode must be an object (codec, bit_depth, chroma, options)");
        if (out) *out = e;
        return issues;
    }
    std::string codecs;
    for (const auto& c : KnownCodecs()) codecs += (codecs.empty() ? "" : " | ") + c.name;
    const CodecCaps* caps = nullptr;
    if (!encode.contains("codec")) {
        bad("codec", J(), "encode.codec is missing (" + codecs + ")");
    } else if (!encode["codec"].is_string()) {
        bad("codec", encode["codec"], "encode.codec must be a string (" + codecs + ")");
    } else {
        e.codec = encode["codec"].get<std::string>();
        caps = FindCodecCaps(e.codec);
        if (!caps) bad("codec", encode["codec"], "encode.codec '" + e.codec + "' is not one of " + codecs);
    }
    if (encode.contains("bit_depth")) {
        const J& v = encode["bit_depth"];
        std::optional<int> d = v.is_number_integer() ? std::optional<int>(v.get<int>()) : v.is_string() ? ParseBitDepthChoice(v.get<std::string>()) : std::nullopt;
        if (d && *d != 0 && *d != 8 && *d != 10 && *d != 12) d.reset();
        if (!d) {
            bad("bit_depth", v, "encode.bit_depth must be auto | 8 | 10 | 12 (got " + Brief(v) + ")");
        } else {
            e.format.bitDepth = *d;
            if (*d != 0 && caps && std::find(caps->bitDepths.begin(), caps->bitDepths.end(), *d) == caps->bitDepths.end()) {
                std::vector<std::string> takes;
                for (const int x : caps->bitDepths) takes.push_back(std::to_string(x));
                bad("bit_depth", v, "encode.bit_depth " + std::to_string(*d) + ": " + caps->title + " does not encode " + std::to_string(*d) + "-bit (it takes " + Join(takes, " | ") + ")");
            }
        }
    }
    if (encode.contains("chroma")) {
        const J& v = encode["chroma"];
        if (!v.is_string() || !ValidChromaChoice(v.get<std::string>())) {
            bad("chroma", v, "encode.chroma must be auto | 420 | 422 | 444 (got " + Brief(v) + ")");
        } else {
            e.format.chroma = v.get<std::string>();
            if (e.format.chroma != "auto" && caps && std::find(caps->chromas.begin(), caps->chromas.end(), e.format.chroma) == caps->chromas.end())
                bad("chroma", v, "encode.chroma " + e.format.chroma + ": " + caps->title + " does not encode it (it takes " + Join(caps->chromas, " | ") + ")");
        }
    }
    if (encode.contains("options")) {
        const J& opts = encode["options"];
        if (!opts.is_object()) {
            bad("options", opts, "encode.options must be an object ({\"b\": \"50M\"} or {})");
        } else {
            for (const auto& [k, v] : opts.items()) {
                if (k != "b") {
                    add(Kind::EncodeField, k, v, "encode.options." + k + " is not a preset setting (the card has no row for it) - dropped").to = "options";
                    continue;
                }
                if (caps && !(LossyCodec(e.codec) && HasBitrateControl(e.codec))) {
                    add(Kind::ForeignKey, "b", v, "encode.options.b: " + caps->title + " takes no bitrate - dropped");
                    continue;
                }
                const std::string s = v.is_string() ? v.get<std::string>() : v.is_number() ? v.dump() : std::string();
                const auto mbps = s.empty() ? std::nullopt : ParseBitrateMbps(s);
                if (!mbps) {
                    bad("b", v, "encode.options.b '" + Brief(v) + "' is not a bitrate (50M, 50000k)");
                    if (read == PresetRead::Slot) e.bitrate = s;
                    continue;
                }
                const double whole = std::round(*mbps);
                if (std::fabs(*mbps - whole) > 0.01 || whole < 1 || whole > 400) {
                    bad("b", v, "encode.options.b " + s + ": the card takes whole 1..400 Mbit/s");
                    if (read == PresetRead::Slot) e.bitrate = s;
                    continue;
                }
                e.bitrate = BitrateOption(whole);
            }
        }
    }
    for (const auto& [k, v] : encode.items())
        if (k != "codec" && k != "bit_depth" && k != "chroma" && k != "options")
            add(Kind::EncodeField, k, v, "encode." + k + " is not a preset setting (the card has no row for it) - dropped");
    if (out) *out = e;
    return issues;
}

PresetLoad ValidatePreset(const nlohmann::json& j, const std::string& forStage, PresetRead read) {
    PresetLoad L;
    auto add = [&](Kind k, std::string text) -> PresetIssue& {
        PresetIssue i;
        i.kind = k;
        i.text = std::move(text);
        L.issues.push_back(std::move(i));
        return L.issues.back();
    };
    if (!j.is_object()) {
        add(Kind::NotJson, "not a JSON object");
        return L;
    }
    if (!j.contains("format") || !j["format"].is_string() || j["format"].get<std::string>() != kPresetFormat) {
        add(Kind::NotPreset, std::string("not a dlssvid-preset (no \"format\": \"") + kPresetFormat + "\")");
        return L;
    }
    if (j.contains("name") && j["name"].is_string()) L.preset.name = TrimPresetName(j["name"].get<std::string>());  // also of a refused file
    const J schemaField = j.contains("schema") ? j["schema"] : J();
    if (!schemaField.is_number_integer() || schemaField.get<int64_t>() < 1) {
        add(Kind::BadSchema, "schema must be a whole number from 1 (this dlssvid reads " + std::to_string(kPresetSchemaVersion) + ")");
        return L;
    }
    if (schemaField.get<int64_t>() > kPresetSchemaVersion) {
        const int64_t n = schemaField.get<int64_t>();
        add(Kind::NewerSchema, "preset schema " + std::to_string(n) + " is newer than this dlssvid reads (" + std::to_string(kPresetSchemaVersion) +
                                   ") - update dlssvid")
            .schema = static_cast<int>(std::min<int64_t>(n, 1 << 30));
        return L;
    }
    // schema 1 needs no migration (kPresetMigrations is empty): a removed key is dropped with a note, a new one takes its default
    const std::string scopeText = j.contains("scope") && j["scope"].is_string() ? j["scope"].get<std::string>() : std::string();
    if (scopeText != "stage" && scopeText != "pipeline") {
        add(Kind::BadScope, "scope must be stage | pipeline (got " + (j.contains("scope") ? Brief(j["scope"]) : std::string("none")) +
                                ") - a newer dlssvid may know it");
        return L;
    }
    const PresetScope scope = scopeText == "pipeline" ? PresetScope::Pipeline : PresetScope::Stage;
    const bool stageScope = scope == PresetScope::Stage;
    L.preset.scope = scope;
    L.scopeKnown = true;

    if (!j.contains("name") || !j["name"].is_string()) add(Kind::BadName, "name must be a string of 1.." + std::to_string(kPresetNameMax) + " characters");
    else if (const std::string p = PresetNameProblem(j["name"].get<std::string>()); !p.empty()) add(Kind::BadName, p);
    for (const char* f : {"created", "app"}) {
        if (!j.contains(f)) continue;
        if (j[f].is_string()) (std::string(f) == "created" ? L.preset.created : L.preset.app) = j[f].get<std::string>();
        else add(Kind::UnknownField, std::string(f) + " is not a string - ignored").key = f;
    }
    static const std::set<std::string> fields = {"format", "schema", "scope", "name", "created", "app", "stages", "encode"};
    for (const auto& [k, v] : j.items())
        if (!fields.count(k)) add(Kind::UnknownField, "field '" + k + "' is not a preset field - ignored").key = k;

    if (!j.contains("stages") || !j["stages"].is_object()) {
        add(Kind::BadStages, "stages must be an object of the stages' settings");
        return L;
    }
    const J& stages = j["stages"];
    const auto& order = PipelineStageNames();
    for (const auto& [name, v] : stages.items())
        if (std::find(order.begin(), order.end(), name) == order.end())
            add(Kind::UnknownStage, "stage '" + name + "' is not one of this dlssvid's " + PipelineStageNamesText() + " - dropped").stage = name;
    std::vector<std::string> names;  // the known stages, in run order
    for (const auto& s : order)
        if (stages.contains(s)) names.push_back(s);
    if (stageScope) {
        if (names.empty()) {
            add(Kind::BadStages, "a card preset needs the settings of one stage " + PipelineStageNamesText());
            return L;
        }
        if (names.size() > 1) {
            add(Kind::BadScope, "a card preset (scope stage) holds exactly one stage, this one has " + Join(names)).stages = names;
            return L;
        }
        if (j.contains("encode")) add(Kind::PipelineFields, "a card preset holds no encode - dropped").key = "encode";
    }

    // the entries' shape
    std::vector<std::string> good;
    for (const auto& s : names) {
        const J& e = stages[s];
        const std::string where = "stages." + s;
        if (!e.is_object() || !e.contains("params") || !e["params"].is_object()) {
            add(Kind::BadStages, where + (stageScope ? " must be {\"params\": {...}}" : " must be {\"enabled\": true | false, \"params\": {...}}")).stage = s;
            continue;
        }
        if (stageScope) {
            if (e.contains("enabled")) {
                PresetIssue& i = add(Kind::PipelineFields, where + ".enabled: a card preset holds no switch - dropped");
                i.stage = s;
                i.key = "enabled";
            }
        } else if (!e.contains("enabled") || !e["enabled"].is_boolean()) {
            add(Kind::BadStages, where + ".enabled must be true | false in a pipeline preset").stage = s;
            continue;
        }
        for (const auto& [k, v] : e.items())
            if (k != "params" && k != "enabled") {
                PresetIssue& i = add(Kind::UnknownField, "field '" + where + "." + k + "' is not a preset field - ignored");
                i.stage = s;
                i.key = k;
            }
        good.push_back(s);
    }
    if (!stageScope && forStage.empty())
        for (const auto& s : order)
            if (!stages.contains(s)) add(Kind::MissingStage, s + " is not in the preset - the project's " + s + " stays").stage = s;

    // one stage of either scope
    if (!forStage.empty()) {
        if (std::find(names.begin(), names.end(), forStage) == names.end()) {
            add(Kind::NoStage, "the preset '" + L.preset.name + "' has no settings for " + forStage).stage = forStage;
            return L;
        }
        if (!stageScope) {
            std::vector<std::string> others;
            for (const auto& s : names)
                if (s != forStage) others.push_back(s);
            if (!others.empty()) {
                PresetIssue& i = add(Kind::OtherStages, "only its " + forStage + " settings are used here; its other stages (" + Join(others) + ") are not");
                i.stages = others;
            }
            add(Kind::PipelineFields, "a pipeline preset read for " + forStage + ": its switches and encode are not used here").stage = forStage;
        }
        good.erase(std::remove_if(good.begin(), good.end(), [&](const std::string& s) { return s != forStage; }), good.end());
    }

    // the settings of each stage: dropped what a preset of this scope does not hold, the values checked
    for (const auto& s : good) {
        const J& e = stages[s];
        const StageSchema* schema = FindStageSchema(s);
        const J params = NormalizeStageParams(s, e["params"]);
        J kept = J::object();
        for (const auto& [k, v] : params.items()) {
            if (const MovedParam* m = FindMovedParam(s, k)) {
                PresetIssue& i = add(Kind::MovedKey, s + "." + k + " moved to " + m->to + " - dropped");
                i.stage = s;
                i.key = k;
                i.to = m->to;
                i.got = v;
                continue;
            }
            const ParamSpec* spec = schema ? schema->Find(k) : nullptr;
            if (!spec) {
                PresetIssue& i = add(Kind::UnknownKey, s + "." + k + " is not a setting of this dlssvid - dropped");
                i.stage = s;
                i.key = k;
                i.got = v;
                continue;
            }
            if (!PresetKeyClass(*spec, scope)) {
                PresetIssue& i = add(Kind::EngineerKey,
                                     stageScope            ? s + "." + k + " is an engineer key - card presets hold the card's settings, dropped"
                                     : IsMachineKey(*spec) ? s + "." + k + " is bound to the machine (a path or a size for its GPU memory) - presets do not hold it, dropped"
                                                           : s + "." + k + " is not a preset setting (nothing to choose) - dropped");
                i.stage = s;
                i.key = k;
                i.got = v;
                continue;
            }
            kept[k] = v;
        }
        for (ParamProblem& p : CheckStageParams(s, kept)) {
            PresetIssue& i = read == PresetRead::Slot ? add(Kind::SlotValue, "the slot keeps a value with an error: " + p.text) : add(Kind::BadValue, p.text);
            i.stage = s;
            i.key = p.key;
            i.got = p.got;
            i.problem = std::move(p);
        }
        for (auto& [k, v] : kept.items()) v = CanonicalizeJson(v);
        if (stageScope && kept.empty()) add(Kind::BadStages, "the preset has no settings of " + s + " left").stage = s;
        PresetStage ps;
        ps.params = std::move(kept);
        ps.enabled = stageScope ? true : e["enabled"].get<bool>();
        L.preset.stages[s] = std::move(ps);
    }

    if (!stageScope && forStage.empty()) {
        if (j.contains("encode")) {
            PresetEncode enc;
            const auto issues = CheckEncodeSection(j["encode"], &enc, read);
            const bool fatal = std::any_of(issues.begin(), issues.end(), [](const PresetIssue& i) { return i.fatal(); });
            L.issues.insert(L.issues.end(), issues.begin(), issues.end());
            if (!fatal) L.preset.encode = enc;
        } else {
            add(Kind::NoEncode, "the preset has no encode - the project's encoder stays");
        }
        if (names.empty() && !j.contains("encode")) add(Kind::BadStages, "the preset has no settings");
    }
    return L;
}

PresetLoad LoadPreset(const std::filesystem::path& file, const std::string& forStage, PresetRead read) {
    JsonFile f = ReadJsonFile(file);
    if (f.issue) {
        PresetLoad L;
        L.issues.push_back(*f.issue);
        return L;
    }
    return ValidatePreset(f.json, forStage, read);
}

Preset MakePreset(const std::string& name, const std::string& stage, const nlohmann::json& current) {
    Preset p;
    p.name = TrimPresetName(name);
    p.created = NowIso8601();
    p.app = DLSSVID_VERSION;
    p.scope = PresetScope::Stage;
    p.stages[stage].params = PresetParams(stage, current, PresetScope::Stage);
    return p;
}

nlohmann::ordered_json PresetToJson(const Preset& p) {
    using O = nlohmann::ordered_json;
    O root;
    root["format"] = kPresetFormat;
    root["schema"] = kPresetSchemaVersion;
    root["scope"] = PresetScopeName(p.scope);
    root["name"] = p.name;
    if (!p.created.empty()) root["created"] = p.created;
    if (!p.app.empty()) root["app"] = p.app;
    O stages = O::object();
    std::vector<std::string> names;
    for (const auto& s : PipelineStageNames())
        if (p.stages.count(s)) names.push_back(s);
    for (const auto& [s, e] : p.stages)
        if (std::find(names.begin(), names.end(), s) == names.end()) names.push_back(s);
    for (const auto& s : names) {
        const PresetStage& e = p.stages.at(s);
        O entry = O::object();
        if (p.scope == PresetScope::Pipeline) entry["enabled"] = e.enabled;
        O params = O::object();
        std::set<std::string> done;
        if (const StageSchema* schema = FindStageSchema(s))
            for (const ParamSpec& spec : schema->params)
                if (e.params.is_object() && e.params.contains(spec.key)) {
                    params[spec.key] = O::parse(CanonicalizeJson(e.params[spec.key]).dump());
                    done.insert(spec.key);
                }
        if (e.params.is_object())
            for (const auto& [k, v] : e.params.items())
                if (!done.count(k)) params[k] = O::parse(CanonicalizeJson(v).dump());
        entry["params"] = std::move(params);
        stages[s] = std::move(entry);
    }
    root["stages"] = std::move(stages);
    if (p.scope == PresetScope::Pipeline && p.encode) {
        O enc;
        enc["codec"] = p.encode->codec;
        enc["bit_depth"] = BitDepthChoiceName(p.encode->format.bitDepth);
        enc["chroma"] = p.encode->format.chroma;
        O opts = O::object();
        if (!p.encode->bitrate.empty()) opts["b"] = p.encode->bitrate;
        enc["options"] = std::move(opts);
        root["encode"] = std::move(enc);
    }
    return root;
}

void SavePreset(const std::filesystem::path& file, const Preset& p, PresetRead read) {
    const std::string where = PathText(file) + ": ";
    const std::string text = PresetToJson(p).dump(2) + "\n";
    if (text.size() > kPresetMaxBytes) Throw(where + "the preset would be larger than 1 MiB");
    const PresetLoad check = ValidatePreset(J::parse(text), {}, read);
    if (!check.ok()) throw PresetError(where + check.problemsText(), check.issues);
    std::error_code ec;
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path(), ec);
    std::filesystem::path tmp = file;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << text;
        out.close();
        if (!out) {
            std::filesystem::remove(tmp, ec);
            Throw(where + "cannot write the file");
        }
    }
    std::filesystem::rename(tmp, file, ec);  // replaces an existing file
    if (ec) {
        std::error_code ignore;
        std::filesystem::remove(tmp, ignore);
        Throw(where + "cannot write the file (system error " + std::to_string(ec.value()) + ")");
    }
}

std::vector<LeftOut> StagePresetLeftOut(const std::string& stage, const nlohmann::json& current, PresetScope scope) {
    std::vector<LeftOut> out;
    const J cur = current.is_object() ? NormalizeStageParams(stage, current) : J::object();
    const StageSchema* schema = FindStageSchema(stage);
    const std::string backend = StageBackend(stage, cur);
    for (const auto& [key, value] : cur.items()) {
        const ParamSpec* spec = schema ? schema->Find(key) : nullptr;
        if (!spec) {
            out.push_back({LeftOut::Why::Unknown, stage, key, value});
            continue;
        }
        if (IsPresetKey(*spec, backend, scope) || !OfBackend(*spec, backend) || SameValue(value, spec->def)) continue;
        LeftOut::Why why = LeftOut::Why::Engineer;
        if (PresetKeyClass(*spec, scope)) why = LeftOut::Why::NoRow;  // in the class, not a preset key: a list key without a list
        else if (IsPathKey(key)) why = LeftOut::Why::Path;
        else if (scope == PresetScope::Pipeline) why = LeftOut::Why::Machine;
        out.push_back({why, stage, key, value});
    }
    return out;
}

StageParamsFile ReadStageParamsFile(const std::string& stage, const std::filesystem::path& file) {
    const std::string where = PathText(file) + ": ";
    const StageSchema* schema = FindStageSchema(stage);
    if (!schema) Throw("unknown stage '" + stage + "'");
    JsonFile f = ReadJsonFile(file);
    if (f.issue) Throw(where + f.issue->text);
    const J& j = f.json;
    if (!j.is_object()) Throw(where + "must be one JSON object of " + stage + " settings, as stages." + stage + ".params of a project, or a dlssvid-preset");
    StageParamsFile out;
    if (j.contains("format") && j["format"] == kPresetFormat) {
        PresetLoad L = ValidatePreset(j, stage);
        if (!L.ok()) throw PresetError(where + L.problemsText(), L.issues);
        out.preset = true;
        out.scope = L.preset.scope;
        out.presetName = L.preset.name;
        out.params = L.preset.stages.at(stage).params;
        for (auto& i : L.issues) out.notes.push_back(std::move(i));
        return out;
    }
    J params = NormalizeStageParams(stage, j);
    for (auto it = params.begin(); it != params.end();) {
        if (schema->Find(it.key())) {
            ++it;
            continue;
        }
        if (const MovedParam* m = FindMovedParam(stage, it.key())) Throw(where + MovedParamProblem(*m));  // not dropped in silence
        out.dropped.push_back(it.key());
        it = params.erase(it);
    }
    out.params = std::move(params);
    return out;
}

}  // namespace dlssvid
