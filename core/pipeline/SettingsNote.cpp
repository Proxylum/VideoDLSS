#include "pipeline/SettingsNote.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <utility>
#include <vector>

#include "passes/Manifest.h"
#include "pipeline/PassVersions.h"

namespace dlssvid {
namespace {

std::optional<Manifest> Load(const std::filesystem::path& dir) {
    if (!Manifest::Exists(dir)) return std::nullopt;
    try {
        return Manifest::Load(dir);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::string IdOf(const Manifest& m) { return m.fingerprint.empty() ? LegacyPassId(m) : m.fingerprint; }

// The version of `pass` the pass `down` was made from: process records it among `inputs` - the current folder when it matches,
// else the history version with that fingerprint (a color_fg kept while nr was recomputed under it, review of PR-11); without
// a record (a standalone pass) the current folder. nullopt: that version is not on disk any more.
std::optional<Manifest> Upstream(const std::filesystem::path& root, const std::string& pass, const Manifest& down, bool& found) {
    found = true;
    const auto it = down.inputs.find(pass);
    std::optional<Manifest> cur = Load(root / pass);
    if (it == down.inputs.end() || it->second.empty()) return cur;
    if (cur && IdOf(*cur) == it->second) return cur;
    for (const auto& v : ListPassVersions(root, pass))
        if (!v.current && v.fingerprint == it->second)
            if (auto m = Load(v.dir)) return m;
    found = false;
    return std::nullopt;
}

// a number as a short text: 2, 1.2, 0.24
std::string Num(const nlohmann::json& v) {
    if (v.is_boolean()) return v.get<bool>() ? "on" : "off";
    if (!v.is_number()) return v.is_string() ? v.get<std::string>() : v.dump();
    const double d = v.get<double>();
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4g", std::abs(d) < 1e-9 ? 0.0 : d);
    return buf;
}

// the keys a segment names, in this order, when the stage wrote them; values only - never a path (dll_dir, files)
void Pick(const nlohmann::json& from, nlohmann::json& to, std::initializer_list<const char*> keys) {
    if (!from.is_object()) return;
    for (const char* k : keys)
        if (from.contains(k) && (from[k].is_number() || from[k].is_boolean() || from[k].is_string())) to[k] = from[k];
}

nlohmann::json Upscale(const Manifest& m) {
    nlohmann::json s = {{"backend", m.stageParams.value("backend", std::string())}};
    Pick(m.stageParams, s, {"scale", "quality", "model"});
    if (m.stageParams.value("fallback", false)) s["fallback"] = true;  // ran on another backend than asked
    return s;
}

nlohmann::json Nr(const Manifest& m) {
    const nlohmann::json& p = m.stageParams;
    nlohmann::json s = {{"backend", p.value("backend", std::string())}};
    if (p.value("bypass", false)) {  // intensity 0 (PR-7): no model ran
        s["intensity"] = 0;
        s["bypass"] = true;
    } else {
        if (p.contains("style") && p["style"].is_number_integer()) {
            static const char* named[] = {"default", "natural", "cinematic"};
            const int st = p["style"].get<int>();
            s["style"] = st >= 0 && st < 3 ? nlohmann::json(named[st]) : nlohmann::json(st);
        }
        Pick(p, s, {"intensity", "passes", "local_tone", "local_structure"});
        if (p.contains("skin_structure") && p["skin_structure"].is_number())
            s["skin_structure"] = p["skin_structure"].get<double>() < 0 ? nlohmann::json("follow") : p["skin_structure"];
        Pick(p, s, {"auto_mask", "guides", "temporal", "model_scale"});
        // what acts only with its switch: the filter's threshold with the filter, the ratio transfer at another model resolution
        if (p.value("temporal", 0.0) > 0.0) Pick(p, s, {"temporal_threshold"});
        if (p.contains("model_scale") && p["model_scale"].is_number() && p["model_scale"].get<double>() != 1.0) Pick(p, s, {"transfer", "max_ratio"});
    }
    if (p.contains("scene_cut_count")) {  // the detector ran (PR-6)
        Pick(p, s, {"scene_cut_threshold"});
        s["cuts"] = p["scene_cut_count"];
    }
    if (p.contains("tonemap") && p["tonemap"].is_object()) {
        const nlohmann::json& t = p["tonemap"];
        if (t.value("curve", std::string("passthrough")) != "passthrough") s["tonemap"] = t["curve"];
        if (t.value("input_transfer", std::string("srgb")) != "srgb") s["input_transfer"] = t["input_transfer"];
        if (t.contains("exposure_ev") && t["exposure_ev"].is_number() && t["exposure_ev"].get<double>() != 0.0) s["exposure"] = t["exposure_ev"];
    }
    if (p.contains("masks") && p["masks"].is_array() && !p["masks"].empty()) {
        std::string masks;
        for (const auto& v : p["masks"])
            if (v.is_string()) masks += (masks.empty() ? "" : ",") + v.get<std::string>();
        s["masks"] = masks;
    }
    return s;
}

nlohmann::json Fg(const Manifest& m) {
    nlohmann::json s = {{"backend", m.stageParams.value("backend", std::string())}};
    Pick(m.stageParams, s, {"multiplier", "model"});
    return s;
}

}  // namespace

nlohmann::json BuildSettingsJson(const std::filesystem::path& root, const std::string& finalPass) {
    nlohmann::json out = {{"app", std::string("VideoDLSS ") + DLSSVID_VERSION}, {"stages", nlohmann::json::object()}};
    nlohmann::json& stages = out["stages"];
    auto mark = [](nlohmann::json s, const Manifest& m) {
        if (m.stageParams.value("adopted", false)) s["adopted"] = true;
        return s;
    };
    // walk back from the encoded pass: each stage names what it read (color_source) and process recorded which version (inputs); a
    // stage it did not read is not in the result, a version no longer on disk is named as such
    const char* stageOf[] = {"color_fg", "fg", "color_nr", "nr", "color_sr", "upscale"};
    auto stage = [&](const std::string& pass) {
        for (size_t i = 0; i < 6; i += 2)
            if (pass == stageOf[i]) return std::string(stageOf[i + 1]);
        return std::string();
    };
    std::string pass = finalPass;
    std::optional<Manifest> m = stage(pass).empty() ? std::nullopt : Load(root / pass);
    while (m) {
        const std::string name = stage(pass);
        stages[name] = mark(name == "fg" ? Fg(*m) : name == "nr" ? Nr(*m) : Upscale(*m), *m);
        std::string src = m->stageParams.value("color_source", std::string());
        if (name == "fg") {
            // pass:<folder> | slot:<pass> | video; "pass" alone (before PR-11) or "pass:" of a folder given with a trailing separator:
            // the input process recorded, color_nr before color_sr. Another folder name: not this root's pass - the walk stops
            if (src.rfind("pass:", 0) == 0 || src.rfind("slot:", 0) == 0) src = src.substr(5);
            if (src == "pass" || src.empty()) src = m->inputs.count("color_nr") ? "color_nr" : m->inputs.count("color_sr") ? "color_sr" : std::string();
        } else if (name == "nr") {
            src = src == "color_sr" || src == "slot" ? "color_sr" : std::string();
        } else {
            src.clear();
        }
        if (src != "color_nr" && src != "color_sr") break;
        bool found = true;
        m = Upstream(root, src, *m, found);
        pass = src;
        if (!found) stages[stage(src)] = {{"backend", "?"}, {"missing", true}};  // made from a version that is gone
    }
    return out;
}

std::string SettingsNoteText(const nlohmann::json& settings) {
    std::string note = settings.value("app", std::string("VideoDLSS"));
    const nlohmann::json& stages = settings.contains("stages") ? settings["stages"] : nlohmann::json::object();
    // the order a reader expects (spec §2g); a key not listed follows, alphabetically
    const std::vector<std::pair<const char*, std::vector<std::string>>> order = {
        {"upscale", {"scale", "quality", "model", "fallback"}},
        {"nr", {"bypass", "style", "intensity", "passes", "local_tone", "local_structure", "skin_structure", "auto_mask", "guides", "scene_cut_threshold", "cuts",
                "temporal", "temporal_threshold", "model_scale", "transfer", "max_ratio", "tonemap", "input_transfer", "exposure", "masks"}},
        {"fg", {"multiplier", "model"}}};
    for (const auto& [name, keys] : order) {
        if (!stages.contains(name)) continue;
        const nlohmann::json& s = stages[name];
        std::string seg = std::string(name) + ": " + s.value("backend", std::string("?"));
        std::vector<std::string> all = keys;
        for (const auto& [k, v] : s.items())
            if (k != "backend" && k != "adopted" && std::find(keys.begin(), keys.end(), k) == keys.end()) all.push_back(k);
        for (const std::string& k : all) {
            if (!s.contains(k)) continue;
            const nlohmann::json& v = s[k];
            if (k == "scale" || k == "multiplier") seg += " x" + Num(v);
            else if (k == "scene_cut_threshold") seg += " scene_cut=" + Num(v);
            else if (k == "bypass") seg += " (no model)";
            else if (k == "missing") seg += " (that version is not on disk)";
            else seg += " " + k + "=" + Num(v);
        }
        if (s.value("adopted", false)) seg += " (adopted)";
        note += " | " + seg;
    }
    for (char& c : note)
        if (static_cast<unsigned char>(c) < 32 || static_cast<unsigned char>(c) > 126) c = '?';  // one ASCII line
    if (note.size() > kSettingsNoteMax) note = note.substr(0, kSettingsNoteMax - 3) + "...";
    return note;
}

}  // namespace dlssvid
