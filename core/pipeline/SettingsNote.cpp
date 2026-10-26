#include "pipeline/SettingsNote.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <utility>
#include <vector>

#include "passes/Manifest.h"

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
        Pick(p, s, {"auto_mask", "guides"});
    }
    if (p.contains("scene_cut_count")) {  // the detector ran (PR-6)
        Pick(p, s, {"scene_cut_threshold"});
        s["cuts"] = p["scene_cut_count"];
    }
    Pick(p, s, {"temporal", "model_scale"});
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
    std::string pass = finalPass;
    // walk back from the encoded pass: each stage names what it read (color_source); a stage it did not read is not in the result
    if (pass == "color_fg") {
        const auto m = Load(root / "color_fg");
        if (!m) return out;
        stages["fg"] = mark(Fg(*m), *m);
        // pass:<folder> | slot:<pass> | video; "pass" alone (before PR-11): the input process fingerprinted, color_nr before color_sr
        std::string src = m->stageParams.value("color_source", std::string());
        if (src.rfind("pass:", 0) == 0) src = src.substr(5);
        else if (src.rfind("slot:", 0) == 0) src = src.substr(5);
        else if (src == "pass") src = m->inputs.count("color_nr") ? "color_nr" : m->inputs.count("color_sr") ? "color_sr" : std::string();
        pass = src == "color_nr" || src == "color_sr" ? src : std::string();
    }
    if (pass == "color_nr") {
        const auto m = Load(root / "color_nr");
        if (!m) return out;
        stages["nr"] = mark(Nr(*m), *m);
        const std::string src = m->stageParams.value("color_source", std::string());
        pass = src == "color_sr" || src == "slot" ? "color_sr" : std::string();
    }
    if (pass == "color_sr") {
        const auto m = Load(root / "color_sr");
        if (m) stages["upscale"] = mark(Upscale(*m), *m);
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
                "temporal", "model_scale", "tonemap", "input_transfer", "exposure", "masks"}},
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
