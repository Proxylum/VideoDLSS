#include "viewport/ViewportState.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <algorithm>

#include "passes/PassImage.h"
#include "util/Error.h"

namespace dlssvid {

namespace {
template <class T, size_t N>
std::optional<T> ParseEnum(std::string_view s, const std::array<std::pair<const char*, T>, N>& table) {
    for (const auto& [name, v] : table)
        if (s == name) return v;
    return std::nullopt;
}
constexpr std::array<std::pair<const char*, ViewMode>, 3> kViewModes{{{"single", ViewMode::Single}, {"overlay", ViewMode::Overlay}, {"grid", ViewMode::Grid}}};
constexpr std::array<std::pair<const char*, DisplayMode>, 9> kDisplayModes{{{"color", DisplayMode::Color},
                                                                             {"grayscale", DisplayMode::Grayscale},
                                                                             {"viridis", DisplayMode::Viridis},
                                                                             {"turbo", DisplayMode::Turbo},
                                                                             {"mv_hsv", DisplayMode::MvHsv},
                                                                             {"mv_arrows", DisplayMode::MvArrows},
                                                                             {"mv_magnitude", DisplayMode::MvMagnitude},
                                                                             {"mask_fill", DisplayMode::MaskFill},
                                                                             {"mask_contour", DisplayMode::MaskContour}}};
constexpr std::array<std::pair<const char*, BlendMode>, 4> kBlendModes{
    {{"normal", BlendMode::Normal}, {"difference", BlendMode::Difference}, {"multiply", BlendMode::Multiply}, {"screen", BlendMode::Screen}}};
}  // namespace

std::string_view ToString(ViewMode m) { return kViewModes[static_cast<size_t>(m)].first; }
std::string_view ToString(DisplayMode m) { return kDisplayModes[static_cast<size_t>(m)].first; }
std::string_view ToString(BlendMode m) { return kBlendModes[static_cast<size_t>(m)].first; }
std::optional<ViewMode> ParseViewMode(std::string_view s) { return ParseEnum(s, kViewModes); }
std::optional<DisplayMode> ParseDisplayMode(std::string_view s) { return ParseEnum(s, kDisplayModes); }
std::optional<BlendMode> ParseBlendMode(std::string_view s) { return ParseEnum(s, kBlendModes); }

DisplayMode LayerState::DefaultDisplayFor(const std::string& sourceName) {
    const std::string source = SplitSourceVersion(sourceName).first;  // a previous version shows like its pass
    if (source == "source" || source == "result") return DisplayMode::Color;
    if (const auto kind = ParsePassKind(source)) {
        switch (*kind) {
            case PassKind::DepthRaw:
            case PassKind::DepthDlss: return DisplayMode::Turbo;
            case PassKind::MvRaw:
            case PassKind::MvDlss: return DisplayMode::MvHsv;
            case PassKind::Mask: return DisplayMode::MaskFill;
            default: return DisplayMode::Color;
        }
    }
    if (source.rfind("mask", 0) == 0) return DisplayMode::MaskFill;
    if (source.rfind("mv", 0) == 0) return DisplayMode::MvHsv;
    if (source.rfind("depth", 0) == 0) return DisplayMode::Turbo;
    return DisplayMode::Color;
}

nlohmann::json LayerState::ToJson() const {
    return {{"source", source},   {"display", std::string(ToString(display))}, {"auto_range", autoRange}, {"min", minValue},
            {"max", maxValue},    {"invert", invert},                          {"opacity", opacity},      {"blend", std::string(ToString(blend))},
            {"visible", visible}, {"solo", solo},                              {"arrow_step", arrowStep}, {"mv_scale", mvScale}};
}

LayerState LayerState::FromJson(const nlohmann::json& j) {
    LayerState l;
    l.source = j.value("source", "source");
    l.display = ParseDisplayMode(j.value("display", "")).value_or(DefaultDisplayFor(l.source));
    l.autoRange = j.value("auto_range", true);
    l.minValue = j.value("min", 0.f);
    l.maxValue = j.value("max", 1.f);
    l.invert = j.value("invert", false);
    l.opacity = std::clamp(j.value("opacity", 1.f), 0.f, 1.f);
    l.blend = ParseBlendMode(j.value("blend", "normal")).value_or(BlendMode::Normal);
    l.visible = j.value("visible", true);
    l.solo = j.value("solo", false);
    l.arrowStep = std::clamp(j.value("arrow_step", 16), 4, 64);
    l.mvScale = j.value("mv_scale", 0.f);
    return l;
}

nlohmann::json ViewportState::ToJson() const {
    nlohmann::json ls = nlohmann::json::array();
    for (const auto& l : layers) ls.push_back(l.ToJson());
    return {{"mode", std::string(ToString(mode))},
            {"layers", ls},
            {"grid_sources", gridSources},
            {"expanded_cell", expandedCell},
            {"wipe", {{"enabled", wipe.enabled}, {"vertical", wipe.vertical}, {"position", wipe.position}, {"layer_a", wipe.layerA}, {"layer_b", wipe.layerB}}},
            {"view", {{"zoom", view.zoom}, {"center_x", view.centerX}, {"center_y", view.centerY}}},
            {"time", time},
            {"selected_layer", selectedLayer}};
}

ViewportState ViewportState::FromJson(const nlohmann::json& j) {
    ViewportState s;
    s.mode = ParseViewMode(j.value("mode", "single")).value_or(ViewMode::Single);
    s.layers.clear();
    for (const auto& l : j.value("layers", nlohmann::json::array())) {
        if (static_cast<int>(s.layers.size()) >= kMaxLayers) break;
        s.layers.push_back(LayerState::FromJson(l));
    }
    if (s.layers.empty()) s.layers.push_back(LayerState{});
    if (j.contains("grid_sources") && j["grid_sources"].is_array()) {
        for (size_t i = 0; i < 4 && i < j["grid_sources"].size(); ++i) s.gridSources[i] = j["grid_sources"][i].get<std::string>();
    }
    s.expandedCell = std::clamp(j.value("expanded_cell", -1), -1, 3);
    if (j.contains("wipe")) {
        const auto& w = j["wipe"];
        s.wipe.enabled = w.value("enabled", false);
        s.wipe.vertical = w.value("vertical", true);
        s.wipe.position = std::clamp(w.value("position", 0.5f), 0.f, 1.f);
        s.wipe.layerA = w.value("layer_a", 0);
        s.wipe.layerB = w.value("layer_b", 1);
    }
    if (j.contains("view")) {
        const auto& v = j["view"];
        s.view.zoom = v.value("zoom", 0.f);
        if (s.view.zoom > 0.f) s.view.zoom = std::clamp(s.view.zoom, kMinZoom, kMaxZoom);
        s.view.centerX = v.value("center_x", 0.f);
        s.view.centerY = v.value("center_y", 0.f);
    }
    s.time = std::max(0.0, j.value("time", 0.0));
    s.legacyFrame = !j.contains("time") && j.contains("frame") && j["frame"].is_number() ? j["frame"].get<int64_t>() : int64_t{-1};
    s.selectedLayer = std::clamp(j.value("selected_layer", 0), 0, static_cast<int>(s.layers.size()) - 1);
    return s;
}

void ViewportState::SetSingleSource(const std::string& source) {
    if (layers.empty()) layers.push_back(LayerState{});
    LayerState& l = layers[0];
    if (l.source != source) {
        l.source = source;
        l.display = LayerState::DefaultDisplayFor(source);
        l.autoRange = true;
    }
}

LayerState* ViewportState::AddOverlayLayer(const std::string& source) {
    if (static_cast<int>(layers.size()) >= kMaxLayers) return nullptr;
    LayerState l;
    l.source = source;
    l.display = LayerState::DefaultDisplayFor(source);
    l.opacity = 1.f;  // a full layer: comparisons wipe, they do not blend (stage 9)
    layers.push_back(l);
    selectedLayer = static_cast<int>(layers.size()) - 1;
    return &layers.back();
}

void ViewportState::RemoveLayer(int index) {
    if (index <= 0 || index >= static_cast<int>(layers.size())) return;  // the base layer stays
    layers.erase(layers.begin() + index);
    selectedLayer = std::clamp(selectedLayer, 0, static_cast<int>(layers.size()) - 1);
    if (wipe.layerA >= static_cast<int>(layers.size())) wipe.layerA = 0;
    if (wipe.layerB >= static_cast<int>(layers.size())) wipe.layerB = std::min(1, static_cast<int>(layers.size()) - 1);
}

std::vector<std::string> ViewportState::NeededSources() const {
    std::vector<std::string> out;
    auto add = [&](const std::string& n) {
        if (std::find(out.begin(), out.end(), n) == out.end()) out.push_back(n);
    };
    if (mode == ViewMode::Grid) {
        for (const auto& g : gridSources) add(g);
    } else {
        for (const auto& l : layers) add(l.source);
    }
    return out;
}

std::pair<std::string, std::string> SplitSourceVersion(const std::string& source) {
    const size_t at = source.find('@');
    if (at == std::string::npos) return {source, {}};
    return {source.substr(0, at), source.substr(at + 1)};
}

std::string VersionSourceName(const std::string& pass, const std::string& version) { return version.empty() ? pass : pass + "@" + version; }

void ViewportState::SetCompare(const std::string& before, const std::string& after) {
    mode = ViewMode::Overlay;
    LayerState a, b;
    a.source = before;
    a.display = LayerState::DefaultDisplayFor(before);
    b.source = after;
    b.display = LayerState::DefaultDisplayFor(after);
    layers = {a, b};
    selectedLayer = 1;
    wipe.enabled = true;
    wipe.vertical = true;
    wipe.position = 0.5f;
    wipe.layerA = 0;
    wipe.layerB = 1;
}

void ViewportState::SetAfterOnly(const std::string& after) {
    mode = ViewMode::Single;
    if (layers.size() > 1) layers.resize(1);
    SetSingleSource(after);
    layers[0].opacity = 1.f;
    layers[0].visible = true;
    layers[0].solo = false;
    selectedLayer = 0;
    wipe.enabled = false;
}

bool ViewportState::CompareConfigured() const {
    if (mode != ViewMode::Overlay || !wipe.enabled) return false;
    const int n = static_cast<int>(layers.size());
    if (wipe.layerA < 0 || wipe.layerB < 0 || wipe.layerA >= n || wipe.layerB >= n || wipe.layerA == wipe.layerB) return false;
    return layers[static_cast<size_t>(wipe.layerA)].source != layers[static_cast<size_t>(wipe.layerB)].source;
}

const LayerState* ViewportState::CompareSide(bool after) const {
    if (!CompareConfigured()) return nullptr;
    return &layers[static_cast<size_t>(after ? wipe.layerB : wipe.layerA)];
}

std::string ViewportState::NextUnusedSource(const std::vector<std::string>& available) const {
    for (const auto& a : available) {
        bool used = false;
        for (const auto& l : layers) used = used || l.source == a;
        if (!used) return a;
    }
    return available.empty() ? std::string("source") : available.back();
}

void ViewportState::ResolveLegacyFrame(double baseFps) {
    if (legacyFrame < 0) return;
    time = baseFps > 0 ? static_cast<double>(legacyFrame) / baseFps : 0.0;
    legacyFrame = -1;
}

std::string FormatTimecode(double seconds) {
    const long long hundredths = std::llround(std::max(0.0, seconds) * 100.0);
    const long long h = hundredths / 360000, m = hundredths / 6000 % 60, s = hundredths / 100 % 60, hh = hundredths % 100;
    char buf[32];
    if (h > 0) std::snprintf(buf, sizeof buf, "%lld:%02lld:%02lld.%02lld", h, m, s, hh);
    else std::snprintf(buf, sizeof buf, "%02lld:%02lld.%02lld", m, s, hh);
    return buf;
}

bool ViewportState::AnySolo() const {
    for (const auto& l : layers)
        if (l.solo) return true;
    return false;
}

}  // namespace dlssvid
