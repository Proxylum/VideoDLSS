#include "stages/depth/DepthModels.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

#include "ml/ModelRegistry.h"
#include "util/Error.h"

namespace dlssvid {

std::string DepthFamilyOf(const std::string& backend) {
    if (backend == "vda" || backend == "worker:vda") return "vda";
    if (backend == "da3" || backend == "worker" || backend == "worker:da3") return "da3";
    return {};
}

std::string DefaultDepthModel(const std::string& family) {
    if (family == "da3") return "da3metric-large";
    if (family == "vda") return "metric-vda-small";
    Throw("depth: no default model for the backend family '" + family + "'");
}

std::filesystem::path DepthRegistryPath(const std::string& modelsDir) {
    return modelsDir.empty() ? ModelRegistry::DefaultRegistryPath() : std::filesystem::path(modelsDir) / "registry.json";
}

namespace {

const char* FamilyTitle(const std::string& family) { return family == "vda" ? "VDA" : family == "da3" ? "DA3" : "depth"; }

std::string FamilyModels(const ModelRegistry& registry, const std::string& family) {
    std::string out;
    for (const ModelEntry& e : registry.Entries()) {
        if (e.stage != "depth" || e.params.value("family", "") != family) continue;
        out += (out.empty() ? "" : ", ") + e.id + (e.license.empty() ? "" : " (" + e.license + ")");
    }
    return out.empty() ? "none" : out;
}

std::string Gb(double mib) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", mib / 1024.0);
    return buf;
}

}  // namespace

int SmallerDepthWindow(int window) { return std::max(4, window / 2 / 4 * 4); }

uint64_t DepthWorkspaceBytes(uint64_t freeBytes) {
    constexpr uint64_t gib = uint64_t{1} << 30;
    return std::clamp<uint64_t>(freeBytes / 2, 2 * gib, 8 * gib);
}

DepthModelChoice ResolveDepthModel(const ModelRegistry& registry, const std::string& backend, const std::string& modelId, int windowOverride,
                                   bool torch) {
    DepthModelChoice c;
    c.backend = backend;
    c.family = DepthFamilyOf(backend);
    c.modelId = modelId;
    if (c.family.empty()) return c;  // stub, icdepth, worker:stub: no registry model
    c.modelId = modelId.empty() ? DefaultDepthModel(c.family) : modelId;
    const char* title = FamilyTitle(c.family);
    const ModelEntry* e = registry.Find(c.modelId);
    if (!e) Throw(std::string("depth: unknown ") + title + " model '" + c.modelId + "'; " + title + " models: " + FamilyModels(registry, c.family));
    if (e->stage != "depth" || e->params.value("family", "") != c.family)
        Throw("depth: '" + c.modelId + "' is not a " + title + " model (backend " + backend + "); " + title + " models: " + FamilyModels(registry, c.family));
    c.license = e->license;
    c.nonCommercial = e->license.find("NC") != std::string::npos;
    c.params = e->params;
    c.metric = e->params.value("metric", false);
    c.registryWindow = std::max(1, e->params.value("window", 1));
    c.registryOverlap = std::max(0, e->params.value("overlap", 0));
    c.maxWindow = std::max(c.registryWindow, e->params.value("max_window", c.registryWindow));
    c.registryInputSize = e->params.value("input_size", 0);
    c.window = c.registryWindow;
    c.overlap = c.registryOverlap;
    if (c.family != "vda") {
        if (windowOverride > 0)
            c.warnings.push_back("depth: window " + std::to_string(windowOverride) + " applies to VDA models only; ignored for " + c.modelId);
        return c;
    }
    int window = windowOverride;
    if (window <= 0 && torch) window = e->params.value("window_torch", 0);
    if (window > 0) {
        if (window > c.maxWindow || (window > 4 && window % 4 != 0))
            Throw("depth: window " + std::to_string(window) + " for " + c.modelId + ": needs 1.." + std::to_string(c.maxWindow) +
                  ", a multiple of 4 above 4 (the DPT head runs micro-batches of 4)");
        c.window = window;
        c.overlap = static_cast<int>(std::lround(static_cast<double>(c.registryOverlap) * window / c.registryWindow));
        c.overlap = std::clamp(c.overlap, 0, std::max(0, window - 1));
    }
    return c;
}

DepthModelChoice ResolveDepthModel(const std::string& modelsDir, const std::string& backend, const std::string& modelId, int windowOverride, bool torch) {
    if (DepthFamilyOf(backend).empty()) {
        DepthModelChoice c;
        c.backend = backend;
        c.modelId = modelId;
        return c;
    }
    return ResolveDepthModel(ModelRegistry::Load(DepthRegistryPath(modelsDir)), backend, modelId, windowOverride, torch);
}

int EffectiveDepthInputSize(const DepthModelChoice& choice, int requested) {
    return requested > 0 && requested == 518 && choice.registryInputSize > 0 ? choice.registryInputSize : requested;
}

std::optional<double> EstimateDepthVramMiB(const DepthModelChoice& choice, bool torch, uint32_t inputWidth, uint32_t inputHeight, bool fp16) {
    const nlohmann::json& v = choice.params.value(torch ? "vram_torch" : "vram", nlohmann::json::object());
    if (!v.is_object() || !v.contains("points") || !v["points"].is_array() || v["points"].empty()) return std::nullopt;
    // window -> (pixels, MiB), sorted by pixels
    std::map<int, std::vector<std::pair<double, double>>> byWindow;
    for (const auto& p : v["points"]) {
        if (!p.contains("window") || !p.contains("input") || !p.contains("mib") || !p["input"].is_array() || p["input"].size() != 2) continue;
        const double px = p["input"][0].get<double>() * p["input"][1].get<double>();
        byWindow[p["window"].get<int>()].emplace_back(px, p["mib"].get<double>());
    }
    if (byWindow.empty()) return std::nullopt;
    for (auto& [w, pts] : byWindow) std::sort(pts.begin(), pts.end());
    const double px = static_cast<double>(inputWidth) * inputHeight;
    const auto atPixels = [&](const std::vector<std::pair<double, double>>& pts) {
        for (const auto& [p, mib] : pts)
            if (p >= px) return mib;  // the smallest measured input with at least as many pixels
        return pts.back().second * px / pts.back().first;  // larger than measured: scaled up by pixels
    };
    const int window = std::max(1, choice.window);
    double mib = 0.0;
    const auto at = byWindow.lower_bound(window);
    if (at != byWindow.end()) {
        mib = atPixels(at->second);
    } else if (byWindow.size() >= 2) {  // beyond the largest measured window: linear through the largest two
        const auto w2 = std::prev(byWindow.end());
        const auto w1 = std::prev(w2);
        const double m2 = atPixels(w2->second), m1 = atPixels(w1->second);
        mib = m2 + (m2 - m1) * (window - w2->first) / std::max(1, w2->first - w1->first);
    } else {
        mib = atPixels(byWindow.rbegin()->second) * window / std::max(1, byWindow.rbegin()->first);
    }
    return mib * 1.1 * (fp16 ? 1.0 : 2.0);
}

DepthAdvice AdviseDepth(const DepthModelChoice& choice, std::optional<double> estimateMiB, uint64_t vramBytes, uint32_t inputWidth, uint32_t inputHeight) {
    DepthAdvice a;
    if (choice.nonCommercial)
        a.warnings.push_back(choice.modelId + ": лицензия " + choice.license + " — только некоммерческое использование");
    for (const auto& w : choice.warnings) a.warnings.push_back(w);
    if (choice.family.empty()) return a;
    if (!estimateMiB) {
        if (choice.family == "vda") a.info = "потребность VRAM для " + choice.modelId + " не замерена";
        return a;
    }
    const double totalMiB = static_cast<double>(vramBytes) / (1024.0 * 1024.0);
    const std::string geometry = "окно " + std::to_string(choice.window) + ", вход " + std::to_string(inputWidth) + "x" + std::to_string(inputHeight);
    if (vramBytes == 0) {
        a.info = "≈ " + Gb(*estimateMiB) + " ГБ видеопамяти (" + geometry + ")";
        return a;
    }
    if (*estimateMiB > totalMiB) {
        a.error = choice.modelId + ": нужно ≈ " + Gb(*estimateMiB) + " ГБ видеопамяти (" + geometry + "), у GPU " + Gb(totalMiB) + " ГБ" +
                  (choice.family == "vda" && choice.window > 4 ? ": уменьшите окно (depth.window / --window " + std::to_string(SmallerDepthWindow(choice.window)) + ")"
                                                                  : ": уменьшите размер входа модели") +
                  " или выберите модель меньше";
    } else if (*estimateMiB > 0.85 * totalMiB) {
        a.warnings.push_back(choice.modelId + ": видеопамяти впритык — ≈ " + Gb(*estimateMiB) + " ГБ из " + Gb(totalMiB) + " (" + geometry + ")");
    } else {
        a.info = "≈ " + Gb(*estimateMiB) + " ГБ видеопамяти из " + Gb(totalMiB);
    }
    return a;
}

std::string CheckDepthVramFree(const DepthModelChoice& choice, double estimateMiB, uint64_t freeBytes, uint64_t totalBytes) {
    const double freeMiB = static_cast<double>(freeBytes) / (1024.0 * 1024.0), totalMiB = static_cast<double>(totalBytes) / (1024.0 * 1024.0);
    if (freeBytes == 0 || estimateMiB <= freeMiB) return {};
    return choice.modelId + ": нужно ≈ " + Gb(estimateMiB) + " ГБ видеопамяти (окно " + std::to_string(choice.window) + "), свободно " + Gb(freeMiB) +
           " из " + Gb(totalMiB) + " ГБ: закройте другие GPU-приложения" +
           (choice.family == "vda" && choice.window > 4 ? " или уменьшите окно (--window " + std::to_string(SmallerDepthWindow(choice.window)) + ")" : "") +
           " (проверку отключает DLSSVID_DEPTH_SKIP_VRAM_CHECK=1)";
}

}  // namespace dlssvid
