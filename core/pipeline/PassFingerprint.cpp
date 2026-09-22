#include "pipeline/PassFingerprint.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <mutex>

#include "ml/ModelRegistry.h"
#include "stages/depth/TrtDepthEstimator.h"
#include "stages/fg/IFrameGenerator.h"
#include "stages/flow/TrtFlowEstimator.h"
#include "stages/nr/INrBackend.h"
#include "stages/upscale/IUpscaler.h"
#include "util/Error.h"
#include "util/Sha256.h"

namespace dlssvid {

namespace {

std::string Str(const nlohmann::json& p, const char* key, const std::string& def) {
    if (!p.contains(key)) return def;
    if (!p[key].is_string()) Throw(std::string("process: stage parameter '") + key + "' must be a string");
    return p[key].get<std::string>();
}

bool Flag(const nlohmann::json& p, const char* key) {
    if (!p.contains(key)) return false;
    if (!p[key].is_boolean()) Throw(std::string("process: stage parameter '") + key + "' must be true or false");
    return p[key].get<bool>();
}

std::string RegistryHash(const std::string& modelId, const std::string& modelsDir) {
    try {
        const auto registry = modelsDir.empty() ? ModelRegistry::DefaultRegistryPath() : std::filesystem::path(modelsDir) / "registry.json";
        const ModelRegistry r = ModelRegistry::Load(registry);
        if (const ModelEntry* e = r.Find(modelId)) return e->sha256;
    } catch (...) {  // no registry here: the model id alone identifies the model
    }
    return {};
}

std::string FormatUtc(std::time_t t, const char* fmt) {
    std::tm tm{};
    gmtime_s(&tm, &t);
    char buf[40];
    std::strftime(buf, sizeof buf, fmt, &tm);
    return buf;
}

}  // namespace

nlohmann::json CanonicalizeJson(const nlohmann::json& j) {
    switch (j.type()) {
        case nlohmann::json::value_t::object: {
            nlohmann::json o = nlohmann::json::object();  // std::map keeps the keys sorted
            for (const auto& [k, v] : j.items()) o[k] = CanonicalizeJson(v);
            return o;
        }
        case nlohmann::json::value_t::array: {
            nlohmann::json a = nlohmann::json::array();
            for (const auto& v : j) a.push_back(CanonicalizeJson(v));
            return a;
        }
        case nlohmann::json::value_t::number_float: {
            const double d = j.get<double>();
            if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 9.0e15) return nlohmann::json(static_cast<int64_t>(d));
            return j;
        }
        case nlohmann::json::value_t::number_unsigned: {
            const uint64_t u = j.get<uint64_t>();
            if (u <= static_cast<uint64_t>(INT64_MAX)) return nlohmann::json(static_cast<int64_t>(u));
            return j;
        }
        default: return j;
    }
}

std::string CanonicalJson(const nlohmann::json& j) { return CanonicalizeJson(j).dump(); }

nlohmann::json ToolInfo::ToJson() const {
    return {{"app", app}, {"backend", backend}, {"model", model}, {"model_hash", modelHash}, {"dll_file", dllFile}, {"dll", dll}};
}

ToolInfo ToolInfo::FromJson(const nlohmann::json& j) {
    ToolInfo t;
    if (!j.is_object()) return t;
    t.app = j.value("app", "");
    t.backend = j.value("backend", "");
    t.model = j.value("model", "");
    t.modelHash = j.value("model_hash", "");
    t.dllFile = j.value("dll_file", "");
    t.dll = j.value("dll", "");
    return t;
}

std::string Sha256FileCached(const std::filesystem::path& path) {
    struct Entry {
        uintmax_t size = 0;
        std::filesystem::file_time_type mtime;
        std::string hash;
    };
    static std::mutex mutex;
    static std::map<std::wstring, Entry> cache;
    std::error_code ec;
    const auto abs = std::filesystem::absolute(path, ec);
    const uintmax_t size = std::filesystem::file_size(abs, ec);
    if (ec) Throw("cannot read " + path.string() + ": " + ec.message());
    const auto mtime = std::filesystem::last_write_time(abs, ec);
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(abs.wstring());
    if (it != cache.end() && it->second.size == size && it->second.mtime == mtime) return it->second.hash;
    Entry e{size, mtime, Sha256File(abs)};
    cache[abs.wstring()] = e;
    return e.hash;
}

ToolInfo ToolForStage(const std::string& stage, const nlohmann::json& p) {
    ToolInfo t;
    t.app = kPassToolVersion;  // not DLSSVID_VERSION: a release does not invalidate the passes (CHANGELOG 0.2.0)
    const char* def = stage == "depth" ? "da3" : stage == "flow" ? "ofa" : stage == "upscale" ? "dlss" : stage == "nr" ? "ngx" : stage == "fg" ? "dlssg" : "";
    t.backend = Str(p, "backend", def);
    const std::string model = Str(p, "model", "");
    const std::filesystem::path dllDir = Str(p, "dll_dir", "");
    if (stage == "depth" && (t.backend == "da3" || t.backend == "vda")) t.model = model.empty() ? TrtDepthEstimator::DefaultModel(t.backend) : model;
    else if (stage == "flow" && t.backend == "searaft") t.model = model.empty() ? TrtFlowEstimator::DefaultModel("searaft") : model;
    else if (!model.empty()) t.model = model;  // worker depth, rife
    if (!t.model.empty()) t.modelHash = RegistryHash(t.model, Str(p, "models_dir", ""));
    std::filesystem::path dll;
    if (stage == "upscale" && t.backend == "dlss") {
        t.dllFile = "nvngx_dlss.dll";
        for (const auto& dir : NvidiaDllSearchPaths(dllDir))
            if (std::filesystem::exists(dir / t.dllFile)) {
                dll = dir / t.dllFile;
                break;
            }
    } else if (stage == "nr" && t.backend == "ngx") {
        t.dllFile = kNrDllName;
        dll = FindNrDll(dllDir);
    } else if (stage == "fg" && t.backend == "dlssg") {
        t.dllFile = kDlssgDllName;
        dll = FindDlssgDll(dllDir);
    }
    if (!dll.empty()) t.dll = "sha256:" + Sha256FileCached(dll);
    return t;
}

std::string ShortFingerprint(const std::string& fingerprint) {
    if (fingerprint.empty()) return "legacy";
    const size_t colon = fingerprint.find(':');
    const std::string hex = colon == std::string::npos ? fingerprint : fingerprint.substr(colon + 1);
    return hex.substr(0, 8);
}

std::string PassFingerprint::Short() const { return ShortFingerprint(value); }

PassFingerprint ComputeFingerprint(const std::string& stage, const std::string& sourceHash, const nlohmann::json& params,
                                   const std::map<std::string, std::string>& inputs, const ToolInfo& tool) {
    PassFingerprint f;
    f.stage = stage;
    f.sourceHash = sourceHash;
    f.params = CanonicalizeJson(params.is_null() ? nlohmann::json::object() : params);
    f.inputs = inputs;
    f.tool = tool;
    const nlohmann::json j = {{"source", sourceHash}, {"stage", stage}, {"params", f.params}, {"inputs", inputs}, {"tool", tool.ToJson()}};
    const std::string s = CanonicalJson(j);
    f.value = "sha256:" + Sha256Hex(s.data(), s.size());
    return f;
}

std::vector<std::string> StageOutputPasses(const std::string& stage, const nlohmann::json& params) {
    if (stage == "depth") return Flag(params, "no_dlss") ? std::vector<std::string>{"depth_raw"} : std::vector<std::string>{"depth_raw", "depth_dlss"};
    if (stage == "flow") return Flag(params, "no_dlss") ? std::vector<std::string>{"mv_raw"} : std::vector<std::string>{"mv_raw", "mv_dlss"};
    if (stage == "upscale") return {"color_sr"};
    if (stage == "nr") return {"color_nr"};
    if (stage == "fg") return {"color_fg"};
    Throw("process: unknown stage '" + stage + "' (depth | flow | upscale | nr | fg)");
}

const std::vector<std::string>& StageFamilyPasses(const std::string& stage) {
    static const std::vector<std::string> depth{"depth_raw", "depth_dlss"}, flow{"mv_raw", "mv_dlss"}, upscale{"color_sr"}, nr{"color_nr"}, fg{"color_fg"}, none;
    if (stage == "depth") return depth;
    if (stage == "flow") return flow;
    if (stage == "upscale") return upscale;
    if (stage == "nr") return nr;
    if (stage == "fg") return fg;
    return none;
}

std::string StageForPass(const std::string& pass) {
    for (const char* stage : {"depth", "flow", "upscale", "nr", "fg"})
        for (const auto& p : StageFamilyPasses(stage))
            if (p == pass) return stage;
    return {};
}

std::vector<std::vector<std::string>> StageInputCandidates(const std::string& stage) {
    if (stage == "flow") return {{"depth_raw"}};
    if (stage == "upscale") return {{"depth_dlss"}, {"mv_dlss"}};
    if (stage == "nr") return {{"color_sr"}, {"depth_dlss"}, {"mv_dlss"}, {"mask_ui"}, {"mask_ignore"}, {"mask_face"}, {"mask_skin"}};
    if (stage == "fg") return {{"color_nr", "color_sr"}, {"depth_dlss"}, {"mv_dlss"}};
    return {};
}

std::string NowIso8601() { return FormatUtc(std::time(nullptr), "%Y-%m-%dT%H:%M:%SZ"); }

std::string FileTimeIso8601(const std::filesystem::path& path) {
    std::error_code ec;
    const auto ft = std::filesystem::last_write_time(path, ec);
    if (ec) return {};
    const auto sys = std::chrono::clock_cast<std::chrono::system_clock>(ft);
    return FormatUtc(std::chrono::system_clock::to_time_t(sys), "%Y-%m-%dT%H:%M:%SZ");
}

std::string CompactTimestamp(const std::string& iso8601) {
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;
    if (std::sscanf(iso8601.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &h, &mi, &s) != 6) return {};
    char buf[32];
    std::snprintf(buf, sizeof buf, "%04d%02d%02d-%02d%02d%02d", y, mo, d, h, mi, s);
    return buf;
}

}  // namespace dlssvid
