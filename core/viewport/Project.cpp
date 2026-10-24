#include "viewport/Project.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include "pipeline/PassFingerprint.h"
#include "util/Error.h"
#include "util/Log.h"
#include "viewport/ViewportProxy.h"

namespace dlssvid {

namespace {
std::string Rel(const std::filesystem::path& p, const std::filesystem::path& base) {
    if (p.empty()) return {};
    if (base.empty()) return p.generic_string();
    std::error_code ec;
    const auto r = std::filesystem::relative(p, base, ec);
    if (ec || r.empty()) return std::filesystem::absolute(p).generic_string();  // other drive / not relatable
    return r.generic_string();
}
std::filesystem::path Abs(const std::string& s, const std::filesystem::path& base) {
    if (s.empty()) return {};
    const std::filesystem::path p(s);
    return p.is_absolute() || base.empty() ? p : (base / p).lexically_normal();
}

// schema_version 1 -> 2 (TASK-0043): nr.exposure was a linear multiplier, it is in stops now - x > 0 becomes log2(x)
// (0 = the identity drops the key), x <= 0 (black frames then) is dropped; nr.skin_structure between -1 and 0 had no
// meaning (-1 = «as the local structure», 0..2 a strength) and becomes -1.
void MigrateNrParamsV1(nlohmann::json& params) {
    if (!params.is_object()) return;
    if (params.contains("exposure") && params["exposure"].is_number()) {
        const double x = params["exposure"].get<double>();
        const double ev = x > 0 ? std::clamp(std::round(std::log2(x) * 100.0) / 100.0, -8.0, 8.0) : 0.0;
        if (ev == 0.0) params.erase("exposure");
        else params["exposure"] = ev;
        Log()->info("project v1: nr.exposure {} (a multiplier) -> {} EV", x, ev);
    }
    if (params.contains("skin_structure") && params["skin_structure"].is_number()) {
        const double v = params["skin_structure"].get<double>();
        if (v < 0.0 && v != -1.0) {
            params["skin_structure"] = -1.0;
            Log()->info("project v1: nr.skin_structure {} -> -1 (as the local structure)", v);
        }
    }
}
}  // namespace

std::vector<StageEntry> Project::DefaultStages() {
    return {StageEntry{"depth", true, {{"backend", "da3"}}}, StageEntry{"flow", true, {{"backend", "ofa"}}},
            StageEntry{"upscale", true, {{"backend", "dlss"}, {"scale", 2}}},
            StageEntry{"nr", true, {{"backend", "ngx"}, {"intensity", 1.0}, {"passes", 1}}},
            StageEntry{"fg", true, {{"backend", "dlssg"}, {"multiplier", 2}}}};
}

Project Project::Create(const std::filesystem::path& sourceVideo, const std::filesystem::path& passesRoot) {
    Project p;
    p.sourceVideo = sourceVideo;
    p.passesRoot = passesRoot.empty() ? sourceVideo.parent_path() / (sourceVideo.stem().string() + "_passes") : passesRoot;
    p.stages = DefaultStages();
    p.viewport = ViewportState{};
    return p;
}

nlohmann::json Project::ToJson(const std::filesystem::path& relativeTo) const {
    nlohmann::json st = nlohmann::json::array();
    for (const auto& s : stages) st.push_back({{"name", s.name}, {"enabled", s.enabled}, {"params", s.params}});
    return {{"schema_version", kProjectSchemaVersion},
            {"source", Rel(sourceVideo, relativeTo)},
            {"passes", Rel(passesRoot, relativeTo)},
            {"result", Rel(resultVideo, relativeTo)},
            {"stages", st},
            {"pass_versions_keep", passVersionsKeep},
            {"encode", {{"codec", codec}, {"options", codecOptions}, {"bit_depth", BitDepthChoiceName(outputFormat.bitDepth)}, {"chroma", outputFormat.chroma}}},
            {"source_hash", {{"value", sourceHash}, {"size", sourceSize}, {"mtime", sourceMtime}}},
            {"viewport", viewport.ToJson()}};
}

Project Project::FromJson(const nlohmann::json& j, const std::filesystem::path& relativeTo) {
    const int version = j.value("schema_version", 0);
    if (version < 1 || version > kProjectSchemaVersion)
        Throw("project: unsupported schema_version " + std::to_string(version) + (version > kProjectSchemaVersion ? " (made by a newer dlssvid)" : ""));
    Project p;
    p.sourceVideo = Abs(j.value("source", ""), relativeTo);
    p.passesRoot = Abs(j.value("passes", ""), relativeTo);
    p.resultVideo = Abs(j.value("result", ""), relativeTo);
    for (const auto& s : j.value("stages", nlohmann::json::array())) {
        StageEntry e;
        e.name = s.value("name", "");
        e.enabled = s.value("enabled", true);
        e.params = s.value("params", nlohmann::json::object());
        if (version < 2 && e.name == "nr") MigrateNrParamsV1(e.params);
        if (!e.name.empty()) p.stages.push_back(e);
    }
    if (p.stages.empty()) p.stages = DefaultStages();
    p.passVersionsKeep = j.value("pass_versions_keep", 2);
    if (j.contains("encode") && j["encode"].is_object()) {
        p.codec = j["encode"].value("codec", "h264_nvenc");
        if (j["encode"].contains("options") && j["encode"]["options"].is_object())
            for (const auto& [k, v] : j["encode"]["options"].items()) p.codecOptions[k] = v.is_string() ? v.get<std::string>() : v.dump();
        const nlohmann::json bd = j["encode"].value("bit_depth", nlohmann::json("auto"));
        const auto depth = bd.is_number_integer() ? std::optional<int>(bd.get<int>()) : ParseBitDepthChoice(bd.is_string() ? bd.get<std::string>() : "auto");
        if (!depth || (*depth != 0 && *depth != 8 && *depth != 10 && *depth != 12)) Throw("project: encode.bit_depth must be auto | 8 | 10 | 12");
        p.outputFormat.bitDepth = *depth;
        p.outputFormat.chroma = j["encode"].value("chroma", "auto");
        if (!ValidChromaChoice(p.outputFormat.chroma)) Throw("project: encode.chroma must be auto | 420 | 422 | 444");
    }
    if (j.contains("source_hash") && j["source_hash"].is_object()) {
        p.sourceHash = j["source_hash"].value("value", "");
        p.sourceSize = j["source_hash"].value("size", uint64_t{0});
        p.sourceMtime = j["source_hash"].value("mtime", "");
    }
    p.viewport = j.contains("viewport") ? ViewportState::FromJson(j["viewport"]) : ViewportState{};
    return p;
}

Project Project::Load(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) Throw("cannot open project " + file.string());
    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        Throw("project parse error: " + std::string(e.what()));
    }
    Project p = FromJson(j, std::filesystem::absolute(file).parent_path());
    p.file = std::filesystem::absolute(file);
    return p;
}

void Project::Save(const std::filesystem::path& target) {
    if (target.empty()) Throw("Project::Save: no file name");
    const auto abs = std::filesystem::absolute(target);
    std::filesystem::create_directories(abs.parent_path());
    std::ofstream out(abs, std::ios::binary);
    if (!out) Throw("cannot write project " + abs.string());
    out << ToJson(abs.parent_path()).dump(2) << "\n";
    file = abs;
}

std::string Project::SourceHash() {
    if (sourceVideo.empty()) return {};
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(sourceVideo, ec);
    if (ec) return {};
    const std::string mtime = FileTimeIso8601(sourceVideo);
    if (!sourceHash.empty() && size == sourceSize && mtime == sourceMtime) return sourceHash;
    sourceHash = "sha256:" + Sha256FileCached(sourceVideo);
    sourceSize = size;
    sourceMtime = mtime;
    return sourceHash;
}

std::vector<ViewportSource> Project::Sources(bool useProxies) const {
    std::vector<ViewportSource> out;
    if (auto s = FrameStore::VideoSource("source", sourceVideo)) out.push_back(*s);
    if (auto r = FrameStore::VideoSource("result", resultVideo)) out.push_back(*r);
    if (useProxies)
        for (auto& v : out)
            if (const auto proxy = ValidProxy(v.path, passesRoot, v.name)) v.proxy = *proxy;
    for (auto& p : FrameStore::DiscoverPasses(passesRoot)) {
        bool dup = false;
        for (const auto& e : out) dup = dup || e.name == p.name;
        if (!dup) out.push_back(std::move(p));
    }
    for (auto& v : FrameStore::DiscoverPassVersions(passesRoot)) out.push_back(std::move(v));  // previous versions: "<pass>@<id>"
    return out;
}

}  // namespace dlssvid
