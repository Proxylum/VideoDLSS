#include "viewport/Project.h"

#include <fstream>

#include "util/Error.h"
#include "util/Log.h"

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
}  // namespace

std::vector<StageEntry> Project::DefaultStages() {
    return {StageEntry{"depth", true, {{"backend", "da3"}}}, StageEntry{"flow", true, {{"backend", "ofa"}}},
            StageEntry{"upscale", true, {{"backend", "rtxvsr"}, {"scale", 2}}},
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
    return {{"schema_version", 1},
            {"source", Rel(sourceVideo, relativeTo)},
            {"passes", Rel(passesRoot, relativeTo)},
            {"result", Rel(resultVideo, relativeTo)},
            {"stages", st},
            {"viewport", viewport.ToJson()}};
}

Project Project::FromJson(const nlohmann::json& j, const std::filesystem::path& relativeTo) {
    if (j.value("schema_version", 0) != 1) Throw("project: unsupported schema_version");
    Project p;
    p.sourceVideo = Abs(j.value("source", ""), relativeTo);
    p.passesRoot = Abs(j.value("passes", ""), relativeTo);
    p.resultVideo = Abs(j.value("result", ""), relativeTo);
    for (const auto& s : j.value("stages", nlohmann::json::array())) {
        StageEntry e;
        e.name = s.value("name", "");
        e.enabled = s.value("enabled", true);
        e.params = s.value("params", nlohmann::json::object());
        if (!e.name.empty()) p.stages.push_back(e);
    }
    if (p.stages.empty()) p.stages = DefaultStages();
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

std::vector<ViewportSource> Project::Sources() const {
    std::vector<ViewportSource> out;
    if (auto s = FrameStore::VideoSource("source", sourceVideo)) out.push_back(*s);
    if (auto r = FrameStore::VideoSource("result", resultVideo)) out.push_back(*r);
    for (auto& p : FrameStore::DiscoverPasses(passesRoot)) {
        bool dup = false;
        for (const auto& e : out) dup = dup || e.name == p.name;
        if (!dup) out.push_back(std::move(p));
    }
    return out;
}

}  // namespace dlssvid
