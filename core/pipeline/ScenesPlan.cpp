#include "pipeline/ScenesPlan.h"

#include "pipeline/PassVersions.h"

namespace dlssvid {

bool ScoresUsable(const Manifest& m, const std::string& sourceHash, const std::string& sourceFormat, int64_t needed) {
    if (m.pass != kScenesPass || !m.IsDataPass() || m.firstFrame != 0) return false;
    if (!sourceHash.empty() && m.sourceHash != sourceHash) return false;
    const nlohmann::json& sp = m.stageParams;
    if (!sp.is_object() || !sp.contains("analysis") || !sp["analysis"].is_object()) return false;
    const nlohmann::json& a = sp["analysis"];
    if (a.value("algorithm", std::string()) != kScenesAlgorithm) return false;
    if (a.value("source_format", std::string()) != sourceFormat) return false;
    const int64_t count = a.value("scores_count", int64_t{-1});
    if (count <= 0) return false;
    return needed > 0 ? count >= needed : a.value("reached_end", false);
}

std::optional<ScenesSource> FindSceneScores(const std::filesystem::path& root, const std::string& sourceHash, const std::string& sourceFormat, int64_t needed) {
    for (const auto& v : ListPassVersions(root, kScenesPass)) {
        try {
            Manifest m = Manifest::Load(v.dir);
            if (!ScoresUsable(m, sourceHash, sourceFormat, needed)) continue;
            auto scores = ReadSceneScores(v.dir, m);
            if (!scores) continue;
            ScenesSource s;
            s.manifest = std::move(m);
            s.scores = std::move(*scores);
            s.version = v.current ? std::string() : v.id;
            s.dir = v.dir;
            return s;
        } catch (const std::exception&) {
            continue;
        }
    }
    return std::nullopt;
}

}  // namespace dlssvid
