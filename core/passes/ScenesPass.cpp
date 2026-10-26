#include "passes/ScenesPass.h"

#include <cmath>

#include "passes/formats/NpzIO.h"
#include "util/Error.h"
#include "util/Sha256.h"

namespace dlssvid {

int64_t SceneList::CountIn(int64_t first, int64_t last) const {
    return std::count_if(cuts.begin(), cuts.end(), [&](int64_t c) { return c >= first && c <= last; });
}

std::string SceneCutsId(const std::vector<int64_t>& cuts) {
    if (cuts.empty()) return {};
    const std::string text = nlohmann::json(cuts).dump();
    return "cuts:" + Sha256Hex(text.data(), text.size());
}

std::vector<int64_t> CutsFromScores(const std::vector<float>& scores, double threshold) {
    std::vector<int64_t> cuts;
    if (threshold <= 0.0) return cuts;
    for (size_t i = 1; i < scores.size(); ++i)
        if (static_cast<double>(scores[i]) > threshold) cuts.push_back(static_cast<int64_t>(i));
    return cuts;
}

std::vector<std::pair<int64_t, float>> NearCuts(const std::vector<float>& scores, double threshold, size_t max) {
    std::vector<std::pair<int64_t, float>> near;
    if (threshold <= 0.0) return near;
    for (size_t i = 1; i < scores.size(); ++i) {
        const double s = scores[i];
        if (s >= 0.6 * threshold && s <= threshold) near.push_back({static_cast<int64_t>(i), scores[i]});  // <= : not a cut (strict >)
    }
    std::stable_sort(near.begin(), near.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    if (near.size() > max) near.resize(max);
    return near;
}

std::optional<SceneList> ReadSceneList(const std::filesystem::path& dir, const std::string& sourceHash, std::string* why) {
    auto fail = [&](std::string reason) -> std::optional<SceneList> {
        if (why) *why = std::move(reason);
        return std::nullopt;
    };
    if (!Manifest::Exists(dir)) return fail("absent");
    Manifest m;
    try {
        m = Manifest::Load(dir);
    } catch (const std::exception& e) {
        return fail(std::string("unreadable: ") + e.what());
    }
    if (m.pass != kScenesPass || !m.channels.empty()) return fail("not a scenes pass");
    if (m.firstFrame != 0) return fail("a part of the clip (frames from " + std::to_string(m.firstFrame) + ")");
    const nlohmann::json& sp = m.stageParams;
    if (!sp.is_object()) return fail("no stage_params");
    if (sp.contains("ab")) return fail("an A/B pass");
    if (!sourceHash.empty() && m.sourceHash != sourceHash) return fail("of another source");
    SceneList l;
    l.frames = m.frameCount;
    l.threshold = sp.value("threshold", 0.0);
    l.off = sp.value("off", false);
    l.reachedEnd = sp.contains("analysis") && sp["analysis"].is_object() && sp["analysis"].value("reached_end", false);
    if (sp.contains("scene_cuts")) {
        if (!sp["scene_cuts"].is_array()) return fail("scene_cuts is not a list");
        for (const auto& v : sp["scene_cuts"]) {
            if (!v.is_number_integer()) return fail("a cut that is not a frame index");
            const int64_t c = v.get<int64_t>();
            if (c < 1 || c >= l.frames) return fail("cut " + std::to_string(c) + " outside frames 1.." + std::to_string(l.frames - 1));
            if (!l.cuts.empty() && c <= l.cuts.back()) return fail("cuts unsorted or repeated");
            l.cuts.push_back(c);
        }
    }
    l.id = SceneCutsId(l.cuts);
    if (sp.value("cuts_id", std::string()) != l.id) return fail("cuts_id does not match its list");
    if (sp.contains("near") && sp["near"].is_array())
        for (const auto& n : sp["near"])
            if (n.is_array() && n.size() == 2 && n[0].is_number_integer() && n[1].is_number()) l.near.push_back({n[0].get<int64_t>(), n[1].get<float>()});
    return l;
}

SceneList ReadSceneListExpect(const std::filesystem::path& dir, const std::string& expectedId, const std::string& sourceHash) {
    std::string why;
    const auto l = ReadSceneList(dir, sourceHash, &why);
    if (!l) Throw("scenes: " + dir.string() + ": " + why);
    if (l->id != expectedId) Throw("scenes: the list in " + dir.string() + " changed (" + (l->id.empty() ? std::string("none") : l->id) + ", planned " + expectedId + ")");
    return *l;
}

std::optional<SceneScores> ReadSceneScores(const std::filesystem::path& dir, const Manifest& m, std::string* why) {
    auto fail = [&](std::string reason) -> std::optional<SceneScores> {
        if (why) *why = std::move(reason);
        return std::nullopt;
    };
    const nlohmann::json& sp = m.stageParams;
    if (!sp.is_object() || !sp.contains("analysis") || !sp["analysis"].is_object()) return fail("no analysis");
    const nlohmann::json& a = sp["analysis"];
    const std::string file = a.value("scores_file", std::string());
    const int64_t count = a.value("scores_count", int64_t{-1});
    if (file.empty() || count < 0) return fail("no scores");
    const auto path = dir / file;
    if (!std::filesystem::exists(path)) return fail(file + " missing");
    if (Sha256File(path) != a.value("scores_sha256", std::string())) return fail(file + " changed (sha256)");
    SceneScores s;
    s.analysis = a;
    if (count == 0) return s;
    try {
        const PassImage img = ReadNpy(path);
        if (img.type != PixelType::F32 || img.ChannelCount() != 1 || static_cast<int64_t>(img.width) * img.height != count) return fail(file + " of another size");
        const float* v = img.As<float>();
        s.scores.assign(v, v + count);
    } catch (const std::exception& e) {
        return fail(file + ": " + e.what());
    }
    return s;
}

void WriteScenesPass(const std::filesystem::path& dir, Manifest base, const SceneList& list, const SceneScores* scores) {
    std::filesystem::create_directories(dir);
    const auto scoresPath = dir / kScenesScoresFile;
    nlohmann::json analysis = scores ? scores->analysis : nlohmann::json::object();
    if (!analysis.is_object()) analysis = nlohmann::json::object();
    std::error_code ec;
    std::filesystem::remove(scoresPath, ec);  // a folder written again: no scores of another analysis left beside the new list
    if (scores) {
        const int64_t n = static_cast<int64_t>(scores->scores.size());
        if (n > 0) {
            PassImage img;
            img.Allocate(static_cast<uint32_t>(n), 1, PixelType::F32, {"score"});
            std::copy(scores->scores.begin(), scores->scores.end(), img.As<float>());
            WriteNpy(scoresPath, img);
            analysis["scores_sha256"] = Sha256File(scoresPath);
        } else {
            analysis.erase("scores_sha256");
        }
        analysis["scores_file"] = kScenesScoresFile;
        analysis["scores_count"] = n;
    }
    Manifest m = std::move(base);
    m.pass = kScenesPass;
    m.channels.clear();  // a data pass: the manifest is the content (Manifest::IsDataPass)
    m.frameCount = list.frames;
    m.firstFrame = 0;
    m.lastFrame = list.frames - 1;
    if (!m.stageParams.is_object()) m.stageParams = nlohmann::json::object();
    m.stageParams["threshold"] = list.threshold;
    m.stageParams["off"] = list.off;
    m.stageParams["scene_cuts"] = list.cuts;
    m.stageParams["scene_cut_count"] = static_cast<int64_t>(list.cuts.size());
    m.stageParams["cuts_id"] = SceneCutsId(list.cuts);
    nlohmann::json near = nlohmann::json::array();
    for (const auto& [f, s] : list.near) near.push_back({f, std::round(static_cast<double>(s) * 1e4) / 1e4});
    m.stageParams["near"] = near;
    analysis["reached_end"] = list.reachedEnd;
    analysis["frames"] = list.frames;
    m.stageParams["analysis"] = analysis;
    m.Save(dir);
}

std::string SceneLabelOf(const Manifest& consumer) {
    const nlohmann::json& sp = consumer.stageParams;
    if (!sp.is_object()) return {};
    if (sp.contains("scene_cuts_id") && sp["scene_cuts_id"].is_string()) return sp["scene_cuts_id"].get<std::string>();
    // PR-6: nr reset its history by its own detector - a run no list describes
    if (consumer.pass == "color_nr" && sp.value("scene_cut_count", int64_t{0}) > 0) return "pr6";
    return {};
}

}  // namespace dlssvid
