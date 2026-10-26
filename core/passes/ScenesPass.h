#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "passes/Manifest.h"

namespace dlssvid {

// The scene-cut list of a clip (PR-13, spec §2d «Общий список склеек»): the stage `scenes` analyses the source once and writes a
// data pass - a manifest with the list and no frames (channels: []) plus scores.npy, the score of every frame against the one
// before it (convert/SceneCut: mean |dY|, 0 for frame 0). The temporal stages read the list: flow zeroes the vectors at a cut,
// upscale resets DLSS SR, fg holds the frame before the cut, nr resets its history.
inline constexpr const char* kScenesPass = "scenes";
inline constexpr const char* kScenesPendingId = "pending:scenes";  // a plan only: the list is analysed first; never on disk
inline constexpr const char* kScenesAlgorithm = "luma-mad.640";      // SceneCutDetector, thumbs <= 640 wide: another one needs new scores
inline constexpr const char* kScenesScoresFile = "scores.npy";

struct SceneList {
    std::vector<int64_t> cuts;  // sorted, unique, absolute source frames that start a shot (never 0)
    std::string id;             // SceneCutsId(cuts): "" when empty - no input of the stages below (spec §2d)
    double threshold = 0.0;
    int64_t frames = 0;  // frames analysed: 0..frames-1
    bool reachedEnd = false, off = false;
    std::vector<std::pair<int64_t, float>> near;  // the highest scores just below the threshold: to check for missed cuts
    bool IsCut(int64_t f) const { return std::binary_search(cuts.begin(), cuts.end(), f); }
    int64_t CountIn(int64_t first, int64_t last) const;  // cuts in [first, last]
};

// "cuts:" + sha256 hex of nlohmann::json(cuts).dump(): the list's identity, the input id of the stages that read it; "" for none
std::string SceneCutsId(const std::vector<int64_t>& cuts);
// {i >= 1 : scores[i] > threshold}; threshold <= 0: none. One rule for a fresh analysis and a rederive (the stored float32 scores),
// so a plan and its run never disagree on a score at the threshold.
std::vector<int64_t> CutsFromScores(const std::vector<float>& scores, double threshold);
// At most `max` frames with the highest score in [0.6 * threshold, threshold), highest first.
std::vector<std::pair<int64_t, float>> NearCuts(const std::vector<float>& scores, double threshold, size_t max = 5);

// The clip's list in `dir`: nullopt (the reason in `why`: "absent" when there is no manifest) when unreadable, not the pass
// `scenes`, a part of the clip, an A/B pass, of another source (sourceHash given), unsorted / repeated / out of [1, frames), or
// its cuts_id differs from the one of its list.
std::optional<SceneList> ReadSceneList(const std::filesystem::path& dir, const std::string& sourceHash, std::string* why = nullptr);
// The same, throwing unless the list's id is expectedId: the list a stage was planned with changed under the run.
SceneList ReadSceneListExpect(const std::filesystem::path& dir, const std::string& expectedId, const std::string& sourceHash);

struct SceneScores {
    std::vector<float> scores;
    nlohmann::json analysis = nlohmann::json::object();  // stage_params.analysis of the pass the scores come from
};
// The scores of the pass in `dir` (manifest `m`): nullopt when the file is missing, of another size or its sha256 differs.
std::optional<SceneScores> ReadSceneScores(const std::filesystem::path& dir, const Manifest& m, std::string* why = nullptr);

// Writes the pass: scores.npy first (when given), the manifest last - a folder without a manifest is no pass. `base` gives the size,
// fps and source; its stage_params (backend, threshold) are kept and the list's fields set.
void WriteScenesPass(const std::filesystem::path& dir, Manifest base, const SceneList& list, const SceneScores* scores);

// The list a consumer pass was made with (stage_params.scene_cuts_id); an nr pass of PR-6 reset by its own detector: "pr6"; else "".
std::string SceneLabelOf(const Manifest& consumer);

}  // namespace dlssvid
