#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "passes/ScenesPass.h"

namespace dlssvid {

// Stored scene-cut scores a threshold can be applied to without decoding (PR-13): the pass they came with and where it lies.
struct ScenesSource {
    Manifest manifest;
    SceneScores scores;
    std::string version;  // history id; "" for the current pass
    std::filesystem::path dir;
};

// The scores in `m` (its folder `dir`) stand for this run: the same source and decoded layout, the current algorithm, and every
// frame the run needs (needed <= 0: the analysis reached the end of the source).
bool ScoresUsable(const Manifest& m, const std::string& sourceHash, const std::string& sourceFormat, int64_t needed);
// The first usable scores of the pass `scenes` under `root`: the current one, then the history; read into memory, so they stay
// usable when the current folder is retired before the run.
std::optional<ScenesSource> FindSceneScores(const std::filesystem::path& root, const std::string& sourceHash, const std::string& sourceFormat, int64_t needed);

}  // namespace dlssvid
