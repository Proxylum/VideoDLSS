#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

#include "io/VideoDecoder.h"
#include "passes/ScenesPass.h"

namespace dlssvid {

// The stage `scenes` (PR-13, spec §2d «Общий список склеек»): one pass over the source's luma on the CPU (convert/SceneCut) - the
// score of every frame against the one before it, stored with the list it gives at the threshold. The threshold is applied to the
// stored scores without decoding again (RederiveScenes): a score does not depend on it. No GPU, no Pipeline.
struct ScenesStageOptions {
    double threshold = 0.24;  // schema scenes.threshold; 0: off - no cut
    std::string backend = "luma";
    std::filesystem::path outputDir;  // the passes root: the pass goes to <outputDir>/scenes
    std::string sourceFile, sourceHash, sourceFormat;
};

struct ScenesRunResult {
    SceneList list;
    std::filesystem::path outDir;
    bool rederived = false;
    double msPerFrame = 0.0;  // the analysis's own time per frame (decode + thumb + score); a rederive keeps the one of its scores
};

// Decodes from wherever the decoder stands (frame 0 for a pass of the clip) at most maxFrames frames (-1: to the end).
ScenesRunResult AnalyseScenes(VideoDecoder& decoder, const ScenesStageOptions& options, int64_t maxFrames = -1,
                              const std::function<void(int64_t)>& progress = {});
// The threshold over stored scores (`from`: the pass they come from - its size, fps and analysis are kept): no decoder.
ScenesRunResult RederiveScenes(const Manifest& from, const SceneScores& scores, const ScenesStageOptions& options);
// Threshold 0 without usable scores: an «off» pass of `knownFrames` frames (0: unknown), no analysis.
ScenesRunResult WriteOffScenes(const ScenesStageOptions& options, uint32_t width, uint32_t height, Rational fps, int64_t knownFrames);

}  // namespace dlssvid
