#include "stages/scenes/ScenesStage.h"

#include <chrono>
#include <cmath>

#include "convert/ColorConvert.h"
#include "convert/SceneCut.h"
#include "util/Log.h"

namespace dlssvid {
namespace {

Manifest Base(const ScenesStageOptions& o, uint32_t width, uint32_t height, Rational fps) {
    Manifest m;
    m.width = std::max(1u, width);  // a manifest needs a size (Manifest::FromJson): the source's
    m.height = std::max(1u, height);
    m.fps = fps;
    m.sourceFile = o.sourceFile;
    m.sourceHash = o.sourceHash;
    m.sourceFormat = o.sourceFormat;
    m.stageParams = {{"backend", o.backend}, {"threshold", o.threshold}};
    return m;
}

void LogCuts(const SceneList& l, const std::vector<float>& scores) {
    for (const int64_t c : l.cuts)
        Log()->info("scenes: cut at frame {} (score {:.2f})", c, c < static_cast<int64_t>(scores.size()) ? scores[static_cast<size_t>(c)] : 0.f);
}

}  // namespace

ScenesRunResult AnalyseScenes(VideoDecoder& decoder, const ScenesStageOptions& o, int64_t maxFrames, const std::function<void(int64_t)>& progress) {
    const auto& info = decoder.Info();
    SceneCutDetector det(ColorInfoFromStream(info));
    SceneScores s;
    CpuFrame frame;
    bool reachedEnd = false;
    const auto t0 = std::chrono::steady_clock::now();
    while (maxFrames < 0 || static_cast<int64_t>(s.scores.size()) < maxFrames) {
        if (!decoder.NextFrame(frame)) {
            reachedEnd = true;
            break;
        }
        s.scores.push_back(static_cast<float>(det.Score(frame)));
        if (progress) progress(static_cast<int64_t>(s.scores.size()));
    }
    const double ms = s.scores.empty() ? 0.0 : std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / static_cast<double>(s.scores.size());
    ScenesRunResult r;
    r.msPerFrame = ms;
    r.list.threshold = o.threshold;
    r.list.frames = static_cast<int64_t>(s.scores.size());
    r.list.reachedEnd = reachedEnd;
    r.list.off = o.threshold <= 0.0;
    r.list.cuts = CutsFromScores(s.scores, o.threshold);
    r.list.id = SceneCutsId(r.list.cuts);
    r.list.nearMisses = NearCuts(s.scores, o.threshold);
    s.analysis = {{"algorithm", kScenesAlgorithm}, {"max_width", 640}, {"ms_per_frame", std::round(ms * 100.0) / 100.0}, {"source_format", o.sourceFormat}};
    r.outDir = o.outputDir / kScenesPass;
    WriteScenesPass(r.outDir, Base(o, info.width, info.height, info.frameRate), r.list, &s);
    LogCuts(r.list, s.scores);
    Log()->info("scenes: {} frames, threshold {:.2f}: {} cut(s) ({:.1f} ms/frame)", r.list.frames, o.threshold, r.list.cuts.size(), ms);
    return r;
}

ScenesRunResult RederiveScenes(const Manifest& from, const SceneScores& scores, const ScenesStageOptions& o) {
    ScenesRunResult r;
    r.rederived = true;
    r.list.threshold = o.threshold;
    r.list.frames = static_cast<int64_t>(scores.scores.size());
    r.list.reachedEnd = scores.analysis.value("reached_end", false);
    r.list.off = o.threshold <= 0.0;
    r.list.cuts = CutsFromScores(scores.scores, o.threshold);
    r.list.id = SceneCutsId(r.list.cuts);
    r.list.nearMisses = NearCuts(scores.scores, o.threshold);
    r.msPerFrame = scores.analysis.value("ms_per_frame", 0.0);  // the decoding these scores cost: what an estimate of the stage needs
    r.outDir = o.outputDir / kScenesPass;
    Manifest base = Base(o, from.width, from.height, from.fps);
    base.stageParams["rederived"] = true;
    WriteScenesPass(r.outDir, base, r.list, &scores);
    LogCuts(r.list, scores.scores);
    Log()->info("scenes: threshold {:.2f} over the stored scores of {} frames (no decoding): {} cut(s)", o.threshold, r.list.frames, r.list.cuts.size());
    return r;
}

ScenesRunResult WriteOffScenes(const ScenesStageOptions& o, uint32_t width, uint32_t height, Rational fps, int64_t knownFrames) {
    ScenesRunResult r;
    r.list.threshold = 0.0;
    r.list.off = true;
    r.list.frames = std::max<int64_t>(knownFrames, 0);
    r.outDir = o.outputDir / kScenesPass;
    WriteScenesPass(r.outDir, Base(o, width, height, fps), r.list, nullptr);
    Log()->info("scenes: threshold 0 - off: no cut, no analysis");
    return r;
}

}  // namespace dlssvid
