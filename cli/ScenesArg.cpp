#include "ScenesArg.h"

#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid::cli {

SceneList ResolveScenes(const std::string& given, bool noScenes, bool noAuto, const std::filesystem::path& outputRoot, const std::string& sourceHash,
                        bool stageReads, const char* stage, int64_t neededFrames, bool* found) {
    if (found) *found = false;
    if (!stageReads) return {};
    std::filesystem::path dir;
    if (!given.empty()) {
        dir = given;
    } else {
        if (noScenes || noAuto) return {};
        dir = outputRoot / kScenesPass;
        if (!Manifest::Exists(dir)) return {};
    }
    std::string why;
    auto l = ReadSceneList(dir, sourceHash, &why);
    if (!l) {
        if (!given.empty()) Throw(std::string(stage) + ": --scenes-dir " + dir.string() + ": " + why);
        Log()->warn("{}: {} is not used ({}): no reset at scene cuts", stage, dir.string(), why);
        return {};
    }
    if (found) *found = true;
    if (!l->off && !l->reachedEnd && (neededFrames <= 0 || l->frames < neededFrames))
        Log()->warn("{}: {} covers frames 0..{} only; cuts after it are unknown", stage, dir.string(), l->frames - 1);
    if (l->off || l->cuts.empty()) {
        Log()->info("{}: {} lists no cut{}", stage, dir.string(), l->off ? " (threshold 0: off)" : "");
        return {};
    }
    Log()->info("{}: using {} ({} cuts, threshold {:.2f})", stage, dir.string(), l->cuts.size(), l->threshold);
    return *l;
}

}  // namespace dlssvid::cli
