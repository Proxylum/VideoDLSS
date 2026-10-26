#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "passes/ScenesPass.h"

namespace dlssvid::cli {

// The scene-cut list a standalone stage command uses (PR-13, the pass of `dlssvid scenes` / `process`): none unless the stage reads
// one (`stageReads`: StageReadsSceneCuts). `given` (--scenes-dir) must be that pass of this source - anything else is an error.
// Else, unless --no-scenes or --no-auto, <outputRoot>/scenes when there: of another source or unreadable it is a warning and no
// list. An empty or «off» list is none (no label). One shorter than the run (neededFrames > 0, or the end not reached) warns: the
// cuts after it are unknown. `found`: a list of this source was read (an empty one too) - the run used it (nr --json scenes_used).
SceneList ResolveScenes(const std::string& given, bool noScenes, bool noAuto, const std::filesystem::path& outputRoot, const std::string& sourceHash,
                        bool stageReads, const char* stage, int64_t neededFrames = 0, bool* found = nullptr);

}  // namespace dlssvid::cli
