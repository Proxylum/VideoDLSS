#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "io/VideoDecoder.h"

namespace dlssvid {

struct Project;

// The whole pipeline as one command (stage 8, ТЗ §4 / §7 / §9): the stages run one after another over the disk
// cache of passes under `passesRoot` (depth -> flow -> upscale -> nr -> fg; each reads what the previous ones wrote),
// a pass that is already complete is reused (`skipExisting`: a rerun with other NR / FG parameters does not
// recompute depth or motion vectors), then the last colour pass (color_fg > color_nr > color_sr) is encoded into
// `output` with the source's audio copied. Stage parameters are the JSON the project file and the GUI hold.
struct ProcessStage {
    std::string name;  // depth | flow | upscale | nr | fg
    bool enabled = true;
    nlohmann::json params = nlohmann::json::object();
};

struct ProcessOptions {
    std::filesystem::path input, output;
    std::filesystem::path passesRoot;  // default: <output stem>_passes next to the output
    std::vector<ProcessStage> stages;  // any order; run in the canonical one
    bool skipExisting = true;
    bool disableUnavailable = false;   // nr / fg without DLL or GPU: skip with a message instead of failing
    std::string codec = "h264_nvenc";
    std::map<std::string, std::string> codecOptions;
    int64_t frames = -1;               // at most N source frames
    bool warp = false;
    std::string hwaccel = "none";      // source decode for the encode / passthrough step
    bool passthrough = false;          // stage 0 behaviour: decode -> GPU -> encode, no stages
    bool gpuRoundtrip = true;          // passthrough: upload / readback each frame
    std::function<void(const std::string& stage, int64_t done, int64_t total)> progress;
};

struct ProcessStageReport {
    std::string name;
    std::string status;  // ran | reused | disabled | passthrough
    std::string reason;
    std::filesystem::path outDir;
    int64_t frames = 0;
    double seconds = 0.0;
    double msPerFrame = 0.0;
    nlohmann::json details = nlohmann::json::object();
    nlohmann::json ToJson() const;
};

struct ProcessResult {
    std::vector<ProcessStageReport> stages;  // stages in run order, then "encode"
    std::filesystem::path finalPass;         // colour pass that was encoded (empty for passthrough)
    std::filesystem::path output;
    int64_t framesIn = 0, framesOut = 0;
    Rational outputFps{0, 1};
    bool audio = false;
    double seconds = 0.0;
    nlohmann::json ToJson() const;
};

ProcessResult RunProcess(const ProcessOptions& options);

// Canonical stage order and defaults (the same as Project::DefaultStages).
const std::vector<std::string>& ProcessStageOrder();
std::vector<ProcessStage> DefaultProcessStages();
std::vector<ProcessStage> StagesFromProject(const Project& project);
// "stage.key=value" -> params (value parsed as JSON when possible, else a string); throws on a bad spec.
void ApplyParamSpec(std::vector<ProcessStage>& stages, const std::string& spec);

// A pass folder is complete when its manifest lists at least `neededFrames` frames and the last one exists. With an
// unknown source length (neededFrames <= 0: containers without a frame count) a pass counts as complete when it was
// written for the same source (`sourceHash`) and its last listed frame exists.
bool PassComplete(const std::filesystem::path& dir, int64_t neededFrames, const std::string& sourceHash = {});

// Encodes a colour pass (RGB frames, sRGB) into a video at the pass fps, copying the source's audio; pass frames
// are paced against the source frames (mult = round(pass fps / source fps)) so audio stays interleaved.
struct EncodeReport {
    int64_t frames = 0;
    int64_t sourceFrames = 0;  // source frames decoded while encoding (audio pacing)
    Rational fps{0, 1};
    double seconds = 0.0;
    bool audio = false;
};
EncodeReport EncodePassToVideo(const std::filesystem::path& passDir, const std::filesystem::path& source, const std::filesystem::path& output, const std::string& codec,
                               const std::map<std::string, std::string>& codecOptions = {}, int64_t maxSourceFrames = -1,
                               const std::function<void(int64_t done, int64_t total)>& progress = {}, const std::string& hwaccel = "none");

}  // namespace dlssvid
