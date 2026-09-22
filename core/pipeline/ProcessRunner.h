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
// then the last colour pass (color_fg > color_nr > color_sr) is encoded into `output` with the source's audio copied.
// Stage parameters are the JSON the project file and the GUI hold.
//
// Stage 9 (MR A): a pass is reused only when its fingerprint (PassFingerprint.h: source, parameters, input passes,
// tool) equals the one the run needs; a pass that would be replaced by another version is moved aside first
// (PassVersions.h) and a matching version found in the history is restored instead of recomputed. `PlanProcess`
// makes exactly these decisions without running anything.
struct ProcessStage {
    std::string name;  // depth | flow | upscale | nr | fg
    bool enabled = true;
    nlohmann::json params = nlohmann::json::object();
};

struct ProcessOptions {
    std::filesystem::path input, output;
    std::filesystem::path passesRoot;  // default: <output stem>_passes next to the output
    std::vector<ProcessStage> stages;  // any order; run in the canonical one
    bool skipExisting = true;          // false: every enabled stage runs (a version with the same fingerprint is overwritten in place)
    bool disableUnavailable = false;   // nr / fg without DLL or GPU: skip with a message instead of failing
    std::string codec = "h264_nvenc";
    std::map<std::string, std::string> codecOptions;
    int64_t frames = -1;               // at most N source frames
    bool warp = false;
    std::string hwaccel = "none";      // source decode for the encode / passthrough step
    bool passthrough = false;          // stage 0 behaviour: decode -> GPU -> encode, no stages
    bool gpuRoundtrip = true;          // passthrough: upload / readback each frame
    int keepVersions = 2;              // pass versions kept per pass after the run, the current one included; 0 = keep all
    std::function<void(const std::string& stage, int64_t done, int64_t total)> progress;
};

struct ProcessStageReport {
    std::string name;
    std::string status;  // ran | reused | disabled | passthrough
    std::string reason;  // why it ran or was reused ("params_changed: intensity: 1 -> 1.4"), or why it is disabled
    std::filesystem::path outDir;
    int64_t frames = 0;
    double seconds = 0.0;
    double msPerFrame = 0.0;
    nlohmann::json details = nlohmann::json::object();
    std::string fingerprint;  // of the pass produced or reused (PassFingerprint.h)
    std::string retired;      // history id the previous version of the pass was moved to
    std::string restored;     // history id the current version was taken from instead of recomputing
    nlohmann::json decision = nlohmann::json::object();  // StageDecision::ToJson() the run acted on
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

// What RunProcess will do for one stage, decided from the passes on disk without running anything.
struct StageDecision {
    std::string name;
    std::string action;  // run | reuse | restore
    std::string kind;    // exact | legacy | missing | incomplete | params_changed | input_changed | source_changed | tool_changed | forced | restored | changed
    std::string detail;  // human-readable: "intensity: 1 -> 1.4", "input changed: color_nr", ...
    nlohmann::json diff = nlohmann::json::object();  // {"params": {key: [old, new]}, "inputs": [pass...], "tool": {field: [old, new]}}
    std::string fingerprint;                         // the pass this run needs ("sha256:<hex>")
    bool current = false;                            // a pass folder with a manifest is on disk
    std::string currentFingerprint;                  // its fingerprint (empty for a legacy pass)
    std::string version;                             // restore: the history id that becomes current
    std::vector<std::string> outputs;                // pass folders the stage writes (depth: depth_raw, depth_dlss)
    std::filesystem::path outDir;                    // the last of them, the one checked for completeness
    int64_t frames = -1;                             // frames the pass needs (-1: every frame of a source of unknown length)
    bool retire = false;                             // the folder on disk goes to the history before the run
    nlohmann::json params = nlohmann::json::object();   // canonical parameters the fingerprint hashed
    std::map<std::string, std::string> inputs;          // input pass -> fingerprint the fingerprint hashed
    nlohmann::json tool = nlohmann::json::object();     // ToolInfo::ToJson()
    nlohmann::json ToJson() const;
};

struct ProcessPlan {
    std::filesystem::path passesRoot;
    std::string sourceHash;
    int64_t frames = -1;                // source frames the run covers (-1: unknown)
    std::vector<StageDecision> stages;  // enabled stages in run order
    std::string finalPass;              // colour pass the encode step would use ("" -> the source is passed through)
    int runCount = 0, reuseCount = 0;   // reuseCount includes restores
    nlohmann::json ToJson() const;
};

// Decides every enabled stage the way RunProcess would, from the manifests under the passes root; hashes the source
// but writes nothing (the passes root is not even created).
ProcessPlan PlanProcess(const ProcessOptions& options);
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
