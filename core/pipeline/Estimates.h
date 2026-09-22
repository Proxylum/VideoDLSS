#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "pipeline/ProcessRunner.h"

namespace dlssvid {

// Estimates for the project screen (stage 9, MR D): how long a run takes and what it adds on disk. Baselines are the
// real-clip runs of docs/benchmarks.md on the reference machine (RTX 4070 Ti SUPER, 1920×800 source: depth 350,
// flow 500, upscale 420, nr 480, fg 780 ms per source frame, encode 150 ms per output frame) scaled by megapixels;
// a pass on disk that was made on this machine overrides the baseline with the ms/frame its manifest recorded.
double BaselineMsPerFrame(const std::string& stage, const std::string& backend, double sourceMegapixels);
// ms/frame the current pass of a stage recorded (manifest stage_params.ms_per_frame), if any.
std::optional<double> RecordedMsPerFrame(const std::filesystem::path& passesRoot, const std::string& stage);
// Bytes one source frame of a stage's outputs takes on disk (the layouts the passes use: colour RGBA16F, depth F32 in
// two folders, motion vectors RG32F in two folders; frame generation writes `multiplier` frames per source frame).
uint64_t BytesPerFrame(const std::string& stage, uint32_t sourceW, uint32_t sourceH, double scale, int multiplier);

struct SourceInfo {
    uint32_t width = 0, height = 0;
    double fps = 0.0;
    int64_t frames = 0;
    bool audio = false;
};

struct StageEstimate {
    std::string stage;
    std::string action;        // run | reuse | restore (from the plan)
    double msPerFrame = 0.0;
    int64_t frames = 0;        // frames the stage writes
    double seconds = 0.0;      // 0 when reused
    uint64_t bytes = 0;        // what it writes (0 when reused)
    bool recorded = false;     // ms/frame from a pass made on this machine
};

struct RunOutcome {
    uint32_t width = 0, height = 0;  // result frame
    double fps = 0.0;                // result rate
    bool audio = false;
    double seconds = 0.0;            // stages that run + the encode
    double fullSeconds = 0.0;        // if every stage ran
    uint64_t newBytes = 0;           // passes that will be written
    int stagesToRun = 0, stagesTotal = 0;  // the encode counts as a stage
    std::vector<StageEstimate> stages;
    std::vector<std::string> reused;  // stages the plan reuses (or restores)
};

// Combines the plan, the enabled stages' parameters and the source: resolution and rate after upscale / frame
// generation, time and disk of what runs, time of a full run.
RunOutcome EstimateRun(const ProcessPlan& plan, const std::vector<ProcessStage>& stages, const SourceInfo& source);

}  // namespace dlssvid
