#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "pipeline/ProcessRunner.h"
#include "stages/upscale/IUpscaler.h"

namespace dlssvid {

// Estimates for the project screen (stage 9, MR D): how long a run takes and what it adds on disk. Baselines are the
// real-clip runs of docs/benchmarks.md on the reference machine (RTX 4070 Ti SUPER, 1920×800 source: depth 350,
// flow 500, upscale 420, nr 480, fg 780 ms per source frame, encode 150 ms per output frame) scaled by megapixels;
// a pass made by `process` on this machine overrides the baseline with the wall ms/frame its manifest recorded (NR: moved
// to the run's passes and model resolution, D8).
// `model`: the depth model (VDA Small / Base / Large differ several times, TASK-0039).
double BaselineMsPerFrame(const std::string& stage, const std::string& backend, double sourceMegapixels, const std::string& model = "");
// Wall ms/frame the current pass of a stage recorded (manifest stage_params.ms_per_frame, the wall time `process` writes), if any.
// nr, upscale and fg write their own timer beside it (ms_per_frame_gpu, ms_per_frame_gpu, ms_per_frame_backend); a record of these
// stages counts only beside that timer (TASK-0051).
// `tool` (StageDecision::tool): a pass made by another backend or model than the one asked for is not used - after a switch to
// VDA Large the time of Small would be shown (TASK-0039); only the fields present and non-empty in `tool` are compared.
// D8: a GPU-only time is not a wall time. An nr record counts only beside the stage's ms_per_frame_gpu: before TASK-0049 NrStage
// wrote its GPU time as ms_per_frame and `process` adopted standalone passes with it (16 against ~410 ms). A wall time below
// the GPU time next to it is not one either. nr moves the record to `params` (EffectiveStageParams: passes, model_scale) over
// `frames` (0: the record's own) with NrMovedMsPerFrame.
std::optional<double> RecordedMsPerFrame(const std::filesystem::path& passesRoot, const std::string& stage, const nlohmann::json& tool = {},
                                         const nlohmann::json& params = {}, int64_t frames = 0);
// nr: the model's GPU time of one pass at `workPixels`: the `passes 2` step of PR-2 E6 (6.4 ms at 2560x1440, Ada)
// scaled by the working pixels; 0 for the stub. Only this per-pass step is scaled. The stage's whole time is not fitted to
// megapixels: model_scale 0.5 cost 13.4 ms GPU against 15.9, which such a fit contradicts.
double NrModelMsPerPass(const std::string& backend, double workPixels);
// nr: a wall ms/frame recorded with `recordedPasses` model passes at `recordedWorkPixels`, moved to `passes` at `workPixels`.
// The model's GPU time per pass goes out and in, and each added pass also creates its feature once, spread over `frames`
// (0: not counted). Decode, pass I/O, tonemap and resolve stay as recorded. The result is never below the passes' GPU time.
double NrMovedMsPerFrame(double recordedMs, int recordedPasses, double recordedWorkPixels, int passes, double workPixels, int64_t frames,
                         const std::string& backend);
// Bytes one source frame of a stage's outputs takes on disk (the layouts the passes use: colour RGBA16F, depth F32 in
// two folders, motion vectors RG32F in two folders; frame generation writes `multiplier` frames per source frame).
uint64_t BytesPerFrame(const std::string& stage, uint32_t sourceW, uint32_t sourceH, double scale, int multiplier);

struct SourceInfo {
    uint32_t width = 0, height = 0;
    double fps = 0.0;
    int64_t frames = 0;
    bool audio = false;
    std::string codec;        // "hevc", "prores" ... (VideoStreamInfo::codecName)
    FrameDesc decoded;        // the layout the source decodes into (TASK-0034)
    FrameDesc layout;         // the stream's own layout as shown (SourceLayout): what the encoder's «auto» follows
    DisplayOrientation orientation;  // TASK-0072: how the source is shown (a phone video turned); width / height are as shown
    std::string orientationFrom;     // "display matrix" | "rotate tag" | ""
    uint32_t codedWidth = 0, codedHeight = 0;  // as stored
    int bitDepth = 8;         // of the source's own pixel format
    std::string chroma = "420";
    // TASK-0073: the proportions of a pixel in the axes of width / height (VideoStreamInfo::sar; codedSar - as stored, as ffprobe says)
    PixelAspect sar, codedSar;
    std::string sarFrom;      // "container" | "bitstream" | ""
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
    // TASK-0073: the frame the colour stages make - the result is it stretched to square pixels for an anamorphic source (the same else)
    uint32_t gridWidth = 0, gridHeight = 0;
    double fps = 0.0;                // result rate
    bool audio = false;
    double seconds = 0.0;            // stages that run + the encode
    double fullSeconds = 0.0;        // if every stage ran
    uint64_t newBytes = 0;           // passes that will be written
    int stagesToRun = 0, stagesTotal = 0;  // the encode counts as a stage
    std::vector<StageEstimate> stages;
    std::vector<std::string> reused;  // stages the plan reuses (or restores)
    std::string codec;                // the result's encoder and layout (filled by the caller from the project: AppModel::refreshPlan)
    FrameDesc format;
};

// The frame the upscale stage makes of a source of that size with these (effective) parameters - as UpscaleStage::Setup: the source's with
// artifact_reduction_only, else the scale or target_width / target_height, even, capped at 3840x2160 by the long side (TASK-0072;
// `capByLongSide = false`: by the axes, as before); `sar` - the source's pixels (TASK-0073: the target and the cap of the frame as shown).
UpscaleTarget UpscaleOutputFor(uint32_t sourceW, uint32_t sourceH, const nlohmann::json& params, bool capByLongSide = true, PixelAspect sar = {});
// TASK-0073: the frame `process` writes of a colour pass w x h of pixels `sar`: stretched to square pixels (PixelAspect::DisplaySize), fit
// into 3840x2160 by the long side when the stretch takes it beyond (an anamorphic 4K grid with no upscale to cap it: NVENC's H.264 stops at
// 4096); square pixels - w x h as is.
std::pair<uint32_t, uint32_t> StretchedResultSize(PixelAspect sar, uint32_t w, uint32_t h);

// Combines the plan, the enabled stages' parameters and the source: resolution and rate after upscale / frame
// generation, time and disk of what runs, time of a full run.
RunOutcome EstimateRun(const ProcessPlan& plan, const std::vector<ProcessStage>& stages, const SourceInfo& source);

}  // namespace dlssvid
