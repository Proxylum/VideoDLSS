#pragma once

#include <string>
#include <vector>

#include "stages/StagePreset.h"
#include "viewport/Project.h"

namespace dlssvid {

// Pipeline presets over a project (PR-9): the switch and the settings of every stage card and the «Кодирование» card at once. Next to
// Project because they read and change it; the GUI, the seeding of new projects and the CLI (BuildStages) share them.

// The «Кодирование» card of a project: the bitrate only when the codec takes one (the card greys it out otherwise).
PresetEncode EncodeOf(const Project& p);
// Every stage of PipelineStageNames() with its switch and PresetParams(Pipeline), and EncodeOf. A stage the project lacks (a v3 file
// without scenes) is written as its card shows it: off, with the built-in defaults.
Preset MakePipelinePreset(const std::string& name, const Project& p);
// A checked preset onto the project. Pipeline: each stage it names by ApplyPreset(Pipeline) and its switch (a stage the project lacks
// is inserted, EnsureStage, with the preset's switch); the encoder's codec, bit depth, chroma and bitrate, the result taking the
// codec's container. A stage or the encoder the preset lacks stays as it is (ValidatePreset gave MissingStage / NoEncode). The other
// encoder options, the settings note, the folders and the viewport are never touched. A card preset (scope stage) goes to its stage
// as a card's (ApplyPreset(Stage), the switch untouched). notes: ForeignKey - keys that do not apply to the stage's backend.
void ApplyPipelinePreset(Project& p, const Preset& preset, std::vector<PresetIssue>* notes = nullptr);

struct StageDiff {
    std::string stage;
    bool enabledWas = false, enabledNow = false;
    std::vector<ParamChange> params;  // DiffStageParams(Pipeline)
    bool empty() const { return enabledWas == enabledNow && params.empty(); }
};
struct EncodeChange {
    std::string field;  // codec | bit_depth | chroma | bitrate
    std::string was, now;  // the project's spelling: "hevc_nvenc", "auto" / "10", "420", "50M" / "" (= «Авто»)
};
struct PipelineDiff {
    std::vector<StageDiff> stages;  // in run order, only the stages with a change
    std::vector<EncodeChange> encode;
    bool empty() const { return stages.empty() && encode.empty(); }
};
// What a pipeline preset changed: the stages by PipelineStageNames() (a stage a project lacks counts as its card shows it - off,
// built-in defaults) and the encoder through EncodeOf on both sides (the other options and the settings note are no part of it).
PipelineDiff DiffPipeline(const Project& before, const Project& after);
// What MakePipelinePreset leaves out (the save dialog, principle 11): StagePresetLeftOut(Pipeline) of every stage, the encoder options
// other than its bitrate and a settings note turned off (stage "encode").
std::vector<LeftOut> PipelinePresetLeftOut(const Project& p);

}  // namespace dlssvid
